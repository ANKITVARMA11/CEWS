"""The OpenAI-compatible chat client every provider is reached through.

One HTTP shape, three concerns kept explicit:

* **No secret ever appears in an error or a log line.** The API key is redacted before any
  exception message or log record is built, the same discipline the ingestion adapters use for
  source API keys.
* **A request never hangs forever.** A timeout and a small retry budget are always in force;
  network and server errors are retried, a bad request is not.
* **Local and hosted are the same code path.** ``LLM_PROVIDER=ollama`` and
  ``LLM_PROVIDER=openai_compatible`` differ only in base URL and whether a key is sent.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from cews.constants import LLMProvider
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0  # extraction wants the same answer twice, not creativity
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Raised when a call to the configured LLM provider cannot be completed.

    The message never contains the API key: it is redacted before this is raised, regardless of
    which layer (network, HTTP status, or response parsing) produced the failure.
    """


@dataclass(frozen=True)
class LLMResponse:
    """What came back from the model."""

    text: str
    model: str
    provider: str
    raw: dict[str, Any]

    def json(self) -> Any:
        """Parse the response text as JSON.

        Raises:
            LLMError: if the text is not valid JSON.
        """
        try:
            return json.loads(self.text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"response was not valid JSON: {exc}") from exc


@dataclass(frozen=True)
class ToolCall:
    """One request from the model to run a tool.

    ``arguments`` is already parsed from the JSON string the model sent, so a caller never has
    to touch ``json.loads`` itself. If the model sent arguments that are not valid JSON, parsing
    fails loudly at the point of the call (see :class:`LLMError` below) rather than silently
    becoming an empty dict, which would hide a real problem as a tool called with no arguments.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    """What came back from one turn of a tool-calling conversation.

    Exactly one of ``text`` and ``tool_calls`` is meaningful at a time, the same way the
    OpenAI-style chat-completions API itself distinguishes them: a model either answers in words,
    or asks to run one or more tools before it can answer. ``tool_calls`` is empty when the model
    just answered; it is non-empty when the model wants tools run first, in which case ``text``
    is usually empty too.
    """

    text: str
    tool_calls: tuple[ToolCall, ...]
    model: str
    provider: str
    raw: dict[str, Any]

    @property
    def wants_tools(self) -> bool:
        """True when the model asked to run tools rather than answering directly."""
        return len(self.tool_calls) > 0


def llm_enabled(settings: Settings) -> bool:
    """Whether any code should attempt an LLM call at all.

    Every caller checks this first. ``LLM_PROVIDER=none`` (the default) means CEWS never imports
    an HTTP client for this purpose and every feature that could use one runs its deterministic
    fallback instead.
    """
    return settings.llm_provider is not LLMProvider.NONE


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED = re.compile(r"<think>.*", re.IGNORECASE | re.DOTALL)
NO_VISIBLE_ANSWER = (
    "The model's response was entirely internal reasoning, with no answer outside it "
    "(it may have run out of room to finish - try a shorter question, or raise LLM_MAX_TOKENS)."
)


def _strip_reasoning(text: str) -> str:
    """Remove a "reasoning" model's chain-of-thought from its visible answer.

    Some models - reasoning-tuned ones served through providers such as NVIDIA NIM are the
    common case here - write their scratch thinking directly into the message content, wrapped
    in ``<think>...</think>``, rather than returning it in a separate field the way a few APIs
    do. Left in place, that scratchpad ("We need to answer... Thus the answer is...") is what
    ends up shown to the person as if it were the answer itself - confirmed against a real reply
    from a real model, not a hypothetical.

    An unclosed ``<think>`` (the model ran out of tokens mid-thought, before ever writing a real
    answer) has everything from that point on removed too, on the view that a stray fragment of
    reasoning is worse to show than nothing; :data:`NO_VISIBLE_ANSWER` is returned in that case if
    nothing else is left.
    """
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or NO_VISIBLE_ANSWER


def _redact(text: str, secret: str | None) -> str:
    """Remove a secret from a string before it can reach a log or an exception."""
    if not secret:
        return text
    return text.replace(secret, "***")


class LLMClient:
    """A chat-completions client for whichever provider ``settings`` names.

    Construct one per call site rather than sharing a global instance; it is cheap (no
    connection is opened until a request is made) and keeps the settings it was built from
    explicit.
    """

    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None) -> None:
        """
        Raises:
            LLMError: if ``settings.llm_provider`` is ``none`` — construct one only after
                checking :func:`llm_enabled`.
        """
        if not llm_enabled(settings):
            raise LLMError("LLM_PROVIDER is 'none'; check llm_enabled(settings) before this call")
        self._provider = settings.llm_provider
        self._model = settings.llm_model
        self._api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
        self._max_retries = settings.llm_max_retries
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        self._client = httpx.Client(
            base_url=settings.llm_base_url,
            headers=headers,
            timeout=settings.llm_timeout_seconds,
            transport=transport,
        )

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying HTTP connection."""
        self._client.close()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> LLMResponse:
        """Send one chat-completion request and return the model's reply.

        Args:
            prompt: the user message.
            system: an optional system message (task instructions, output format).
            json_schema: when given, asks the server to constrain output to this JSON Schema.
                Supported by Ollama and by OpenAI-compatible servers that implement structured
                outputs; servers that do not understand the field ignore it, so this is safe to
                pass everywhere, but the response should still be validated by the caller.
            max_tokens: the reply length cap.
            temperature: sampling temperature; 0 for deterministic extraction.

        Raises:
            LLMError: for a network failure, a non-2xx response, or an unreadable response body.
                The message never contains the API key.
        """
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": json_schema, "strict": True},
            }

        body = self._request_with_retries(payload)
        try:
            choice = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {self._provider}: {exc}") from exc
        return LLMResponse(
            text=_strip_reasoning(choice),
            model=self._model,
            provider=self._provider.value,
            raw=body,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> ChatResponse:
        """One turn of a multi-turn, tool-calling conversation.

        Unlike :meth:`complete`, the caller builds and keeps the whole ``messages`` list itself
        (system prompt, prior user/assistant turns, and any tool results), because a tool-calling
        conversation is stateful across several calls to this method — one per turn of the
        LLM-decides / tool-runs / LLM-answers loop an agent drives (see
        :mod:`cews.agents.core`). ``complete`` stays single-shot and untouched by this method;
        existing callers (announcement extraction) are unaffected.

        Args:
            messages: the full conversation so far, each item ``{"role": ..., "content": ...}``,
                plus (for a tool result) ``{"role": "tool", "tool_call_id": ..., "content": ...}``.
            tools: tool definitions in the OpenAI function-calling shape — see
                :func:`cews.agents.tool_schema.mcp_tools_to_openai_functions`, which builds this
                list from an MCP server's own tool listing so the two are never hand-kept in sync.
            max_tokens, temperature: same meaning as in :meth:`complete`.

        Raises:
            LLMError: for a network failure, a non-2xx response, an unreadable response body, or
                (new here) a tool call whose arguments are not valid JSON. The message never
                contains the API key.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools

        body = self._request_with_retries(payload)
        try:
            message = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape from {self._provider}: {exc}") from exc

        raw_calls = message.get("tool_calls") or []
        tool_calls = tuple(self._parse_tool_call(call) for call in raw_calls)
        content = message.get("content") or ""
        # A reasoning model's <think> block is stripped only from a final answer - what a person
        # actually sees. When tool_calls are present, run_agent_turn feeds this same text back to
        # the model as its OWN prior turn on the next call (the ordinary tool-calling protocol
        # requires the assistant's earlier message to be preserved), and that round-trip is left
        # untouched: it is an internal detail between the model and itself, not something a
        # person reads, and a model may expect to see its own reasoning intact there.
        text = content if tool_calls else _strip_reasoning(content)
        return ChatResponse(
            text=text,
            tool_calls=tool_calls,
            model=self._model,
            provider=self._provider.value,
            raw=body,
        )

    def _parse_tool_call(self, call: dict[str, Any]) -> ToolCall:
        """Turn one OpenAI-shaped ``tool_calls`` entry into a :class:`ToolCall`.

        Raises:
            LLMError: if the entry is missing a piece, or its arguments are not valid JSON — a
                model that returns malformed tool-call arguments is a real problem to surface,
                not to paper over with an empty dict.
        """
        try:
            function = call["function"]
            arguments_text = function["arguments"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"{self._provider} sent a malformed tool call: {call!r}") from exc
        try:
            arguments = json.loads(arguments_text) if arguments_text else {}
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self._provider} sent unparseable tool-call arguments for "
                f"{function.get('name', '?')!r}: {exc}"
            ) from exc
        if not isinstance(arguments, dict):
            raise LLMError(
                f"{self._provider} sent tool-call arguments that are not a JSON object "
                f"for {function.get('name', '?')!r}"
            )
        return ToolCall(
            id=str(call.get("id") or ""), name=str(function.get("name") or ""), arguments=arguments
        )

    def _request_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        # INFO, not DEBUG, on purpose: this is the one line that lets someone watching the
        # terminal see that a request is actually going out - message *content* is never
        # logged (it may hold a person's own attached document), only its shape.
        LOGGER.info(
            "LLM request -> %s (%s): %d message(s)%s",
            self._provider,
            self._model,
            len(payload.get("messages", [])),
            ", with tools" if payload.get("tools") else "",
        )
        attempt = 0
        while True:
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                if attempt >= self._max_retries:
                    LOGGER.warning("LLM request to %s timed out", self._provider)
                    raise LLMError(f"{self._provider} did not respond in time") from exc
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    LOGGER.warning("LLM request to %s failed: %s", self._provider, exc)
                    raise LLMError(
                        f"could not reach {self._provider}: {_redact(str(exc), self._api_key)}"
                    ) from exc
            else:
                if response.status_code < 300:
                    try:
                        body = dict(response.json())
                    except ValueError as exc:
                        raise LLMError(f"{self._provider} returned invalid JSON: {exc}") from exc
                    message = (body.get("choices") or [{}])[0].get("message", {})
                    calls = message.get("tool_calls") or []
                    if calls:
                        names = ", ".join(
                            (call.get("function") or {}).get("name", "?") for call in calls
                        )
                        outcome = f"requested tool call(s): {names}"
                    else:
                        outcome = "final answer"
                    LOGGER.info("LLM response <- %s: %s", self._provider, outcome)
                    return body
                if response.status_code not in RETRYABLE_STATUS or attempt >= self._max_retries:
                    detail = _redact(response.text[:300], self._api_key)
                    LOGGER.warning(
                        "LLM request to %s got HTTP %d", self._provider, response.status_code
                    )
                    raise LLMError(
                        f"{self._provider} returned HTTP {response.status_code}: {detail}"
                    )
            attempt += 1
            time.sleep(min(2**attempt * 0.25, 4.0))
