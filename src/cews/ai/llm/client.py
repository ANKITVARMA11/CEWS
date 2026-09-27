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


def llm_enabled(settings: Settings) -> bool:
    """Whether any code should attempt an LLM call at all.

    Every caller checks this first. ``LLM_PROVIDER=none`` (the default) means CEWS never imports
    an HTTP client for this purpose and every feature that could use one runs its deterministic
    fallback instead.
    """
    return settings.llm_provider is not LLMProvider.NONE


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
        return LLMResponse(text=choice, model=self._model, provider=self._provider.value, raw=body)

    def _request_with_retries(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as exc:
                if attempt >= self._max_retries:
                    raise LLMError(f"{self._provider} did not respond in time") from exc
            except httpx.HTTPError as exc:
                if attempt >= self._max_retries:
                    raise LLMError(
                        f"could not reach {self._provider}: {_redact(str(exc), self._api_key)}"
                    ) from exc
            else:
                if response.status_code < 300:
                    try:
                        return dict(response.json())
                    except ValueError as exc:
                        raise LLMError(f"{self._provider} returned invalid JSON: {exc}") from exc
                if response.status_code not in RETRYABLE_STATUS or attempt >= self._max_retries:
                    detail = _redact(response.text[:300], self._api_key)
                    raise LLMError(
                        f"{self._provider} returned HTTP {response.status_code}: {detail}"
                    )
            attempt += 1
            time.sleep(min(2**attempt * 0.25, 4.0))
