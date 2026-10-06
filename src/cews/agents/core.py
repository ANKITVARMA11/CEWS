"""The agent loop: an LLM decides which MCP tools to call, this runs them, and repeats.

This is deliberately the *only* piece of control flow in the whole agents package. Reading it
top to bottom is reading the whole idea of a tool-calling agent:

1. Send the conversation so far, plus the persona's allowed tools, to the LLM.
2. If the LLM answered in words, that answer is the result — stop.
3. If the LLM asked for one or more tools instead, run each one against the real MCP server,
   turn each result into a ``role: "tool"`` message, and go back to step 1.
4. Stop early (with a plain, honest message) if this goes on for more turns than
   ``AGENT_MAX_TOOL_TURNS`` allows, so a confused model can never loop forever.

Nothing here is CEWS-specific. It would work unchanged against any MCP server and any persona;
what makes an agent "the Analyst" versus "the Fact-checker" lives entirely in
:mod:`cews.agents.personas`, not here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.types import Tool

from cews.agents.tool_schema import mcp_tools_to_openai_functions
from cews.ai.llm import LLMClient

LOGGER = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 6_000  # keeps one very large tool result from crowding out everything else


@dataclass
class ToolCallLog:
    """One tool call an agent made, for showing "what it checked" alongside the answer."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    summary: str  # a short, human-readable note - the full result, not this, goes to the model


def format_tool_call(call: ToolCallLog) -> str:
    """One line describing a tool call: its name, arguments and whether it succeeded.

    Shared so every place that shows "what was checked" - the CLI's own output and the
    dashboard's chat panel alike - renders a call identically rather than slowly drifting apart
    into two different formats, the way they briefly did before this existed.
    """
    arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
    mark = "ok" if call.ok else "FAILED"
    return f"{call.name}({arguments}) [{mark}]"


@dataclass
class AgentReply:
    """What one question to an agent produced."""

    text: str
    tool_calls: list[ToolCallLog] = field(default_factory=list)
    stopped_early: bool = False  # hit the turn limit without a final answer

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, for the web chat and the CLI's ``--json`` output alike."""
        return {
            "text": self.text,
            "tool_calls": [
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "ok": call.ok,
                    "summary": call.summary,
                }
                for call in self.tool_calls
            ],
            "stopped_early": self.stopped_early,
        }


def _tool_result_text(result: Any) -> str:
    """A tool result as plain text for the model: the structured JSON if there is one, else the
    first text block MCP always includes as a fallback."""
    structured = getattr(result, "structuredContent", None)
    if structured is not None:
        text = json.dumps(structured)
    else:
        blocks = getattr(result, "content", []) or []
        text = " ".join(getattr(block, "text", "") for block in blocks) or "(no content)"
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = (
            text[:MAX_TOOL_RESULT_CHARS] + f"... (truncated at {MAX_TOOL_RESULT_CHARS} characters)"
        )
    return text


async def run_agent_turn(
    llm: LLMClient,
    session: ClientSession,
    *,
    system_prompt: str,
    history: list[dict[str, Any]],
    user_message: str,
    tools: list[Tool],
    allowed_tools: frozenset[str],
    max_turns: int,
    max_tokens: int,
) -> AgentReply:
    """Answer one message, calling MCP tools as needed, and return the result.

    ``history`` is the conversation so far in OpenAI message shape (``role``/``content``); it is
    read but not mutated, so the caller decides what becomes part of the session's saved history
    (see :mod:`cews.agents.session`) — typically the user's message and the final answer, not the
    tool-call back-and-forth in between, which would only bloat every later turn's prompt.

    Raises:
        LLMError: if the LLM cannot be reached at all (a network failure, not a tool failing:
            a failing tool is reported back to the model as a normal tool result, per the whole
            point of this loop — the model gets to see and react to a failure, same as a success).
    """
    function_schemas = mcp_tools_to_openai_functions(tools, allowed=allowed_tools)
    messages: list[dict[str, Any]] = (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": user_message}]
    )
    made_calls: list[ToolCallLog] = []

    for _turn in range(max_turns):
        response = llm.chat(messages, tools=function_schemas, max_tokens=max_tokens)
        if not response.wants_tools:
            return AgentReply(text=response.text, tool_calls=made_calls)

        # The assistant's own tool-call request is part of the conversation the model needs to
        # see again next turn, exactly as the model sent it (OpenAI-style tool calling requires
        # the request and its results to both be present for the follow-up call to make sense).
        messages.append(
            {
                "role": "assistant",
                "content": response.text or None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                    }
                    for call in response.tool_calls
                ],
            }
        )

        for call in response.tool_calls:
            if call.name not in allowed_tools:
                # The model asked for a tool this persona was never even shown (a malformed or
                # hallucinated call). Refuse it the same way a real MCP server refuses an unknown
                # tool: report it back as a failed call, do not execute anything.
                result_text = f"error: '{call.name}' is not a tool available to this agent"
                log_entry = ToolCallLog(call.name, call.arguments, ok=False, summary=result_text)
            else:
                try:
                    result = await session.call_tool(call.name, call.arguments)
                except Exception as exc:  # a transport-level failure, not a tool-reported error
                    LOGGER.exception("MCP call to %s failed", call.name)
                    result_text = f"error: could not reach the tool: {exc}"
                    log_entry = ToolCallLog(
                        call.name, call.arguments, ok=False, summary=result_text
                    )
                else:
                    result_text = _tool_result_text(result)
                    ok = not getattr(result, "isError", False)
                    summary = result_text if ok else f"error: {result_text}"
                    log_entry = ToolCallLog(call.name, call.arguments, ok=ok, summary=summary[:200])
            made_calls.append(log_entry)
            # The one place a tool call's actual outcome (arguments, ok/failed) reaches the
            # terminal - the LLM client's own logging only ever sees that a tool was *requested*,
            # not whether running it worked, since that happens here, after the MCP call returns.
            # format_tool_call is the same formatting the CLI and the dashboard caption use, so
            # all three show identical detail for the same call.
            LOGGER.info("checked: %s", format_tool_call(log_entry))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_text})

    LOGGER.warning("agent stopped after %s turns without a final answer", max_turns)
    return AgentReply(
        text=(
            "I wasn't able to reach an answer within the allowed number of tool calls. "
            "Try asking a narrower question, or check `cews jobs` if this keeps happening."
        ),
        tool_calls=made_calls,
        stopped_early=True,
    )
