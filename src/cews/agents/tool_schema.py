"""Turning an MCP tool listing into the function-calling schema an LLM expects.

MCP tools already describe their own inputs as JSON Schema (``tool.inputSchema``) — that is
exactly the shape OpenAI-style function calling wants under ``function.parameters`` too, so there
is no format to invent here, only a small wrapper to write. Keeping this as one small function,
instead of hand-writing each tool's schema a second time somewhere in the agent code, is the
point: the model always sees exactly what the server itself advertises, so the two can never
silently drift apart.
"""

from __future__ import annotations

from typing import Any

from mcp.types import Tool


def mcp_tool_to_openai_function(tool: Tool) -> dict[str, Any]:
    """One MCP :class:`~mcp.types.Tool` as one OpenAI-style function-calling entry."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def mcp_tools_to_openai_functions(
    tools: list[Tool], *, allowed: frozenset[str] | None = None
) -> list[dict[str, Any]]:
    """Every tool in ``tools``, or only the ones named in ``allowed`` if given.

    A persona passes its own tool allowlist here (see :mod:`cews.agents.personas`) so the model
    is never even shown a tool that persona should not use — the safest form of "the model won't
    call it": it cannot pick a name it never saw.
    """
    chosen = tools if allowed is None else [tool for tool in tools if tool.name in allowed]
    return [mcp_tool_to_openai_function(tool) for tool in chosen]
