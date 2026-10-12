"""Connecting CrewAI's agents to the real CEWS MCP server.

One ``MCPServerAdapter`` connection is opened for a whole crew run and shared by every agent in
it - not one subprocess per agent. ``crewai_tools.MCPServerAdapter`` returns every tool the
server publishes; splitting that single collection by name in Python (see
:func:`tools_for`) is what actually gives each agent its own subset, since the adapter itself
has no notion of "this agent gets these, that one gets those."
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from crewai.tools import BaseTool
from crewai_tools.adapters.tool_collection import ToolCollection
from mcp import StdioServerParameters


def server_params(env_file: str | Path | None) -> StdioServerParameters:
    """Launch ``cews mcp-serve`` exactly as :mod:`cews.agents.session` does.

    ``env=dict(os.environ)`` matters here for the same reason it does there: the MCP SDK passes
    a spawned server only a short built-in whitelist of environment variables by default, not
    the parent process's own - breaking a development setup such as this project's own
    ``PYTHONPATH=src`` convention. See ``cews.agents.session._default_server_params`` for the
    full explanation; duplicated here rather than imported, since the two packages are meant to
    stay independent of each other (see this package's own docstring).
    """
    args = ["-m", "cews.cli", "mcp-serve"]
    if env_file:
        args += ["--env-file", str(env_file)]
    return StdioServerParameters(command=sys.executable, args=args, env=dict(os.environ))


def tools_for(tools: ToolCollection[BaseTool], names: frozenset[str]) -> list[BaseTool]:
    """The subset of an already-connected tool collection whose names are in ``names``.

    Raises:
        ValueError: if a name in ``names`` is not actually published by the server - almost
            always a typo in a crew's own agent definition, worth failing loudly on rather than
            silently handing that agent fewer tools than its definition says it should have.
    """
    by_name = {tool.name: tool for tool in tools}
    missing = names - by_name.keys()
    if missing:
        raise ValueError(f"the MCP server does not publish these tools: {sorted(missing)}")
    return [by_name[name] for name in names]
