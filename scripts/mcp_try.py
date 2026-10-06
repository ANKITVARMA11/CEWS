"""Try the CEWS MCP server from Python, no Claude Desktop needed.

This is a minimal MCP *client*: it launches ``cews mcp-serve`` as a subprocess, connects over
stdio, and shows each step of the protocol, which is the best way to see what an agent sees.

    python scripts/mcp_try.py                      # list everything, then call two tools
    python scripts/mcp_try.py --tool get_entity --args '{"kind": "topic", "reference": "crispr"}'
    python scripts/mcp_try.py --env-file path/to/.env
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def show(title: str, payload: Any) -> None:
    print(f"\n=== {title}")
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str))


async def run(env_file: str | None, tool: str | None, arguments: dict[str, Any]) -> None:
    command = [sys.executable, "-m", "cews.cli", "mcp-serve"]
    if env_file:
        command += ["--env-file", env_file]
    # The SDK passes a server only a short whitelist of environment variables by default; a local
    # learning client can safely hand over its own, so the server finds the same packages.
    params = StdioServerParameters(command=command[0], args=command[1:], env=dict(os.environ))

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        # 1. Handshake: the client and server agree on a protocol version and exchange names.
        init = await session.initialize()
        show(
            "1. initialize", f"connected to {init.serverInfo.name}, protocol {init.protocolVersion}"
        )

        # 2. Discovery: the server describes what it can do; a model reads exactly this.
        tools = (await session.list_tools()).tools
        show(
            "2. tools the server offers",
            [f"{t.name}: {(t.description or '').splitlines()[0]}" for t in tools],
        )
        resources = (await session.list_resources()).resources
        show("   resources", [str(r.uri) for r in resources])
        prompts = (await session.list_prompts()).prompts
        show("   prompts", [p.name for p in prompts])

        # 3. A tool call: the model picks a tool and arguments; the server answers with data.
        calls = (
            [(tool, arguments)] if tool else [("get_overview", {}), ("list_trends", {"limit": 3})]
        )
        for name, args in calls:
            result = await session.call_tool(name, args)
            label = f"3. call_tool {name}({json.dumps(args)})"
            show(
                label,
                result.structuredContent if not result.isError else f"ERROR: {result.content}",
            )

        # 4. A resource: readable context a client can attach without a tool call.
        methodology = await session.read_resource("cews://methodology")  # type: ignore[arg-type]
        show("4. resource cews://methodology (first lines)", "\n".join(methodology.contents[0].text.splitlines()[:6]))  # type: ignore[union-attr]


def main() -> int:
    parser = argparse.ArgumentParser(description="Try the CEWS MCP server from Python.")
    parser.add_argument(
        "--env-file", help="dotenv file for CEWS (default: .env in the project root)"
    )
    parser.add_argument("--tool", help="call just this tool")
    parser.add_argument("--args", default="{}", help="JSON arguments for --tool")
    options = parser.parse_args()
    try:
        asyncio.run(run(options.env_file, options.tool, json.loads(options.args)))
    except BaseExceptionGroup as group:
        print(
            "could not talk to the server. Check that it starts on its own first "
            "(run: cews mcp-serve, then press Ctrl+C), that requirements-mcp.txt is installed, "
            f"and that the database exists (cews db-init).\ndetails: {group.exceptions[0]!r}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
