"""A second, independent agent system: CrewAI, collaborating over the same MCP server.

Everything in :mod:`cews.agents` (the Analyst/Fact-checker/Briefing-writer personas, the
hand-rolled tool-calling loop, the dashboard chat widget) stays exactly as it is - this package
does not replace it. What it adds is something that package was never built to do: several
agents working on one task *together*, each one's output automatically feeding the next, rather
than a person picking one persona and asking it one question.

Both systems point at the very same thing underneath: the real ``cews mcp-serve`` server. There
is exactly one implementation of "what CEWS's data looks like as tools" in this whole project
(:mod:`cews.mcp_server`); this package is a second kind of *client* for it, the same way
:mod:`cews.agents` is one. Confirmed by actually connecting `crewai-tools`' MCP adapter to the
real server and watching it discover all 13 real tools, not assumed from documentation.

Install with ``pip install -r requirements-crew.txt`` (``crewai`` plus ``crewai-tools[mcp]``,
which also needs the separate ``mcpadapt`` package - easy to miss, since the adapter's own error
message for not having it is a vague "you are missing the 'mcp' package" that is actually about
the missing ``mcpadapt``, not ``mcp`` itself, which may well already be installed).
"""
