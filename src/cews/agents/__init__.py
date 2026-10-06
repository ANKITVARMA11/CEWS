"""Agents that use CEWS's MCP server to answer questions in plain language.

**How this fits together, for anyone reading this to learn MCP:**

- ``cews mcp-serve`` (see :mod:`cews.mcp_server`) is the MCP *server*. It has no idea agents
  exist; it just answers tool calls, the same for a person testing it with
  ``scripts/mcp_try.py`` as for the code in this package.
- Everything in this package is an MCP *client* plus a loop that lets an LLM decide, on its own,
  which of the server's tools to call and in what order. That loop lives in :mod:`.core` and is
  the one piece of real "agent" logic here — small enough to read start to finish.
- A **persona** (:mod:`.personas`) is nothing clever: a system prompt plus a short list of which
  of the server's tools that persona is allowed to use. Three ship here — Analyst, Fact-checker,
  Briefing writer — each with only the tools its job needs. An MCP server can offer far more than
  any one client should be handed at once, and restricting a client to a subrole's actual needs
  ("least privilege") is worth seeing for real, not just reading about.
- :mod:`.documents` reads an uploaded file into plain text, whole, so an agent can read a
  person's own document directly rather than through any kind of search index.

Every agent here is read-only, the same as the MCP server underneath it: nothing in this package
can approve, rate, fetch, or change CEWS data. An agent that could act on a person's behalf would
need a different, carefully considered design; this one only answers questions.
"""
