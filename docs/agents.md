# Agents and the chat window

Three agents sit on top of the MCP server (`docs/mcp_server.md`), sharing one small, deliberately
readable engine. This document is written for someone learning MCP: it explains what each piece
does and why, not just how to run it.

## Install

```
pip install -r requirements-mcp.txt
```

Same install as the MCP server: `mcp`, plus `pypdf` and `python-docx` for reading an attached
document. You also need an LLM configured (`LLM_PROVIDER` and `LLM_MODEL` in `.env`) - Ollama
running locally, or an OpenAI-compatible endpoint such as NVIDIA NIM. With `LLM_PROVIDER=none`
(the default), every agent command refuses to start rather than pretending to answer.

**Important limit on what has actually been verified:** nothing in this project has ever talked
to a real language model - there is no internet access in the environment this was built in. The
tool-calling loop, the MCP wiring, and the dashboard widget are all tested against a scripted
HTTP server standing in for the model (see "What's tested" below), and the full pipeline -
real MCP subprocess, real HTTP call, real tool call, real answer - was checked once by hand
against a small local server. It has never been tried against an actual Ollama or NIM model.
Please tell me how it goes with a real one.

## How the loop works (`cews/agents/core.py`)

This is the one piece of real "agent" logic in the whole package - everything else is
configuration around it:

1. Send the conversation so far, plus the tools this agent is allowed to use, to the LLM.
2. If it answered in words, that is the result.
3. If it asked for one or more tools, run each one for real against the MCP server, turn each
   result into a message, and go back to step 1.
4. Stop after `AGENT_MAX_TOOL_TURNS` (default 6) turns if no final answer has come yet, so a
   confused model can never loop forever.

A tool failing (an unknown entity, say) is fed back to the model as an ordinary tool result, the
same as a success - the model can see the failure and react to it, the same way a person would
read an error and try something else. The LLM itself being unreachable is different: there is
nothing to feed a network failure back to, so that is raised instead and each caller (the CLI,
the dashboard widget) shows a plain error message.

## The three personas (`cews/agents/personas.py`)

A persona is nothing more than a system prompt plus a list of which MCP tools it may use - there
is no separate code path per persona, all three run on the one loop above.

| Persona | Job | Tools it can use |
| --- | --- | --- |
| **Analyst** | Answers open-ended questions. Wired to the dashboard's chat window. | Every read-only tool: trends, opportunities, competitors, entity detail, evidence, insights, anomalies, data quality, source health, evaluation, review queue. |
| **Fact-checker** | Given one insight id, checks its claims against its own evidence. | `get_insight` only. |
| **Briefing writer** | Drafts a short weekly briefing from the newest insights. | `get_overview`, `list_insights`, `get_insight`. |

Restricting each persona's tools is worth noticing if you are learning MCP: the server offers far
more than any one persona needs, and a persona is never even shown a tool outside its list - the
model cannot call what it never saw. Trying anyway (say, a hallucinated tool name) is refused the
same way an unknown MCP tool is refused, without being run.

## Reading an attached document (`cews/agents/documents.py`)

When someone attaches a file to a chat, the whole document is read into text and handed to the
model directly - no chunking, no embeddings, no retrieval step, the way you would hand a
colleague a printout. The only limit is size: `AGENT_DOCUMENT_CHAR_LIMIT` (default 40,000
characters, roughly 10,000 tokens) caps how much goes in, and the agent is always told plainly
when a document was cut, so it never claims to have read more than it saw.

Docling was considered and rejected: its default install pulls in torch, torchvision and
multi-GB layout-detection models, which fights this project's low-resource, offline-first design
from the ground up. `pypdf`, `python-docx` and `openpyxl` (already a CEWS dependency) do the same
job - pull the text out - as plain Python, no model download.

Supported: PDF, Word (.docx), Excel (.xlsx/.xlsm), CSV, plain text and Markdown, up to 25 MB.

**Nothing about an attached document is written to CEWS's database.** The dashboard widget reads
an upload's bytes into text once and keeps only the text, in memory, for that browser tab; the
raw file is never saved to disk.

## Using it from the command line

```
cews agent ask "what's trending in oncology, and how confident are we?"
cews agent verify 42          # fact-check insight id 42
cews agent brief              # draft the weekly briefing
```

Each of these opens its own MCP connection for the one question and closes it afterwards. For a
back-and-forth conversation, use the dashboard's chat window instead, which keeps one connection
open across every question in the session.

## The dashboard chat window

A floating icon at the bottom-left of every dashboard page; clicking it opens a small panel with
the Analyst. No Flask, no second server, no JavaScript: the widget is plain Streamlit, embedded
directly in `dashboards/streamlit/app.py`, and calls the agent in the same process the dashboard
already runs in.

**How the floating position works**, since it is the one non-obvious trick here: Streamlit gives
`st.container(key=...)` a CSS class of `st-key-<key>` on its outer element. That is confirmed by
rendering a real dashboard in a headless browser and reading the class directly from the DOM, not
assumed from documentation alone. Setting `position: fixed` on that class in one injected
`<style>` block is enough to make it float over the page - no custom component, no iframe.

One trade-off worth knowing: Streamlit's own sidebar sits at the true left edge of the page too,
at a z-index that had to be matched and cleared (checked the same way, by reading its computed
style in a real browser). The chat icon sits low enough to never overlap the sidebar's own
controls, but the open panel is wide enough to cover the sidebar while it is open - normal for a
floating overlay, and it goes away the instant the panel closes.

Why in-process rather than a small server: a long-lived `AgentRuntime` (one MCP connection, one
background thread with its own asyncio event loop) is built once with `@st.cache_resource` -
the same pattern the dashboard already uses for its database connection - and shared by every
browser tab that opens the widget. Each tab still gets a private conversation:
`st.session_state` is already isolated per browser session, so no extra bookkeeping was needed
for that.

## Reasoning models (e.g. NVIDIA NIM)

Some models - reasoning-tuned ones served through providers such as NVIDIA NIM are the case this
was actually seen with - write their scratch thinking directly into the answer, wrapped in
`<think>...</think>`, instead of returning it separately. Left alone, that scratchpad
("We need to answer... Thus the answer is...") is what a person sees instead of the actual
answer. `LLMClient` strips this from anything shown to a person; it is only left alone on the
turn a model requests a tool, since that exact text is fed back to the model as its own prior
turn on the next call, and it may expect to see its own reasoning there.

If a model's response is cut off entirely mid-thought (it ran out of tokens before ever writing a
visible answer), you will see a plain note saying so rather than a fragment of reasoning; try a
shorter question or raise `LLM_MAX_TOKENS`.

The terminal running `cews agent` or the dashboard now logs one line per request and one per
response (`LLM request -> ...`, `LLM response <- ...`) - message content and the API key are
never in these lines, only the provider, model, message count, and whether a tool was requested.

## What's tested

- The tool-calling loop, against a **real** MCP server (in-process, no subprocess) with a
  **scripted** LLM transport: a model asking for one tool, several tools in one turn, several
  turns in a row, a tool outside a persona's allowed list being refused without running, a tool
  reporting a failure being relayed rather than hidden, and a model that never stops calling
  tools being cut off at the turn limit.
- Stripping a reasoning model's `<think>` block from a final answer, from a `complete()` call, and
  from a chat reply - including a closed block, an unclosed one, and one with no answer left at
  all - and confirming it is *not* stripped from the message fed back to the model on its own
  tool-calling turn. Request/response logging is confirmed to appear, and to never contain a
  prompt's own text or an API key.
- `AgentRuntime` (the async-to-sync bridge): starting, stopping, two full lifecycles in a row,
  every error path (connection failure, calling before start, a stopped runtime). One real bug
  was caught this way, not by inspection: an earlier version connected and disconnected the MCP
  session from two different asyncio tasks, which the underlying `anyio` library forbids and
  raises on shutdown.
- The dashboard widget, with Streamlit's own `AppTest` harness (the same tool the other seven
  dashboard pages are tested with): the icon renders, opening and closing it works, the
  "not configured" message shows correctly with no LLM set, and a full scripted conversation -
  including a follow-up question and clearing the chat - works through the real widget code.
- The CLI's argument validation and its refusal paths (no LLM configured, database not
  initialized) - against a real MCP subprocess, so the process actually launches; this project's
  own test policy blocks every real network call in a test, so the "and then gets a real answer"
  half of that path was checked by hand instead (see the note at the top of this document).

Not tested: a real language model. Please try `cews agent ask` against your own Ollama or NIM
setup and tell me what breaks.
