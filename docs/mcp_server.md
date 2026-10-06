# CEWS MCP server

`cews mcp-serve` publishes CEWS's stored results to any MCP client (Claude Desktop, an agent
you write, ...). It is **read-only**: the database connection itself refuses writes, and there
is no tool that approves, rates, fetches or changes anything.

## Install

```
python -m pip install -r requirements-mcp.txt
```

## Try it without any AI client

```
python scripts/mcp_try.py
python scripts/mcp_try.py --tool get_entity --args "{\"kind\": \"topic\", \"reference\": \"crispr\"}"
```

## Connect Claude Desktop (Windows)

Edit `%APPDATA%\Claude\claude_desktop_config.json` and restart Claude Desktop. Use the full
path of the Python inside your virtual environment (`python -c "import sys; print(sys.executable)"`):

```json
{
  "mcpServers": {
    "cews": {
      "command": "C:\\projects\\CEWS\\venv\\Scripts\\python.exe",
      "args": ["-m", "cews.cli", "mcp-serve", "--env-file", "C:\\projects\\CEWS\\.env"]
    }
  }
}
```

## What it offers

| Kind | Name | Purpose |
| --- | --- | --- |
| tool | `get_overview` | Counts and whether the data is synthetic. Start here. |
| tool | `list_trends`, `list_opportunities` | Topics ranked, with confidence and failed rules. |
| tool | `list_competitors` | By `monitoring_priority` or `innovation`. |
| tool | `get_entity`, `get_evidence` | One topic or competitor: scores and why, forecast, evidence. |
| tool | `list_insights`, `get_insight` | Findings, and one in full with its evidence. |
| tool | `list_anomalies` | Unusual months, each labelled. |
| tool | `get_data_quality`, `get_source_health` | State of the data and its sources. |
| tool | `get_latest_evaluation` | Algorithm, backtest and expert validation, kept apart. |
| tool | `list_review_queue` | Decisions waiting for a person (resolve with `cews review`). |
| resource | `cews://methodology` | How scores are built, with live weights. |
| resource | `cews://overview` | The overview as a document. |
| prompt | `weekly_briefing`, `explain_entity`, `verify_insight` | Ready-made instructions for common jobs. |

## Safety choices

- Every answer says whether the data is synthetic (`is_synthetic`, `data_origin`).
- Record titles come from outside sources, so they are cleaned, length-capped and marked as
  untrusted data (`notice`), the standard defence against text in a record steering an agent.
- Every list has a hard maximum of 50, so one call cannot flood a model's context.
- Console logging is moved to stderr, because stdout carries the protocol.
- The server has no authentication because it uses stdio: only the program that launched it can
  talk to it. Do not expose it over a network without adding authentication.
