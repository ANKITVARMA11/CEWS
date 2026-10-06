"""A read-only MCP server over CEWS's stored results.

MCP (Model Context Protocol) lets an AI client, such as Claude Desktop or an agent you write,
discover and call tools on a server. This package is that server: it publishes what CEWS has
already computed (scores, forecasts, insights, evidence, evaluation results) and nothing else.

* ``tools`` is the service layer: plain functions, no MCP dependency, reusable elsewhere.
* ``server`` wraps it with the MCP SDK (an optional dependency, see ``requirements-mcp.txt``).

Two rules hold throughout. The server can never change data (the database connection itself is
read-only), and every answer says whether the numbers come from synthetic demo data.
"""
