"""End-to-end: launch ``cews mcp-serve`` as a subprocess and talk MCP to it over stdio.

This is what Claude Desktop or an agent framework does. The database is built by the real
pipeline (seed, normalize, features, score, insights), so the answers come from real stored
results rather than hand-made rows. Skipped when the optional ``mcp`` package is missing.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from cews.cli import main  # noqa: E402
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE  # noqa: E402

pytestmark = pytest.mark.integration

T = TypeVar("T")
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project directory whose database was built by the real pipeline."""
    root = tmp_path_factory.mktemp("mcp_project")
    env = root / ".env"
    env.write_text(
        f"PROJECT_ROOT={root}\nSQLITE_PATH=./data/cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n"
        f"SOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "ENABLE_CLINICAL_TRIALS_GOV=false\nENABLE_PUBMED=false\nENABLE_EUROPE_PMC=false\n",
        encoding="utf-8",
    )
    for command in (
        ["db-init"],
        ["seed-demo", "--scale", "0.2"],
        ["normalize"],
        ["competitors"],
        ["features", "--as-of", "2026-09-01"],
        ["score", "--as-of", "2026-09-01"],
        ["insights"],
    ):
        assert main([command[0], "--env-file", str(env), *command[1:]]) == 0, command
    return env


def connect(env: Path, action: Callable[[ClientSession], Awaitable[T]]) -> T:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cews.cli", "mcp-serve", "--env-file", str(env)],
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )

    async def go() -> T:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            return await action(session)

    return asyncio.run(asyncio.wait_for(go(), timeout=90))


def call(env: Path, tool: str, arguments: dict[str, Any] | None = None) -> Any:
    return connect(env, lambda session: session.call_tool(tool, arguments or {}))


def test_the_server_starts_and_lists_its_tools_over_stdio(project: Path) -> None:
    tools = connect(project, lambda s: s.list_tools()).tools
    assert len(tools) == 13 and all(t.annotations and t.annotations.readOnlyHint for t in tools)


def test_the_overview_reports_real_stored_counts_and_synthetic_origin(project: Path) -> None:
    result = call(project, "get_overview")
    assert result.isError is False
    content = result.structuredContent
    assert content["is_synthetic"] is True and content["summary"]["records"] > 0
    assert content["summary"]["topics"] > 0


def test_trends_come_back_ranked_with_confidence(project: Path) -> None:
    content = call(project, "list_trends", {"limit": 5}).structuredContent
    scores = [row["score"] for row in content["results"]]
    assert scores == sorted(scores, reverse=True) and content["results"]
    assert all("confidence" in row and "rules_failed" in row for row in content["results"])


def test_an_entity_can_be_explained_with_evidence(project: Path) -> None:
    content = call(
        project, "get_entity", {"kind": "topic", "reference": "crispr"}
    ).structuredContent
    assert content["entity"] == "CRISPR gene editing"
    assert "trend" in content["scores"] and content["evidence"]
    assert content["notice"].startswith("Record titles")


def test_an_insight_can_be_fetched_by_id_from_the_list(project: Path) -> None:
    async def action(session: ClientSession) -> tuple[Any, Any]:
        listed = await session.call_tool("list_insights", {"limit": 3})
        assert listed.structuredContent is not None
        first_id = listed.structuredContent["insights"][0]["id"]
        return listed, await session.call_tool("get_insight", {"insight_id": first_id})

    listed, detail = connect(project, action)
    assert detail.isError is False
    assert (
        detail.structuredContent["insight"]["id"] == listed.structuredContent["insights"][0]["id"]
    )
    assert detail.structuredContent["evidence"]  # an insight without evidence is never stored


def test_a_bad_request_comes_back_as_an_error_not_a_crash(project: Path) -> None:
    async def action(session: ClientSession) -> tuple[bool, bool]:
        bad = await session.call_tool("get_entity", {"kind": "topic", "reference": "zzzz"})
        good = await session.call_tool("get_overview")  # the server is still alive afterwards
        return bad.isError, good.isError

    assert connect(project, action) == (True, False)


def test_the_data_quality_check_runs_through_the_server(project: Path) -> None:
    report = call(project, "get_data_quality").structuredContent["report"]
    assert report["passed"] is True


def test_the_methodology_resource_reflects_the_real_configuration(project: Path) -> None:
    async def action(session: ClientSession) -> str:
        content = await session.read_resource("cews://methodology")  # type: ignore[arg-type]
        return content.contents[0].text  # type: ignore[union-attr]

    text = connect(project, action)
    assert "Trend = 0.3 velocity + 0.25 momentum" in text


def table_hashes(database: Path) -> dict[str, str]:
    """A hash of the full contents of every table, so any insert, update or delete shows up."""
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        hashes = {}
        for table in tables:
            digest = hashlib.sha256()
            for row in connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'):  # noqa: S608
                digest.update(repr(row).encode())
            hashes[table] = digest.hexdigest()
        return hashes
    finally:
        connection.close()


def test_the_fingerprint_would_notice_a_change(project: Path) -> None:
    """The next test is only worth anything if this check can fail."""
    database = project.parent / "data" / "cews.db"
    before = table_hashes(database)
    writer = sqlite3.connect(database)
    try:
        writer.execute("UPDATE topics SET canonical_name = canonical_name || ' ' WHERE id = 1")
        writer.commit()
        assert table_hashes(database) != before
        writer.execute("UPDATE topics SET canonical_name = TRIM(canonical_name) WHERE id = 1")
        writer.commit()
    finally:
        writer.close()
    assert table_hashes(database) == before


def test_using_the_server_never_changes_the_database(project: Path) -> None:
    database = project.parent / "data" / "cews.db"
    before = table_hashes(database)

    async def action(session: ClientSession) -> None:
        for tool in (
            "get_overview",
            "list_trends",
            "list_opportunities",
            "list_competitors",
            "get_entity",
            "list_insights",
            "list_anomalies",
            "get_data_quality",
            "get_source_health",
            "get_latest_evaluation",
            "list_review_queue",
        ):
            arguments = {"kind": "topic", "reference": "crispr"} if tool == "get_entity" else {}
            result = await session.call_tool(tool, arguments)
            assert result.isError is False, tool

    connect(project, action)
    assert table_hashes(database) == before


def test_a_missing_database_makes_the_server_exit_with_a_clear_message(tmp_path: Path) -> None:
    import subprocess

    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [sys.executable, "-m", "cews.cli", "mcp-serve", "--env-file", str(env)],
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert done.returncode == 1
    assert "cews db-init" in done.stderr
    assert done.stdout == ""  # nothing but protocol may ever be written to stdout
    assert not (tmp_path / "absent.db").exists()
