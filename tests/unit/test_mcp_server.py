"""Unit tests for the MCP wiring: what the server advertises and how it answers a real client.

An in-process MCP client is connected to the server, so the actual protocol messages (tool
listing, schema validation, structured results, errors) are exercised without a subprocess.
Skipped when the optional ``mcp`` package is not installed.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, date, datetime
from typing import Any, TypeVar

import pytest
from sqlalchemy.orm import Session, sessionmaker

pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from cews.database.connection import (
    create_memory_engine,
    create_session_factory,
    session_scope,
)  # noqa: E402
from cews.database.migrations import upgrade_database  # noqa: E402
from cews.database.models import Insight, Score, SourceRecord, Topic  # noqa: E402
from cews.mcp_server.server import INSTRUCTIONS, build_server  # noqa: E402
from cews.settings import load_settings  # noqa: E402
from support import SCORING_FILE  # noqa: E402

pytestmark = pytest.mark.unit

T = TypeVar("T")
EXPECTED_TOOLS = {
    "get_overview",
    "list_trends",
    "list_opportunities",
    "list_competitors",
    "get_entity",
    "get_evidence",
    "list_insights",
    "get_insight",
    "list_anomalies",
    "get_data_quality",
    "get_source_health",
    "get_latest_evaluation",
    "list_review_queue",
}


@pytest.fixture
def server() -> Iterator[Any]:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    with session_scope(factory) as session:
        topic = Topic(key="a", canonical_name="Alpha", topic_type="technology")
        session.add(topic)
        session.flush()
        session.add(
            Score(
                score_date=date(2026, 8, 1),
                entity_type="topic",
                entity_id=topic.id,
                score_type="trend",
                score_value=70.0,
                confidence_score=90.0,
                scoring_version="1.0.0",
                component_json={
                    "qualified": True,
                    "category": "Emerging",
                    "sample_size": 50,
                    "explanation": "why",
                    "components": {},
                },
            )
        )
        session.add(
            Insight(
                insight_date=date(2026, 8, 1),
                severity="watch",
                insight_type="emerging_trend",
                entity_type="topic",
                entity_id=topic.id,
                title="T",
                observed_fact="f",
                interpretation="i",
                recommended_review="r",
                confidence_score=80.0,
            )
        )
        session.add(
            SourceRecord(
                source="s",
                source_record_id="r",
                record_type="publication",
                fetched_at=datetime(2026, 8, 1, tzinfo=UTC),
                content_hash="a" * 64,
                is_synthetic=True,
            )
        )
    yield build_server(settings, factory)
    engine.dispose()


def run(server: Any, action: Callable[[ClientSession], Awaitable[T]]) -> T:
    async def go() -> T:
        async with create_connected_server_and_client_session(server) as session:
            return await action(session)

    return asyncio.run(go())


# --------------------------------------------------------------------------------------
# What is advertised
# --------------------------------------------------------------------------------------
def test_exactly_the_expected_tools_are_offered(server: Any) -> None:
    tools = run(server, lambda s: s.list_tools()).tools
    assert {tool.name for tool in tools} == EXPECTED_TOOLS


def test_every_tool_is_declared_read_only(server: Any) -> None:
    for tool in run(server, lambda s: s.list_tools()).tools:
        assert tool.annotations is not None, tool.name
        assert tool.annotations.readOnlyHint is True and tool.annotations.destructiveHint is False


def test_no_tool_could_change_anything_by_its_name(server: Any) -> None:
    """The server offers reads only; approving, rating and fetching stay at the command line."""
    for tool in run(server, lambda s: s.list_tools()).tools:
        assert tool.name.startswith(("get_", "list_")), tool.name


def test_every_tool_has_a_description(server: Any) -> None:
    for tool in run(server, lambda s: s.list_tools()).tools:
        assert tool.description and len(tool.description) > 20, tool.name


def test_the_limit_argument_is_bounded_in_the_schema(server: Any) -> None:
    tools = {t.name: t for t in run(server, lambda s: s.list_tools()).tools}
    limit = tools["list_trends"].inputSchema["properties"]["limit"]
    assert limit["minimum"] == 1 and limit["maximum"] == 50


def test_the_instructions_tell_a_client_to_check_for_synthetic_data(server: Any) -> None:
    assert (
        "is_synthetic" in INSTRUCTIONS
        and "never present it as a real market finding" in INSTRUCTIONS
    )
    assert "read-only" in INSTRUCTIONS


# --------------------------------------------------------------------------------------
# Answering
# --------------------------------------------------------------------------------------
def test_a_tool_call_returns_structured_data_stating_its_origin(server: Any) -> None:
    result = run(server, lambda s: s.call_tool("list_trends", {"limit": 5}))
    assert result.isError is False
    assert result.structuredContent is not None
    assert result.structuredContent["is_synthetic"] is True
    assert result.structuredContent["results"][0]["entity"] == "Alpha"


def test_get_entity_answers_by_name(server: Any) -> None:
    result = run(
        server, lambda s: s.call_tool("get_entity", {"kind": "topic", "reference": "alpha"})
    )
    assert result.structuredContent is not None and result.structuredContent["entity"] == "Alpha"


def test_a_bad_request_is_an_error_the_client_can_read(server: Any) -> None:
    result = run(
        server, lambda s: s.call_tool("get_entity", {"kind": "topic", "reference": "nothing"})
    )
    assert result.isError is True
    assert "no topic matches" in result.content[0].text  # type: ignore[union-attr]


def test_a_limit_above_the_maximum_is_rejected_before_it_reaches_the_service(server: Any) -> None:
    result = run(server, lambda s: s.call_tool("list_trends", {"limit": 5000}))
    assert result.isError is True


def test_a_missing_required_argument_is_an_error(server: Any) -> None:
    assert run(server, lambda s: s.call_tool("get_entity", {"kind": "topic"})).isError is True


def test_an_unknown_tool_is_an_error(server: Any) -> None:
    async def action(session: ClientSession) -> bool:
        try:
            result = await session.call_tool("approve_everything", {})
        except Exception:
            return True
        return bool(result.isError)

    assert run(server, action) is True


# --------------------------------------------------------------------------------------
# Resources and prompts
# --------------------------------------------------------------------------------------
def test_the_methodology_resource_is_served(server: Any) -> None:
    async def action(session: ClientSession) -> str:
        listed = await session.list_resources()
        assert "cews://methodology" in {str(r.uri) for r in listed.resources}
        content = await session.read_resource("cews://methodology")  # type: ignore[arg-type]
        return content.contents[0].text  # type: ignore[union-attr]

    text = run(server, action)
    assert text.startswith("# How CEWS scores are built") and "Trend = " in text


def test_the_prompts_are_offered_and_filled_in(server: Any) -> None:
    async def action(session: ClientSession) -> tuple[set[str], str]:
        names = {p.name for p in (await session.list_prompts()).prompts}
        prompt = await session.get_prompt("explain_entity", {"kind": "topic", "reference": "Alpha"})
        return names, prompt.messages[0].content.text  # type: ignore[union-attr]

    names, text = run(server, action)
    assert names == {"weekly_briefing", "explain_entity", "verify_insight"}
    assert "get_entity(kind='topic', reference='Alpha')" in text


def test_the_briefing_prompt_forbids_unsupported_claims(server: Any) -> None:
    async def action(session: ClientSession) -> str:
        return (await session.get_prompt("weekly_briefing")).messages[0].content.text  # type: ignore[union-attr]

    text = run(server, action)
    assert "synthetic" in text and "Do not add claims the tools did not return" in text


# --------------------------------------------------------------------------------------
# stdio safety: nothing but the protocol may reach stdout
# --------------------------------------------------------------------------------------
def test_mcp_serve_moves_console_logging_off_stdout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from cews import cli
    from cews.database.connection import create_db_engine
    from cews.mcp_server import server as server_module

    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\nSCORING_CONFIG_FILE={SCORING_FILE}\n"
        f"TOPIC_TAXONOMY_FILE={SCORING_FILE.parent / 'topic_taxonomy.yaml'}\n"
        f"SOURCE_REGISTRY_FILE={SCORING_FILE.parent / 'source_registry.yaml'}\n",
        encoding="utf-8",
    )
    assert cli.main(["db-init", "--env-file", str(env)]) == 0
    create_db_engine(load_settings(env_file=env)).dispose()

    ran: dict[str, bool] = {}

    class FakeServer:
        def run(self, transport: str) -> None:
            ran["transport"] = transport == "stdio"
            handlers = [
                h
                for h in logging.getLogger("cews").handlers
                if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            ]
            ran["console_handlers"] = bool(handlers)
            ran["all_on_stderr"] = all(h.stream is sys.stderr for h in handlers)

    monkeypatch.setattr(server_module, "build_server", lambda *_a, **_k: FakeServer())
    assert cli.main(["mcp-serve", "--env-file", str(env)]) == 0
    assert ran == {"transport": True, "console_handlers": True, "all_on_stderr": True}
