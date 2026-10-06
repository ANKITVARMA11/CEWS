"""Unit tests for the LLM-to-MCP tool-calling loop (cews.agents.core), and the schema converter.

An in-process MCP server (the same real server code the app uses) is connected to a real
ClientSession; only the LLM side is mocked, since it is the only part that would otherwise need a
live model. This proves the loop actually drives real MCP tool calls, not a stand-in.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from datetime import date
from typing import Any, TypeVar

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session, sessionmaker

from cews.agents.core import AgentReply, ToolCallLog, format_tool_call, run_agent_turn
from cews.agents.tool_schema import mcp_tool_to_openai_function, mcp_tools_to_openai_functions
from cews.ai.llm import LLMClient
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Score, Topic
from cews.mcp_server.server import build_server
from cews.settings import load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

T = TypeVar("T")


@pytest.fixture
def factory() -> sessionmaker[Session]:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
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
                confidence_score=88.0,
                scoring_version="1.0.0",
                component_json={
                    "qualified": True,
                    "category": "Emerging",
                    "sample_size": 40,
                    "explanation": "steady growth",
                    "components": {},
                },
            )
        )
    return factory


@pytest.fixture
def server(factory: sessionmaker[Session]) -> Any:
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    return build_server(settings, factory)


def make_llm(handler: Callable[[httpx.Request], httpx.Response]) -> LLMClient:
    settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def scripted(*responses: dict[str, Any]) -> Callable[[httpx.Request], httpx.Response]:
    """A handler that replies with each of ``responses`` in turn, one per call."""
    calls = iter(responses)

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": next(calls)}]})

    return handler


def tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def text(content: str) -> dict[str, Any]:
    return {"content": content}


def run(server: Any, action: Callable[[Any], Awaitable[T]]) -> T:
    async def go() -> T:
        async with create_connected_server_and_client_session(server) as session:
            return await action(session)

    return asyncio.run(go())


# --------------------------------------------------------------------------------------
# The loop against a real MCP server
# --------------------------------------------------------------------------------------
def test_a_direct_answer_needs_no_tool_call(server: Any) -> None:
    with make_llm(scripted(text("Hi there."))) as llm:
        reply: AgentReply = run(
            server,
            lambda session: run_agent_turn(
                llm,
                session,
                system_prompt="You are helpful.",
                history=[],
                user_message="hi",
                tools=[],
                allowed_tools=frozenset(),
                max_turns=3,
                max_tokens=100,
            ),
        )
    assert reply.text == "Hi there." and reply.tool_calls == [] and reply.stopped_early is False


def test_one_real_tool_call_then_a_final_answer(server: Any) -> None:
    with make_llm(
        scripted(tool_call("c1", "list_trends", {"limit": 3}), text("Alpha is trending."))
    ) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="what's trending?",
                tools=tools,
                allowed_tools=frozenset({"list_trends"}),
                max_turns=4,
                max_tokens=100,
            )

        reply = run(server, action)
    assert reply.text == "Alpha is trending."
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.name == "list_trends" and call.ok is True and "Alpha" in call.summary


def test_several_tool_calls_in_one_turn_all_run(server: Any) -> None:
    both = {
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_overview", "arguments": "{}"},
            },
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "list_trends", "arguments": "{}"},
            },
        ],
    }
    with make_llm(scripted(both, text("done"))) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="q",
                tools=tools,
                allowed_tools=frozenset({"get_overview", "list_trends"}),
                max_turns=4,
                max_tokens=100,
            )

        reply = run(server, action)
    assert [c.name for c in reply.tool_calls] == ["get_overview", "list_trends"]
    assert all(c.ok for c in reply.tool_calls)


def test_several_turns_of_tool_calls_chain_correctly(server: Any) -> None:
    with make_llm(
        scripted(
            tool_call("c1", "get_overview", {}),
            tool_call("c2", "list_trends", {"limit": 1}),
            text("Based on both, Alpha leads."),
        )
    ) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="q",
                tools=tools,
                allowed_tools=frozenset({"get_overview", "list_trends"}),
                max_turns=5,
                max_tokens=100,
            )

        reply = run(server, action)
    assert reply.text == "Based on both, Alpha leads."
    assert [c.name for c in reply.tool_calls] == ["get_overview", "list_trends"]


# --------------------------------------------------------------------------------------
# The tool-turn limit
# --------------------------------------------------------------------------------------
def test_a_model_that_never_stops_calling_tools_is_cut_off(server: Any) -> None:
    forever = tool_call("c1", "get_overview", {})

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": forever}]})

    with make_llm(handler) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="q",
                tools=tools,
                allowed_tools=frozenset({"get_overview"}),
                max_turns=3,
                max_tokens=100,
            )

        reply = run(server, action)
    assert reply.stopped_early is True
    assert len(reply.tool_calls) == 3  # exactly max_turns worth of calls were made, not more
    assert "allowed number of tool calls" in reply.text


# --------------------------------------------------------------------------------------
# Safety: a persona's tool allowlist is enforced here too, not only by tool visibility
# --------------------------------------------------------------------------------------
def test_a_tool_outside_the_allowlist_is_refused_without_being_called(server: Any) -> None:
    """Even if a model somehow names a tool it was never shown, it must not run."""
    with make_llm(scripted(tool_call("c1", "get_data_quality", {}), text("ok"))) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="q",
                tools=tools,
                allowed_tools=frozenset({"get_overview"}),  # get_data_quality is NOT in here
                max_turns=4,
                max_tokens=100,
            )

        reply = run(server, action)
    assert reply.tool_calls[0].name == "get_data_quality"
    assert reply.tool_calls[0].ok is False
    assert "not a tool available" in reply.tool_calls[0].summary


def test_a_tool_that_reports_an_error_is_relayed_not_hidden(server: Any) -> None:
    with make_llm(
        scripted(
            tool_call("c1", "get_entity", {"kind": "topic", "reference": "nonexistent"}),
            text("That topic was not found."),
        )
    ) as llm:

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            return await run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=[],
                user_message="q",
                tools=tools,
                allowed_tools=frozenset({"get_entity"}),
                max_turns=4,
                max_tokens=100,
            )

        reply = run(server, action)
    assert reply.tool_calls[0].ok is False and "error" in reply.tool_calls[0].summary.lower()
    assert reply.text == "That topic was not found."


# --------------------------------------------------------------------------------------
# History and message construction
# --------------------------------------------------------------------------------------
def test_history_is_included_but_not_mutated(server: Any) -> None:
    history = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    original = list(history)
    with make_llm(scripted(text("ok"))) as llm:
        run(
            server,
            lambda session: run_agent_turn(
                llm,
                session,
                system_prompt="x",
                history=history,
                user_message="new question",
                tools=[],
                allowed_tools=frozenset(),
                max_turns=2,
                max_tokens=100,
            ),
        )
    assert history == original  # the caller's list was not appended to in place


def test_large_tool_results_are_truncated_before_reaching_the_model(server: Any) -> None:
    from cews.agents import core

    huge_result = type("R", (), {"structuredContent": {"data": "x" * 50_000}, "isError": False})()
    assert len(core._tool_result_text(huge_result)) <= core.MAX_TOOL_RESULT_CHARS + 60
    assert "truncated" in core._tool_result_text(huge_result)


# --------------------------------------------------------------------------------------
# The MCP-to-OpenAI schema conversion
# --------------------------------------------------------------------------------------
def test_every_tool_becomes_a_valid_function_schema(server: Any) -> None:
    tools = run(server, lambda session: session.list_tools()).tools
    functions = mcp_tools_to_openai_functions(tools)
    assert len(functions) == len(tools)
    for entry in functions:
        assert entry["type"] == "function"
        assert entry["function"]["name"] and entry["function"]["description"]
        assert isinstance(entry["function"]["parameters"], dict)


def test_the_allowlist_actually_filters(server: Any) -> None:
    tools = run(server, lambda session: session.list_tools()).tools
    functions = mcp_tools_to_openai_functions(tools, allowed=frozenset({"get_overview"}))
    assert [f["function"]["name"] for f in functions] == ["get_overview"]


def test_an_unknown_allowed_name_yields_nothing_rather_than_erroring(server: Any) -> None:
    tools = run(server, lambda session: session.list_tools()).tools
    assert mcp_tools_to_openai_functions(tools, allowed=frozenset({"no_such_tool"})) == []


def test_one_tool_conversion_matches_the_bulk_conversion(server: Any) -> None:
    tools = run(server, lambda session: session.list_tools()).tools
    single = mcp_tool_to_openai_function(tools[0])
    bulk = mcp_tools_to_openai_functions(tools, allowed=frozenset({tools[0].name}))
    assert single == bulk[0]


# --------------------------------------------------------------------------------------
# format_tool_call: the one place "what was checked" is rendered, shared by the CLI and the
# dashboard so the two can never again show different detail for the same kind of thing
# --------------------------------------------------------------------------------------
def test_format_tool_call_shows_name_arguments_and_success() -> None:
    call = ToolCallLog("list_trends", {"limit": 5, "qualified_only": True}, True, "ok")
    assert format_tool_call(call) == "list_trends(limit=5, qualified_only=True) [ok]"


def test_format_tool_call_marks_a_failed_call_clearly() -> None:
    call = ToolCallLog("get_entity", {"kind": "topic"}, False, "error: not found")
    assert format_tool_call(call) == "get_entity(kind='topic') [FAILED]"


def test_format_tool_call_with_no_arguments() -> None:
    call = ToolCallLog("get_overview", {}, True, "ok")
    assert format_tool_call(call) == "get_overview() [ok]"


def test_format_tool_call_quotes_string_arguments_but_not_numbers_or_booleans() -> None:
    call = ToolCallLog("x", {"name": "GSK", "limit": 3, "flag": False}, True, "ok")
    assert format_tool_call(call) == "x(name='GSK', limit=3, flag=False) [ok]"


# --------------------------------------------------------------------------------------
# The terminal shows a tool's actual outcome, not just that one was requested
# --------------------------------------------------------------------------------------
def test_a_tool_call_logs_its_full_detail_to_the_terminal(
    server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The LLM client's own logging only ever sees that a tool was *requested* - whether it
    actually succeeded is decided here, after the MCP call returns, so this is the one place
    that outcome can be logged at all."""
    import logging

    with make_llm(scripted(
        tool_call("c1", "list_trends", {"limit": 5}), text("done"),
    )) as llm:  # fmt: skip

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            with caplog.at_level(logging.INFO, logger="cews.agents.core"):
                return await run_agent_turn(
                    llm,
                    session,
                    system_prompt="x",
                    history=[],
                    user_message="q",
                    tools=tools,
                    allowed_tools=frozenset({"list_trends"}),
                    max_turns=4,
                    max_tokens=100,
                )

        run(server, action)
    assert "checked: list_trends(limit=5) [ok]" in caplog.text


def test_a_failed_tool_call_logs_as_failed_too(
    server: Any, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with make_llm(scripted(
        tool_call("c1", "get_entity", {"kind": "topic", "reference": "nonexistent"}), text("not found"),
    )) as llm:  # fmt: skip

        async def action(session: Any) -> AgentReply:
            tools = (await session.list_tools()).tools
            with caplog.at_level(logging.INFO, logger="cews.agents.core"):
                return await run_agent_turn(
                    llm,
                    session,
                    system_prompt="x",
                    history=[],
                    user_message="q",
                    tools=tools,
                    allowed_tools=frozenset({"get_entity"}),
                    max_turns=4,
                    max_tokens=100,
                )

        run(server, action)
    assert "[FAILED]" in caplog.text and "get_entity" in caplog.text
