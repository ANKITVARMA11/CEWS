"""Unit tests for AgentRuntime (the sync-to-async bridge) and SessionStore (per-conversation state)."""

from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session, sessionmaker

from cews.agents.documents import DocumentText
from cews.agents.personas import ANALYST, FACT_CHECKER
from cews.agents.session import (
    AgentRuntime,
    ChatSession,
    RuntimeNotReadyError,
    SessionStore,
)
from cews.ai.llm import LLMClient
from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.mcp_server.server import build_server
from cews.settings import load_settings

pytestmark = pytest.mark.unit


def real_server() -> Any:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
    settings = load_settings(env_file=None)
    return build_server(settings, factory)


def mock_llm(handler: Any) -> LLMClient:
    settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def text_reply(content: str) -> Any:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


@pytest.fixture
def runtime() -> Any:
    server = real_server()
    settings = load_settings(env_file=None)
    made = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: mock_llm(text_reply("ok")),
    )
    yield made
    made.stop()


# --------------------------------------------------------------------------------------
# AgentRuntime lifecycle
# --------------------------------------------------------------------------------------
def test_start_connects_and_lists_real_tools() -> None:
    server = real_server()
    settings = load_settings(env_file=None)
    runtime = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: mock_llm(text_reply("ok")),
    )
    runtime.start(timeout=10)
    try:
        assert len(runtime.tools) == 13
        assert {t.name for t in runtime.tools} >= {"get_overview", "list_trends"}
    finally:
        runtime.stop()


def test_ask_runs_on_the_background_thread_not_the_callers_thread() -> None:
    server = real_server()
    settings = load_settings(env_file=None)
    llm_settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["thread"] = threading.current_thread().name
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    runtime = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: LLMClient(llm_settings, transport=httpx.MockTransport(handler)),
    )
    runtime.start(timeout=10)
    try:
        reply = runtime.ask(ANALYST, "hello")
        assert reply.text == "answer"
        assert seen["thread"] != threading.main_thread().name
    finally:
        runtime.stop()


def test_ask_before_start_is_refused_clearly() -> None:
    settings = load_settings(env_file=None)
    runtime = AgentRuntime(
        settings, session_factory=lambda: create_connected_server_and_client_session(real_server())
    )
    with pytest.raises(RuntimeNotReadyError, match="start"):
        runtime.ask(ANALYST, "hi")


def test_a_connection_failure_at_start_is_raised_to_the_caller() -> None:
    @asynccontextmanager
    async def broken() -> Any:
        raise ConnectionError("no server")
        yield  # pragma: no cover - required for the generator shape, never reached

    settings = load_settings(env_file=None)
    runtime = AgentRuntime(settings, session_factory=broken)
    with pytest.raises(ConnectionError, match="no server"):
        runtime.start(timeout=10)
    runtime.stop()  # must not crash even though it never fully started


def test_stop_is_safe_to_call_repeatedly_and_before_start() -> None:
    settings = load_settings(env_file=None)
    runtime = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(real_server()),
        llm_client_factory=lambda: mock_llm(text_reply("ok")),
    )
    runtime.stop()
    runtime.stop()
    runtime.start(timeout=10)
    runtime.stop()
    runtime.stop()


def test_two_full_lifecycles_in_a_row_both_work() -> None:
    """Regression: an earlier version of AgentRuntime failed to shut down cleanly because it
    connected and disconnected in different asyncio tasks, which anyio's task groups forbid."""
    settings = load_settings(env_file=None)

    def make_runtime() -> AgentRuntime:
        server = real_server()

        def session_factory() -> Any:
            return create_connected_server_and_client_session(server)

        return AgentRuntime(
            settings,
            session_factory=session_factory,
            llm_client_factory=lambda: mock_llm(text_reply("ok")),
        )

    for _ in range(2):
        runtime = make_runtime()
        runtime.start(timeout=10)
        assert len(runtime.tools) == 13
        runtime.stop()


def test_a_real_tool_call_and_answer_round_trips_correctly(runtime: AgentRuntime) -> None:
    server = real_server()
    settings = load_settings(env_file=None)
    llm_settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )

    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "get_overview", "arguments": "{}"},
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "There is no data yet."}}]}
        )

    live = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: LLMClient(llm_settings, transport=httpx.MockTransport(handler)),
    )
    live.start(timeout=10)
    try:
        reply = live.ask(ANALYST, "how much data is there?")
        assert reply.text == "There is no data yet."
        assert reply.tool_calls[0].name == "get_overview" and reply.tool_calls[0].ok
    finally:
        live.stop()


def test_a_persona_with_a_narrower_allowlist_cannot_use_a_tool_the_analyst_can(
    runtime: AgentRuntime,
) -> None:
    server = real_server()
    settings = load_settings(env_file=None)
    llm_settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "list_trends", "arguments": "{}"},
                                }
                            ],
                        }
                    }
                ]
            },
        )

    live = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: LLMClient(llm_settings, transport=httpx.MockTransport(handler)),
    )
    live.start(timeout=10)
    try:
        reply = live.ask(FACT_CHECKER, "what's trending?")  # fact-checker cannot use list_trends
        assert reply.stopped_early is True
        assert all(not call.ok for call in reply.tool_calls)
    finally:
        live.stop()


def test_a_document_note_is_appended_to_the_message_the_model_sees(runtime: AgentRuntime) -> None:
    server = real_server()
    settings = load_settings(env_file=None)
    llm_settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen["last_user_message"] = body["messages"][-1]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    live = AgentRuntime(
        settings,
        session_factory=lambda: create_connected_server_and_client_session(server),
        llm_client_factory=lambda: LLMClient(llm_settings, transport=httpx.MockTransport(handler)),
    )
    live.start(timeout=10)
    try:
        live.ask(
            ANALYST,
            "what does this say?",
            document_note="--- report.txt ---\nConfidential figures here.",
        )
        assert "what does this say?" in seen["last_user_message"]
        assert "Confidential figures here." in seen["last_user_message"]
    finally:
        live.stop()


# --------------------------------------------------------------------------------------
# ChatSession
# --------------------------------------------------------------------------------------
def test_a_new_session_starts_on_the_analyst_persona_with_no_history() -> None:
    session = ChatSession(session_id="s1")
    assert session.persona == "analyst" and session.history == [] and session.documents == []


def test_recording_a_turn_appends_user_then_assistant() -> None:
    from cews.agents.core import AgentReply

    session = ChatSession(session_id="s1")
    session.record_turn("hello", AgentReply(text="hi there"))
    assert session.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_is_capped_and_drops_the_oldest_turns_first() -> None:
    from cews.agents.core import AgentReply

    session = ChatSession(session_id="s1")
    for i in range(20):
        session.record_turn(f"q{i}", AgentReply(text=f"a{i}"))
    from cews.agents.session import MAX_HISTORY_TURNS

    assert len(session.history) == MAX_HISTORY_TURNS * 2
    assert session.history[0]["content"] == "q8"  # the oldest kept turn, not q0
    assert session.history[-1]["content"] == "a19"


def test_a_turn_with_no_tool_calls_logs_an_empty_entry() -> None:
    from cews.agents.core import AgentReply

    session = ChatSession(session_id="s1")
    session.record_turn("hello", AgentReply(text="hi there"))
    assert session.tool_log == [()]


def test_a_turn_with_tool_calls_logs_the_full_detail_in_order_not_deduplicated() -> None:
    """Unlike an earlier version of this log (names only, deduplicated), the same tool called
    twice with different arguments is two genuinely different lookups and both are kept, in the
    order they happened - matching what `cews agent` has always printed on the command line."""
    from cews.agents.core import AgentReply, ToolCallLog

    session = ChatSession(session_id="s1")
    first = ToolCallLog("list_trends", {"limit": 5}, True, "ok")
    second = ToolCallLog("get_overview", {}, True, "ok")
    third = ToolCallLog("list_trends", {"limit": 10}, True, "ok")  # same tool, different args
    session.record_turn("what's up?", AgentReply(text="here", tool_calls=[first, second, third]))
    assert session.tool_log == [(first, second, third)]


def test_a_failed_tool_call_is_still_logged_as_checked() -> None:
    """Whether it succeeded is not the point of this log - it is a record of what was looked
    at, and a failed lookup was still looked at. Its own ok=False is preserved, not hidden."""
    from cews.agents.core import AgentReply, ToolCallLog

    session = ChatSession(session_id="s1")
    call = ToolCallLog("get_entity", {}, False, "error: not found")
    session.record_turn("q", AgentReply(text="not found", tool_calls=[call]))
    assert session.tool_log == [(call,)]
    assert session.tool_log[0][0].ok is False


def test_tool_log_stays_lined_up_with_history_one_entry_per_turn() -> None:
    from cews.agents.core import AgentReply, ToolCallLog

    session = ChatSession(session_id="s1")
    session.record_turn("q1", AgentReply(text="a1"))
    call = ToolCallLog("get_overview", {}, True, "ok")
    session.record_turn("q2", AgentReply(text="a2", tool_calls=[call]))
    assert len(session.tool_log) == len(session.history) // 2 == 2
    assert session.tool_log[0] == () and session.tool_log[1] == (call,)


def test_tool_log_is_trimmed_in_step_with_a_capped_history() -> None:
    from cews.agents.core import AgentReply, ToolCallLog
    from cews.agents.session import MAX_HISTORY_TURNS

    session = ChatSession(session_id="s1")
    for i in range(20):
        calls = [ToolCallLog(f"tool_{i}", {}, True, "ok")]
        session.record_turn(f"q{i}", AgentReply(text=f"a{i}", tool_calls=calls))
    assert len(session.tool_log) == MAX_HISTORY_TURNS == len(session.history) // 2
    # tool_log[0] must describe history[0]/history[1] (the oldest KEPT turn, "q8"/"a8"), not
    # some entry left over from a turn that was already dropped.
    assert session.tool_log[0][0].name == "tool_8"
    assert session.tool_log[-1][0].name == "tool_19"


def test_document_context_is_none_with_no_documents() -> None:
    assert ChatSession(session_id="s1").document_context() is None


def test_document_context_includes_the_note_and_the_text() -> None:
    session = ChatSession(session_id="s1")
    session.documents.append(
        DocumentText(filename="a.txt", text="secret content", full_length=14, truncated=False)
    )
    context = session.document_context()
    assert context is not None
    assert (
        "a.txt" in context and "secret content" in context and "full document included" in context
    )


def test_several_documents_are_all_included() -> None:
    session = ChatSession(session_id="s1")
    session.documents.append(DocumentText("a.txt", "content A", 9, False))
    session.documents.append(DocumentText("b.txt", "content B", 9, False))
    context = session.document_context() or ""
    assert "content A" in context and "content B" in context


# --------------------------------------------------------------------------------------
# SessionStore
# --------------------------------------------------------------------------------------
def test_a_new_session_id_creates_a_fresh_session() -> None:
    store = SessionStore()
    session = store.get_or_create(None)
    assert session.session_id and session.history == []


def test_the_same_id_returns_the_same_session() -> None:
    store = SessionStore()
    first = store.get_or_create(None)
    first.history.append({"role": "user", "content": "hi"})
    second = store.get_or_create(first.session_id)
    assert second is first and second.history == [{"role": "user", "content": "hi"}]


def test_an_unknown_id_gets_a_fresh_session_not_an_error() -> None:
    store = SessionStore()
    session = store.get_or_create("nonexistent-id")
    assert session.session_id == "nonexistent-id" and session.history == []


def test_clear_removes_a_session() -> None:
    store = SessionStore()
    session = store.get_or_create(None)
    assert store.clear(session.session_id) is True
    assert store.clear(session.session_id) is False  # already gone
    fresh = store.get_or_create(session.session_id)
    assert fresh.history == []  # a genuinely new session, not the cleared one revived


def test_idle_sessions_are_evicted() -> None:
    store = SessionStore(idle_timeout=timedelta(milliseconds=50))
    session = store.get_or_create(None)
    session_id = session.session_id
    time.sleep(0.1)
    revived = store.get_or_create(session_id)
    assert revived is not session  # evicted, so a new one was made under the same id


def test_using_a_session_resets_its_idle_clock() -> None:
    store = SessionStore(idle_timeout=timedelta(milliseconds=200))
    session = store.get_or_create(None)
    session_id = session.session_id
    time.sleep(0.1)
    store.get_or_create(session_id)  # touches it, resetting the clock
    time.sleep(0.15)
    still_there = store.get_or_create(session_id)
    assert still_there is session  # 0.25s has passed, but the touch reset the 0.2s window


def test_sessions_are_independent_of_each_other() -> None:
    store = SessionStore()
    a = store.get_or_create("a")
    b = store.get_or_create("b")
    a.history.append({"role": "user", "content": "only in a"})
    assert b.history == []
