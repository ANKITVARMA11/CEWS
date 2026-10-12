"""Tests for the chat page (``pages/chat.py``), using Streamlit's own test harness.

This is where the real chat behaviour lives - the main dashboard (``app.py``) only shows a link
to it (see ``test_chat_widget.py``). This page is served by the same Streamlit process and port
as the dashboard (Streamlit's own multipage ``pages/`` convention), not a second standalone app -
that is what fixed "This site can't be reached" from clicking the chat icon, since there used to
be nothing that actually started a second process on that port. These tests cover the same ground
the old embedded widget's tests did: a script-level mistake (the page never calling
``render_chat_page``, the "not configured" message vanishing, a tool call going unlogged) is
caught the same way a break in any dashboard page would be. They cannot check the floating
launcher's actual position in a browser; that was verified separately by rendering the real
dashboard in a headless browser (see ``docs/agents.md``).

Monkeypatching ``chat_widget._runtime`` before ``AppTest.run()`` is what lets the "LLM
configured" tests inject a real in-process MCP session (no subprocess) and a mocked LLM
transport, instead of every test needing a live model: confirmed to actually take effect across
an ``AppTest`` run before relying on it here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from cews.cli import EXIT_OK, main
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

CHAT_PAGE = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "pages" / "chat.py"
# Import exactly the way the page itself does ("from components import chat_widget", with
# dashboards/streamlit/ on sys.path) so this file's `chat_widget` and the page's are the *same*
# module object under the same sys.modules key - see test_chat_widget.py's note on why this
# matters for monkeypatching to actually take effect.
_STREAMLIT_DIR = CHAT_PAGE.resolve().parents[1]
if str(_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(_STREAMLIT_DIR))


def start(env_file: Path) -> AppTest:
    app = AppTest.from_file(str(CHAT_PAGE), default_timeout=120)
    app.query_params["env_file"] = str(env_file)
    return app.run()


def with_llm_configured(env_file: Path) -> Path:
    """A copy of ``env_file`` with an LLM provider declared."""
    configured = env_file.with_name("with_llm.env")
    configured.write_text(
        env_file.read_text(encoding="utf-8") + "\nLLM_PROVIDER=ollama\nLLM_BASE_URL=http://x/v1\n",
        encoding="utf-8",
    )
    return configured


def patch_runtime(
    monkeypatch: pytest.MonkeyPatch, *, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Make the page's cached runtime use a real in-process MCP session and a mocked LLM.

    Imported here, not at module scope, so a test file that never calls this never needs the
    dashboard's own database/MCP fixtures at collection time.
    """
    from components import chat_widget
    from mcp.shared.memory import create_connected_server_and_client_session
    from sqlalchemy.orm import sessionmaker

    from cews.agents.session import AgentRuntime
    from cews.ai.llm import LLMClient
    from cews.database.connection import create_db_engine, create_session_factory
    from cews.mcp_server.server import build_server
    from cews.settings import load_settings

    def fake_runtime(env_file: str) -> AgentRuntime:
        settings = load_settings(env_file=env_file or None)
        factory: sessionmaker = create_session_factory(create_db_engine(settings))
        server = build_server(settings, factory)
        llm_settings = load_settings(
            env_file=env_file or None,
            overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"},
        )
        runtime = AgentRuntime(
            settings,
            session_factory=lambda: create_connected_server_and_client_session(server),
            llm_client_factory=lambda: LLMClient(
                llm_settings, transport=httpx.MockTransport(handler)
            ),
        )
        runtime.start(timeout=15)
        return runtime

    monkeypatch.setattr(chat_widget, "_runtime", fake_runtime)


def text_only(content: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    return handler


def tool_call_then_text(
    tool_name: str, final_text: str
) -> Callable[[httpx.Request], httpx.Response]:
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        message: dict[str, Any]
        if calls["n"] == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            }
        else:
            message = {"content": final_text}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return handler


def two_rounds_of_tools_then_text(
    first_tool: str, second_tool: str, final_text: str
) -> Callable[[httpx.Request], httpx.Response]:
    """Two separate tool-calling turns before a final answer - what was actually seen from a
    real model (NVIDIA NIM's nemotron-3-super-120b), not a hypothetical."""
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        message: dict[str, Any]
        if calls["n"] == 1:
            tool_name = first_tool
        elif calls["n"] == 2:
            tool_name = second_tool
        else:
            tool_name = ""
        if tool_name:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{calls['n']}",
                        "type": "function",
                        "function": {"name": tool_name, "arguments": "{}"},
                    }
                ],
            }
        else:
            message = {"content": final_text}
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return handler


# --------------------------------------------------------------------------------------
# The page is present and titled
# --------------------------------------------------------------------------------------
def test_the_chat_page_starts(env_file: Path) -> None:
    app = start(env_file)
    assert not app.exception
    assert app.title[0].value == "CEWS Analyst"


def test_the_native_sidebar_page_nav_is_hidden_here_too(env_file: Path) -> None:
    app = start(env_file)
    css = "".join(m.value for m in app.markdown)
    assert 'data-testid="stSidebarNav"' in css and "display: none" in css


# --------------------------------------------------------------------------------------
# The review queue is reachable from this same page, with its own total/paging - but collapsed
# by default (regression: it used to be open on every page load, which is not what anyone asking
# a plain question in the chat wants to see first).
# --------------------------------------------------------------------------------------
def test_the_review_queue_exists_but_is_collapsed_by_default(env_file: Path) -> None:
    app = start(env_file)
    assert not app.exception
    expanders = [e for e in app.get("expander") if "Review queue" in e.label]
    assert len(expanders) == 1
    assert expanders[0].proto.expanded is False


def test_asking_about_the_review_queue_offers_to_open_the_panel(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression-shaped: the fix for "always shown" must not make the panel unreachable from
    the chat - once the agent actually consults the review queue to answer, the page should
    offer to open it, and clicking "Yes, show me" should actually expand it."""
    patch_runtime(
        monkeypatch,
        handler=tool_call_then_text("list_review_queue", "There are 3 items pending."),
    )
    app = start(with_llm_configured(env_file))
    app = app.chat_input[0].set_value("how many documents are left to review?").run()
    assert not app.exception
    assert any("review those now" in i.value for i in app.info)

    yes = next(b for b in app.button if b.key == "cews_chat_review_yes")
    app = yes.click().run()
    assert not app.exception
    expanders = [e for e in app.get("expander") if "Review queue" in e.label]
    assert expanders[0].proto.expanded is True


def test_a_question_unrelated_to_review_never_offers_the_panel(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("Nothing is trending yet."))
    app = start(with_llm_configured(env_file))
    app = app.chat_input[0].set_value("what's trending?").run()
    assert not app.exception
    assert not any("review those now" in i.value for i in app.info)
    expanders = [e for e in app.get("expander") if "Review queue" in e.label]
    assert expanders[0].proto.expanded is False


# --------------------------------------------------------------------------------------
# The upload control sits next to the chat input, not in a bar across the top of the page
# --------------------------------------------------------------------------------------
def test_the_attach_control_exists_next_to_the_chat_input(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression-shaped: the uploader used to be a wide ``st.file_uploader`` drawn before
    anything else on the page. It now lives inside a small ``st.popover`` placed immediately
    above the chat input (only drawn once the chat agent itself is available, same as the chat
    input it sits next to)."""
    patch_runtime(monkeypatch, handler=text_only("answer"))
    app = start(with_llm_configured(env_file))
    assert not app.exception
    uploaders = app.get("file_uploader")
    assert len(uploaders) == 1
    assert uploaders[0].key == "cews_chat_upload"


# --------------------------------------------------------------------------------------
# No LLM configured
# --------------------------------------------------------------------------------------
def test_with_no_llm_configured_the_page_explains_that_instead_of_a_chat_box(
    env_file: Path,
) -> None:
    app = start(env_file)
    assert not app.exception
    assert any("No LLM is configured" in m.value for m in app.info)
    assert app.chat_input == []  # no chat box offered when nothing can answer it


# --------------------------------------------------------------------------------------
# A real (mocked-transport) exchange through the page
# --------------------------------------------------------------------------------------
def test_a_question_gets_a_real_answer(env_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_runtime(monkeypatch, handler=text_only("Nothing is trending yet."))
    app = start(with_llm_configured(env_file))
    assert not app.exception
    assert len(app.chat_input) == 1

    app.chat_input[0].set_value("what's trending?").run()
    assert not app.exception
    messages = {m.name: m for m in app.chat_message}
    assert "user" in messages and "assistant" in messages
    assistant_text = "".join(el.value for el in messages["assistant"].markdown)
    assert "Nothing is trending yet." in assistant_text


def test_tool_names_from_several_separate_tool_calling_rounds_are_all_shown(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression-shaped: a real exchange with a real model made two separate tool-calling
    round trips (list_trends, then get_overview) before its final answer, and every tool used
    across the whole turn - not just the last round trip - must show up in the caption."""
    patch_runtime(
        monkeypatch,
        handler=two_rounds_of_tools_then_text("list_trends", "get_overview", "Here is the answer."),
    )
    app = start(with_llm_configured(env_file))
    app.chat_input[0].set_value("what's trending, and how much data is there overall?").run()
    assert not app.exception
    caption = next(c.value for c in app.caption if "Checked:" in c.value)
    assert "list_trends" in caption and "get_overview" in caption


def test_which_tool_was_checked_is_still_shown_after_the_rerun_that_follows_each_answer(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the widget used to draw a "Checked: ..." caption in the same script run that
    then immediately called st.rerun(), which throws that run's output away before a person
    could ever see it - so the note effectively never appeared. It has to be read back from
    persisted state (ChatSession.tool_log) on the run that actually renders the page."""
    patch_runtime(monkeypatch, handler=tool_call_then_text("get_overview", "There is no data yet."))
    app = start(with_llm_configured(env_file))
    app.chat_input[0].set_value("how much data is there?").run()
    assert not app.exception
    caption = next(c.value for c in app.caption if "Checked:" in c.value)
    assert "get_overview" in caption and "[ok]" in caption


def test_a_turn_with_no_tool_calls_shows_no_checked_caption(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("A plain answer, no lookup needed."))
    app = start(with_llm_configured(env_file))
    app.chat_input[0].set_value("hello").run()
    assert not any("Checked:" in c.value for c in app.caption)


def test_asking_a_question_logs_a_request_and_response_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: unlike every `cews` command (which sets this up through cli.py's own
    _load()), nothing configured logging when this app was launched directly with
    `streamlit run` - so cews.ai.llm.client's request/response lines, and every other cews.*
    logger, had no handler at all and were silently dropped, at any level.

    Checked against real stdout (capsys), not the `caplog` fixture: `configure_logging` calls
    `logging.config.dictConfig`, which replaces the `cews` logger's handlers outright each time
    it runs - discarding a handler `caplog` had attached before that point - so `caplog` cannot
    see these records even though they are genuinely written to the terminal `capsys` reads
    from, which is what "visible in the terminal" actually means here.

    Uses its own database and env file, under this test's own `tmp_path`, rather than the
    `env_file` fixture other tests here share: `_configure` is `@st.cache_resource`, keyed on
    the env file's path, and persists across separate `AppTest` runs within one pytest process -
    a shared path already "warmed" by an earlier test would skip `configure_logging` here
    entirely.
    """
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    patch_runtime(monkeypatch, handler=text_only("An answer."))
    app = start(env)
    app.chat_input[0].set_value("hi").run()
    assert not app.exception
    out = capsys.readouterr().out
    assert "LLM request ->" in out and "LLM response <-" in out


def test_a_second_question_in_the_same_session_keeps_the_first_in_view(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("answer"))
    app = start(with_llm_configured(env_file))
    app.chat_input[0].set_value("first question").run()
    app.chat_input[0].set_value("second question").run()
    user_texts = [
        "".join(el.value for el in m.markdown) for m in app.chat_message if m.name == "user"
    ]
    assert "first question" in user_texts and "second question" in user_texts


def test_clear_conversation_empties_the_history(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("answer"))
    app = start(with_llm_configured(env_file))
    app.chat_input[0].set_value("a question").run()
    assert any(m.name == "user" for m in app.chat_message)

    next(b for b in app.button if b.key == "cews_chat_clear").click().run()
    assert not any(m.name == "user" for m in app.chat_message)
