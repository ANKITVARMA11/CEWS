"""Tests for the floating chat widget, using Streamlit's own test harness.

These run the real ``app.py`` (through ``AppTest``, the same tool ``test_app.py`` uses for the
other seven pages) so a script-level mistake - the widget never being called, the toggle wiring
breaking, the "not configured" message vanishing - is caught the same way a break in any other
page would be. They cannot check the actual floating position in a browser; that was verified
separately by rendering the real dashboard in a headless browser and reading the button's
computed CSS position directly (see ``docs/agents.md`` for what was checked and how).

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
from streamlit.testing.v1.element_tree import Button

from cews.cli import EXIT_OK, main
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "app.py"
# Import exactly the way app.py itself does ("from components import chat_widget", with
# app.py's own directory on sys.path) so this file's `chat_widget` and app.py's are the *same*
# module object under the same sys.modules key. Setting up sys.path differently here - e.g.
# adding the components/ directory itself and doing a bare "import chat_widget" - registers a
# second, distinct module under a different name, and patching that one has no effect on what
# app.py actually calls; this was caught by a test that silently launched a real subprocess
# instead of using the mocked LLM, rather than by any static check.
if str(APP.parent) not in sys.path:
    sys.path.insert(0, str(APP.parent))


def _click(app: AppTest, key: str) -> Button:
    """``get_by_key`` returns a plain ``Node``; every key used here names a real button."""
    widget = app.get_by_key(key)
    assert isinstance(widget, Button), f"{key!r} is not a button: {type(widget)}"
    return widget.click()


def start(env_file: Path) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["env_file"] = str(env_file)
    return app.run()


def start_with_empty_config_box(monkeypatch: pytest.MonkeyPatch, real_env_file: Path) -> AppTest:
    """Load the whole dashboard - not just the widget - the way a never-touched page actually
    loads: the "Configuration file" sidebar box left blank, exactly as it is on first render.

    Regression test: ``resolve_env_file`` is what stands between an empty box and
    ``load_settings(env_file=None)``, which means "skip dotenv files entirely" rather than "try
    the project's own .env" - a real, once-shipped bug where the dashboard (unlike `cews`
    commands, which have always defaulted correctly) silently never loaded ANY `.env`, blank box
    or not, and both the main dashboard and the chat widget quietly ran on bare defaults instead.
    """
    monkeypatch.setattr("cews.settings.DEFAULT_ENV_FILE", real_env_file)
    app = AppTest.from_file(str(APP), default_timeout=120)
    return app.run()  # session_state["env_file"] is never touched: stays unset, like a fresh load


def test_leaving_the_config_box_empty_still_loads_the_real_database(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = start_with_empty_config_box(monkeypatch, env_file)
    assert not app.exception
    # Under the bug, settings silently fall back to bare defaults (a database path that does not
    # exist), and this exact message is what app.py shows for that - confirmed by reverting the
    # fix and reading what actually rendered, not guessed.
    assert not any("has not been created yet" in m.value for m in app.error)
    # And positively: a real, non-empty count from the prepared database actually appears.
    assert any(m.value.strip().isdigit() and int(m.value) > 0 for m in app.metric)


def test_leaving_the_config_box_empty_still_finds_a_configured_llm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_env = tmp_path / ".env"
    real_env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(real_env)]) == EXIT_OK
    app = start_with_empty_config_box(monkeypatch, real_env)
    _click(app, "cews_chat_toggle_button").run()
    assert not any("No LLM is configured" in m.value for m in app.info)


def with_llm_configured(env_file: Path) -> Path:
    """A copy of ``env_file`` with an LLM provider declared.

    ``render()`` checks ``llm_enabled(settings)`` - read from the real env file, not from
    ``_runtime`` - before it ever calls ``_runtime`` at all, so a test that wants to reach the
    mocked-LLM path needs a provider actually set here; ``patch_runtime`` below only replaces
    what that provider then talks to.
    """
    configured = env_file.with_name("with_llm.env")
    configured.write_text(
        env_file.read_text(encoding="utf-8") + "\nLLM_PROVIDER=ollama\nLLM_BASE_URL=http://x/v1\n",
        encoding="utf-8",
    )
    return configured


def patch_runtime(
    monkeypatch: pytest.MonkeyPatch, *, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Make the widget's cached runtime use a real in-process MCP session and a mocked LLM.

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


# --------------------------------------------------------------------------------------
# The widget is present and wired into the dashboard
# --------------------------------------------------------------------------------------
def test_the_toggle_button_is_on_every_page_load(env_file: Path) -> None:
    app = start(env_file)
    assert not app.exception
    assert app.get_by_key("cews_chat_toggle_button") is not None


def test_the_floating_css_is_injected(env_file: Path) -> None:
    """Only checks the style rules are present - actual floating position was verified by
    rendering the real dashboard in a headless browser, not by this test."""
    app = start(env_file)
    css = "".join(m.value for m in app.markdown)
    assert "st-key-cews-chat-toggle" in css and "position: fixed" in css


def test_the_panel_is_closed_by_default(env_file: Path) -> None:
    app = start(env_file)
    assert app.get_by_key("cews_chat_toggle_button") is not None
    # the panel's own contents (its heading) should not be findable in a fresh, unopened session
    assert not any("CEWS Analyst" in m.value for m in app.markdown)


def test_clicking_the_icon_opens_the_panel(env_file: Path) -> None:
    app = start(env_file)
    _click(app, "cews_chat_toggle_button").run()
    assert any("CEWS Analyst" in m.value for m in app.markdown)


def test_clicking_the_icon_twice_closes_it_again(env_file: Path) -> None:
    app = start(env_file)
    _click(app, "cews_chat_toggle_button").run()
    _click(app, "cews_chat_toggle_button").run()
    assert not any("CEWS Analyst" in m.value for m in app.markdown)


# --------------------------------------------------------------------------------------
# No LLM configured
# --------------------------------------------------------------------------------------
def test_with_no_llm_configured_the_panel_explains_that_instead_of_a_chat_box(
    env_file: Path,
) -> None:
    app = start(env_file)
    _click(app, "cews_chat_toggle_button").run()
    assert not app.exception
    assert any("No LLM is configured" in m.value for m in app.info)
    assert app.chat_input == []  # no chat box offered when nothing can answer it


# --------------------------------------------------------------------------------------
# A real (mocked-transport) exchange through the widget
# --------------------------------------------------------------------------------------
def test_a_question_gets_a_real_answer_through_the_widget(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("Nothing is trending yet."))
    app = start(with_llm_configured(env_file))
    _click(app, "cews_chat_toggle_button").run()
    assert not app.exception
    assert len(app.chat_input) == 1

    app.chat_input[0].set_value("what's trending?").run()
    assert not app.exception
    messages = {m.name: m for m in app.chat_message}
    assert "user" in messages and "assistant" in messages
    assistant_text = "".join(el.value for el in messages["assistant"].markdown)
    assert "Nothing is trending yet." in assistant_text


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
    _click(app, "cews_chat_toggle_button").run()
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
    _click(app, "cews_chat_toggle_button").run()
    app.chat_input[0].set_value("how much data is there?").run()
    assert not app.exception
    caption = next(c.value for c in app.caption if "Checked:" in c.value)
    assert "get_overview" in caption and "[ok]" in caption


def test_a_turn_with_no_tool_calls_shows_no_checked_caption(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("A plain answer, no lookup needed."))
    app = start(with_llm_configured(env_file))
    _click(app, "cews_chat_toggle_button").run()
    app.chat_input[0].set_value("hello").run()
    assert not any("Checked:" in c.value for c in app.caption)


def test_asking_a_question_logs_a_request_and_response_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: unlike every `cews` command (which sets this up through cli.py's own
    _load()), nothing configured logging when the dashboard was launched directly with
    `streamlit run` - so cews.ai.llm.client's request/response lines, and every other cews.*
    logger, had no handler at all and were silently dropped, at any level.

    Checked against real stdout (capsys), not the `caplog` fixture: `configure_logging` calls
    `logging.config.dictConfig`, which replaces the `cews` logger's handlers outright each time
    it runs - discarding a handler `caplog` had attached before that point - so `caplog` cannot
    see these records even though they are genuinely written to the terminal `capsys` reads
    from, which is what "visible in the terminal" actually means here.

    Uses its own database and env file, under this test's own `tmp_path`, rather than the
    `env_file` fixture other tests here share: `_session_factory` is `@st.cache_resource`, keyed
    on the env file's path, and persists across separate `AppTest` runs within one pytest
    process - a shared path already "warmed" by an earlier test would skip configure_logging
    here entirely (it only runs on a fresh cache entry), and even if it did not, the console
    handler it bound during that earlier test would be writing into a stdout object pytest has
    since swapped out from under it, invisible to this test's own `capsys`.
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
    _click(app, "cews_chat_toggle_button").run()
    app.chat_input[0].set_value("hi").run()
    assert not app.exception
    out = capsys.readouterr().out
    assert "LLM request ->" in out and "LLM response <-" in out


def test_a_second_question_in_the_same_session_keeps_the_first_in_view(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch_runtime(monkeypatch, handler=text_only("answer"))
    app = start(with_llm_configured(env_file))
    _click(app, "cews_chat_toggle_button").run()
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
    _click(app, "cews_chat_toggle_button").run()
    app.chat_input[0].set_value("a question").run()
    assert any(m.name == "user" for m in app.chat_message)

    _click(app, "cews_chat_clear").run()
    assert not any(m.name == "user" for m in app.chat_message)
