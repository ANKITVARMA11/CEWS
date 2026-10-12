"""Tests for the floating chat launcher embedded in the main dashboard.

The chat itself lives on its own Streamlit *page* now (see ``test_chat_page.py``), served on the
same process and port as this dashboard - not a second standalone app. All this file needs to
check is that the dashboard shows a relative link to that page, that the link opens in a new tab,
that it carries over whichever ``.env`` the dashboard's sidebar is using, and that the dashboard
kicks off a background pre-warm of the chat runtime rather than waiting for someone to click the
icon before starting it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import quote

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "app.py"
if str(APP.parent) not in sys.path:
    sys.path.insert(0, str(APP.parent))


def start(env_file: Path) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["env_file"] = str(env_file)
    return app.run()


def test_the_launcher_link_is_on_every_page_load(env_file: Path) -> None:
    app = start(env_file)
    assert not app.exception
    html = "".join(m.value for m in app.markdown)
    assert "st-key-cews-chat-launcher" in html


def test_the_launcher_opens_in_a_new_tab(env_file: Path) -> None:
    app = start(env_file)
    html = "".join(m.value for m in app.markdown)
    assert 'target="_blank"' in html


def test_the_launcher_points_at_the_chat_page_relatively(env_file: Path) -> None:
    """Regression-shaped: the link must be a plain relative path to the chat page served by this
    same app, never a hardcoded ``http://host:port`` address - there is no second process or port
    to point at any more."""
    app = start(env_file)
    html = "".join(m.value for m in app.markdown)
    assert 'href="chat' in html
    assert "http://" not in html


def test_the_launcher_carries_the_env_file_along_in_the_link(env_file: Path) -> None:
    app = start(env_file)
    html = "".join(m.value for m in app.markdown)
    assert f"env_file={quote(str(env_file))}" in html


def test_the_floating_css_is_injected(env_file: Path) -> None:
    """Only checks the style rules are present - actual floating position was verified by
    rendering the real dashboard in a headless browser, not by this test."""
    app = start(env_file)
    css = "".join(m.value for m in app.markdown)
    assert "st-key-cews-chat-launcher" in css and "position: fixed" in css


def test_the_native_sidebar_page_nav_is_hidden(env_file: Path) -> None:
    """Once `pages/chat.py` exists, Streamlit auto-adds its own page nav to the sidebar; the
    dashboard hides it so it doesn't sit next to the custom `st.radio` nav."""
    app = start(env_file)
    css = "".join(m.value for m in app.markdown)
    assert 'data-testid="stSidebarNav"' in css and "display: none" in css


def test_opening_the_dashboard_prewarms_the_chat_runtime(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of pre-warming: by the time someone clicks the chat icon, the runtime
    should already be built rather than starting cold at that moment."""
    from components import chat_widget

    calls: list[str] = []

    def fake_runtime(env: str) -> None:
        calls.append(env)
        return None

    monkeypatch.setattr(chat_widget, "_runtime", fake_runtime)
    app = start(env_file)
    assert not app.exception
    for _ in range(50):
        if calls:
            break
        time.sleep(0.1)
    assert calls == [str(env_file)]


def test_prewarm_only_starts_once_per_session(
    env_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dashboard rerun (changing a widget, reloading data) must not spawn a second background
    build of the same runtime."""
    from components import chat_widget

    calls: list[str] = []

    def fake_runtime(env: str) -> None:
        calls.append(env)
        return None

    monkeypatch.setattr(chat_widget, "_runtime", fake_runtime)
    app = start(env_file)
    app.run()  # a second rerun of the same session
    for _ in range(50):
        if calls:
            break
        time.sleep(0.1)
    time.sleep(0.2)  # give a stray second thread a chance to show up, if there were one
    assert calls == [str(env_file)]
