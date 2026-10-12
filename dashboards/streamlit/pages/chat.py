"""The CEWS Analyst chat, as a Streamlit *page* - opened in a separate browser tab.

This is a page under ``pages/``, Streamlit's own multipage convention: Streamlit serves it at
the relative URL ``/chat``, on the exact same server and port as ``app.py`` (confirmed by reading
Streamlit 1.65's ``source_util.py``/``app_session.py``) - there is no second process to start and
nothing to connect to, which is what made the previous two-process design fail with "connection
refused" whenever the second process had not been started by hand.

The dashboard's floating chat icon (``components/chat_widget.render_chat_launcher``) links here
with ``target="_blank"``, so clicking it opens a genuinely separate browser tab with its own
Streamlit session - a long conversation here never forces the seven-page dashboard to rerun, and
a dashboard rerun never interrupts a conversation.

``?env_file=...`` on the URL carries over whichever ``.env`` the dashboard's sidebar was using,
so this page reads the same configuration without it being typed twice; see
:func:`components.chat_widget.chat_page_url`, which builds that URL on the dashboard side.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:  # running through streamlit, not as an installed package
    sys.path.insert(0, str(ROOT / "src"))
COMPONENTS_DIR = Path(__file__).resolve().parents[1]
if str(COMPONENTS_DIR) not in sys.path:  # so "components" resolves explicitly, rather than
    sys.path.insert(0, str(COMPONENTS_DIR))  # relying on Streamlit's own path setup

from components import chat_widget  # noqa: E402

from cews.logging_config import configure_logging  # noqa: E402
from cews.settings import load_settings, resolve_env_file  # noqa: E402

st.set_page_config(page_title="CEWS Analyst", page_icon="💬", layout="centered")
st.markdown(chat_widget.HIDE_NATIVE_PAGE_NAV_CSS, unsafe_allow_html=True)
st.title("CEWS Analyst")
st.caption(
    "Ask about trends, competitors, insights and evidence. This chat runs in its own tab and "
    "session, separate from the dashboard - closing this tab does not affect it."
)


@st.cache_resource
def _configure(env_file: str) -> None:
    # Mirrors app.py's own one-time logging setup (see its docstring): `streamlit run` never
    # calls cli.py's _load(), which is what every `cews` command relies on to configure logging,
    # so without this nothing a cews.* logger writes - including the LLM request/response lines
    # - would reach the terminal at all.
    configure_logging(load_settings(env_file=resolve_env_file(env_file)))


_env_file = st.query_params.get("env_file", "")
_configure(_env_file)
chat_widget.render_chat_page(_env_file)
