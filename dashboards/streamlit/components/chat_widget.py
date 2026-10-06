"""A floating chat widget, embedded directly in the Streamlit dashboard.

**How the floating position works, since this is the one non-obvious trick in this file:**
Streamlit gives ``st.container(key=...)`` a CSS class of ``st-key-<key>`` on its outer element
(confirmed by rendering a real Streamlit app in a headless browser and inspecting the DOM before
writing this — see the note in the test file). That is an ordinary block element in the normal
page flow; setting ``position: fixed`` on it via one injected ``<style>`` block is enough to make
it float over the rest of the page, pinned to the browser viewport rather than the page's normal
layout. No JavaScript, no custom component, no iframe trick is needed for this.

**Why this lives in the dashboard process rather than a separate server:** the widget calls
:func:`cews.agents.session.AgentRuntime.ask` directly, in the same Python process Streamlit is
already running. ``@st.cache_resource`` (the same idiom ``app.py`` already uses for the database
connection) keeps one long-lived :class:`AgentRuntime` — one MCP connection, one background
thread — alive for as long as the dashboard runs, shared by every browser tab that opens it. Each
browser tab still gets its own private conversation: Streamlit's own ``st.session_state`` is
already isolated per session, so a :class:`~cews.agents.session.ChatSession` kept there needs no
extra bookkeeping of its own.

Nothing here writes to CEWS's database. An attached document is read into text in memory
(:func:`cews.agents.documents.read_document`) and lives only in ``st.session_state`` for that
browser tab; the file's raw bytes are discarded the moment its text has been extracted.
"""

from __future__ import annotations

import logging

import streamlit as st

from cews.agents.core import AgentReply, format_tool_call
from cews.agents.documents import DocumentReadError, read_document
from cews.agents.personas import ANALYST
from cews.agents.session import AgentRuntime, ChatSession, RuntimeNotReadyError
from cews.ai.llm import LLMError, llm_enabled
from cews.settings import Settings, load_settings, resolve_env_file

LOGGER = logging.getLogger(__name__)

TOGGLE_KEY = "cews-chat-toggle"
PANEL_KEY = "cews-chat-panel"
STATE_OPEN = "cews_chat_open"
STATE_SESSION = "cews_chat_session"
STATE_DOCUMENT_IDS = "cews_chat_document_ids"

# One block of CSS, injected once, that turns the two keyed containers above into a floating
# button and a floating panel above it. Written as plain CSS on purpose - see the module
# docstring for why this needs no JavaScript.
#
# The z-index needs a specific note: Streamlit's own sidebar (`[data-testid="stSidebar"]`) sits
# at the true left edge of the page too, at z-index 999991 (checked directly by rendering this
# dashboard in a real browser and reading the computed style - not guessed). Bottom-left, as
# asked for, is therefore *inside* the sidebar's horizontal span whenever the sidebar is open, so
# this widget's z-index has to clear the sidebar's or clicks land on the sidebar instead of the
# button, even though the button is visually drawn on top. The toggle icon itself sits low enough
# that it never overlaps the sidebar's own controls; the open panel is wider and does cover the
# sidebar while it is open, which is normal for a floating overlay and goes away the moment it is
# closed.
_ABOVE_STREAMLIT_SIDEBAR = 1_000_000
_CSS = f"""
<style>
div.st-key-{TOGGLE_KEY} {{
    position: fixed;
    left: 20px;
    bottom: 20px;
    z-index: {_ABOVE_STREAMLIT_SIDEBAR};
    width: 56px;
}}
div.st-key-{TOGGLE_KEY} button {{
    border-radius: 50%;
    width: 56px;
    height: 56px;
    font-size: 1.4rem;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}}
div.st-key-{PANEL_KEY} {{
    position: fixed;
    left: 20px;
    bottom: 88px;
    z-index: {_ABOVE_STREAMLIT_SIDEBAR - 1};
    width: 1000px;
    max-width: calc(100vw - 40px);
    max-height: 65vh;
    overflow-y: auto;
    background: var(--background-color, #333333);
    border: 1px solid rgba(128, 128, 128, 0.3);
    border-radius: 12px;
    padding: 0.75rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}
</style>
"""


@st.cache_resource(show_spinner=False)
def _runtime(env_file: str) -> AgentRuntime | None:
    """One :class:`AgentRuntime` per distinct ``env_file``, kept for the app's lifetime.

    Returns None (rather than raising) when no LLM is configured at all, so the rest of the
    widget can show a plain "not set up" message instead of every rerun trying and failing to
    start a runtime that was never going to work.

    ``env_file`` (a plain string, the cache key ``@st.cache_resource`` hashes on) is loaded into
    settings here rather than the caller passing a ``Settings`` object in, so this cache never has
    to hash or compare a whole settings object - just the one string that actually varies.
    """
    settings = load_settings(env_file=resolve_env_file(env_file))
    if not llm_enabled(settings):
        return None
    runtime = AgentRuntime(settings, env_file=env_file or None)
    runtime.start(timeout=60)
    return runtime


def _ensure_state() -> ChatSession:
    if STATE_OPEN not in st.session_state:
        st.session_state[STATE_OPEN] = False
    if STATE_SESSION not in st.session_state:
        st.session_state[STATE_SESSION] = ChatSession(session_id="dashboard")
    if STATE_DOCUMENT_IDS not in st.session_state:
        st.session_state[STATE_DOCUMENT_IDS] = set()
    session: ChatSession = st.session_state[STATE_SESSION]
    return session


def _handle_upload(session: ChatSession, settings: Settings) -> None:
    uploaded = st.file_uploader(
        "Attach a document (private to this conversation; never sent to CEWS's database)",
        type=["pdf", "docx", "xlsx", "xlsm", "csv", "txt", "md"],
        key="cews_chat_upload",
    )
    if uploaded is None:
        return
    seen: set[str] = st.session_state[STATE_DOCUMENT_IDS]
    if uploaded.file_id in seen:
        return  # already processed this exact upload; a rerun must not re-read it
    try:
        document = read_document(
            uploaded.name, uploaded.getvalue(), char_limit=settings.agent_document_char_limit
        )
    except DocumentReadError as exc:
        st.error(str(exc))
        return
    session.documents.append(document)
    seen.add(uploaded.file_id)
    st.caption(document.note)


def _render_history(session: ChatSession) -> None:
    for index, turn in enumerate(session.history):
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn["role"] == "assistant":
                # history holds one user + one assistant entry per turn, in that order, so the
                # assistant at position `index` is turn `index // 2` in tool_log - kept in step
                # with history by record_turn (see ChatSession). format_tool_call is the same
                # formatting `cews agent` itself prints, so the two never show different detail
                # for the same kind of thing.
                calls = session.tool_log[index // 2]
                if calls:
                    lines = "\n".join(f"- `{format_tool_call(call)}`" for call in calls)
                    st.caption(f"Checked:\n{lines}")


def _ask(runtime: AgentRuntime, session: ChatSession, message: str) -> AgentReply:
    document_note = session.document_context()
    try:
        return runtime.ask(ANALYST, message, history=session.history, document_note=document_note)
    except TimeoutError:
        return AgentReply(text="The analyst took too long to answer. Try a shorter question.")
    except RuntimeNotReadyError:
        return AgentReply(text="The chat agent is not ready yet. Try again in a moment.")
    except LLMError as exc:
        # A tool failing is fed back to the model as a normal tool result (see
        # cews.agents.core), so the model can react to it; the model itself being unreachable
        # has nothing left to feed the message to, so it is caught here instead and shown as an
        # ordinary chat reply rather than crashing the whole dashboard page.
        LOGGER.warning("LLM call failed during a chat turn: %s", exc)
        return AgentReply(text=f"I couldn't reach the language model: {exc}")


def render(env_file: str) -> None:
    """Draw the floating chat icon and, if open, the chat panel. Call once per page render."""
    session = _ensure_state()
    st.markdown(_CSS, unsafe_allow_html=True)

    with st.container(key=TOGGLE_KEY):
        # No `help=` tooltip here on purpose: Streamlit renders a second, hidden copy of a
        # button's markup to support a hover tooltip, which only complicates this element's DOM
        # for no real benefit on a single self-explanatory icon.
        icon = "✕" if st.session_state[STATE_OPEN] else "💬"
        if st.button(icon, key="cews_chat_toggle_button"):
            st.session_state[STATE_OPEN] = not st.session_state[STATE_OPEN]
            st.rerun()

    if not st.session_state[STATE_OPEN]:
        return

    with st.container(key=PANEL_KEY):
        st.markdown("**CEWS Analyst**")
        settings = load_settings(env_file=resolve_env_file(env_file))
        if not llm_enabled(settings):
            st.info(
                "No LLM is configured (`LLM_PROVIDER=none`), so the chat agent cannot run. "
                "Set `LLM_PROVIDER` and `LLM_MODEL` in your `.env` to use it."
            )
            return

        with st.spinner("Connecting to the CEWS server..."):
            runtime = _runtime(env_file)
        if runtime is None:  # llm_enabled changed between checks is the only realistic way here
            st.info("No LLM is configured, so the chat agent cannot run.")
            return

        _handle_upload(session, settings)
        _render_history(session)

        message = st.chat_input("Ask about trends, competitors or insights...")
        if message:
            with st.spinner("Thinking..."):
                reply = _ask(runtime, session, message)
            # record_turn saves which tools were used too, and _render_history (called again
            # right after this rerun) is what actually displays it - a caption drawn here,
            # before rerun() throws this script run away, would never be visible to a person.
            session.record_turn(message, reply)
            st.rerun()

        if st.button("Clear conversation", key="cews_chat_clear"):
            st.session_state[STATE_SESSION] = ChatSession(session_id="dashboard")
            st.session_state[STATE_DOCUMENT_IDS] = set()
            st.rerun()
