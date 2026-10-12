"""The CEWS Analyst chat page, and the small launcher icon the dashboard shows for it.

This file has two halves, both used from the SAME Streamlit process as the seven-page
dashboard (``app.py``):

* :func:`render_chat_page` draws the full chat (upload, history, input, review queue) and is
  called by ``pages/chat.py`` - a Streamlit *page* (not a separate process) that Streamlit's own
  multipage mechanism serves at the relative URL ``/chat``, on the exact same server and port as
  ``app.py``. Opening it in a new browser tab (the launcher's ``target="_blank"``) gives a
  genuinely separate browser tab with its own session, so a long conversation never forces the
  rest of the dashboard to rerun - without needing a second OS process or a second port, which is
  what made the previous design fail with "connection refused" (nothing ever started that second
  process automatically).
* :func:`render_chat_launcher` draws only the small floating icon the main dashboard shows
  (``app.py``) - a plain relative link to ``chat``, opening in a new tab.
* :func:`prewarm` kicks off building the :class:`~cews.agents.session.AgentRuntime` in a
  background thread as soon as the dashboard itself starts, so that by the time someone actually
  clicks the chat icon, the MCP server and the runtime are already warm instead of making them
  wait out a ~60s cold start. Because ``@st.cache_resource`` caches per *process*, not per page or
  per browser tab, the warm runtime built here is the same one ``pages/chat.py`` gets - that
  sharing is the whole reason this moved off a second process, where ``st.cache_resource`` could
  never be shared across the two.

**How the floating position works, since this is the one non-obvious trick in this file:**
Streamlit gives ``st.container(key=...)`` a CSS class of ``st-key-<key>`` on its outer element
(confirmed by rendering a real Streamlit app in a headless browser and inspecting the DOM before
writing this — see the note in the test file). That is an ordinary block element in the normal
page flow; setting ``position: fixed`` on it via one injected ``<style>`` block is enough to make
it float over the rest of the page, pinned to the browser viewport rather than the page's normal
layout. No JavaScript, no custom component, no iframe trick is needed for this.

Nothing here writes to CEWS's database except the review panel's approve/reject, which goes
through ``cews.normalization.review_actions`` directly (see ``components/review_panel.py``) -
never through the LLM, whose job is to answer questions, not to be trusted to carry out a write
it was asked to make. An attached document is read into text in memory
(:func:`cews.agents.documents.read_document`) and lives only in ``st.session_state`` for that
browser tab; the file's raw bytes are discarded the moment its text has been extracted.
"""

from __future__ import annotations

import logging
import threading
from urllib.parse import quote

import streamlit as st

from cews.agents.core import AgentReply, format_tool_call
from cews.agents.documents import DocumentReadError, read_document
from cews.agents.personas import ANALYST
from cews.agents.session import AgentRuntime, ChatSession, RuntimeNotReadyError
from cews.ai.llm import LLMError, llm_enabled
from cews.settings import Settings, load_settings, resolve_env_file

LOGGER = logging.getLogger(__name__)

LAUNCHER_KEY = "cews-chat-launcher"
STATE_SESSION = "cews_chat_session"
STATE_DOCUMENT_IDS = "cews_chat_document_ids"
_PREWARM_FLAG_PREFIX = "cews_chat_prewarm_started"
# Whether the review queue panel is currently expanded. Starts False on every fresh session -
# the panel must never be open by default (see render_chat_page's docstring) - and is only set
# True by the person clicking "Yes, show me" below, or by manually expanding the panel itself.
STATE_SHOW_REVIEW = "cews_chat_show_review"
# Which turn (an index into ChatSession.tool_log) the open-the-review-queue prompt was last
# shown or answered for, so it is offered once per turn that actually touched the review queue,
# not on every rerun of the page for as long as that remains the latest turn.
STATE_REVIEW_PROMPT_TURN = "cews_chat_review_prompt_turn"

# The launcher button only ever needs to float bottom-left and look like a round icon; it is not
# paired with a panel - clicking it opens the chat page in a new tab instead.
_ABOVE_STREAMLIT_SIDEBAR = 1_000_000
_LAUNCHER_CSS = f"""
<style>
div.st-key-{LAUNCHER_KEY} {{
    position: fixed;
    left: 20px;
    bottom: 20px;
    z-index: {_ABOVE_STREAMLIT_SIDEBAR};
}}
div.st-key-{LAUNCHER_KEY} a {{
    display: flex;
    align-items: center;
    justify-content: center;
    width: 56px;
    height: 56px;
    border-radius: 50%;
    font-size: 1.4rem;
    text-decoration: none;
    background: var(--primary-color, #0072B2);
    color: white !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
}}
</style>
"""

# Hides Streamlit's own auto-generated multipage nav (which appears in the sidebar the moment a
# `pages/` directory exists) so it does not sit next to, and clash with, the dashboard's own
# custom `st.radio` nav in `app.py`. Confirmed selector: `[data-testid="stSidebarNav"]` is present
# in this Streamlit version's bundled JS.
HIDE_NATIVE_PAGE_NAV_CSS = """
<style>
[data-testid="stSidebarNav"] { display: none; }
</style>
"""


def chat_page_url(env_file: str) -> str:
    """A relative link to the chat page, carrying the dashboard's own ``env_file`` along so both
    pages read the same configuration without the person having to type it twice. Relative (not
    ``http://host:port/...``) on purpose - the chat page is served by this same app, not a
    separate server, so a relative link is both simpler and immune to the dashboard running on a
    different host/port than whatever was hardcoded."""
    url = "chat"
    if env_file:
        url += f"?env_file={quote(env_file)}"
    return url


def render_chat_launcher(env_file: str) -> None:
    """Draw the floating button that opens the chat page in a new browser tab.

    Call once per dashboard page render. This never starts an :class:`AgentRuntime` or checks
    the database - it is just a link - so it costs nothing on pages that never open it.
    """
    st.markdown(_LAUNCHER_CSS, unsafe_allow_html=True)
    url = chat_page_url(env_file)
    with st.container(key=LAUNCHER_KEY):
        st.markdown(
            f'<a href="{url}" target="_blank" rel="noopener" title="Open the CEWS Analyst chat '
            f'in a new tab">💬</a>',
            unsafe_allow_html=True,
        )


@st.cache_resource(show_spinner=False)
def _runtime(env_file: str) -> AgentRuntime | None:
    """One :class:`AgentRuntime` per distinct ``env_file``, kept for the app's lifetime.

    Returns None (rather than raising) when no LLM is configured at all, so the rest of the
    page can show a plain "not set up" message instead of every rerun trying and failing to
    start a runtime that was never going to work.

    ``env_file`` (a plain string, the cache key ``@st.cache_resource`` hashes on) is loaded into
    settings here rather than the caller passing a ``Settings`` object in, so this cache never has
    to hash or compare a whole settings object - just the one string that actually varies.

    Cached per *process*, not per page or browser tab - this is why :func:`prewarm`, called from
    ``app.py`` at dashboard startup, and :func:`render_chat_page`, called from the separate chat
    page, end up sharing the exact same warmed runtime instead of each building their own.
    """
    settings = load_settings(env_file=resolve_env_file(env_file))
    if not llm_enabled(settings):
        return None
    runtime = AgentRuntime(settings, env_file=env_file or None)
    runtime.start(timeout=60)
    return runtime


def prewarm(env_file: str) -> None:
    """Start building the chat runtime in the background, so it is already warm by the time
    someone clicks the chat icon, instead of them hitting a ~60s cold start (MCP server startup
    plus tool discovery) at that moment.

    Streamlit reruns the whole script on almost every interaction, so a per-session flag makes
    sure this only kicks off the background thread once per browser tab/session, not on every
    rerun. The thread calls only :func:`_runtime`, which makes no ``st.*`` calls of its own -
    Streamlit's APIs are not safe to call from a thread outside the main script run, so this
    deliberately keeps the background thread free of any of them.
    """
    flag_key = f"{_PREWARM_FLAG_PREFIX}:{env_file}"
    if st.session_state.get(flag_key):
        return
    st.session_state[flag_key] = True

    def _build() -> None:
        try:
            _runtime(env_file)
        except Exception:  # pragma: no cover - best-effort warmup, never allowed to crash
            LOGGER.exception("Background pre-warm of the chat runtime failed")

    threading.Thread(target=_build, daemon=True, name="cews-chat-prewarm").start()


def _ensure_state() -> ChatSession:
    if STATE_SESSION not in st.session_state:
        st.session_state[STATE_SESSION] = ChatSession(session_id="chat-app")
    if STATE_DOCUMENT_IDS not in st.session_state:
        st.session_state[STATE_DOCUMENT_IDS] = set()
    session: ChatSession = st.session_state[STATE_SESSION]
    return session


def _handle_upload(session: ChatSession, settings: Settings) -> None:
    """The attach control, drawn inside a small popover right next to the chat input (not a wide
    bar at the top of the page) so it reads as part of the conversation rather than a separate
    section above it."""
    with st.popover("📎 Attach", help="Attach a document to ask questions about it"):
        uploaded = st.file_uploader(
            "Attach a document (private to this conversation; never sent to CEWS's database)",
            type=["pdf", "docx", "xlsx", "xlsm", "csv", "txt", "md"],
            key="cews_chat_upload",
            label_visibility="collapsed",
        )
        if uploaded is None:
            return
        seen: set[str] = st.session_state[STATE_DOCUMENT_IDS]
        if uploaded.file_id in seen:
            st.caption(f"Already attached: {uploaded.name}")
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


def _latest_turn_touched_review_queue(session: ChatSession) -> bool:
    """True only when the most recent answer actually called ``list_review_queue`` - never from
    guessing at the person's wording. That is what the open-the-panel prompt is gated on, so
    asking "what's trending?" never triggers it just because the word "review" appeared
    somewhere, and asking about the review queue always does, regardless of phrasing."""
    if not session.tool_log:
        return False
    return any(call.name == "list_review_queue" for call in session.tool_log[-1])


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
        # ordinary chat reply rather than crashing the whole page.
        LOGGER.warning("LLM call failed during a chat turn: %s", exc)
        return AgentReply(text=f"I couldn't reach the language model: {exc}")


def render_chat_page(env_file: str) -> None:
    """Draw the full chat page: history, input with a nearby attach popover, and - collapsed by
    default, below the conversation - the same review queue the dashboard's Review page shows,
    with its own Details popup and Approve/Reject buttons.

    The review panel is NOT shown by default: it opens only when the person clicks "Yes, show
    me" on the prompt that appears right after the chat agent actually consults the review queue
    (``list_review_queue``), or when they manually expand it themselves. The chat agent itself is
    read-only and can only describe what is pending (see ``cews.agents.personas.COMMON_RULES``);
    the actual approve/reject action happens in the panel, which writes to the database directly
    - never through the LLM.

    Call once per page render. This is the whole content of ``pages/chat.py``.
    """
    # Local imports: avoids a circular import with app.py, and keeps this function's two halves
    # (chat, review queue) each importable without pulling in the other's dependencies at
    # module load time.
    from cews.database.connection import create_db_engine, create_session_factory
    from components import review_panel

    session = _ensure_state()
    settings = load_settings(env_file=resolve_env_file(env_file))

    # The review queue below does not depend on the chat agent at all - it reads and writes the
    # database directly - so a missing/misconfigured LLM must only disable the conversation half
    # of this page, never hide the review queue along with it.
    if not llm_enabled(settings):
        st.info(
            "No LLM is configured (`LLM_PROVIDER=none`), so the chat agent cannot run. "
            "Set `LLM_PROVIDER` and `LLM_MODEL` in your `.env` to use it."
        )
    else:
        with st.spinner("Connecting to the CEWS server..."):
            runtime = _runtime(env_file)
        if runtime is None:  # llm_enabled changing between checks is the only realistic way here
            st.info("No LLM is configured, so the chat agent cannot run.")
        else:
            _render_history(session)

            attach_col, _ = st.columns([1, 6])
            with attach_col:
                _handle_upload(session, settings)
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
                st.session_state[STATE_SESSION] = ChatSession(session_id="chat-app")
                st.session_state[STATE_DOCUMENT_IDS] = set()
                st.session_state[STATE_SHOW_REVIEW] = False
                st.session_state.pop(STATE_REVIEW_PROMPT_TURN, None)
                st.rerun()

            # Offer to open the review panel only right after the agent actually looked at the
            # review queue to answer this turn - and only once per turn, not on every rerun for
            # as long as that stays the latest turn (a widget key alone can't tell "already
            # answered" from "not asked yet", so the turn index itself is tracked instead).
            latest_turn = len(session.tool_log) - 1
            already_prompted = st.session_state.get(STATE_REVIEW_PROMPT_TURN) == latest_turn
            if (
                latest_turn >= 0
                and _latest_turn_touched_review_queue(session)
                and not already_prompted
            ):
                st.info("Would you like to review those now?")
                yes_col, no_col, _ = st.columns([1, 1, 4])
                if yes_col.button("Yes, show me", key="cews_chat_review_yes"):
                    st.session_state[STATE_SHOW_REVIEW] = True
                    st.session_state[STATE_REVIEW_PROMPT_TURN] = latest_turn
                    st.rerun()
                if no_col.button("No thanks", key="cews_chat_review_no"):
                    st.session_state[STATE_REVIEW_PROMPT_TURN] = latest_turn
                    st.rerun()

    st.divider()
    st.session_state.setdefault(STATE_SHOW_REVIEW, False)
    with st.expander("📋 Review queue", expanded=st.session_state[STATE_SHOW_REVIEW]):
        st.caption(
            "The chat agent above can tell you how many items are pending and what they are, "
            "but it cannot approve or reject anything itself - it is read-only. Do that here: "
            'open "Details" on an item for its full record and source evidence, then Approve '
            "or Reject."
        )
        engine = create_db_engine(settings)
        factory = create_session_factory(engine)
        review_panel.render_review_panel(factory, key_prefix="chat_review")
