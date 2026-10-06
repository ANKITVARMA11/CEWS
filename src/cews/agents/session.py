"""Bridging a synchronous web server (Flask) to the async agent loop, and per-conversation state.

Two ideas live here, kept separate on purpose:

* :class:`AgentRuntime` owns exactly one long-lived MCP connection (the same subprocess
  ``cews mcp-serve`` launches for ``scripts/mcp_try.py``) for the whole life of the web app, and
  runs it on a background thread with its own asyncio event loop. Flask's request handlers are
  ordinary synchronous functions; :meth:`AgentRuntime.ask` lets one call into the async agent
  loop and block for its answer, without the rest of the app needing to know anything is async.
  Opening a fresh MCP connection (a new subprocess) per chat message would work too, but would
  make every message pay a subprocess-startup cost for no reason — one connection, reused, is
  both simpler to reason about and faster.

* :class:`SessionStore` holds each open conversation's message history and any document text a
  person has attached, in memory only, for as long as the process runs. Nothing here is written
  to disk or to CEWS's database: an uploaded file's *bytes* are never kept at all (see
  :mod:`cews.agents.documents` — only the extracted text lives here, and only in memory), and a
  session disappears the moment it has been idle for ``SESSION_IDLE_TIMEOUT`` or the process
  restarts. That is the deliberate design for "this may not be public data": the safest place to
  keep something private is not to persist it anywhere at all.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

from cews.agents.core import AgentReply, ToolCallLog, run_agent_turn
from cews.agents.documents import DocumentText
from cews.agents.personas import PERSONAS, Persona
from cews.ai.llm import LLMClient
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

SESSION_IDLE_TIMEOUT = timedelta(hours=4)
MAX_HISTORY_TURNS = 12  # user+assistant pairs kept; older turns are dropped, not summarized


class RuntimeNotReadyError(RuntimeError):
    """Raised if the agent runtime is used before :meth:`AgentRuntime.start` has finished."""


class AgentRuntime:
    """One background thread, one asyncio event loop, one MCP connection, for the app's lifetime."""

    def __init__(
        self,
        settings: Settings,
        *,
        env_file: str | Path | None = None,
        session_factory: Callable[[], AbstractAsyncContextManager[ClientSession]] | None = None,
        llm_client_factory: Callable[[], LLMClient] | None = None,
    ) -> None:
        """
        Args:
            env_file: the ``--env-file`` to launch ``cews mcp-serve`` with, so the subprocess
                sees the same configuration this process was given. Ignored if
                ``session_factory`` is passed directly.
            session_factory: an async context manager (called with no arguments) that yields a
                ready, initialized :class:`ClientSession`. Defaults to launching
                ``cews mcp-serve`` as a real subprocess over stdio (matches
                ``scripts/mcp_try.py``); a test supplies an in-process server here instead (see
                ``create_connected_server_and_client_session`` in the ``mcp`` package, the same
                helper :mod:`tests.unit.test_mcp_server` uses), so the fast unit tests never pay
                for spawning a real process, while a separate integration test still exercises
                the real subprocess path this defaults to.
            llm_client_factory: builds the :class:`LLMClient` used for every question; defaults
                to one built straight from ``settings``. A test overrides this to inject a
                mocked transport instead of making a real network call.
        """
        self._settings = settings
        self._session_factory = session_factory or (
            lambda: _stdio_session(_default_server_params(env_file))
        )
        self._llm_client_factory = llm_client_factory or (lambda: LLMClient(settings))
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._llm: LLMClient | None = None
        self._tools: list[Tool] = []
        self._stack = AsyncExitStack()
        self._ready = threading.Event()
        self._start_error: BaseException | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def tools(self) -> list[Tool]:
        """The MCP server's published tools, fetched once at startup."""
        return self._tools

    def start(self, *, timeout: float = 30.0) -> None:
        """Start the background loop and connect to the MCP server. Blocks until ready.

        Raises:
            RuntimeError: if the connection could not be established within ``timeout`` seconds,
                or (re-raised) whatever the connection attempt itself raised.
        """

        def run_loop() -> None:
            # On Windows, the event loop this background thread creates must be a Proactor
            # loop, explicitly - not whatever asyncio.new_event_loop() would otherwise build.
            # Streamlit's own server is built on Tornado, and Tornado sets the *process-wide*
            # asyncio event loop policy to WindowsSelectorEventLoopPolicy on Windows, because
            # its own networking code does not support ProactorEventLoop. That process-wide
            # policy is what asyncio.new_event_loop() would otherwise inherit here too, on this
            # thread - and SelectorEventLoop cannot spawn a subprocess on Windows at all, which
            # is exactly what this runtime needs to do to launch `cews mcp-serve`. Building this
            # thread's own loop from an explicit Proactor policy sidesteps the main thread's
            # (Streamlit's) policy entirely, rather than inheriting it.
            #
            # This has not been run on a real Windows machine; it is the standard, documented
            # fix for this exact conflict (see docs/agents.md).
            if sys.platform == "win32":
                loop = asyncio.WindowsProactorEventLoopPolicy().new_event_loop()
            else:
                loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            self._stop_event = asyncio.Event()
            # One coroutine, run to completion by this one call, is the connection's whole
            # lifetime: it connects, then waits for stop() to signal it, then disconnects. This
            # matters because MCP's connection setup uses anyio task groups internally, and
            # those must be entered and exited from the SAME asyncio task - splitting "connect"
            # and "disconnect" across two separately-scheduled tasks (an earlier version of this
            # method did that) makes anyio raise "cancel scope in a different task" on shutdown.
            # While this coroutine is suspended on stop_event.wait(), the loop is still very much
            # alive and free to run the short-lived tasks ask() schedules onto it meanwhile.
            loop.run_until_complete(self._lifecycle())

        self._thread = threading.Thread(target=run_loop, name="cews-agent-runtime", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(f"agent runtime did not start within {timeout}s")
        if self._start_error is not None:
            raise self._start_error

    async def _lifecycle(self) -> None:
        assert self._stop_event is not None  # set by run_loop just before this coroutine starts
        try:
            session = await self._stack.enter_async_context(self._session_factory())
            self._tools = (await session.list_tools()).tools
            self._session = session
            self._llm = self._llm_client_factory()
            LOGGER.info("agent runtime connected: %s tool(s) available", len(self._tools))
        except (
            BaseException
        ) as exc:  # noqa: BLE001 - reported to the starting thread, not swallowed
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        await self._stop_event.wait()
        await self._stack.aclose()
        if self._llm is not None:
            self._llm.close()

    def ask(
        self,
        persona: Persona,
        message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        document_note: str | None = None,
        timeout: float = 90.0,
    ) -> AgentReply:
        """Answer one question with the given persona. Safe to call from any (sync) thread.

        Args:
            document_note: text describing any attached document(s) (see
                :meth:`ChatSession.document_context`), appended to the user's message so the
                model sees it as part of what was asked, not as a separate hidden system detail.

        Raises:
            RuntimeNotReadyError: if called before :meth:`start` has completed successfully.
            TimeoutError: if the agent does not answer within ``timeout`` seconds.
        """
        if self._loop is None or self._session is None or self._llm is None:
            raise RuntimeNotReadyError("call start() before ask()")
        full_message = message if not document_note else f"{message}\n\n{document_note}"
        future = asyncio.run_coroutine_threadsafe(
            run_agent_turn(
                self._llm,
                self._session,
                system_prompt=persona.system_prompt,
                history=history or [],
                user_message=full_message,
                tools=self._tools,
                allowed_tools=persona.allowed_tools,
                max_turns=self._settings.agent_max_tool_turns,
                max_tokens=self._settings.agent_max_tokens,
            ),
            self._loop,
        )
        return future.result(timeout=timeout)

    def stop(self, *, timeout: float = 15.0) -> None:
        """Signal the lifecycle coroutine to disconnect and stop the background thread.

        Safe to call more than once, and safe to call even if :meth:`start` never finished
        successfully.
        """
        if self._loop is None or self._thread is None or self._stop_event is None:
            return
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            LOGGER.warning("agent runtime thread did not stop within %ss", timeout)
        self._loop = None
        self._thread = None


def _default_server_params(env_file: str | Path | None) -> StdioServerParameters:
    """Launch ``cews mcp-serve`` exactly as ``scripts/mcp_try.py`` does.

    ``env=dict(os.environ)`` matters: the MCP SDK passes a spawned server only a short built-in
    whitelist of environment variables by default, not the parent process's own environment. That
    is fine for a "python" found on PATH with cews properly installed, but silently breaks a
    subprocess that needs anything from the parent's environment the whitelist does not cover
    (for example this project's own ``PYTHONPATH=src`` development setup) - the child fails to
    even import ``cews``, and the only symptom on this side is "Connection closed". Caught by
    actually running this command, not by any of the mocked-transport tests, which never spawn a
    real subprocess and so never exercise this.
    """
    import os

    args = ["-m", "cews.cli", "mcp-serve"]
    if env_file:
        args += ["--env-file", str(env_file)]
    return StdioServerParameters(command=sys.executable, args=args, env=dict(os.environ))


@asynccontextmanager
async def _stdio_session(params: StdioServerParameters) -> AsyncGenerator[ClientSession, None]:
    """The default session factory: launch the real server over stdio and initialize it.

    ``errlog`` is passed explicitly as ``sys.__stderr__`` (Python's permanent reference to the
    real original stream, untouched by output capturing) rather than leaving ``stdio_client`` to
    its own default. That default is a *mutable default argument*, bound once to whatever
    ``sys.stderr`` is the moment ``mcp.client.stdio`` is first imported - and because this whole
    module is only imported lazily, from inside ``cews.cli``'s command functions, "the moment
    it's first imported" can land inside a test that has swapped ``sys.stderr`` for a
    non-file-backed capture object (``pytest``'s ``capsys``), which has no real file descriptor.
    ``stdio_client`` needs one, to hand the spawned server process a real stderr to write to, and
    raises a bare ``fileno`` error otherwise. Found by an integration test failing with exactly
    that error, not by inspection.
    """
    async with (
        stdio_client(params, errlog=sys.__stderr__ or sys.stderr) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


# ------------------------------------------------------------------------------------------
# Per-conversation state
# ------------------------------------------------------------------------------------------
@dataclass
class ChatSession:
    """One open conversation: its history, persona, and any documents attached to it."""

    session_id: str
    persona: str = PERSONAS["analyst"].name
    history: list[dict[str, Any]] = field(default_factory=list)
    # Every tool call an answer actually made, kept only for display (a "checked: ..." note
    # next to the message it belongs to) - one entry per turn, in step with `history`, holding
    # the FULL cews.agents.core.ToolCallLog for each call (name, arguments, ok, summary) rather
    # than just a name, so the dashboard can show the same detail cews.cli._print_agent_reply
    # already shows for the command line: what was actually asked of each tool, and whether it
    # succeeded. Calls are kept in order and NOT deduplicated by name, again to match the CLI:
    # the same tool called twice with different arguments is two genuinely different lookups,
    # both worth showing. This is deliberately NOT part of `history` itself: everything in
    # `history` is sent back to the LLM as the conversation so far, and a model has no use for -
    # and should not be shown - a UI-only annotation about what a *previous* answer looked up.
    tool_log: list[tuple[ToolCallLog, ...]] = field(default_factory=list)
    documents: list[DocumentText] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_active_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def record_turn(self, user_message: str, reply: AgentReply) -> None:
        """Add one exchange to the saved history, keeping only the most recent turns.

        Only the user's message and the agent's final text are kept in ``history`` — not the
        tool-call back-and-forth that produced it (see
        :func:`cews.agents.core.run_agent_turn`), so the history sent with each new question
        stays short and does not repeat every past tool call. Which tools were used, and with
        what arguments and outcome, is kept separately, in ``tool_log``, purely for display -
        see its own docstring above.
        """
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply.text})
        self.tool_log.append(tuple(reply.tool_calls))
        overflow = len(self.history) - MAX_HISTORY_TURNS * 2
        if overflow > 0:
            del self.history[:overflow]
            del self.tool_log[: overflow // 2]

    def document_context(self) -> str | None:
        """The attached documents' text and truncation notes, ready to append to a question.

        Returns None when nothing is attached, so a question with no document looks exactly like
        it always did.
        """
        if not self.documents:
            return None
        parts = [
            "The person has attached the following document(s) to this conversation. Their full "
            "text is already here - answer questions about them directly from this text; no "
            "tool call can read them."
        ]
        for doc in self.documents:
            parts.append(f"\n--- {doc.note} ---\n{doc.text}")
        return "\n".join(parts)


class SessionStore:
    """In-memory sessions, keyed by id, evicted after they have been idle too long."""

    def __init__(self, *, idle_timeout: timedelta = SESSION_IDLE_TIMEOUT) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> ChatSession:
        """The session for ``session_id``, or a brand new one if it is missing or unknown."""
        with self._lock:
            self._evict_idle_locked()
            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.last_active_at = datetime.now(UTC)
                return session
            new_id = session_id or uuid.uuid4().hex
            session = ChatSession(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def clear(self, session_id: str) -> bool:
        """Remove one session (history and any attached documents) entirely. Returns whether it
        existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _evict_idle_locked(self) -> None:
        cutoff = datetime.now(UTC) - self._idle_timeout
        stale = [key for key, session in self._sessions.items() if session.last_active_at < cutoff]
        for key in stale:
            del self._sessions[key]
