"""Job locks and persisted job history.

Two cycles must never run at once: they would fight over the same database and double-write
the same records. The lock is an **operating-system file lock**, not a row in the database, for
three reasons that all came from a real failure:

* SQLite allows one writer at a time, and a long step (a big normalize or forecast is a single
  write transaction) can hold it for minutes. A lock kept in the database had to be renewed by
  writing to it, and that write timed out while the step ran.
* An OS lock needs no expiry and no heartbeat: it is held exactly as long as the process is
  alive, and the operating system releases it the instant the process dies, however it dies.
* It holds across a scheduler, a manual ``cews refresh`` in another terminal, and any future API
  call, since they all look at the same file.

Who holds the lock is written to a small ``.info`` file beside it, only so ``cews jobs`` can say.
The lock itself is the source of truth: a stale ``.info`` file with no lock held means nothing.

Every cycle also leaves an audit row (``job_runs``) with its outcome, so what the scheduler did,
and when it last succeeded, survives a restart.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import portalocker
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import session_scope
from cews.database.models import JobRun
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

REFRESH_JOB = "refresh"
REFRESH_LOCK = "refresh"
INTERRUPTED = "interrupted: the process stopped before this run finished"

_HELD = 0  # how many locks this process holds right now
_HELD_GUARD = threading.Lock()


class JobAlreadyRunningError(RuntimeError):
    """Raised when a cycle is asked to start while another still holds the lock."""

    def __init__(self, name: str, holder: dict[str, Any] | None = None) -> None:
        who = ""
        if holder:
            who = f" (process {holder.get('pid', '?')} since {str(holder.get('since', '?'))[:16]} UTC)"
        super().__init__(f"{name!r} is already running{who}")
        self.name = name
        self.holder = holder


def _now() -> datetime:
    return datetime.now(UTC)


def lock_directory(settings: Settings) -> Path:
    """The folder holding lock files: beside the database, so everything that shares one
    database shares one lock."""
    return settings.sqlite_path.parent / "locks"


def lock_file(directory: Path, name: str) -> Path:
    """Where the lock for ``name`` lives."""
    return directory / f"{name}.lock"


def _info_file(directory: Path, name: str) -> Path:
    return directory / f"{name}.info"


def read_holder(directory: Path, name: str) -> dict[str, Any] | None:
    """The details the current holder wrote, or None if there are none or they are unreadable."""
    try:
        loaded = json.loads(_info_file(directory, name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return dict(loaded) if isinstance(loaded, dict) else None


def lock_status(directory: Path, name: str) -> dict[str, Any] | None:
    """Who holds the lock right now, or None if it is free.

    The answer comes from the lock itself, by trying to take it: if that works, nobody holds it
    (and it is released again at once); if it does not, someone does. A leftover ``.info`` file
    from a process that died is therefore never mistaken for a live holder.
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle: IO[bytes] = lock_file(directory, name).open("a+b")
    try:
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.exceptions.LockException:
            return read_holder(directory, name) or {"pid": None, "since": None}
        portalocker.unlock(handle)
        return None
    finally:
        handle.close()


@dataclass
class LockHandle:
    """A held lock."""

    directory: Path
    name: str
    owner: str


@contextmanager
def job_lock(directory: Path, name: str = REFRESH_LOCK) -> Iterator[LockHandle]:
    """Hold the named lock for the duration of a ``with`` block.

    Raises:
        JobAlreadyRunningError: if another process (or another thread here) holds it.
    """
    global _HELD
    directory.mkdir(parents=True, exist_ok=True)
    handle: IO[bytes] = lock_file(directory, name).open("a+b")
    try:
        portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
    except portalocker.exceptions.LockException:
        handle.close()
        raise JobAlreadyRunningError(name, read_holder(directory, name)) from None

    owner = uuid.uuid4().hex[:12]
    info = _info_file(directory, name)
    try:
        info.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "owner": owner,
                    "since": _now().isoformat(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:  # the info file is a convenience; never fail the cycle over it
        LOGGER.warning("could not write %s", info)
    with _HELD_GUARD:
        _HELD += 1
    try:
        yield LockHandle(directory, name, owner)
    finally:
        with _HELD_GUARD:
            _HELD -= 1
        info.unlink(missing_ok=True)
        try:
            portalocker.unlock(handle)
        finally:
            handle.close()


def held_lock_count() -> int:
    """How many locks this process holds right now (that is, whether a cycle is in flight)."""
    with _HELD_GUARD:
        return _HELD


# ----------------------------------------------------------------------------------------
# Persisted history
# ----------------------------------------------------------------------------------------
def start_job_run(
    factory: sessionmaker[Session],
    job_name: str,
    *,
    trigger: str,
    dry_run: bool = False,
    now: datetime | None = None,
) -> str:
    """Record that a cycle has started, and return its id."""
    job_id = uuid.uuid4().hex
    with session_scope(factory) as session:
        session.add(
            JobRun(
                job_id=job_id,
                job_name=job_name,
                trigger=trigger,
                status=RunStatus.RUNNING.value,
                dry_run=dry_run,
                started_at=now or _now(),
            )
        )
    return job_id


def finish_job_run(
    factory: sessionmaker[Session],
    job_id: str,
    status: RunStatus,
    *,
    summary: dict[str, Any] | None = None,
    error: str | None = None,
    now: datetime | None = None,
) -> None:
    """Record how a cycle ended."""
    with session_scope(factory) as session:
        row = session.scalars(select(JobRun).where(JobRun.job_id == job_id)).one()
        row.status = status.value
        row.finished_at = now or _now()
        row.summary_json = summary
        row.error_summary = error


def mark_interrupted_runs(
    factory: sessionmaker[Session],
    job_names: str | Sequence[str],
    *,
    now: datetime | None = None,
) -> int:
    """Close out runs still marked running whose process is gone.

    Only call this while holding the lock: if it is held, no other cycle can legitimately be
    running, so anything still marked running was left behind by a process that stopped.
    ``job_names`` may name several kinds (a cycle also records its inner ``fetch`` run, and that
    is left dangling too when the process dies). Returns how many were closed.
    """
    names = [job_names] if isinstance(job_names, str) else list(job_names)
    moment = now or _now()
    with session_scope(factory) as session:
        result = session.execute(
            update(JobRun)
            .where(JobRun.job_name.in_(names), JobRun.status == RunStatus.RUNNING.value)
            .values(status=RunStatus.FAILED.value, finished_at=moment, error_summary=INTERRUPTED)
        )
    return int(getattr(result, "rowcount", 0) or 0)


def recent_job_runs(
    session: Session, *, job_name: str | None = None, limit: int = 10
) -> list[JobRun]:
    """The newest runs first, optionally of one kind."""
    query = select(JobRun).order_by(JobRun.started_at.desc(), JobRun.id.desc()).limit(limit)
    if job_name:
        query = query.where(JobRun.job_name == job_name)
    return list(session.scalars(query))


def last_success(session: Session, job_name: str = REFRESH_JOB) -> JobRun | None:
    """The most recent run of this kind that fully succeeded."""
    return session.scalars(
        select(JobRun)
        .where(JobRun.job_name == job_name, JobRun.status == RunStatus.SUCCEEDED.value)
        .order_by(JobRun.started_at.desc(), JobRun.id.desc())
        .limit(1)
    ).first()


def consecutive_failures(session: Session, job_name: str = REFRESH_JOB) -> int:
    """How many of the most recent finished runs failed in a row (0 if the latest succeeded)."""
    count = 0
    for run in recent_job_runs(session, job_name=job_name, limit=50):
        if run.status == RunStatus.RUNNING.value:
            continue
        if run.status != RunStatus.FAILED.value:
            break
        count += 1
    return count
