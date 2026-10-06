"""Unit tests for the job lock and persisted job history."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import JobLock, JobRun
from cews.scheduler.job_state import (
    INTERRUPTED,
    JobAlreadyRunningError,
    consecutive_failures,
    finish_job_run,
    held_lock_count,
    job_lock,
    last_success,
    lock_directory,
    lock_file,
    lock_status,
    mark_interrupted_runs,
    read_holder,
    recent_job_runs,
    start_job_run,
)
from cews.settings import load_settings

pytestmark = pytest.mark.unit

T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    settings = load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./jobs.db")}
    )
    engine = create_db_engine(settings)
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def locks(tmp_path: Path) -> Path:
    return tmp_path / "locks"


# --------------------------------------------------------------------------------------
# The lock
# --------------------------------------------------------------------------------------
def test_a_free_lock_can_be_taken_and_is_free_again_afterwards(locks: Path) -> None:
    assert lock_status(locks, "refresh") is None
    with job_lock(locks, "refresh"):
        assert lock_status(locks, "refresh") is not None
    assert lock_status(locks, "refresh") is None


def test_a_held_lock_refuses_a_second_taker_with_a_clear_message(locks: Path) -> None:
    with (
        job_lock(locks, "refresh"),
        pytest.raises(JobAlreadyRunningError, match="already running"),
        job_lock(locks, "refresh"),
    ):
        pass


def test_the_refusal_says_which_process_holds_it(locks: Path) -> None:
    with (
        job_lock(locks, "refresh"),
        pytest.raises(JobAlreadyRunningError) as caught,
        job_lock(locks, "refresh"),
    ):
        pass
    assert caught.value.holder is not None and caught.value.holder["pid"] == os.getpid()
    assert f"process {os.getpid()}" in str(caught.value)


def test_different_names_do_not_block_each_other(locks: Path) -> None:
    with job_lock(locks, "refresh"), job_lock(locks, "other"):
        assert lock_status(locks, "other") is not None


def test_the_lock_is_released_when_the_block_fails(locks: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), job_lock(locks, "refresh"):
        raise RuntimeError("boom")
    assert lock_status(locks, "refresh") is None
    with job_lock(locks, "refresh"):  # and can be taken again straight away
        pass


def test_the_holder_is_recorded_while_held_and_removed_after(locks: Path) -> None:
    with job_lock(locks, "refresh"):
        holder = read_holder(locks, "refresh")
        assert holder is not None and holder["pid"] == os.getpid() and holder["since"]
    assert read_holder(locks, "refresh") is None


def test_lock_status_reports_the_holder(locks: Path) -> None:
    with job_lock(locks, "refresh"):
        status = lock_status(locks, "refresh")
        assert status is not None and status["pid"] == os.getpid()


def test_a_leftover_info_file_with_no_lock_is_not_mistaken_for_a_live_holder(locks: Path) -> None:
    """A process that died leaves its info file behind; only the lock itself is the truth."""
    locks.mkdir()
    (locks / "refresh.info").write_text(
        json.dumps({"pid": 99999, "since": "2026-01-01"}), encoding="utf-8"
    )
    assert lock_status(locks, "refresh") is None
    with job_lock(locks, "refresh"):  # and it can simply be taken over
        holder = read_holder(locks, "refresh")
        assert holder is not None and holder["pid"] == os.getpid()


@pytest.mark.parametrize("content", ["", "not json", "[1, 2]", "\x00\x01"])
def test_an_unreadable_info_file_is_ignored(locks: Path, content: str) -> None:
    locks.mkdir()
    (locks / "refresh.info").write_text(content, encoding="utf-8")
    assert read_holder(locks, "refresh") is None


def test_the_lock_directory_is_created_on_demand(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "locks"
    with job_lock(deep, "refresh"):
        assert lock_file(deep, "refresh").is_file()


def test_the_lock_lives_beside_the_database(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./data/cews.db")}
    )
    assert lock_directory(settings) == tmp_path / "data" / "locks"


def test_held_lock_count_follows_the_context(locks: Path) -> None:
    before = held_lock_count()
    with job_lock(locks, "refresh"):
        assert held_lock_count() == before + 1
    assert held_lock_count() == before


def test_exactly_one_of_many_simultaneous_callers_gets_the_lock(locks: Path) -> None:
    """The point of a lock: a real race, from many threads, has one winner."""
    winners: list[str] = []
    refused: list[str] = []
    barrier = threading.Barrier(8)
    hold = threading.Event()

    def contend(name: str) -> None:
        barrier.wait()
        try:
            with job_lock(locks, "refresh"):
                winners.append(name)
                hold.wait(timeout=5)
        except JobAlreadyRunningError:
            refused.append(name)

    threads = [threading.Thread(target=contend, args=(f"t{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    time.sleep(0.5)
    hold.set()
    for thread in threads:
        thread.join(timeout=30)
    assert len(winners) == 1 and len(refused) == 7


# --------------------------------------------------------------------------------------
# Regression: the failure that prompted this design
# --------------------------------------------------------------------------------------
def test_the_lock_works_while_the_database_is_write_locked(tmp_path: Path, locks: Path) -> None:
    """A long step holds SQLite's single writer for minutes. The lock must not need the database
    at all, or (as it once did) renewing it fails with 'database is locked' mid-cycle."""
    database = tmp_path / "busy.db"
    writer = sqlite3.connect(database, isolation_level=None)
    writer.execute("CREATE TABLE t (x INTEGER)")
    writer.execute("BEGIN IMMEDIATE")  # take the writer lock and keep it
    writer.execute("INSERT INTO t VALUES (1)")
    try:
        with job_lock(locks, "refresh"):
            time.sleep(0.3)
            assert lock_status(locks, "refresh") is not None
        assert lock_status(locks, "refresh") is None
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_the_lock_never_writes_to_the_database(factory: sessionmaker[Session], locks: Path) -> None:
    with job_lock(locks, "refresh"):
        lock_status(locks, "refresh")
    with session_scope(factory) as session:
        assert session.query(JobLock).count() == 0


def test_a_long_held_lock_needs_no_renewal(locks: Path) -> None:
    """There is no expiry: held for longer than the old 10-minute limit's scaled-down equivalent,
    nobody else can take it."""
    with job_lock(locks, "refresh"):
        time.sleep(1.2)
        with pytest.raises(JobAlreadyRunningError), job_lock(locks, "refresh"):
            pass


def wait_until_free(locks: Path, seconds: float = 15.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if lock_status(locks, "refresh") is None:
            return True
        time.sleep(0.1)
    return False


def test_a_killed_process_frees_the_lock_at_once(locks: Path) -> None:
    """The operating system releases the lock however the holder dies; there is no wait for an
    expiry, and no need for anyone to clean up.

    The holder reports its own pid. On Windows a virtual environment's python.exe is a launcher
    that starts the real interpreter as a child, so the pid of the process we started is not the
    pid of the process holding the lock, and killing the launcher would not kill the holder.
    """
    code = (
        "import os, sys, time; from pathlib import Path; from cews.scheduler.job_state import job_lock\n"
        "with job_lock(Path(sys.argv[1]), 'refresh'):\n"
        "    print('held', os.getpid(), flush=True); time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(locks)],
        stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(SRC)},
    )  # fmt: skip
    holder_pid = 0
    try:
        assert child.stdout is not None
        word, pid_text = child.stdout.readline().split()
        assert word == "held"
        holder_pid = int(pid_text)
        status = lock_status(locks, "refresh")
        assert status is not None and status["pid"] == holder_pid  # held by the other process
        with (
            pytest.raises(JobAlreadyRunningError, match=str(holder_pid)),
            job_lock(locks, "refresh"),
        ):
            pass
        os.kill(holder_pid, getattr(signal, "SIGKILL", signal.SIGTERM))  # no chance to clean up
        assert wait_until_free(locks), "the lock was not freed after the holder was killed"
        with job_lock(locks, "refresh"):
            pass
    finally:
        if holder_pid:
            with contextlib.suppress(OSError):  # it may already be gone
                os.kill(holder_pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        if child.poll() is None:
            child.kill()
        child.wait(timeout=15)


# --------------------------------------------------------------------------------------
# Persisted history
# --------------------------------------------------------------------------------------
def test_a_run_is_recorded_from_start_to_finish(factory: sessionmaker[Session]) -> None:
    job_id = start_job_run(factory, "refresh", trigger="schedule", now=T0)
    with session_scope(factory) as session:
        row = session.query(JobRun).one()
        assert row.status == "running" and row.trigger == "schedule" and row.finished_at is None
    finish_job_run(
        factory, job_id, RunStatus.SUCCEEDED, summary={"steps": 3}, now=T0 + timedelta(minutes=5)
    )
    with session_scope(factory) as session:
        row = session.query(JobRun).one()
        assert row.status == "succeeded" and row.summary_json == {"steps": 3}
        assert row.finished_at is not None


def test_a_failure_keeps_its_reason(factory: sessionmaker[Session]) -> None:
    job_id = start_job_run(factory, "refresh", trigger="manual")
    finish_job_run(factory, job_id, RunStatus.FAILED, error="step 'score' failed")
    with session_scope(factory) as session:
        assert session.query(JobRun).one().error_summary == "step 'score' failed"


def test_runs_left_marked_running_by_a_dead_process_are_closed(
    factory: sessionmaker[Session],
) -> None:
    stale = start_job_run(factory, "refresh", trigger="schedule", now=T0)
    other = start_job_run(factory, "fetch", trigger="manual", now=T0)  # a different kind of job
    done = start_job_run(factory, "refresh", trigger="manual", now=T0)
    finish_job_run(factory, done, RunStatus.SUCCEEDED)
    assert mark_interrupted_runs(factory, "refresh", now=T0 + timedelta(hours=1)) == 1
    with session_scope(factory) as session:
        rows = {r.job_id: r for r in session.query(JobRun)}
    assert rows[stale].status == "failed" and rows[stale].error_summary == INTERRUPTED
    assert rows[other].status == "running"  # not this job's, so left alone
    assert rows[done].status == "succeeded"


def test_several_kinds_of_dangling_run_can_be_closed_together(
    factory: sessionmaker[Session],
) -> None:
    """A cycle also records an inner fetch run, and it is left dangling when the process dies."""
    refresh = start_job_run(factory, "refresh", trigger="schedule", now=T0)
    fetch = start_job_run(factory, "fetch", trigger="schedule", now=T0)
    unrelated = start_job_run(factory, "something_else", trigger="manual", now=T0)
    assert mark_interrupted_runs(factory, ("refresh", "fetch"), now=T0 + timedelta(hours=1)) == 2
    with session_scope(factory) as session:
        rows = {r.job_id: r.status for r in session.query(JobRun)}
    assert rows[refresh] == "failed" and rows[fetch] == "failed" and rows[unrelated] == "running"


def test_recent_runs_are_newest_first_limited_and_filterable(
    factory: sessionmaker[Session],
) -> None:
    for index in range(5):
        start_job_run(factory, "refresh", trigger="schedule", now=T0 + timedelta(hours=index))
    start_job_run(factory, "fetch", trigger="manual", now=T0 + timedelta(hours=9))
    with session_scope(factory) as session:
        runs = recent_job_runs(session, job_name="refresh", limit=3)
        assert [r.started_at.hour for r in runs] == [16, 15, 14]
        assert recent_job_runs(session, limit=100)[0].job_name == "fetch"


def test_the_last_success_ignores_failures_and_running_runs(factory: sessionmaker[Session]) -> None:
    ok = start_job_run(factory, "refresh", trigger="schedule", now=T0)
    finish_job_run(factory, ok, RunStatus.SUCCEEDED)
    bad = start_job_run(factory, "refresh", trigger="schedule", now=T0 + timedelta(hours=2))
    finish_job_run(factory, bad, RunStatus.FAILED)
    start_job_run(
        factory, "refresh", trigger="schedule", now=T0 + timedelta(hours=4)
    )  # still running
    with session_scope(factory) as session:
        success = last_success(session)
        assert success is not None and success.job_id == ok


def test_no_success_at_all_is_none(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        assert last_success(session) is None


def test_consecutive_failures_count_back_to_the_last_non_failure(
    factory: sessionmaker[Session],
) -> None:
    def add(status: RunStatus, hours: int) -> None:
        job_id = start_job_run(
            factory, "refresh", trigger="schedule", now=T0 + timedelta(hours=hours)
        )
        finish_job_run(factory, job_id, status)

    add(RunStatus.SUCCEEDED, 0)
    add(RunStatus.FAILED, 1)
    add(RunStatus.FAILED, 2)
    add(RunStatus.PARTIAL, 3)  # a partial run is not a failure and ends the streak
    add(RunStatus.FAILED, 4)
    add(RunStatus.FAILED, 5)
    add(RunStatus.FAILED, 6)
    with session_scope(factory) as session:
        assert consecutive_failures(session) == 3


def test_a_latest_success_means_no_failing_streak(factory: sessionmaker[Session]) -> None:
    for hours, status in ((0, RunStatus.FAILED), (1, RunStatus.SUCCEEDED)):
        job_id = start_job_run(
            factory, "refresh", trigger="schedule", now=T0 + timedelta(hours=hours)
        )
        finish_job_run(factory, job_id, status)
    with session_scope(factory) as session:
        assert consecutive_failures(session) == 0
