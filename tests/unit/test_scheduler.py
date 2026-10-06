"""Unit tests for the interval loop: its timing, its safety limits, and its retries."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.ingestion.results import RunReport
from cews.scheduler.job_state import (
    JobAlreadyRunningError,
    held_lock_count,
    job_lock,
    lock_status,
)
from cews.scheduler.jobs import RefreshReport, run_refresh
from cews.scheduler.scheduler import (
    REFRESH_JOB_ID,
    RETRY_JOB_ID,
    SchedulerDisabledError,
    build_scheduler,
    make_cycle,
    retry_delay,
    stop_scheduler,
)
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit


@pytest.fixture
def factory(tmp_path: Path) -> Iterator[sessionmaker[Session]]:
    settings = load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./sched.db")}
    )
    engine = create_db_engine(settings)
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def locks(tmp_path: Path) -> Path:
    return tmp_path / "locks"


def make_settings(**overrides: Any) -> Settings:
    return load_settings(env_file=None, overrides=overrides)


def report(status: RunStatus) -> RefreshReport:
    now = datetime.now(UTC)
    return RefreshReport(
        job_id="abcdef123456", trigger="schedule", dry_run=False, started_at=now, status=status
    )


class FakeScheduler:
    """Records add_job calls so the retry logic can be checked without real timers."""

    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []

    def add_job(self, func: Any, trigger: Any = None, **kwargs: Any) -> None:
        self.jobs.append({"func": func, "trigger": trigger, **kwargs})


# --------------------------------------------------------------------------------------
# Retry delays
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("failures", "minutes"), [(1, 5), (2, 10), (3, 20), (4, 40), (5, 80), (6, 120), (20, 120)]
)
def test_retries_double_and_never_exceed_the_interval(failures: int, minutes: int) -> None:
    assert retry_delay(failures, cap_minutes=120) == timedelta(minutes=minutes)


def test_a_short_interval_caps_the_retry_sooner() -> None:
    assert retry_delay(5, cap_minutes=30) == timedelta(minutes=30)


@pytest.mark.parametrize("failures", [0, -1])
def test_there_is_nothing_to_retry_without_a_failure(failures: int) -> None:
    with pytest.raises(ValueError, match="at least one failure"):
        retry_delay(failures, cap_minutes=120)


# --------------------------------------------------------------------------------------
# Building the scheduler
# --------------------------------------------------------------------------------------
def test_a_disabled_scheduler_refuses_to_start_and_says_why(factory: sessionmaker[Session]) -> None:
    with pytest.raises(SchedulerDisabledError, match=r"ENABLE_SCHEDULER=false.*cews refresh"):
        build_scheduler(make_settings(enable_scheduler=False), factory)


def scheduled(
    settings: Settings, factory: sessionmaker[Session], **kwargs: Any
) -> tuple[BackgroundScheduler, Any]:
    engine = BackgroundScheduler()
    build_scheduler(settings, factory, scheduler=engine, **kwargs)
    engine.start(paused=True)
    return engine, engine.get_job(REFRESH_JOB_ID)


def test_the_job_runs_on_the_configured_interval(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(fetch_interval_minutes=45), factory)
    try:
        assert isinstance(job.trigger, IntervalTrigger)
        assert job.trigger.interval == timedelta(minutes=45)
    finally:
        engine.shutdown(wait=False)


def test_the_default_interval_is_two_hours(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(), factory)
    try:
        assert job.trigger.interval == timedelta(minutes=120)
    finally:
        engine.shutdown(wait=False)


def test_a_cycle_can_never_overlap_itself_or_pile_up(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(fetch_interval_minutes=30), factory)
    try:
        assert job.max_instances == 1  # never two at once
        assert job.coalesce is True  # missed slots collapse into one run, not a queue
        assert job.misfire_grace_time == 30 * 60
    finally:
        engine.shutdown(wait=False)


def test_the_misfire_grace_is_capped_for_long_intervals(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(fetch_interval_minutes=1440), factory)
    try:
        assert job.misfire_grace_time == 3600
    finally:
        engine.shutdown(wait=False)


def test_by_default_the_first_cycle_waits_one_interval(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(fetch_interval_minutes=60), factory)
    try:
        wait = job.next_run_time - datetime.now(job.next_run_time.tzinfo)
        assert timedelta(minutes=58) < wait <= timedelta(minutes=60)
    finally:
        engine.shutdown(wait=False)


def test_run_now_starts_immediately(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(fetch_interval_minutes=60), factory, run_now=True)
    try:
        assert abs(job.next_run_time - datetime.now(job.next_run_time.tzinfo)) < timedelta(
            seconds=5
        )
    finally:
        engine.shutdown(wait=False)


def test_the_startup_setting_is_honoured(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(run_fetch_on_startup=True), factory)
    try:
        assert abs(job.next_run_time - datetime.now(job.next_run_time.tzinfo)) < timedelta(
            seconds=5
        )
    finally:
        engine.shutdown(wait=False)


def test_an_explicit_run_now_overrides_the_startup_setting(factory: sessionmaker[Session]) -> None:
    engine, job = scheduled(make_settings(run_fetch_on_startup=True), factory, run_now=False)
    try:
        assert job.next_run_time - datetime.now(job.next_run_time.tzinfo) > timedelta(minutes=100)
    finally:
        engine.shutdown(wait=False)


def test_registering_twice_does_not_duplicate_the_job(factory: sessionmaker[Session]) -> None:
    engine = BackgroundScheduler()
    settings = make_settings()
    build_scheduler(settings, factory, scheduler=engine)
    build_scheduler(settings, factory, scheduler=engine)
    engine.start(paused=True)
    try:
        assert len(engine.get_jobs()) == 1
    finally:
        engine.shutdown(wait=False)


# --------------------------------------------------------------------------------------
# The cycle the scheduler calls
# --------------------------------------------------------------------------------------
def test_a_cycle_runs_a_refresh_and_returns_its_report(factory: sessionmaker[Session]) -> None:
    calls: list[dict[str, Any]] = []

    def fake_refresh(_factory: Any, _settings: Any, **kwargs: Any) -> RefreshReport:
        calls.append(kwargs)
        return report(RunStatus.SUCCEEDED)

    result = make_cycle(make_settings(), factory, refresh=fake_refresh)()
    assert result is not None and result.status is RunStatus.SUCCEEDED
    assert calls == [{"trigger": "schedule"}]


def test_a_cycle_that_finds_another_running_is_skipped_quietly(
    factory: sessionmaker[Session],
) -> None:
    def busy(*_a: Any, **_k: Any) -> RefreshReport:
        raise JobAlreadyRunningError("refresh", {"pid": 1234})

    assert make_cycle(make_settings(), factory, refresh=busy)() is None


def test_a_crashing_cycle_never_takes_the_scheduler_down(factory: sessionmaker[Session]) -> None:
    def broken(*_a: Any, **_k: Any) -> RefreshReport:
        raise RuntimeError("anything at all")

    assert make_cycle(make_settings(), factory, refresh=broken)() is None  # returns, does not raise


def test_a_successful_cycle_schedules_no_retry(factory: sessionmaker[Session]) -> None:
    fake = FakeScheduler()
    make_cycle(
        make_settings(), factory, fake, refresh=lambda *_a, **_k: report(RunStatus.SUCCEEDED)
    )()
    assert fake.jobs == []


def test_a_partial_cycle_schedules_no_retry(factory: sessionmaker[Session]) -> None:
    fake = FakeScheduler()
    make_cycle(
        make_settings(), factory, fake, refresh=lambda *_a, **_k: report(RunStatus.PARTIAL)
    )()
    assert fake.jobs == []


def failing_fetch(*_a: Any, **_k: Any) -> RunReport:
    raise RuntimeError("no network")


def test_a_failed_cycle_schedules_a_sooner_retry_that_backs_off(
    factory: sessionmaker[Session], locks: Path
) -> None:
    """Uses the real cycle, so the failing streak is read from the persisted audit rows."""
    fake = FakeScheduler()
    settings = make_settings(fetch_interval_minutes=120)

    def refresh(f: Any, s: Any, **kwargs: Any) -> RefreshReport:
        return run_refresh(f, s, fetch=failing_fetch, analysis_steps=[], lock_dir=locks, **kwargs)

    cycle = make_cycle(settings, factory, fake, refresh=refresh)
    delays = []
    for _ in range(3):
        before = datetime.now(UTC)
        cycle()
        job = fake.jobs[-1]
        assert (
            job["id"] == RETRY_JOB_ID
            and job["replace_existing"] is True
            and job["trigger"] == "date"
        )
        delays.append(round((job["run_date"] - before).total_seconds() / 60))
    assert delays == [5, 10, 20]


def test_a_success_after_failures_ends_the_retries(
    factory: sessionmaker[Session], locks: Path
) -> None:
    fake = FakeScheduler()
    outcomes = iter([True, False])  # fail, then succeed

    def refresh(f: Any, s: Any, **kwargs: Any) -> RefreshReport:
        fetch = (
            failing_fetch
            if next(outcomes)
            else (lambda *_a, **_k: RunReport(job_id="j", started_at=datetime.now(UTC)))
        )
        return run_refresh(f, s, fetch=fetch, analysis_steps=[], lock_dir=locks, **kwargs)

    cycle = make_cycle(make_settings(), factory, fake, refresh=refresh)
    cycle()
    assert len(fake.jobs) == 1
    cycle()
    assert len(fake.jobs) == 1  # no second retry


# --------------------------------------------------------------------------------------
# The real APScheduler
# --------------------------------------------------------------------------------------
def test_a_real_scheduler_fires_the_cycle_off_the_main_thread(
    factory: sessionmaker[Session],
) -> None:
    fired = threading.Event()
    seen: dict[str, Any] = {}

    def fake_refresh(_f: Any, _s: Any, **kwargs: Any) -> RefreshReport:
        seen["thread"] = threading.current_thread().name
        seen["trigger"] = kwargs["trigger"]
        fired.set()
        return report(RunStatus.SUCCEEDED)

    engine = BackgroundScheduler()
    build_scheduler(make_settings(), factory, scheduler=engine, refresh=fake_refresh, run_now=True)
    engine.start()
    try:
        assert fired.wait(timeout=10), "the scheduler never ran the job"
    finally:
        engine.shutdown(wait=True)
    assert seen["trigger"] == "schedule" and seen["thread"] != threading.main_thread().name


def test_a_slow_cycle_is_not_started_a_second_time_by_the_scheduler(
    factory: sessionmaker[Session],
) -> None:
    """max_instances=1: a second trigger while one runs is dropped, not run in parallel."""
    running = threading.Event()
    release = threading.Event()
    starts: list[float] = []

    def slow(_f: Any, _s: Any, **_k: Any) -> RefreshReport:
        starts.append(time.monotonic())
        running.set()
        release.wait(timeout=10)
        return report(RunStatus.SUCCEEDED)

    engine = BackgroundScheduler()
    build_scheduler(make_settings(), factory, scheduler=engine, refresh=slow, run_now=True)
    engine.start()
    try:
        assert running.wait(timeout=10)
        engine.get_job(REFRESH_JOB_ID).modify(
            next_run_time=datetime.now(UTC)
        )  # force a second trigger now
        time.sleep(1.0)
        assert len(starts) == 1
    finally:
        release.set()
        engine.shutdown(wait=True)


# --------------------------------------------------------------------------------------
# Stopping without ever letting two cycles overlap
# --------------------------------------------------------------------------------------
class FakeEngine:
    def __init__(self, *, interrupt: bool = False) -> None:
        self.running = True
        self.interrupt = interrupt
        self.shutdown_calls: list[bool] = []

    def shutdown(self, wait: bool = True) -> None:
        self.shutdown_calls.append(wait)
        if self.interrupt:
            raise KeyboardInterrupt


def test_a_normal_stop_waits_for_the_running_cycle_and_leaves_its_lock_alone(locks: Path) -> None:
    engine, exits = FakeEngine(), []  # type: tuple[FakeEngine, list[int]]
    with job_lock(locks, "refresh"):
        stop_scheduler(engine, hard_exit=exits.append)
        assert lock_status(locks, "refresh") is not None  # still held: its cycle is still running
    assert engine.shutdown_calls == [True] and exits == []


def test_a_second_interrupt_exits_130_and_leaves_lock_release_to_the_operating_system(
    locks: Path,
) -> None:
    engine, exits = FakeEngine(interrupt=True), []  # type: tuple[FakeEngine, list[int]]
    with job_lock(locks, "refresh"):
        stop_scheduler(engine, hard_exit=exits.append)
    assert exits == [130]


def test_stopping_an_engine_that_is_not_running_does_nothing() -> None:
    engine = FakeEngine()
    engine.running = False
    stop_scheduler(engine)
    assert engine.shutdown_calls == []


def test_the_scheduler_notices_when_a_cycle_is_in_flight(locks: Path) -> None:
    before = held_lock_count()
    with job_lock(locks, "refresh"):
        assert held_lock_count() == before + 1


def test_a_real_stop_does_not_return_until_the_running_cycle_has_finished(
    locks: Path, factory: sessionmaker[Session]
) -> None:
    """The point: letting go early would allow a manual refresh to start beside the cycle."""
    started, release = threading.Event(), threading.Event()
    stopped_at: list[float] = []

    def slow(_f: Any, _s: Any, **_k: Any) -> RefreshReport:
        with job_lock(locks, "refresh"):
            started.set()
            release.wait(timeout=10)
        return report(RunStatus.SUCCEEDED)

    engine = BackgroundScheduler()
    build_scheduler(make_settings(), factory, scheduler=engine, refresh=slow, run_now=True)
    engine.start()
    assert started.wait(timeout=10)

    def stop() -> None:
        stop_scheduler(engine)
        stopped_at.append(time.monotonic())

    stopper = threading.Thread(target=stop)
    stopper.start()
    time.sleep(0.5)
    assert stopped_at == []  # still waiting: the cycle has not finished
    assert lock_status(locks, "refresh") is not None  # and the lock is still held
    finished = time.monotonic()
    release.set()
    stopper.join(timeout=10)
    assert stopped_at and stopped_at[0] >= finished
    assert lock_status(locks, "refresh") is None
