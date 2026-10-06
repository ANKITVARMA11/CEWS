"""The interval loop: run a refresh cycle every ``FETCH_INTERVAL_MINUTES``.

This is a thin layer over APScheduler. It only decides *when* to run; what a cycle does, and the
guarantee that two never overlap, live in ``jobs`` and ``job_state``.

Behaviours worth knowing:

* **Disabled means disabled.** With ``ENABLE_SCHEDULER=false`` the loop refuses to start, and says
  so, rather than quietly doing nothing.
* **A cycle can never overlap itself**, three ways over: APScheduler allows one instance of the
  job, a slow cycle that misses its slot is coalesced into one run rather than a pile-up, and the
  database lock stops a scheduled cycle overlapping a manual ``cews refresh``.
* **One bad cycle never stops the scheduler.** Every error is caught and logged, and the next
  slot still happens.
* **A failed cycle is retried sooner than the next slot**, at 5, 10, 20... minutes (never later
  than the interval), so a brief outage does not cost a whole interval. A success ends the retries.
* **No cycle runs at startup unless asked** (``RUN_FETCH_ON_STARTUP=true`` or ``--now``): the
  first one happens one interval after the scheduler starts.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import session_scope
from cews.scheduler.job_state import (
    JobAlreadyRunningError,
    consecutive_failures,
    held_lock_count,
)
from cews.scheduler.jobs import RefreshReport, run_refresh
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

REFRESH_JOB_ID = "cews-refresh"
RETRY_JOB_ID = "cews-refresh-retry"
RETRY_BASE_MINUTES = 5
MAX_MISFIRE_GRACE_SECONDS = 3600

RefreshFunction = Callable[..., RefreshReport]


class SchedulerDisabledError(RuntimeError):
    """Raised when the scheduler is asked to start while ``ENABLE_SCHEDULER`` is false."""


def retry_delay(
    failures: int, *, cap_minutes: int, base_minutes: int = RETRY_BASE_MINUTES
) -> timedelta:
    """How long to wait before retrying after ``failures`` failed cycles in a row.

    Doubles each time (5, 10, 20, ... minutes) and never exceeds ``cap_minutes``, the normal
    interval: once a retry would be as late as the next scheduled slot, the slot takes over.

    Raises:
        ValueError: if ``failures`` is below 1 (there is nothing to retry).
    """
    if failures < 1:
        raise ValueError("retry_delay needs at least one failure")
    minutes = min(base_minutes * 2 ** (failures - 1), cap_minutes)
    return timedelta(minutes=minutes)


def make_cycle(
    settings: Settings,
    factory: sessionmaker[Session],
    scheduler: BaseScheduler | None = None,
    *,
    refresh: RefreshFunction = run_refresh,
) -> Callable[[], RefreshReport | None]:
    """The function the scheduler calls each slot. It never raises.

    Returns the cycle's report, or None if the cycle was skipped (another one running) or broke.
    """
    tz = ZoneInfo(settings.timezone)

    def cycle() -> RefreshReport | None:
        try:
            report = refresh(factory, settings, trigger="schedule")
        except JobAlreadyRunningError as exc:
            LOGGER.warning("scheduled refresh skipped: %s", exc)
            return None
        except Exception:  # the scheduler must survive anything a cycle throws
            LOGGER.exception("scheduled refresh crashed")
            return None

        LOGGER.info("scheduled refresh %s: %s", report.job_id[:8], report.status.value)
        if report.status is RunStatus.FAILED and scheduler is not None:
            try:
                with session_scope(factory) as session:
                    failures = consecutive_failures(session)
                delay = retry_delay(max(failures, 1), cap_minutes=settings.fetch_interval_minutes)
                scheduler.add_job(
                    cycle,
                    "date",
                    run_date=datetime.now(tz) + delay,
                    id=RETRY_JOB_ID,
                    replace_existing=True,
                    max_instances=1,
                )
                LOGGER.warning(
                    "refresh failed %s time(s) in a row; retrying in %s", failures, delay
                )
            except Exception:
                LOGGER.exception("could not schedule a retry")
        return report

    return cycle


def build_scheduler(
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    scheduler: BaseScheduler | None = None,
    refresh: RefreshFunction = run_refresh,
    run_now: bool | None = None,
) -> BaseScheduler:
    """A scheduler with the refresh job registered (not yet started).

    Args:
        scheduler: an existing scheduler to register on (a ``BlockingScheduler`` by default).
        run_now: run a cycle immediately at start; default follows ``RUN_FETCH_ON_STARTUP``.

    Raises:
        SchedulerDisabledError: if ``ENABLE_SCHEDULER`` is false.
    """
    if not settings.enable_scheduler:
        raise SchedulerDisabledError(
            "the scheduler is disabled (ENABLE_SCHEDULER=false in .env); "
            "use 'cews refresh' to run a cycle by hand"
        )
    tz = ZoneInfo(settings.timezone)
    engine = scheduler or BlockingScheduler(timezone=tz)
    interval = settings.fetch_interval_minutes
    start_immediately = settings.run_fetch_on_startup if run_now is None else run_now
    kwargs = {"next_run_time": datetime.now(tz)} if start_immediately else {}
    engine.add_job(
        make_cycle(settings, factory, engine, refresh=refresh),
        IntervalTrigger(minutes=interval, timezone=tz),
        id=REFRESH_JOB_ID,
        name="CEWS refresh",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=min(interval * 60, MAX_MISFIRE_GRACE_SECONDS),
        replace_existing=True,
        **kwargs,
    )
    return engine


def stop_scheduler(engine: BaseScheduler, *, hard_exit: Callable[[int], object] = os._exit) -> None:
    """Stop the scheduler without ever letting two cycles overlap.

    No new cycle starts. A cycle already running is allowed to finish (its lock releases itself
    when it does), because releasing the lock while it still runs would let a manual
    ``cews refresh`` start beside it. A **second** interrupt while waiting aborts: the process
    exits at once (Python would otherwise wait for the cycle's thread), and the operating system
    releases the lock as it goes. The half-finished run is closed as interrupted by the next cycle.
    """
    if held_lock_count():
        LOGGER.warning("waiting for the running cycle to finish; press Ctrl+C again to abort it")
    try:
        if engine.running:
            engine.shutdown(wait=True)
    except KeyboardInterrupt:
        LOGGER.warning("aborting the running cycle")
        hard_exit(130)  # the operating system releases the file lock the moment we exit


def run_scheduler(
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    run_now: bool | None = None,
) -> None:
    """Start the scheduler and block until it is stopped (Ctrl+C).

    Raises:
        SchedulerDisabledError: if ``ENABLE_SCHEDULER`` is false.
    """
    engine = build_scheduler(settings, factory, run_now=run_now)
    LOGGER.info("scheduler started: a refresh every %s minute(s)", settings.fetch_interval_minutes)
    try:
        engine.start()
    except (KeyboardInterrupt, SystemExit):
        LOGGER.info("scheduler stopping")
    finally:
        stop_scheduler(engine)
