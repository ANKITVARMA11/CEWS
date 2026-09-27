"""Source checkpoints, the circuit breaker, and collection-window planning.

A checkpoint is stored as JSON in ``source_checkpoints.checkpoint_json``::

    {"watermark": "<end of the last fully collected slice>",
     "window_start": "...", "window_end": "...", "complete": true, "mode": "incremental"}

Rules:

* The checkpoint only moves forward over slices that were collected completely, so a failure
  never skips data. ``attempted_checkpoint_json`` keeps what the last run was aiming for.
* After ``failure_threshold`` failing runs in a row the circuit opens and the source is skipped
  until the cooldown ends. The next run after the cooldown is a trial: success closes the
  circuit, another failure reopens it immediately.
* :func:`plan_window` decides what each run collects: an initial full collection, an
  incremental window that overlaps the watermark by ``INCREMENTAL_LOOKBACK_DAYS``, a resume
  of an unfinished window, or (for ``full`` refresh sources) a full refresh when it is due.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import RunStatus
from cews.database.models import SourceCheckpoint
from cews.ingestion.registry import SourceConfig
from cews.ingestion.results import CollectionWindow, SourceResult
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

RequestedMode = Literal["auto", "full", "incremental"]
MAX_ERROR_LENGTH = 500


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True)
class CheckpointState:
    """A read-only snapshot of one source's checkpoint row (defaults if it never ran)."""

    source: str
    checkpoint: dict[str, Any] | None = None
    attempted_checkpoint: dict[str, Any] | None = None
    last_attempted_at: datetime | None = None
    last_success_at: datetime | None = None
    last_full_refresh_at: datetime | None = None
    last_status: str | None = None
    consecutive_failures: int = 0
    circuit_open_until: datetime | None = None
    last_error: str | None = None

    @property
    def watermark(self) -> datetime | None:
        """End of the last fully collected slice, or None."""
        return _parse_time((self.checkpoint or {}).get("watermark"))

    @property
    def complete(self) -> bool:
        """True unless the last collection stopped before the end of its window."""
        return bool((self.checkpoint or {}).get("complete", True))

    def circuit_open(self, now: datetime) -> bool:
        """True while the source is paused after repeated failures."""
        return self.circuit_open_until is not None and now < self.circuit_open_until


def _snapshot(row: SourceCheckpoint | None, source: str) -> CheckpointState:
    if row is None:
        return CheckpointState(source=source)
    return CheckpointState(
        source=row.source,
        checkpoint=dict(row.checkpoint_json) if row.checkpoint_json else None,
        attempted_checkpoint=(
            dict(row.attempted_checkpoint_json) if row.attempted_checkpoint_json else None
        ),
        last_attempted_at=row.last_attempted_at,
        last_success_at=row.last_success_at,
        last_full_refresh_at=row.last_full_refresh_at,
        last_status=row.last_status,
        consecutive_failures=row.consecutive_failures,
        circuit_open_until=row.circuit_open_until,
        last_error=row.last_error,
    )


def _row(session: Session, source: str) -> SourceCheckpoint | None:
    return session.scalar(select(SourceCheckpoint).where(SourceCheckpoint.source == source))


def read_checkpoint(session: Session, source: str) -> CheckpointState:
    """Return the checkpoint state of ``source`` (an empty state if it never ran)."""
    return _snapshot(_row(session, source), source)


def save_checkpoint(
    session: Session,
    source: str,
    checkpoint: dict[str, Any],
    *,
    now: datetime | None = None,
    full_refresh_completed: bool = False,
) -> CheckpointState:
    """Store a successfully reached checkpoint.

    Raises:
        ValueError: if ``checkpoint`` has no valid ``watermark``.
    """
    if _parse_time(checkpoint.get("watermark")) is None:
        raise ValueError("checkpoint must contain an ISO 'watermark' timestamp")
    now = now or datetime.now(UTC)
    row = _row(session, source)
    if row is None:
        row = SourceCheckpoint(source=source, consecutive_failures=0)
        session.add(row)
    row.checkpoint_json = dict(checkpoint)
    row.last_success_at = now
    if full_refresh_completed:
        row.last_full_refresh_at = now
    row.updated_at = now
    session.flush()
    return _snapshot(row, source)


def is_failed_run(result: SourceResult) -> bool:
    """A run counts as failed for the circuit breaker when any page could not be fetched."""
    return result.status is RunStatus.FAILED or result.page_error_count > 0


def record_attempt(
    session: Session,
    result: SourceResult,
    *,
    failure_threshold: int,
    cooldown_minutes: int,
    full_refresh_completed: bool = False,
    now: datetime | None = None,
) -> CheckpointState:
    """Update checkpoint and circuit-breaker state after a collection run.

    Skipped runs must not be passed here; they leave the state unchanged.

    Raises:
        ValueError: for a skipped result or non-positive breaker settings.
    """
    if result.status is RunStatus.SKIPPED:
        raise ValueError("skipped runs do not change checkpoint state")
    if failure_threshold < 1 or cooldown_minutes < 1:
        raise ValueError("failure_threshold and cooldown_minutes must be at least 1")
    now = now or datetime.now(UTC)
    row = _row(session, result.source_name)
    if row is None:
        row = SourceCheckpoint(source=result.source_name, consecutive_failures=0)
        session.add(row)

    row.last_attempted_at = now
    row.last_status = result.status.value
    row.attempted_checkpoint_json = dict(result.checkpoint) if result.checkpoint else None
    if result.checkpoint_advanced and result.checkpoint:
        row.checkpoint_json = dict(result.checkpoint)
        row.last_success_at = now
        if full_refresh_completed:
            row.last_full_refresh_at = now

    if is_failed_run(result):
        row.consecutive_failures += 1
        row.last_error = "; ".join(result.errors)[:MAX_ERROR_LENGTH] or "collection failed"
        if row.consecutive_failures >= failure_threshold:
            row.circuit_open_until = now + timedelta(minutes=cooldown_minutes)
            LOGGER.warning(
                "%s paused until %s after %d failed runs",
                result.source_name,
                row.circuit_open_until.isoformat(),
                row.consecutive_failures,
            )
    else:
        row.consecutive_failures = 0
        row.circuit_open_until = None
        row.last_error = None
    row.updated_at = now
    session.flush()
    return _snapshot(row, result.source_name)


# --------------------------------------------------------------------------------------
# Window planning
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CollectionPlan:
    """What a run should collect, or why it should not run."""

    window: CollectionWindow | None
    skip_reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def full_refresh(self) -> bool:
        """True when completing this window counts as a completed full refresh."""
        return self.window is not None and self.window.mode in ("full", "resume")


def plan_window(
    config: SourceConfig,
    state: CheckpointState,
    settings: Settings,
    now: datetime,
    requested: RequestedMode = "auto",
) -> CollectionPlan:
    """Decide the collection window for one source.

    Raises:
        ValueError: if ``now`` is naive.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    earliest = now - timedelta(days=settings.default_lookback_days)
    full = CollectionWindow(earliest, now, "full")

    if requested == "full":
        return CollectionPlan(full, notes=("full collection requested",))

    watermark = state.watermark
    if watermark is not None and not state.complete and watermark < now:
        start = max(watermark, earliest)
        return CollectionPlan(
            CollectionWindow(start, now, "resume"),
            notes=(f"resuming an unfinished collection from {start.date().isoformat()}",),
        )

    if config.refresh_mode == "full":
        days = config.full_refresh_window_days or 1
        last = state.last_full_refresh_at
        if last is None:
            return CollectionPlan(full, notes=("first full refresh",))
        due = last + timedelta(days=days)
        if now >= due:
            return CollectionPlan(full, notes=(f"full refresh due (every {days} days)",))
        return CollectionPlan(None, skip_reason=f"next full refresh due {due.date().isoformat()}")

    if watermark is None:
        return CollectionPlan(
            full, notes=("no checkpoint yet; running the initial full collection",)
        )
    start = max(watermark - timedelta(days=settings.incremental_lookback_days), earliest)
    if start >= now:
        start = now - timedelta(days=settings.incremental_lookback_days)
    return CollectionPlan(CollectionWindow(start, now, "incremental"))
