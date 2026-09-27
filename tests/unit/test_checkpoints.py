"""Unit tests for checkpoints, the circuit breaker, and collection-window planning."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import RunStatus
from cews.database.models import SourceCheckpoint
from cews.ingestion.checkpoints import (
    CheckpointState,
    is_failed_run,
    plan_window,
    read_checkpoint,
    record_attempt,
    save_checkpoint,
)
from cews.ingestion.results import CollectionWindow, SourceResult
from cews.settings import Settings, load_settings
from support_adapters import example_config

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
SOURCE = "example_source"


@pytest.fixture
def cfg_settings() -> Settings:
    return load_settings(
        env_file=None, overrides={"default_lookback_days": 365, "incremental_lookback_days": 7}
    )


def _checkpoint(watermark: datetime, complete: bool = True) -> dict[str, object]:
    return {"watermark": watermark.isoformat(), "complete": complete}


def _result(
    status: RunStatus,
    *,
    advanced: bool = False,
    watermark: datetime = NOW,
    page_errors: int = 0,
    complete: bool = True,
) -> SourceResult:
    return SourceResult(
        source_name=SOURCE,
        collection_start=NOW,
        status=status,
        checkpoint=_checkpoint(watermark, complete),
        checkpoint_advanced=advanced,
        page_error_count=page_errors,
        error_count=page_errors,
        errors=["HTTP 503"] if page_errors else [],
    )


def _attempt(
    session: Session, result: SourceResult, now: datetime = NOW, **kwargs: object
) -> CheckpointState:
    return record_attempt(
        session, result, failure_threshold=3, cooldown_minutes=60, now=now, **kwargs
    )


# --------------------------------------------------------------------------------------
# Reading and saving
# --------------------------------------------------------------------------------------
def test_unknown_source_has_an_empty_state(session: Session) -> None:
    state = read_checkpoint(session, SOURCE)
    assert state.checkpoint is None and state.watermark is None
    assert state.complete is True and state.consecutive_failures == 0
    assert state.circuit_open(NOW) is False


def test_save_and_read_checkpoint(session: Session) -> None:
    save_checkpoint(session, SOURCE, _checkpoint(NOW - timedelta(days=1)), now=NOW)
    state = read_checkpoint(session, SOURCE)
    assert state.watermark == NOW - timedelta(days=1)
    assert state.last_success_at == NOW and state.last_full_refresh_at is None
    save_checkpoint(session, SOURCE, _checkpoint(NOW), now=NOW, full_refresh_completed=True)
    state = read_checkpoint(session, SOURCE)
    assert state.watermark == NOW and state.last_full_refresh_at == NOW
    assert session.scalar(select(func.count()).select_from(SourceCheckpoint)) == 1


@pytest.mark.parametrize("bad", [{}, {"watermark": "yesterday"}, {"watermark": 5}])
def test_save_requires_a_valid_watermark(session: Session, bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="watermark"):
        save_checkpoint(session, SOURCE, bad)


def test_naive_watermark_is_read_as_utc() -> None:
    state = CheckpointState(SOURCE, checkpoint={"watermark": "2026-01-01T00:00:00"})
    assert state.watermark == datetime(2026, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# record_attempt and the circuit breaker
# --------------------------------------------------------------------------------------
def test_success_advances_checkpoint_and_records_attempt(session: Session) -> None:
    state = _attempt(session, _result(RunStatus.SUCCEEDED, advanced=True))
    assert state.checkpoint == state.attempted_checkpoint
    assert state.last_success_at == NOW and state.last_attempted_at == NOW
    assert state.last_status == "succeeded" and state.consecutive_failures == 0


def test_failure_keeps_old_checkpoint_but_stores_attempt(session: Session) -> None:
    earlier = NOW - timedelta(days=10)
    _attempt(session, _result(RunStatus.SUCCEEDED, advanced=True, watermark=earlier))
    state = _attempt(session, _result(RunStatus.FAILED, page_errors=1, watermark=NOW))
    assert state.watermark == earlier
    assert state.attempted_checkpoint == _checkpoint(NOW)
    assert state.consecutive_failures == 1 and state.last_error == "HTTP 503"
    assert state.last_success_at == NOW  # from the earlier success


def test_partial_with_page_error_counts_as_failure_but_keeps_progress(session: Session) -> None:
    state = _attempt(
        session, _result(RunStatus.PARTIAL, advanced=True, page_errors=1, complete=False)
    )
    assert state.consecutive_failures == 1
    assert state.checkpoint is not None and state.checkpoint["complete"] is False


def test_record_errors_only_do_not_trip_the_breaker(session: Session) -> None:
    result = _result(RunStatus.PARTIAL, advanced=True)
    result.error_count = 4
    assert is_failed_run(result) is False
    assert _attempt(session, result).consecutive_failures == 0


def test_circuit_opens_after_threshold_and_closes_after_success(session: Session) -> None:
    for i in range(2):
        state = _attempt(
            session, _result(RunStatus.FAILED, page_errors=1), now=NOW + timedelta(minutes=i)
        )
        assert state.circuit_open_until is None
    state = _attempt(
        session, _result(RunStatus.FAILED, page_errors=1), now=NOW + timedelta(minutes=2)
    )
    assert state.consecutive_failures == 3
    assert state.circuit_open_until == NOW + timedelta(minutes=62)
    assert state.circuit_open(NOW + timedelta(minutes=30)) is True
    assert state.circuit_open(NOW + timedelta(minutes=63)) is False

    reopened = _attempt(
        session, _result(RunStatus.FAILED, page_errors=1), now=NOW + timedelta(minutes=70)
    )
    assert reopened.circuit_open_until == NOW + timedelta(
        minutes=130
    )  # trial failed: reopen at once

    closed = _attempt(
        session, _result(RunStatus.SUCCEEDED, advanced=True), now=NOW + timedelta(minutes=140)
    )
    assert (
        closed.consecutive_failures == 0
        and closed.circuit_open_until is None
        and closed.last_error is None
    )


def test_full_refresh_flag_is_stored_only_when_checkpoint_advances(session: Session) -> None:
    assert (
        _attempt(
            session, _result(RunStatus.FAILED, page_errors=1), full_refresh_completed=True
        ).last_full_refresh_at
        is None
    )
    assert (
        _attempt(
            session, _result(RunStatus.SUCCEEDED, advanced=True), full_refresh_completed=True
        ).last_full_refresh_at
        == NOW
    )


def test_record_attempt_rejects_skipped_and_bad_settings(session: Session) -> None:
    with pytest.raises(ValueError, match="skipped"):
        _attempt(session, _result(RunStatus.SKIPPED))
    with pytest.raises(ValueError):
        record_attempt(session, _result(RunStatus.FAILED), failure_threshold=0, cooldown_minutes=5)


# --------------------------------------------------------------------------------------
# plan_window
# --------------------------------------------------------------------------------------
def test_first_run_collects_the_full_lookback(cfg_settings: Settings) -> None:
    plan = plan_window(example_config(), CheckpointState(SOURCE), cfg_settings, NOW)
    assert plan.window == CollectionWindow(NOW - timedelta(days=365), NOW, "full")
    assert plan.full_refresh is True and "initial" in plan.notes[0]


def test_incremental_window_overlaps_the_watermark(cfg_settings: Settings) -> None:
    state = CheckpointState(SOURCE, checkpoint=_checkpoint(NOW - timedelta(hours=2)))
    plan = plan_window(example_config(), state, cfg_settings, NOW)
    assert plan.window is not None and plan.window.mode == "incremental"
    assert plan.window.start == NOW - timedelta(hours=2) - timedelta(days=7)
    assert plan.full_refresh is False


def test_incremental_start_never_precedes_the_lookback(cfg_settings: Settings) -> None:
    state = CheckpointState(SOURCE, checkpoint=_checkpoint(NOW - timedelta(days=900)))
    plan = plan_window(example_config(), state, cfg_settings, NOW)
    assert plan.window is not None and plan.window.start == NOW - timedelta(days=365)


def test_watermark_in_the_future_still_gives_a_valid_window(cfg_settings: Settings) -> None:
    state = CheckpointState(SOURCE, checkpoint=_checkpoint(NOW + timedelta(days=30)))
    plan = plan_window(example_config(), state, cfg_settings, NOW)
    assert plan.window is not None and plan.window.start == NOW - timedelta(days=7)


def test_unfinished_collection_resumes_from_the_watermark(cfg_settings: Settings) -> None:
    state = CheckpointState(
        SOURCE, checkpoint=_checkpoint(NOW - timedelta(days=200), complete=False)
    )
    plan = plan_window(example_config(), state, cfg_settings, NOW)
    assert plan.window == CollectionWindow(NOW - timedelta(days=200), NOW, "resume")
    assert plan.full_refresh is True


def test_full_refresh_source_skips_until_due(cfg_settings: Settings) -> None:
    config = example_config(refresh_mode="full", full_refresh_window_days=7)
    never = plan_window(config, CheckpointState(SOURCE), cfg_settings, NOW)
    assert never.window is not None and never.window.mode == "full"

    recent = CheckpointState(
        SOURCE, checkpoint=_checkpoint(NOW), last_full_refresh_at=NOW - timedelta(days=2)
    )
    skipped = plan_window(config, recent, cfg_settings, NOW)
    assert skipped.window is None and skipped.skip_reason == "next full refresh due 2026-09-06"

    old = CheckpointState(
        SOURCE, checkpoint=_checkpoint(NOW), last_full_refresh_at=NOW - timedelta(days=7)
    )
    due = plan_window(config, old, cfg_settings, NOW)
    assert due.window is not None and due.window.mode == "full" and "due" in due.notes[0]


def test_full_refresh_source_resumes_an_unfinished_refresh(cfg_settings: Settings) -> None:
    config = example_config(refresh_mode="full", full_refresh_window_days=30)
    state = CheckpointState(
        SOURCE,
        checkpoint=_checkpoint(NOW - timedelta(days=100), complete=False),
        last_full_refresh_at=NOW - timedelta(days=1),
    )
    plan = plan_window(config, state, cfg_settings, NOW)
    assert plan.window is not None and plan.window.mode == "resume"


def test_requested_full_ignores_the_checkpoint(cfg_settings: Settings) -> None:
    state = CheckpointState(SOURCE, checkpoint=_checkpoint(NOW - timedelta(hours=1)))
    plan = plan_window(example_config(), state, cfg_settings, NOW, requested="full")
    assert plan.window is not None and plan.window.mode == "full"


def test_plan_requires_aware_now(cfg_settings: Settings) -> None:
    with pytest.raises(ValueError):
        plan_window(example_config(), CheckpointState(SOURCE), cfg_settings, datetime(2026, 1, 1))


# --------------------------------------------------------------------------------------
# CollectionWindow
# --------------------------------------------------------------------------------------
def test_window_slices_cover_the_window_exactly() -> None:
    window = CollectionWindow(NOW - timedelta(days=65), NOW, "full")
    pieces = window.slices(30)
    assert [(p.end - p.start).days for p in pieces] == [30, 30, 5]
    assert pieces[0].start == window.start and pieces[-1].end == window.end
    assert all(a.end == b.start for a, b in zip(pieces, pieces[1:], strict=False))
    assert window.slices(None) == [window]


@pytest.mark.parametrize(
    ("start", "end"),
    [(NOW, NOW), (NOW, NOW - timedelta(days=1)), (datetime(2026, 1, 1), NOW)],
)
def test_invalid_windows(start: datetime, end: datetime) -> None:
    with pytest.raises(ValueError):
        CollectionWindow(start, end)


def test_slice_length_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CollectionWindow(NOW - timedelta(days=1), NOW).slices(0)
