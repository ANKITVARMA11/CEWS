"""Unit tests for one refresh cycle: what runs, in what order, and what a failure does."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import JobRun, SourceRecord
from cews.database.repositories import MixedDataError
from cews.ingestion.results import RunReport, SourceResult
from cews.scheduler.job_state import (
    INTERRUPTED,
    JobAlreadyRunningError,
    lock_directory,
    lock_status,
    start_job_run,
)
from cews.scheduler.jobs import (
    ANALYSIS_STEP_NAMES,
    StepFunction,
    default_analysis_steps,
    run_analysis_steps,
    run_refresh,
)
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


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
def settings(tmp_path: Path) -> Settings:
    """Settings whose database (and therefore lock folder) live in the test's own directory."""
    return load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./jobs.db")}
    )


def fetch_report(*statuses: RunStatus) -> RunReport:
    report = RunReport(job_id="fetchjob1", started_at=NOW)
    for index, status in enumerate(statuses):
        report.results.append(
            SourceResult(source_name=f"source{index}", collection_start=NOW, status=status)
        )
    return report


def fake_fetch(*statuses: RunStatus, calls: list[dict[str, Any]] | None = None) -> Any:
    def fetch(settings: Settings, factory: Any, **kwargs: Any) -> RunReport:
        if calls is not None:
            calls.append(kwargs)
        return fetch_report(*statuses)

    return fetch


def recording_steps(
    log: list[str], *, fail_at: str | None = None
) -> list[tuple[str, StepFunction]]:
    def make(name: str) -> StepFunction:
        def step(factory: Any, settings: Any, now: datetime, synthetic: bool) -> dict[str, Any]:
            log.append(name)
            if name == fail_at:
                raise RuntimeError(f"{name} broke")
            return {"ran": name, "as_of": now.date(), "synthetic": synthetic}

        return step

    return [(name, make(name)) for name in ("normalize", "score", "insights")]


def stored_run(factory: sessionmaker[Session]) -> JobRun:
    with session_scope(factory) as session:
        return session.query(JobRun).order_by(JobRun.id.desc()).first()  # type: ignore[return-value]


# --------------------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------------------
def test_a_clean_cycle_fetches_then_runs_every_step_in_order(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    log: list[str] = []
    report = run_refresh(
        factory, settings, fetch=fake_fetch(RunStatus.SUCCEEDED),
        analysis_steps=recording_steps(log), now=NOW,
    )  # fmt: skip
    assert report.status is RunStatus.SUCCEEDED and report.error is None
    assert log == ["normalize", "score", "insights"]
    assert [s.status for s in report.steps] == ["succeeded"] * 3
    assert report.fetch is not None and report.fetch["status"] == "succeeded"


def test_the_cycle_leaves_an_audit_row_and_releases_the_lock(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    report = run_refresh(
        factory, settings, trigger="schedule", fetch=fake_fetch(RunStatus.SUCCEEDED),
        analysis_steps=recording_steps([]), now=NOW,
    )  # fmt: skip
    row = stored_run(factory)
    assert row.job_id == report.job_id and row.status == "succeeded" and row.trigger == "schedule"
    assert row.summary_json is not None
    assert [s["name"] for s in row.summary_json["steps"]] == ["normalize", "score", "insights"]
    assert row.finished_at is not None
    assert lock_status(lock_directory(settings), "refresh") is None


def test_step_details_are_stored_as_plain_json(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    run_refresh(factory, settings, fetch=fake_fetch(RunStatus.SUCCEEDED),
                analysis_steps=recording_steps([]), now=NOW)  # fmt: skip
    summary = stored_run(factory).summary_json
    assert summary is not None
    detail = summary["steps"][0]["detail"]
    assert detail["as_of"] == "2026-09-01" and detail["ran"] == "normalize"


def test_the_steps_are_told_when_the_data_is_synthetic(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        session.add(
            SourceRecord(
                source="s",
                source_record_id="r",
                record_type="publication",
                fetched_at=NOW,
                content_hash="a" * 64,
                is_synthetic=True,
            )
        )
    report = run_refresh(
        factory, settings, skip_fetch=True, analysis_steps=recording_steps([]), now=NOW
    )
    assert report.steps[0].detail["synthetic"] is True


def test_the_steps_are_told_when_the_data_is_live(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        session.add(
            SourceRecord(
                source="s",
                source_record_id="r",
                record_type="publication",
                fetched_at=NOW,
                content_hash="b" * 64,
                is_synthetic=False,
            )
        )
    report = run_refresh(
        factory, settings, skip_fetch=True, analysis_steps=recording_steps([]), now=NOW
    )
    assert report.steps[0].detail["synthetic"] is False


def test_the_default_steps_are_in_dependency_order(settings: Settings) -> None:
    assert [name for name, _ in default_analysis_steps(settings)] == list(ANALYSIS_STEP_NAMES)
    assert ANALYSIS_STEP_NAMES.index("features") < ANALYSIS_STEP_NAMES.index("score")
    assert ANALYSIS_STEP_NAMES.index("score") < ANALYSIS_STEP_NAMES.index("insights")
    assert ANALYSIS_STEP_NAMES.index("normalize") < ANALYSIS_STEP_NAMES.index("competitors")


def test_the_power_bi_export_only_runs_when_enabled() -> None:
    off = load_settings(env_file=None, overrides={"generate_powerbi_exports": False})
    assert "export" not in [name for name, _ in default_analysis_steps(off)]
    assert "export" in [name for name, _ in default_analysis_steps(load_settings(env_file=None))]


# --------------------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------------------
def test_a_failing_step_stops_the_ones_after_it_and_keeps_the_ones_before(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """Scoring on features that were never recomputed would publish stale numbers as fresh."""
    log: list[str] = []
    report = run_refresh(
        factory, settings, fetch=fake_fetch(RunStatus.SUCCEEDED),
        analysis_steps=recording_steps(log, fail_at="score"), now=NOW,
    )  # fmt: skip
    assert log == ["normalize", "score"]  # insights never ran
    assert [(s.name, s.status) for s in report.steps] == [
        ("normalize", "succeeded"), ("score", "failed"), ("insights", "skipped"),
    ]  # fmt: skip
    assert "score broke" in (report.steps[1].error or "")
    assert report.status is RunStatus.PARTIAL  # new data arrived and one step finished
    assert report.error is not None and "'score' failed" in report.error


def test_a_cycle_where_nothing_succeeded_is_a_plain_failure(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    report = run_refresh(factory, settings, skip_fetch=True,
                         analysis_steps=recording_steps([], fail_at="normalize"), now=NOW)  # fmt: skip
    assert report.status is RunStatus.FAILED
    assert [s.status for s in report.steps] == ["failed", "skipped", "skipped"]


def test_when_every_source_fails_the_analysis_is_not_re_run(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    log: list[str] = []
    report = run_refresh(factory, settings, fetch=fake_fetch(RunStatus.FAILED, RunStatus.FAILED),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert report.status is RunStatus.FAILED and log == []
    assert report.error is not None and "every source that ran failed" in report.error


def test_one_failing_source_does_not_stop_the_cycle(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    log: list[str] = []
    report = run_refresh(factory, settings, fetch=fake_fetch(RunStatus.SUCCEEDED, RunStatus.FAILED),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert log == ["normalize", "score", "insights"]  # the analysis still ran on what arrived
    assert report.status is RunStatus.PARTIAL and report.fetch is not None
    assert report.fetch["sources"] == {"source0": "succeeded", "source1": "failed"}


def test_no_enabled_sources_still_runs_the_analysis(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    log: list[str] = []
    report = run_refresh(factory, settings, fetch=fake_fetch(RunStatus.SKIPPED),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert log == ["normalize", "score", "insights"] and report.status is RunStatus.SUCCEEDED


def test_writing_live_data_into_a_demo_database_is_refused_clearly(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    def refuse(*_a: Any, **_k: Any) -> RunReport:
        raise MixedDataError("this database holds demo data; live data would mix with it")

    log: list[str] = []
    report = run_refresh(
        factory, settings, fetch=refuse, analysis_steps=recording_steps(log), now=NOW
    )
    assert report.status is RunStatus.FAILED and log == []
    assert report.error is not None and "demo data" in report.error


def test_an_unknown_source_is_reported_by_name(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    def unknown(*_a: Any, **_k: Any) -> RunReport:
        raise KeyError("unknown source 'nonsense'")

    report = run_refresh(factory, settings, fetch=unknown, sources=["nonsense"],
                         analysis_steps=recording_steps([]), now=NOW)  # fmt: skip
    assert report.status is RunStatus.FAILED and report.error == "unknown source 'nonsense'"


def test_an_unexpected_crash_still_closes_the_audit_row_and_frees_the_lock(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    def crash(*_a: Any, **_k: Any) -> RunReport:
        raise RuntimeError("the network adapter exploded")

    report = run_refresh(
        factory, settings, fetch=crash, analysis_steps=recording_steps([]), now=NOW
    )
    assert report.status is RunStatus.FAILED and "exploded" in (report.error or "")
    row = stored_run(factory)
    assert row.status == "failed" and row.finished_at is not None  # never left saying "running"
    assert lock_status(lock_directory(settings), "refresh") is None


# --------------------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------------------
def test_a_dry_run_fetches_without_writing_and_skips_the_analysis(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    calls: list[dict[str, Any]] = []
    log: list[str] = []
    report = run_refresh(factory, settings, dry_run=True, fetch=fake_fetch(RunStatus.SUCCEEDED, calls=calls),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert calls[0]["dry_run"] is True and log == []
    assert report.status is RunStatus.SUCCEEDED and report.dry_run is True
    assert stored_run(factory).dry_run is True


def test_skip_fetch_runs_only_the_analysis(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    calls: list[dict[str, Any]] = []
    log: list[str] = []
    report = run_refresh(factory, settings, skip_fetch=True, fetch=fake_fetch(RunStatus.SUCCEEDED, calls=calls),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert calls == [] and log == ["normalize", "score", "insights"] and report.fetch is None


def test_skip_analysis_runs_only_the_fetch(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    log: list[str] = []
    report = run_refresh(factory, settings, skip_analysis=True, fetch=fake_fetch(RunStatus.SUCCEEDED),
                         analysis_steps=recording_steps(log), now=NOW)  # fmt: skip
    assert log == [] and report.steps == [] and report.status is RunStatus.SUCCEEDED


def test_the_source_list_and_trigger_reach_the_fetch(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    calls: list[dict[str, Any]] = []
    run_refresh(factory, settings, sources=["pubmed"], allow_mixed=True, trigger="schedule",
                fetch=fake_fetch(RunStatus.SUCCEEDED, calls=calls), analysis_steps=[], now=NOW)  # fmt: skip
    assert calls[0]["sources"] == ["pubmed"] and calls[0]["allow_mixed"] is True
    assert calls[0]["trigger"] == "schedule"


# --------------------------------------------------------------------------------------
# Never two at once
# --------------------------------------------------------------------------------------
def test_a_second_cycle_is_refused_while_one_is_running(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    refused: list[Exception] = []

    def fetch_that_tries_a_second_cycle(*_a: Any, **_k: Any) -> RunReport:
        try:
            run_refresh(factory, settings, skip_fetch=True, analysis_steps=[], now=NOW)
        except JobAlreadyRunningError as exc:
            refused.append(exc)
        return fetch_report(RunStatus.SUCCEEDED)

    run_refresh(
        factory, settings, fetch=fetch_that_tries_a_second_cycle, analysis_steps=[], now=NOW
    )
    assert len(refused) == 1 and "already running" in str(refused[0])
    with session_scope(factory) as session:
        assert session.query(JobRun).count() == 1  # the refused attempt did no work and left no row


def test_a_cycle_started_from_two_threads_at_once_runs_exactly_once(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    started = threading.Barrier(2)
    outcomes: list[str] = []

    def slow_fetch(*_a: Any, **_k: Any) -> RunReport:
        threading.Event().wait(0.3)
        return fetch_report(RunStatus.SUCCEEDED)

    def attempt() -> None:
        started.wait()
        try:
            run_refresh(factory, settings, fetch=slow_fetch, analysis_steps=[], now=NOW)
            outcomes.append("ran")
        except JobAlreadyRunningError:
            outcomes.append("refused")

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert sorted(outcomes) == ["ran", "refused"]


def test_a_run_left_running_by_a_crashed_process_is_closed_at_the_next_cycle(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    dead = start_job_run(factory, "refresh", trigger="schedule", now=NOW - timedelta(hours=3))
    run_refresh(factory, settings, skip_fetch=True, analysis_steps=[], now=NOW)
    with session_scope(factory) as session:
        old = session.query(JobRun).filter(JobRun.job_id == dead).one()
        assert old.status == "failed" and old.error_summary == INTERRUPTED


def test_the_inner_fetch_run_left_dangling_by_a_crash_is_closed_too(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """Seen for real: after a killed cycle, 'fetch ... running' rows stayed open forever."""
    dead_refresh = start_job_run(
        factory, "refresh", trigger="schedule", now=NOW - timedelta(hours=3)
    )
    dead_fetch = start_job_run(factory, "fetch", trigger="schedule", now=NOW - timedelta(hours=3))
    run_refresh(factory, settings, skip_fetch=True, analysis_steps=[], now=NOW)
    with session_scope(factory) as session:
        rows = {r.job_id: r for r in session.query(JobRun)}
    for job_id in (dead_refresh, dead_fetch):
        assert rows[job_id].status == "failed" and rows[job_id].error_summary == INTERRUPTED


# --------------------------------------------------------------------------------------
# run_analysis_steps directly
# --------------------------------------------------------------------------------------
def test_an_empty_step_list_does_nothing(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    assert run_analysis_steps(factory, settings, [], now=NOW, synthetic=False) == []


def test_step_timings_are_recorded(factory: sessionmaker[Session], settings: Settings) -> None:
    outcomes = run_analysis_steps(factory, settings, recording_steps([]), now=NOW, synthetic=False)
    assert all(o.seconds >= 0 for o in outcomes)
    assert outcomes[0].as_dict()["status"] == "succeeded"


def test_the_report_is_json_friendly(factory: sessionmaker[Session], settings: Settings) -> None:
    import json

    report = run_refresh(factory, settings, fetch=fake_fetch(RunStatus.SUCCEEDED),
                         analysis_steps=recording_steps([], fail_at="score"), now=NOW)  # fmt: skip
    payload = json.loads(json.dumps(report.as_dict()))
    assert payload["status"] == "partial" and payload["steps"][1]["status"] == "failed"
