"""Integration tests for the ingestion pipeline: orchestrator, checkpoints, breaker, audit.

A :class:`Harness` wires several copies of the reference adapter to their own fake APIs and a
file-backed SQLite database (so sources really run on concurrent threads). Time is passed in
explicitly, so checkpoint and circuit-breaker behaviour is tested without waiting.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import func, select

from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import IngestionRun, JobRun, SourceCheckpoint, SourceRecord
from cews.database.repositories import MixedDataError, SourceRecordData, upsert_source_record
from cews.ingestion.base import SourceAdapter, build_http_client
from cews.ingestion.checkpoints import CheckpointState, read_checkpoint
from cews.ingestion.orchestrator import (
    fetch_all_enabled_sources,
    fetch_incremental,
    fetch_source,
)
from cews.ingestion.registry import SourceConfig, SourceRegistry
from cews.ingestion.results import RunReport
from cews.settings import Settings, load_settings
from support_adapters import ExampleAdapter, FakeApi, example_config, make_items, no_sleep

pytestmark = pytest.mark.integration

NOW = datetime(2026, 6, 1, tzinfo=UTC)
# With a 180-day lookback and 30-day slices the window [2025-12-03, 2026-06-01) has six slices.
# make_items(23) spans 2026-01-01 .. 2026-03-08, so requests are (page size 5):
#   1: slice 1 (1 item) | 2-3: slice 2 (10 items) | 4-5: slice 3 (10 items) | 6: slice 4 | 7-8.
BASE_SETTINGS: dict[str, Any] = {
    "default_lookback_days": 180,
    "source_max_retries": 0,
    "circuit_breaker_failure_threshold": 2,
    "circuit_breaker_cooldown_minutes": 60,
}


class Harness:
    """Registry, fake APIs and database for one test."""

    def __init__(self, tmp_path: Path, **settings: Any) -> None:
        self.settings: Settings = load_settings(
            env_file=None, overrides={**BASE_SETTINGS, **settings}
        )
        self.engine = create_db_engine(f"sqlite:///{tmp_path / 'ingest.db'}")
        upgrade_database(self.engine)
        self.factory = create_session_factory(self.engine)
        self.configs: list[SourceConfig] = []
        self.apis: dict[str, FakeApi] = {}
        self.classes: dict[str, type[SourceAdapter]] = {}

    def add(
        self,
        source_id: str,
        *,
        items: int = 23,
        cls: type[SourceAdapter] | None = None,
        **config: Any,
    ) -> FakeApi:
        """Register a source served by its own fake API."""
        api = FakeApi(items=make_items(items))
        adapter_cls = cls or type(f"{source_id}_adapter", (ExampleAdapter,), {})
        adapter_cls.source_name = source_id  # type: ignore[misc]
        self.configs.append(example_config(id=source_id, **config))
        self.apis[source_id] = api
        self.classes[source_id] = adapter_cls
        return api

    def factory_fn(self, config: SourceConfig, settings: Settings) -> SourceAdapter | None:
        cls = self.classes.get(config.id)
        if cls is None:
            return None
        http = build_http_client(
            settings, config, transport=self.apis[config.id].transport(), sleep=no_sleep
        )
        return cls(settings, config, http)

    def run(self, *, now: datetime = NOW, **kwargs: Any) -> RunReport:
        return fetch_all_enabled_sources(
            self.settings,
            self.factory,
            registry=SourceRegistry(1, tuple(self.configs)),
            adapter_factory=self.factory_fn,
            now=now,
            **kwargs,
        )

    def state(self, source: str) -> CheckpointState:
        with session_scope(self.factory) as session:
            return read_checkpoint(session, source)

    def count(self, model: type = SourceRecord, **where: Any) -> int:
        with session_scope(self.factory) as session:
            query = select(func.count()).select_from(model)
            for name, value in where.items():
                query = query.where(getattr(model, name) == value)
            return int(session.scalar(query) or 0)

    def runs(self, source: str) -> list[IngestionRun]:
        with session_scope(self.factory) as session:
            return list(
                session.scalars(
                    select(IngestionRun)
                    .where(IngestionRun.source == source)
                    .order_by(IngestionRun.id)
                )
            )

    def close(self) -> None:
        self.engine.dispose()


@pytest.fixture
def harness(tmp_path: Path) -> Any:
    instance = Harness(tmp_path)
    yield instance
    instance.close()


def _by_source(report: RunReport) -> dict[str, Any]:
    return {result.source_name: result for result in report.results}


# --------------------------------------------------------------------------------------
# Happy path, idempotency, incremental windows
# --------------------------------------------------------------------------------------
def test_fetch_all_collects_every_enabled_source(harness: Harness) -> None:
    for name in ("alpha", "beta", "gamma"):
        harness.add(name)
    report = harness.run()
    assert report.status is RunStatus.SUCCEEDED
    assert report.count_by_status() == {"succeeded": 3}
    assert report.totals()["records_inserted"] == 69
    for name in ("alpha", "beta", "gamma"):
        assert harness.count(source=name) == 23
        state = harness.state(name)
        assert state.watermark == NOW and state.complete
        assert state.consecutive_failures == 0 and state.last_status == "succeeded"
    with session_scope(harness.factory) as session:
        job = session.scalars(select(JobRun)).one()
        assert job.status == "succeeded" and job.job_id == report.job_id
        assert job.summary_json is not None and job.summary_json["records_inserted"] == 69


def test_rerun_is_idempotent_and_uses_an_incremental_window(harness: Harness) -> None:
    api = harness.add("alpha")
    first = _by_source(harness.run())["alpha"]
    assert first.mode == "full" and first.records_inserted == 23
    requests_after_first = len(api.requests)

    later = NOW + timedelta(days=1)
    second = _by_source(harness.run(now=later))["alpha"]
    assert second.status is RunStatus.SUCCEEDED
    assert second.mode == "incremental"
    assert (second.records_inserted, second.records_updated) == (0, 0)
    assert harness.count(source="alpha") == 23
    lower_bounds = {
        datetime.fromisoformat(r.url.params["from"]) for r in api.requests[requests_after_first:]
    }
    overlap = timedelta(days=harness.settings.incremental_lookback_days)
    assert min(lower_bounds) == NOW - overlap  # watermark minus the overlap, not the full lookback
    assert harness.state("alpha").watermark == later


def test_fetch_incremental_and_fetch_source_can_run_one_source(harness: Harness) -> None:
    harness.add("alpha")
    config = harness.configs[0]
    adapter = harness.factory_fn(config, harness.settings)
    try:
        result = fetch_incremental(
            config, adapter, harness.settings, harness.factory, job_id="job-1", now=NOW
        )
    finally:
        assert adapter is not None
        adapter.close()
    assert result.status is RunStatus.SUCCEEDED and result.records_inserted == 23
    assert [run.job_id for run in harness.runs("alpha")] == ["job-1"]


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------
def test_one_failing_source_does_not_stop_the_others(harness: Harness) -> None:
    harness.add("alpha")
    harness.add("broken").fail_always = True
    harness.add("gamma")
    report = harness.run()
    results = _by_source(report)
    assert report.status is RunStatus.PARTIAL
    assert results["alpha"].status is RunStatus.SUCCEEDED
    assert results["gamma"].status is RunStatus.SUCCEEDED
    assert results["broken"].status is RunStatus.FAILED and results["broken"].errors
    assert harness.count(source="broken") == 0
    assert harness.state("broken").consecutive_failures == 1
    assert harness.state("broken").checkpoint is None
    audit = harness.runs("broken")[-1]
    assert audit.status == "failed" and audit.error_summary
    with session_scope(harness.factory) as session:
        job = session.scalars(select(JobRun)).one()
        assert job.status == "partial" and "broken" in (job.error_summary or "")


def test_adapter_that_crashes_while_collecting_fails_alone(harness: Harness) -> None:
    class Crashing(ExampleAdapter):
        def collect(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("adapter bug")

    harness.add("alpha")
    harness.add("crashing", cls=Crashing)
    results = _by_source(harness.run())
    assert results["alpha"].status is RunStatus.SUCCEEDED
    assert results["crashing"].status is RunStatus.FAILED
    assert "adapter bug" in results["crashing"].errors[0]


def test_adapter_that_cannot_be_created_fails_alone(harness: Harness) -> None:
    class Unbuildable(ExampleAdapter):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ValueError("bad adapter setup")

    harness.add("alpha")
    harness.add("unbuildable", cls=Unbuildable)
    results = _by_source(harness.run())
    assert results["alpha"].status is RunStatus.SUCCEEDED
    assert results["unbuildable"].status is RunStatus.FAILED
    assert "bad adapter setup" in results["unbuildable"].errors[0]
    assert harness.runs("unbuildable")[-1].status == "failed"  # still audited


def test_crashing_configuration_check_fails_only_that_source(harness: Harness) -> None:
    class BadCheck(ExampleAdapter):
        def validate_configuration(self) -> list[str]:
            raise KeyError("missing setting")

    harness.add("alpha")
    harness.add("badcheck", cls=BadCheck)
    results = _by_source(harness.run())
    assert results["alpha"].status is RunStatus.SUCCEEDED
    assert results["badcheck"].status is RunStatus.SKIPPED
    assert "configuration check failed" in results["badcheck"].warnings[0]


def test_transient_errors_are_retried(tmp_path: Path) -> None:
    harness = Harness(tmp_path, source_max_retries=3)
    try:
        api = harness.add("alpha")
        api.rate_limit_first = 2  # two HTTP 429 responses before the API answers
        result = _by_source(harness.run())["alpha"]
    finally:
        harness.close()
    assert result.status is RunStatus.SUCCEEDED and result.records_inserted == 23
    assert result.rate_limit["retries"] == 2
    assert result.rate_limit["rate_limited"] == 2


def test_checkpoint_never_moves_past_a_failure(harness: Harness) -> None:
    api = harness.add("alpha")
    api.fail_pages = {4}  # first page of slice 3; slices 1-2 (11 records) finish first
    first = _by_source(harness.run())["alpha"]
    assert first.status is RunStatus.PARTIAL
    assert first.records_inserted == 11
    state = harness.state("alpha")
    assert state.watermark == NOW - timedelta(days=180) + timedelta(days=60)  # end of slice 2
    assert state.complete is False
    assert state.consecutive_failures == 1

    api.fail_pages = set()
    second = _by_source(harness.run(now=NOW + timedelta(hours=2)))["alpha"]
    assert second.status is RunStatus.SUCCEEDED
    assert second.mode == "resume"
    assert second.records_inserted == 12
    assert harness.count(source="alpha") == 23
    assert harness.state("alpha").complete is True
    assert harness.state("alpha").consecutive_failures == 0


def test_invalid_records_are_counted_without_stopping_the_run(harness: Harness) -> None:
    api = harness.add("alpha")
    api.malformed_ids = {"EX-0003", "EX-0017"}  # served without a title
    result = _by_source(harness.run())["alpha"]
    assert result.status is RunStatus.PARTIAL
    assert result.records_inserted == 21 and result.record_error_count == 2
    assert any("EX-0003" in e or "title" in e for e in result.errors)
    state = harness.state("alpha")
    assert state.complete is True  # bad records are rejected, not retried forever
    assert state.consecutive_failures == 0  # record errors do not trip the circuit breaker


def test_circuit_breaker_pauses_a_failing_source_and_recovers(harness: Harness) -> None:
    api = harness.add("alpha")
    api.fail_always = True
    harness.run(now=NOW)
    harness.run(now=NOW + timedelta(hours=2))  # second failure opens the circuit (threshold 2)
    state = harness.state("alpha")
    assert state.consecutive_failures == 2
    assert state.circuit_open_until == NOW + timedelta(hours=2, minutes=60)

    requests_before = len(api.requests)
    paused = _by_source(harness.run(now=NOW + timedelta(hours=2, minutes=30)))["alpha"]
    assert paused.status is RunStatus.SKIPPED and "paused" in paused.warnings[0]
    assert len(api.requests) == requests_before  # the source was not contacted

    api.fail_always = False
    recovered = _by_source(harness.run(now=NOW + timedelta(hours=4)))["alpha"]
    assert recovered.status is RunStatus.SUCCEEDED
    state = harness.state("alpha")
    assert state.consecutive_failures == 0 and state.circuit_open_until is None


# --------------------------------------------------------------------------------------
# Scheduling policies and limits
# --------------------------------------------------------------------------------------
def test_full_refresh_source_waits_for_its_window(harness: Harness) -> None:
    harness.add("patents", refresh_mode="full", full_refresh_window_days=7)
    first = _by_source(harness.run(now=NOW))["patents"]
    assert first.status is RunStatus.SUCCEEDED and first.mode == "full"

    early = _by_source(harness.run(now=NOW + timedelta(days=1)))["patents"]
    assert early.status is RunStatus.SKIPPED and "next full refresh due" in early.warnings[0]

    due = _by_source(harness.run(now=NOW + timedelta(days=8)))["patents"]
    assert due.status is RunStatus.SUCCEEDED and due.mode == "full"


def test_page_limit_stops_early_and_later_runs_resume(harness: Harness) -> None:
    harness.add("alpha", max_pages_per_run=3)
    now = NOW
    first = _by_source(harness.run(now=now))["alpha"]
    assert first.truncated is True and any("max_pages_per_run" in w for w in first.warnings)
    assert harness.state("alpha").complete is False
    for _ in range(5):
        if harness.state("alpha").complete:
            break
        now += timedelta(hours=2)
        harness.run(now=now)
    assert harness.state("alpha").complete is True
    assert harness.count(source="alpha") == 23


def test_sources_run_concurrently_while_writes_stay_consistent(tmp_path: Path) -> None:
    harness = Harness(tmp_path, max_concurrent_source_jobs=3)
    active = 0
    peak = 0
    lock = threading.Lock()
    try:
        for name in ("a1", "a2", "a3", "a4"):
            api = harness.add(name)
            original = api.handler

            def slow(request: httpx.Request, handler: Any = original) -> httpx.Response:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(0.02)
                try:
                    return handler(request)
                finally:
                    with lock:
                        active -= 1

            api.handler = slow  # type: ignore[method-assign]
        report = harness.run()
        assert report.status is RunStatus.SUCCEEDED
        assert harness.count() == 4 * 23
    finally:
        harness.close()
    assert 2 <= peak <= 3  # overlapped, but never more than MAX_CONCURRENT_SOURCE_JOBS


# --------------------------------------------------------------------------------------
# Skips, dry runs, selection, data-origin guard, audit
# --------------------------------------------------------------------------------------
def test_dry_run_writes_nothing_at_all(harness: Harness) -> None:
    harness.add("alpha")
    report = harness.run(dry_run=True)
    result = _by_source(report)["alpha"]
    assert report.dry_run is True and result.records_inserted == 23
    for model in (SourceRecord, IngestionRun, JobRun, SourceCheckpoint):
        assert harness.count(model) == 0


def test_disabled_unimplemented_and_misconfigured_sources_are_skipped(harness: Harness) -> None:
    class NeedsKey(ExampleAdapter):
        def validate_configuration(self) -> list[str]:
            return ["needs_key: API key missing"]

    harness.add("alpha")
    keyed_api = harness.add("needs_key", cls=NeedsKey)
    harness.configs.append(example_config(id="not_built_yet"))  # no adapter registered
    harness.configs.append(example_config(id="switched_off", env_flag="ENABLE_OPENALEX"))

    results = _by_source(harness.run())
    assert set(results) == {"alpha", "needs_key", "not_built_yet"}  # disabled ones not selected
    assert results["needs_key"].status is RunStatus.SKIPPED
    assert "API key missing" in results["needs_key"].warnings[0]
    assert keyed_api.requests == []
    assert results["not_built_yet"].warnings == ["no adapter implemented yet"]

    explicit = _by_source(harness.run(sources=["switched_off"]))["switched_off"]
    assert explicit.status is RunStatus.SKIPPED
    assert "ENABLE_OPENALEX=false" in explicit.warnings[0]


def test_nothing_runnable_writes_no_job_row(harness: Harness) -> None:
    harness.configs.append(example_config(id="not_built_yet"))
    report = harness.run()
    assert report.status is RunStatus.SKIPPED
    assert harness.count(JobRun) == 0


def test_unknown_source_is_rejected(harness: Harness) -> None:
    harness.add("alpha")
    with pytest.raises(KeyError, match="unknown source"):
        harness.run(sources=["does_not_exist"])


def test_live_fetch_refuses_a_database_with_demo_data(harness: Harness) -> None:
    harness.add("alpha")
    with session_scope(harness.factory) as session:
        upsert_source_record(
            session,
            SourceRecordData(
                source="synthetic_publications",
                source_record_id="SYN-1",
                record_type="publication",
                content_hash="a" * 64,
                fetched_at=NOW,
                is_synthetic=True,
            ),
        )
    with pytest.raises(MixedDataError):
        harness.run()
    assert harness.count(source="alpha") == 0
    report = harness.run(allow_mixed=True)
    assert report.status is RunStatus.SUCCEEDED


def test_audit_rows_record_every_counter(harness: Harness) -> None:
    api = harness.add("alpha")
    api.malformed_ids = {"EX-0005"}
    result = _by_source(harness.run())["alpha"]
    audit = harness.runs("alpha")[-1]
    expected: Mapping[str, Any] = {
        "status": result.status.value,
        "records_received": result.records_received,
        "records_inserted": result.records_inserted,
        "records_updated": result.records_updated,
        "records_skipped": result.records_skipped,
        "duplicate_count": result.duplicate_count,
        "error_count": result.error_count,
        "collection_mode": "full",
    }
    for name, value in expected.items():
        assert getattr(audit, name) == value, name
    assert audit.checkpoint_json == result.checkpoint
    assert audit.rate_limit_json is not None and audit.rate_limit_json["requests"] >= 1
    assert audit.end_time is not None and audit.start_time <= audit.end_time
    assert audit.is_synthetic is False


def test_skipped_sources_do_not_touch_checkpoints(harness: Harness) -> None:
    harness.configs.append(example_config(id="not_built_yet"))
    harness.run()
    assert harness.count(SourceCheckpoint) == 0
    assert harness.runs("not_built_yet")[-1].status == "skipped"


def test_fetch_source_dry_run_returns_without_writing(harness: Harness) -> None:
    harness.add("alpha")
    config = harness.configs[0]
    adapter = harness.factory_fn(config, harness.settings)
    assert adapter is not None
    try:
        result = fetch_source(
            config, adapter, harness.settings, harness.factory, job_id="j", dry_run=True, now=NOW
        )
    finally:
        adapter.close()
    assert result.dry_run and result.records_inserted == 23
    assert harness.count(IngestionRun) == 0
