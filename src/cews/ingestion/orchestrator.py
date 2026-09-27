"""Run source collections: one source (:func:`fetch_source`) or all enabled sources.

For every source the orchestrator:

1. skips it (with a reason) if it has no adapter yet, is misconfigured, is paused by the
   circuit breaker, or its full refresh is not due;
2. plans the collection window from its checkpoint;
3. runs the adapter, catching every error so one source can never stop the others;
4. updates the checkpoint and circuit breaker, and writes an ``ingestion_runs`` audit row.

Sources run concurrently (``MAX_CONCURRENT_SOURCE_JOBS``). On SQLite, database access is
serialized with a lock while network requests still overlap. Dry runs write nothing at all.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import session_scope
from cews.database.models import IngestionRun, JobRun
from cews.database.repositories import assert_origin_homogeneous
from cews.ingestion.base import SourceAdapter, WriteGate, adapter_classes, build_http_client
from cews.ingestion.checkpoints import (
    RequestedMode,
    plan_window,
    read_checkpoint,
    record_attempt,
)
from cews.ingestion.registry import SourceConfig, SourceRegistry, load_source_registry
from cews.ingestion.results import RunReport, SourceResult
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

AdapterFactory = Callable[[SourceConfig, Settings], SourceAdapter | None]
MAX_ERROR_SUMMARY = 1000


def create_adapter(
    config: SourceConfig,
    settings: Settings,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SourceAdapter | None:
    """Instantiate the adapter registered for ``config.id`` (None if none exists yet)."""
    cls = adapter_classes().get(config.id)
    if cls is None:
        return None
    http = build_http_client(settings, config, transport=transport, sleep=sleep)
    return cls(settings, config, http)


def configuration_problems(adapter: SourceAdapter) -> list[str]:
    """Return the adapter's configuration problems; a crashing check counts as a problem."""
    try:
        return list(adapter.validate_configuration())
    except Exception as exc:  # isolation: a broken check must not stop other sources
        LOGGER.exception("%s configuration check crashed", adapter.source_name)
        return [f"configuration check failed: {type(exc).__name__}: {str(exc)[:200]}"]


def _failed(source: str, error: str, *, now: datetime, dry_run: bool) -> SourceResult:
    return SourceResult(
        source,
        now,
        collection_end=datetime.now(UTC),
        status=RunStatus.FAILED,
        dry_run=dry_run,
        error_count=1,
        page_error_count=1,
        errors=[error],
    )


def _skipped(source: str, reason: str, *, now: datetime, dry_run: bool) -> SourceResult:
    return SourceResult(
        source_name=source,
        collection_start=now,
        collection_end=now,
        status=RunStatus.SKIPPED,
        dry_run=dry_run,
        warnings=[reason],
    )


def record_ingestion_run(session: Session, result: SourceResult, *, job_id: str) -> IngestionRun:
    """Write the audit row for one source result."""
    run = IngestionRun(
        job_id=job_id,
        source=result.source_name,
        start_time=result.collection_start,
        end_time=result.collection_end,
        status=result.status.value,
        records_requested=result.records_requested,
        records_received=result.records_received,
        records_inserted=result.records_inserted,
        records_updated=result.records_updated,
        records_skipped=result.records_skipped,
        duplicate_count=result.duplicate_count,
        error_count=result.error_count,
        checkpoint_json=result.checkpoint,
        warnings_json=list(result.warnings) or None,
        error_summary="; ".join(result.errors)[:MAX_ERROR_SUMMARY] or None,
        collection_mode=result.mode,
        rate_limit_json=result.rate_limit or None,
        is_synthetic=False,
    )
    session.add(run)
    session.flush()
    return run


def fetch_source(
    config: SourceConfig,
    adapter: SourceAdapter | None,
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    job_id: str,
    mode: RequestedMode = "auto",
    dry_run: bool = False,
    now: datetime | None = None,
    write_lock: WriteGate | None = None,
) -> SourceResult:
    """Collect one source and record the outcome. Never raises for source problems."""
    now = now or datetime.now(UTC)
    gate = write_lock if write_lock is not None else contextlib.nullcontext()

    result: SourceResult | None = None
    full_refresh = False
    if not config.is_enabled(settings):
        result = _skipped(
            config.id, f"disabled ({config.env_flag}=false)", now=now, dry_run=dry_run
        )
    elif adapter is None:
        result = _skipped(config.id, "no adapter implemented yet", now=now, dry_run=dry_run)
    else:
        problems = configuration_problems(adapter)
        if problems:
            result = _skipped(config.id, "; ".join(problems), now=now, dry_run=dry_run)

    if result is None and adapter is not None:
        with gate, session_scope(session_factory) as session:
            state = read_checkpoint(session, config.id)
        if state.circuit_open(now) and state.circuit_open_until is not None:
            reason = (
                f"paused after {state.consecutive_failures} failed runs until "
                f"{state.circuit_open_until.isoformat(timespec='minutes')}"
            )
            result = _skipped(config.id, reason, now=now, dry_run=dry_run)
        else:
            plan = plan_window(config, state, settings, now, requested=mode)
            if plan.window is None:
                result = _skipped(
                    config.id, plan.skip_reason or "not due", now=now, dry_run=dry_run
                )
            else:
                full_refresh = plan.full_refresh
                try:
                    result = adapter.collect(
                        session_factory, plan.window, dry_run=dry_run, write_lock=gate
                    )
                except Exception as exc:  # isolation: a crashing adapter fails only itself
                    LOGGER.exception("%s crashed", config.id)
                    result = SourceResult(
                        config.id,
                        now,
                        collection_end=datetime.now(UTC),
                        status=RunStatus.FAILED,
                        mode=plan.window.mode,
                        dry_run=dry_run,
                        error_count=1,
                        page_error_count=1,
                        errors=[f"{type(exc).__name__}: {str(exc)[:300]}"],
                    )
                result.warnings[:0] = list(plan.notes)

    assert result is not None
    if dry_run:
        return result
    with gate, session_scope(session_factory) as session:
        if result.status is not RunStatus.SKIPPED:
            complete = bool(result.checkpoint and result.checkpoint.get("complete"))
            record_attempt(
                session,
                result,
                failure_threshold=settings.circuit_breaker_failure_threshold,
                cooldown_minutes=settings.circuit_breaker_cooldown_minutes,
                full_refresh_completed=full_refresh and complete and result.checkpoint_advanced,
                now=now,
            )
        record_ingestion_run(session, result, job_id=job_id)
    return result


def fetch_incremental(
    config: SourceConfig,
    adapter: SourceAdapter | None,
    settings: Settings,
    session_factory: sessionmaker[Session],
    **kwargs: Any,
) -> SourceResult:
    """Collect one source from its checkpoint onwards (see :func:`fetch_source`)."""
    return fetch_source(config, adapter, settings, session_factory, mode="incremental", **kwargs)


def _select_sources(
    registry: SourceRegistry, settings: Settings, sources: Sequence[str] | None
) -> list[SourceConfig]:
    if not sources:
        return registry.enabled(settings)
    return [registry.get(source_id) for source_id in dict.fromkeys(sources)]


def fetch_all_enabled_sources(
    settings: Settings,
    session_factory: sessionmaker[Session],
    *,
    sources: Sequence[str] | None = None,
    mode: RequestedMode = "auto",
    dry_run: bool = False,
    allow_mixed: bool = False,
    trigger: str = "manual",
    now: datetime | None = None,
    registry: SourceRegistry | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> RunReport:
    """Collect every enabled source (or the ``sources`` given) and return a report.

    Raises:
        RegistryError: if the registry file is invalid.
        KeyError: if ``sources`` names an unknown source.
        MixedDataError: if live data would be written into a demo database (unless
            ``allow_mixed``).
    """
    now = now or datetime.now(UTC)
    registry = registry or load_source_registry(settings.source_registry_file)
    configs = _select_sources(registry, settings, sources)
    factory = adapter_factory or create_adapter
    report = RunReport(job_id=uuid.uuid4().hex, started_at=now, trigger=trigger, dry_run=dry_run)

    engine = session_factory.kw.get("bind")
    on_sqlite = engine is not None and engine.dialect.name == "sqlite"
    gate: WriteGate = threading.Lock() if on_sqlite else contextlib.nullcontext()

    adapters: dict[str, SourceAdapter | None] = {}
    creation_errors: dict[str, str] = {}
    will_write = False
    try:
        for config in configs:
            if not config.is_enabled(settings):
                adapters[config.id] = None
                continue
            try:
                adapters[config.id] = factory(config, settings)
            except Exception as exc:  # isolation: one broken adapter fails only itself
                LOGGER.exception("could not create the %s adapter", config.id)
                adapters[config.id] = None
                creation_errors[config.id] = (
                    f"adapter could not be created: {type(exc).__name__}: {str(exc)[:300]}"
                )
        will_write = not dry_run and (
            bool(creation_errors)
            or any(
                adapter is not None and not configuration_problems(adapter)
                for adapter in adapters.values()
            )
        )
        if will_write:
            with gate, session_scope(session_factory) as session:
                assert_origin_homogeneous(session, synthetic=False, allow_mixed=allow_mixed)
                session.add(
                    JobRun(
                        job_id=report.job_id,
                        job_name="fetch",
                        trigger=trigger,
                        status=RunStatus.RUNNING.value,
                        dry_run=False,
                        started_at=now,
                    )
                )

        def run(config: SourceConfig) -> SourceResult:
            if config.id in creation_errors:
                failed = _failed(config.id, creation_errors[config.id], now=now, dry_run=dry_run)
                if not dry_run:
                    with gate, session_scope(session_factory) as session:
                        record_ingestion_run(session, failed, job_id=report.job_id)
                return failed
            try:
                return fetch_source(
                    config,
                    adapters.get(config.id),
                    settings,
                    session_factory,
                    job_id=report.job_id,
                    mode=mode,
                    dry_run=dry_run,
                    now=now,
                    write_lock=gate,
                )
            except Exception as exc:  # last line of defence; e.g. the database went away
                LOGGER.exception("fetching %s failed outside the adapter", config.id)
                return _failed(
                    config.id, f"{type(exc).__name__}: {str(exc)[:300]}", now=now, dry_run=dry_run
                )

        workers = max(1, min(settings.max_concurrent_source_jobs, len(configs)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cews-fetch") as pool:
            report.results = list(pool.map(run, configs))
    finally:
        for adapter in adapters.values():
            if adapter is not None:
                adapter.close()

    report.finished_at = datetime.now(UTC)
    if will_write:
        with gate, session_scope(session_factory) as session:
            job = session.scalars(select(JobRun).where(JobRun.job_id == report.job_id)).one()
            job.status = report.status.value
            job.finished_at = report.finished_at
            job.summary_json = {"statuses": report.count_by_status(), **report.totals()}
            failed = [r for r in report.results if r.status is RunStatus.FAILED]
            job.error_summary = (
                "; ".join(f"{r.source_name}: {'; '.join(r.errors)}" for r in failed)[
                    :MAX_ERROR_SUMMARY
                ]
                or None
            )
    LOGGER.info("fetch %s finished: %s", report.job_id, report.count_by_status())
    return report
