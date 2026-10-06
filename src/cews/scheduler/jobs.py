"""One refresh cycle: collect new data, then bring every result up to date.

A cycle is two phases:

1. **Fetch** every enabled source (each source isolated, so one failing never stops the rest;
   retries, rate limits, checkpoints and circuit breakers all live in the ingestion layer).
2. **Analyse**, in dependency order: normalize, competitors, features, scores, forecasts,
   insights, and (if enabled) the Power BI export.

The analysis steps depend on each other, so the first one to fail stops those after it: scoring
on features that were never recomputed would publish stale numbers as fresh. Earlier steps keep
what they finished. Each step commits on its own, so a failure never rolls back work that
already succeeded.

Only one cycle runs at a time, however it was started (scheduler, ``cews refresh``, or anything
later): a database lock enforces it, and every cycle leaves an audit row.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import session_scope
from cews.database.repositories import MixedDataError, count_records_by_origin
from cews.discovery.competitor_discovery import discover_competitors
from cews.exports.powerbi_export import export_powerbi
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import compute_features
from cews.forecasting.pipeline import run_forecasting
from cews.ingestion.orchestrator import fetch_all_enabled_sources
from cews.ingestion.results import RunReport
from cews.insights.rule_engine import generate_insights
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import load_taxonomy, sync_taxonomy
from cews.scheduler.job_state import (
    REFRESH_JOB,
    REFRESH_LOCK,
    finish_job_run,
    job_lock,
    lock_directory,
    mark_interrupted_runs,
    start_job_run,
)
from cews.scoring.pipeline import run_scoring
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

StepFunction = Callable[[sessionmaker[Session], Settings, datetime, bool], dict[str, Any]]
FetchFunction = Callable[..., RunReport]
ANALYSIS_STEP_NAMES = (
    "normalize",
    "competitors",
    "features",
    "score",
    "forecast",
    "insights",
    "export",
)


# ----------------------------------------------------------------------------------------
# The analysis steps
# ----------------------------------------------------------------------------------------
def _normalize(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    taxonomy = load_taxonomy(settings.topic_taxonomy_file)
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        return normalize_records(session, settings, taxonomy, commit_each_batch=True).as_dict()


def _competitors(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        result = discover_competitors(session, settings, as_of=now)
        return {"monitored": len(result.monitored), "warnings": result.warnings[:5]}


def _features(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        aggregate_monthly_activity(session, as_of=now, is_synthetic=synthetic)
        return compute_features(session, settings, as_of=now, is_synthetic=synthetic).as_dict()


def _score(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        run = run_scoring(session, settings, as_of=now, store=True, is_synthetic=synthetic)
        return {
            "score_date": run.score_date,
            "scores": len(run.results),
            "warnings": run.warnings[:5],
        }


def _forecast(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        return run_forecasting(
            session, settings, as_of=now, store=True, is_synthetic=synthetic
        ).as_dict()


def _insights(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        return generate_insights(session, settings, store=True, is_synthetic=synthetic).as_dict()


def _export(
    factory: sessionmaker[Session], settings: Settings, now: datetime, synthetic: bool
) -> dict[str, Any]:
    with session_scope(factory) as session:
        return export_powerbi(session, settings.export_directory / "powerbi").as_dict()


ANALYSIS_STEPS: dict[str, StepFunction] = {
    "normalize": _normalize,
    "competitors": _competitors,
    "features": _features,
    "score": _score,
    "forecast": _forecast,
    "insights": _insights,
    "export": _export,
}


def default_analysis_steps(settings: Settings) -> list[tuple[str, StepFunction]]:
    """The analysis steps in order; the Power BI export only when it is enabled."""
    return [
        (name, ANALYSIS_STEPS[name])
        for name in ANALYSIS_STEP_NAMES
        if name != "export" or settings.generate_powerbi_exports
    ]


# ----------------------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepOutcome:
    """How one step of a cycle went."""

    name: str
    status: str  # "succeeded", "failed" or "skipped"
    seconds: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 2),
            "detail": self.detail,
            "error": self.error,
        }


@dataclass
class RefreshReport:
    """The outcome of one refresh cycle."""

    job_id: str
    trigger: str
    dry_run: bool
    started_at: datetime
    status: RunStatus = RunStatus.RUNNING
    finished_at: datetime | None = None
    fetch: dict[str, Any] | None = None
    steps: list[StepOutcome] = field(default_factory=list)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form (also what is stored in ``job_runs.summary_json``)."""
        return {
            "job_id": self.job_id,
            "trigger": self.trigger,
            "dry_run": self.dry_run,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "fetch": self.fetch,
            "steps": [step.as_dict() for step in self.steps],
            "error": self.error,
        }


def _fetch_summary(report: RunReport) -> dict[str, Any]:
    return {
        "job_id": report.job_id,
        "status": report.status.value,
        "sources": {result.source_name: result.status.value for result in report.results},
        "totals": report.totals(),
    }


def _jsonable(value: Any) -> Any:
    """Make step details safe to store as JSON (dates become text)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def run_analysis_steps(
    factory: sessionmaker[Session],
    settings: Settings,
    steps: Sequence[tuple[str, StepFunction]],
    *,
    now: datetime,
    synthetic: bool,
) -> list[StepOutcome]:
    """Run the steps in order; the first failure stops the rest, which are reported as skipped."""
    outcomes: list[StepOutcome] = []
    failed = False
    for name, function in steps:
        if failed:
            outcomes.append(StepOutcome(name, "skipped", error="an earlier step failed"))
            continue
        started = time.monotonic()
        try:
            detail = _jsonable(function(factory, settings, now, synthetic))
        except Exception as exc:  # a step failing must be reported, not crash the scheduler
            LOGGER.exception("refresh step %s failed", name)
            failed = True
            outcomes.append(
                StepOutcome(
                    name,
                    "failed",
                    seconds=time.monotonic() - started,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            )
            continue
        outcomes.append(
            StepOutcome(name, "succeeded", seconds=time.monotonic() - started, detail=detail)
        )
    return outcomes


# ----------------------------------------------------------------------------------------
# The cycle
# ----------------------------------------------------------------------------------------
def run_refresh(
    factory: sessionmaker[Session],
    settings: Settings,
    *,
    trigger: str = "manual",
    dry_run: bool = False,
    sources: Sequence[str] | None = None,
    skip_fetch: bool = False,
    skip_analysis: bool = False,
    allow_mixed: bool = False,
    fetch: FetchFunction = fetch_all_enabled_sources,
    analysis_steps: Sequence[tuple[str, StepFunction]] | None = None,
    lock_dir: Path | None = None,
    now: datetime | None = None,
) -> RefreshReport:
    """Run one refresh cycle and return what happened.

    Args:
        trigger: ``schedule`` or ``manual``, recorded in the audit row.
        dry_run: fetch without writing anything and skip the analysis (there is nothing new to
            analyse, and analysis writes).
        sources: limit the fetch to these source ids.
        skip_fetch, skip_analysis: run only one phase.
        fetch, analysis_steps: replaceable for testing; the defaults are the real ones.
        lock_dir: where the lock file lives (default: beside the database).

    Raises:
        JobAlreadyRunningError: if another cycle holds the lock. Nothing is recorded for the
            refused attempt, since it did no work.
    """
    moment = now or datetime.now(UTC)
    with job_lock(lock_dir or lock_directory(settings), REFRESH_LOCK):
        closed = mark_interrupted_runs(factory, (REFRESH_JOB, "fetch"))
        if closed:
            LOGGER.warning("marked %s earlier refresh run(s) as interrupted", closed)
        job_id = start_job_run(factory, REFRESH_JOB, trigger=trigger, dry_run=dry_run, now=moment)
        report = RefreshReport(job_id=job_id, trigger=trigger, dry_run=dry_run, started_at=moment)
        try:
            _execute(
                report, factory, settings, sources=sources, skip_fetch=skip_fetch,
                skip_analysis=skip_analysis, allow_mixed=allow_mixed, fetch=fetch,
                analysis_steps=analysis_steps, moment=moment,
            )  # fmt: skip
        except Exception as exc:  # never leave the audit row saying "running"
            LOGGER.exception("refresh cycle failed unexpectedly")
            report.status = RunStatus.FAILED
            report.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        report.finished_at = datetime.now(UTC)
        finish_job_run(
            factory,
            job_id,
            report.status,
            summary=_jsonable(report.as_dict()),
            error=report.error,
            now=report.finished_at,
        )
    return report


def _execute(
    report: RefreshReport,
    factory: sessionmaker[Session],
    settings: Settings,
    *,
    sources: Sequence[str] | None,
    skip_fetch: bool,
    skip_analysis: bool,
    allow_mixed: bool,
    fetch: FetchFunction,
    analysis_steps: Sequence[tuple[str, StepFunction]] | None,
    moment: datetime,
) -> None:
    fetch_status: RunStatus | None = None
    if not skip_fetch:
        try:
            fetched = fetch(
                settings,
                factory,
                sources=list(sources) if sources else None,
                dry_run=report.dry_run,
                allow_mixed=allow_mixed,
                trigger=report.trigger,
            )
        except MixedDataError as exc:
            report.status, report.error = RunStatus.FAILED, str(exc)
            return
        except KeyError as exc:  # an unknown --source id
            report.status, report.error = RunStatus.FAILED, str(exc.args[0])
            return
        report.fetch = _fetch_summary(fetched)
        fetch_status = fetched.status
        if fetch_status is RunStatus.FAILED:
            report.status = RunStatus.FAILED
            report.error = "every source that ran failed; the analysis was not re-run"
            return

    if skip_analysis or report.dry_run:
        report.status = (
            RunStatus.PARTIAL if fetch_status is RunStatus.PARTIAL else RunStatus.SUCCEEDED
        )
        return

    steps = list(analysis_steps) if analysis_steps is not None else default_analysis_steps(settings)
    with session_scope(factory) as session:
        synthetic = count_records_by_origin(session)["synthetic"] > 0
    report.steps = run_analysis_steps(factory, settings, steps, now=moment, synthetic=synthetic)

    failed = [step for step in report.steps if step.status == "failed"]
    if failed:
        # Partial if anything at all was achieved (new data, or a step that finished); otherwise
        # the cycle achieved nothing and is a plain failure.
        achieved = fetch_status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL) or any(
            step.status == "succeeded" for step in report.steps
        )
        report.status = RunStatus.PARTIAL if achieved else RunStatus.FAILED
        report.error = f"step {failed[0].name!r} failed: {failed[0].error}"
    elif fetch_status is RunStatus.PARTIAL:
        report.status = RunStatus.PARTIAL
    else:
        report.status = RunStatus.SUCCEEDED
