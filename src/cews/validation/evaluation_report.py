"""Every evaluation in one run, with three kinds of evidence kept visibly apart.

The output is grouped, not flattened, because the three groups answer different questions and
must never be blended into one number:

* **algorithm** - what the system says about itself: data quality, forecast fit, and how robust
  the ranking is to modelling choices. Nothing here has been checked against the outside world.
* **backtest** - how the ranking would have performed on history, and where the benchmark
  topics landed. This is measured against what actually happened, not against anyone's opinion.
* **expert_validation** - what people made of it: alert ratings, discovered-topic review, the
  labelled announcement set, and a completed expert sheet if one is supplied.

A section that cannot run (a missing config file, no scores yet) is recorded as an error and the
rest still run; a real bug is not swallowed and will raise. Each successful section is stored in
``evaluation_runs`` with the configuration it ran under.
"""

from __future__ import annotations

import logging
import statistics
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.ai.llm import LLMClient
from cews.database.models import EvaluationRun, Forecast
from cews.database.repositories import count_records_by_origin
from cews.settings import Settings
from cews.validation.ai_ablation import run_ai_ablation
from cews.validation.alert_metrics import compute_alert_metrics
from cews.validation.backtest import BacktestReport, run_topic_ranking_backtest
from cews.validation.benchmark_topics import evaluate_benchmarks, load_benchmark_topics
from cews.validation.data_quality import run_data_quality_checks
from cews.validation.expert_review import summarize_expert_ratings
from cews.validation.robustness import run_robustness_checks

LOGGER = logging.getLogger(__name__)

GROUPS: dict[str, tuple[str, ...]] = {
    "algorithm": ("data_quality", "forecast", "robustness"),
    "backtest": ("backtest", "benchmarks"),
    "expert_validation": ("alerts", "ai_ablation", "expert_sheet"),
}
SECTIONS: tuple[str, ...] = tuple(name for names in GROUPS.values() for name in names)
DEFAULT_SECTIONS: tuple[str, ...] = tuple(name for name in SECTIONS if name != "expert_sheet")
EXPECTED_FAILURES = (ValueError, OSError)  # bad config, missing file; real bugs still raise


@dataclass
class EvaluationReport:
    """The result of one evaluation run."""

    evaluation_id: str
    evaluation_date: date
    is_synthetic: bool
    sections: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def grouped(self) -> dict[str, dict[str, Any]]:
        """The sections that ran, under the three headings they must stay apart under."""
        return {
            group: {name: self.sections[name] for name in names if name in self.sections}
            for group, names in GROUPS.items()
        }

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "evaluation_id": self.evaluation_id,
            "evaluation_date": self.evaluation_date.isoformat(),
            "is_synthetic": self.is_synthetic,
            **self.grouped(),
            "errors": dict(self.errors),
        }


def summarize_forecast_metrics(session: Session) -> dict[str, Any]:
    """How the most recent stored forecast for each entity fared in its own backtest.

    Only the distribution is reported. The stored error (MASE) scales by a one-step-ahead naive
    forecast, so at a multi-month horizon a value above 1 is ordinary and must not be read as
    "worse than naive"; comparing against the naive model's own replay is done when a model is
    chosen, not here.

    Raises:
        ValueError: if no forecasts are stored yet.
    """
    latest = (
        select(
            Forecast.entity_type, Forecast.entity_id, func.max(Forecast.forecast_date).label("d")
        )
        .group_by(Forecast.entity_type, Forecast.entity_id)
        .subquery()
    )
    rows = session.execute(
        select(
            Forecast.entity_type, Forecast.entity_id, Forecast.model_name, Forecast.backtest_metric
        )
        .join(
            latest,
            (Forecast.entity_type == latest.c.entity_type)
            & (Forecast.entity_id == latest.c.entity_id)
            & (Forecast.forecast_date == latest.c.d),
        )
        .distinct()
    ).all()
    if not rows:
        raise ValueError("no forecasts stored yet; run: cews forecast")
    metrics = [float(metric) for *_rest, metric in rows if metric is not None]
    return {
        "entities": len(rows),
        "models_chosen": dict(Counter(model for _t, _i, model, _m in rows)),
        "metric": "mase",
        "with_backtest_metric": len(metrics),
        "median_mase": round(statistics.median(metrics), 4) if metrics else None,
        "mean_mase": round(statistics.fmean(metrics), 4) if metrics else None,
        "best_mase": round(min(metrics), 4) if metrics else None,
        "worst_mase": round(max(metrics), 4) if metrics else None,
    }


def _backtest_section(report: BacktestReport) -> dict[str, Any]:
    payload = report.as_dict()
    payload["fold_detail"] = [fold.as_dict() for fold in report.folds]
    return payload


def run_evaluation(
    session: Session,
    settings: Settings,
    *,
    sections: tuple[str, ...] | None = None,
    as_of: datetime | None = None,
    horizon_months: int = 3,
    k: int = 5,
    cleanup_backtest: bool = True,
    llm_client: LLMClient | None = None,
    expert_file: Path | None = None,
    store: bool = True,
) -> EvaluationReport:
    """Run the requested evaluation sections and (optionally) store each result.

    Args:
        sections: which to run (default: everything except ``expert_sheet``, which needs a file).
        horizon_months, k: the backtest's look-ahead and Precision@K cutoff.
        cleanup_backtest: remove the historical feature/score rows the backtest writes, so an
            evaluation leaves the database as it found it.
        expert_file: a completed expert review sheet; runs the ``expert_sheet`` section.
        store: write each successful section to ``evaluation_runs``.

    Raises:
        ValueError: for an unknown section name, or a non-positive ``horizon_months``/``k``.
    """
    wanted = tuple(sections) if sections else DEFAULT_SECTIONS
    unknown = [name for name in wanted if name not in SECTIONS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    if expert_file is not None and "expert_sheet" not in wanted:
        wanted = (*wanted, "expert_sheet")
    if horizon_months < 1 or k < 1:
        raise ValueError("horizon_months and k must be positive")

    moment = as_of or datetime.now(UTC)
    synthetic = count_records_by_origin(session)["synthetic"] > 0
    report = EvaluationReport(
        evaluation_id=f"eval-{moment:%Y%m%d%H%M%S}-{uuid.uuid4().hex[:6]}",
        evaluation_date=datetime.now(UTC).date(),
        is_synthetic=synthetic,
    )
    backtest_report: BacktestReport | None = None

    for name in SECTIONS:
        if name not in wanted:
            continue
        try:
            if name == "data_quality":
                payload = run_data_quality_checks(session, as_of=moment).as_dict()
            elif name == "forecast":
                payload = summarize_forecast_metrics(session)
            elif name == "robustness":
                payload = run_robustness_checks(session, settings, as_of=moment).as_dict()
            elif name == "backtest":
                backtest_report = run_topic_ranking_backtest(
                    session,
                    settings,
                    horizon_months=horizon_months,
                    k=k,
                    is_synthetic=synthetic,
                    cleanup=cleanup_backtest,
                )
                payload = _backtest_section(backtest_report)
            elif name == "benchmarks":
                if backtest_report is None:
                    backtest_report = run_topic_ranking_backtest(
                        session,
                        settings,
                        horizon_months=horizon_months,
                        k=k,
                        is_synthetic=synthetic,
                        cleanup=cleanup_backtest,
                    )
                benchmarks = load_benchmark_topics(settings.benchmark_topics_file)
                payload = evaluate_benchmarks(session, backtest_report, benchmarks).as_dict()
            elif name == "alerts":
                payload = compute_alert_metrics(session).as_dict()
            elif name == "ai_ablation":
                payload = run_ai_ablation(
                    session, settings, settings.announcement_labels_file, llm_client=llm_client
                ).as_dict()
            else:  # expert_sheet
                if expert_file is None:
                    raise ValueError("expert_sheet needs a completed review file")
                payload = summarize_expert_ratings(expert_file).as_dict()
        except EXPECTED_FAILURES as exc:
            LOGGER.warning("evaluation section %s could not run: %s", name, exc)
            report.errors[name] = str(exc)
            continue

        report.sections[name] = payload
        if store:
            session.add(
                EvaluationRun(
                    evaluation_id=f"{report.evaluation_id}-{name}",
                    evaluation_type=name,
                    evaluation_date=report.evaluation_date,
                    period_start=(
                        min(f.cutoff for f in backtest_report.folds)
                        if name == "backtest" and backtest_report and backtest_report.folds
                        else None
                    ),
                    period_end=(
                        max(f.cutoff for f in backtest_report.folds)
                        if name == "backtest" and backtest_report and backtest_report.folds
                        else None
                    ),
                    configuration_json={
                        "horizon_months": horizon_months,
                        "k": k,
                        "as_of": moment.isoformat(),
                        "cleanup_backtest": cleanup_backtest,
                        "llm_provider": settings.llm_provider.value,
                    },
                    metrics_json=payload,
                    is_synthetic=synthetic,
                )
            )
    if store:
        session.flush()
    return report


def latest_stored_evaluations(session: Session) -> dict[str, dict[str, Any]]:
    """The most recent stored result of each kind of evaluation, keyed by section name."""
    latest: dict[str, dict[str, Any]] = {}
    for row in session.scalars(
        select(EvaluationRun).order_by(EvaluationRun.evaluation_date, EvaluationRun.id)
    ):
        latest[row.evaluation_type] = {
            "evaluation_id": row.evaluation_id,
            "evaluation_date": row.evaluation_date.isoformat(),
            "is_synthetic": row.is_synthetic,
            "metrics": row.metrics_json,
        }
    return latest
