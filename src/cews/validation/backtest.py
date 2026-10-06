"""Historical backtest: would CEWS's trend ranking have predicted what actually grew?

At each historical cutoff, features and trend scores are computed using only records published
on or before that date - exactly what ``cews features``/``cews score`` would have shown on that
day. The topics that *actually* grew fastest afterward are then measured from data that has since
arrived, and the two rankings are compared.

**No future leakage is possible by construction**, not just by care: the computation at a cutoff
goes through the same ``compute_features``/``run_scoring`` functions the live system uses, which
already stop at the month before ``as_of``. The only date-tricky part is on the evaluation side,
and it deliberately goes the other way: it reads activity *after* the cutoff, which is looking
back at history now, never information the scorer itself saw.

Running this writes real ``feature_values``/``scores`` rows at each historical cutoff, the same
as running ``cews features``/``cews score`` for that date would. That is a deliberate choice, not
a side effect to work around: those rows are an accurate record of what the system would have
said at the time, and are useful to inspect afterward. Pass ``cleanup=True`` to remove them again
once the backtest is done, for a caller that wants no trace left behind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from cews.constants import EntityType, ScoreType
from cews.database.models import ActivityAggregate, FeatureValue, Score, Topic
from cews.features.activity_counts import aggregate_monthly_activity, monthly_series
from cews.features.growth import calculate_growth_rate
from cews.features.pipeline import compute_features
from cews.features.time_windows import add_months, month_range
from cews.scoring.pipeline import run_scoring
from cews.settings import Settings
from cews.validation.ranking_metrics import (
    CorrelationResult,
    kendall_correlation,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    spearman_correlation,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_HORIZON_MONTHS = 3
DEFAULT_K = 5
DEFAULT_MIN_HISTORY_MONTHS = 15  # matches select_best_model's own floor for a meaningful window
MIN_FOLDS_FOR_AGGREGATE = 1


@dataclass(frozen=True)
class BacktestFold:
    """One replay: what the trend score predicted at a cutoff, and what actually happened next."""

    cutoff: date
    horizon_months: int
    k: int
    predicted: tuple[tuple[int, str, float], ...]  # (topic_id, name, trend score), ranked
    actual_growth: dict[int, float]  # topic_id -> realized log-growth over the horizon
    precision_at_k: float
    recall_at_k: float
    ndcg_at_k: float
    spearman: CorrelationResult
    kendall: CorrelationResult

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "cutoff": self.cutoff.isoformat(),
            "horizon_months": self.horizon_months,
            "k": self.k,
            "topics_evaluated": len(self.predicted),
            "predicted_top_k": [
                {"topic_id": topic_id, "name": name, "score": round(score, 2)}
                for topic_id, name, score in self.predicted[: self.k]
            ],
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "spearman": self.spearman.as_dict(),
            "kendall": self.kendall.as_dict(),
        }


@dataclass
class BacktestReport:
    """Every fold, plus the averages across them."""

    horizon_months: int
    k: int
    folds: list[BacktestFold] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def _mean(self, values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def mean_precision_at_k(self) -> float | None:
        """Average Precision@K across every fold that could be scored."""
        return self._mean([fold.precision_at_k for fold in self.folds])

    @property
    def mean_recall_at_k(self) -> float | None:
        """Average Recall@K across every fold that could be scored."""
        return self._mean([fold.recall_at_k for fold in self.folds])

    @property
    def mean_ndcg_at_k(self) -> float | None:
        """Average NDCG@K across every fold that could be scored."""
        return self._mean([fold.ndcg_at_k for fold in self.folds])

    @property
    def mean_spearman(self) -> float | None:
        """Average Spearman correlation across folds where it was defined."""
        return self._mean(
            [
                fold.spearman.coefficient
                for fold in self.folds
                if fold.spearman.coefficient is not None
            ]
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "horizon_months": self.horizon_months,
            "k": self.k,
            "folds": len(self.folds),
            "mean_precision_at_k": self._round(self.mean_precision_at_k),
            "mean_recall_at_k": self._round(self.mean_recall_at_k),
            "mean_ndcg_at_k": self._round(self.mean_ndcg_at_k),
            "mean_spearman": self._round(self.mean_spearman),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def _round(value: float | None) -> float | None:
        return None if value is None else round(value, 4)


def _default_cutoffs(
    session: Session, *, horizon_months: int, min_history_months: int
) -> list[date]:
    """Every month with enough history before it and enough room for a full horizon after it.

    Based on ``activity_aggregates`` (the underlying data), not ``feature_values`` - the backtest
    is what populates feature_values for a historical date, so that table cannot also be used to
    decide which dates are worth trying.
    """
    query = select(ActivityAggregate.period).where(ActivityAggregate.period_type == "month")
    earliest = session.scalar(query.order_by(ActivityAggregate.period))
    latest = session.scalar(query.order_by(ActivityAggregate.period.desc()))
    if earliest is None or latest is None:
        return []
    cutoffs: list[date] = []
    candidate = add_months(earliest, min_history_months)
    limit = add_months(latest, -horizon_months)
    while candidate <= limit:
        cutoffs.append(candidate)
        candidate = add_months(candidate, 1)
    return cutoffs


def _actual_growth(
    session: Session, topic_ids: list[int], *, cutoff: date, horizon_months: int
) -> dict[int, float]:
    """Realized log-growth for each topic from just before the cutoff to just after it."""
    before_months = month_range(add_months(cutoff, -1), horizon_months)
    after_months = [add_months(cutoff, step) for step in range(horizon_months)]
    growth: dict[int, float] = {}
    for topic_id in topic_ids:
        before = sum(
            sum(series)
            for series in monthly_series(session, before_months, topic_id=topic_id).values()
        )
        after = sum(
            sum(series)
            for series in monthly_series(session, after_months, topic_id=topic_id).values()
        )
        growth[topic_id] = calculate_growth_rate(after, before).log_growth
    return growth


def _cleanup_cutoff(session: Session, cutoff: date) -> None:
    """Remove the feature/score rows a fold created, for a caller that wants a clean database."""
    session.execute(delete(Score).where(Score.score_date == cutoff))
    session.execute(delete(FeatureValue).where(FeatureValue.feature_date == cutoff))
    session.flush()


def run_topic_ranking_backtest(
    session: Session,
    settings: Settings,
    *,
    cutoffs: list[date] | None = None,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    k: int = DEFAULT_K,
    min_history_months: int = DEFAULT_MIN_HISTORY_MONTHS,
    is_synthetic: bool = False,
    cleanup: bool = False,
) -> BacktestReport:
    """Replay the trend score over history and compare it with what actually grew afterward.

    Args:
        cutoffs: the historical dates to test from (default: every month with enough history
            before it and a full horizon of data after it, so the comparison is never based on
            a partial window).
        horizon_months: how far past each cutoff to look for the realized growth.
        k: the cutoff for Precision@K, Recall@K and NDCG@K.
        min_history_months: how much history a cutoff needs before it to be tested at all.
        is_synthetic: mark the feature/score rows this writes as demo data.
        cleanup: delete the feature/score rows this backtest wrote once each fold is scored.

    Raises:
        ValueError: if ``horizon_months`` or ``k`` is not positive.
    """
    if horizon_months < 1 or k < 1:
        raise ValueError("horizon_months and k must be positive")
    report = BacktestReport(horizon_months=horizon_months, k=k)
    # compute_features and _actual_growth both read activity_aggregates directly; this is the
    # same call `cews features` makes, and is idempotent, so it is safe to make again here.
    aggregate_monthly_activity(session, is_synthetic=is_synthetic)
    wanted_cutoffs = cutoffs or _default_cutoffs(
        session, horizon_months=horizon_months, min_history_months=min_history_months
    )
    if not wanted_cutoffs:
        report.warnings.append(
            "not enough history for any backtest fold; run: cews features (repeatedly, over time)"
        )
        return report

    # What was already stored before this backtest began. A fold must never overwrite it, and
    # cleanup must never delete it: these rows may be real output from `cews features`/`cews
    # score`, not something this backtest created.
    stored_before = set(session.scalars(select(FeatureValue.feature_date).distinct())) | set(
        session.scalars(select(Score.score_date).distinct())
    )
    evaluated: set[date] = set()
    for nominal_cutoff in wanted_cutoffs:
        moment = datetime(nominal_cutoff.year, nominal_cutoff.month, nominal_cutoff.day, tzinfo=UTC)
        # compute_features excludes the month containing `moment`, so the date actually scored is
        # the last complete month *before* the nominal cutoff. Working it out here, before
        # anything is written, is what lets a fold refuse to run instead of running and then
        # having to guess which rows were its own.
        expected = add_months(date(moment.year, moment.month, 1), -1)
        if expected in stored_before and cleanup:
            report.warnings.append(
                f"{expected}: scores or features already stored for this date; fold skipped so "
                "they are not overwritten and then deleted by cleanup"
            )
            continue
        if any(expected < stored <= nominal_cutoff for stored in stored_before):
            # run_scoring reads the latest stored feature date on or before the cutoff, so a
            # stored date in this gap would be scored instead of the one just computed.
            report.warnings.append(
                f"{expected}: a later date up to the cutoff already has stored features, which "
                "scoring would pick up instead; fold skipped"
            )
            continue

        feature_run = compute_features(
            session, settings, as_of=moment, store=True, is_synthetic=is_synthetic
        )
        scoring_run = run_scoring(
            session, settings, as_of=moment, store=True, is_synthetic=is_synthetic
        )
        cutoff = scoring_run.score_date
        if cutoff is None or cutoff in evaluated:
            report.warnings.append(f"{nominal_cutoff}: no features available yet; fold skipped")
            continue
        if cutoff != feature_run.feature_date or cutoff != expected:
            report.warnings.append(
                f"{nominal_cutoff}: scored {cutoff} but computed {feature_run.feature_date}; fold skipped"
            )
            if cleanup and feature_run.feature_date not in stored_before:
                _cleanup_cutoff(session, feature_run.feature_date)
            continue
        evaluated.add(cutoff)

        trend_scores = [
            result
            for result in scoring_run.of_type(ScoreType.TREND.value)
            if result.entity_type == EntityType.TOPIC.value
        ]
        if len(trend_scores) < 2:
            report.warnings.append(f"{cutoff}: fewer than 2 scored topics; fold skipped")
            if cleanup:
                _cleanup_cutoff(session, cutoff)
            continue

        names = {
            row.id: row.canonical_name
            for row in session.scalars(
                select(Topic).where(Topic.id.in_([result.entity_id for result in trend_scores]))
            )
        }
        predicted = sorted(
            (
                (result.entity_id, names.get(result.entity_id, "?"), result.value)
                for result in trend_scores
            ),
            key=lambda item: item[2],
            reverse=True,
        )
        growth = _actual_growth(
            session, [item[0] for item in predicted], cutoff=cutoff, horizon_months=horizon_months
        )

        predicted_ids = [item[0] for item in predicted]
        actually_grew_most = {
            topic_id
            for topic_id, _ in sorted(growth.items(), key=lambda item: item[1], reverse=True)[:k]
        }
        actual_by_predicted_order = [growth[topic_id] for topic_id in predicted_ids]
        predicted_scores = [item[2] for item in predicted]

        report.folds.append(
            BacktestFold(
                cutoff=cutoff,
                horizon_months=horizon_months,
                k=k,
                predicted=tuple(predicted),
                actual_growth=growth,
                precision_at_k=precision_at_k(predicted_ids, actually_grew_most, k),
                recall_at_k=recall_at_k(predicted_ids, actually_grew_most, k),
                ndcg_at_k=ndcg_at_k(
                    predicted_ids,
                    {topic_id: max(value, 0.0) for topic_id, value in growth.items()},
                    k,
                ),
                spearman=spearman_correlation(predicted_scores, actual_by_predicted_order),
                kendall=kendall_correlation(predicted_scores, actual_by_predicted_order),
            )
        )
        if cleanup:
            _cleanup_cutoff(session, cutoff)

    LOGGER.info("topic ranking backtest: %s", report.as_dict())
    return report
