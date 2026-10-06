"""How much the trend ranking moves when a modelling choice is nudged.

A ranking that reorders itself completely for a small change in a weight, a window, or which
source happens to be available is not a ranking leadership can trust between runs. Each check
here perturbs exactly one thing, recomputes the same trend ranking, and reports how stable the
result was - via the same rank correlation and top-K overlap metrics ``ranking_metrics`` uses
elsewhere, so a robustness number and a backtest number mean the same thing when compared.

Every perturbation here is computed **in memory**. Weight, normalization-method and
source-removal sensitivity reuse the exact functions the real scoring pipeline calls
(``normalize_feature_cohort``, ``combine``, ``calculate_velocity`` and friends) directly on the
raw features from one ``compute_features(store=False)`` call, so nothing is written to the
database and the arithmetic matches production exactly. Window sensitivity is the one exception:
it genuinely needs a second, differently-configured feature pass, done through a temporary copy
of the scoring config file rather than the live one.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType, ScoreType
from cews.database.models import Score
from cews.database.repositories import count_records_by_origin
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.consistency import calculate_consistency
from cews.features.growth import growth_from_series
from cews.features.momentum import calculate_momentum
from cews.features.pipeline import (
    DEFAULT_COMPOSITE_WEIGHTS,
    EntityFeatures,
    _composite,  # the exact source-weighting logic production uses
    compute_features,
)
from cews.features.velocity import calculate_velocity
from cews.scoring.base import combine
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.normalization import normalize_feature_cohort
from cews.scoring.trend_score import COMPONENT_FEATURES
from cews.settings import Settings
from cews.validation.ranking_metrics import RankStability, rank_stability

DEFAULT_TOP_K = 10
NORMALIZATION_METHODS = ("percentile", "minmax", "robust_zscore", "winsorized_minmax")


@dataclass(frozen=True)
class SensitivityResult:
    """One perturbation, and how much the top-K trend ranking moved because of it."""

    dimension: str
    description: str
    baseline_top: tuple[str, ...]
    perturbed_top: tuple[str, ...]
    stability: RankStability

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "dimension": self.dimension,
            "description": self.description,
            "baseline_top": list(self.baseline_top),
            "perturbed_top": list(self.perturbed_top),
            "stability": self.stability.as_dict(),
        }


@dataclass
class RobustnessReport:
    """Every sensitivity check run together."""

    results: list[SensitivityResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "checks": [result.as_dict() for result in self.results],
            "warnings": list(self.warnings),
        }


def _ranking(
    topics: Sequence[EntityFeatures], weights: Mapping[str, float], *, method: str = "percentile"
) -> list[tuple[int, str, float]]:
    """Score every topic's trend, cohort-normalizing each raw feature across the same topics
    ``normalize_feature_cohort`` and ``combine`` would - the exact functions the real trend
    score uses, called directly rather than through the database."""
    raw_by_feature: dict[str, dict[str, float]] = {}
    raw_per_entity: dict[int, dict[str, float]] = {}
    for entity in topics:
        values = entity.raw_values()
        raw_per_entity[entity.entity_id] = values
        for feature_name, _source in COMPONENT_FEATURES.values():
            if feature_name in values:
                raw_by_feature.setdefault(feature_name, {})[str(entity.entity_id)] = values[
                    feature_name
                ]

    normalized_by_feature = {
        feature_name: normalize_feature_cohort(raw, method=method)
        for feature_name, raw in raw_by_feature.items()
    }

    results: list[tuple[int, str, float]] = []
    for entity in topics:
        key = str(entity.entity_id)
        parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {}
        for component, (feature_name, source) in COMPONENT_FEATURES.items():
            if component not in weights:
                continue
            if source is not None and source not in entity.available_sources:
                parts[component] = (raw_per_entity[entity.entity_id].get(feature_name), None, {})
                continue
            cohort = normalized_by_feature.get(feature_name, {})
            if key in cohort:
                parts[component] = (cohort[key].raw_value, cohort[key].normalized_value, {})
            else:
                parts[component] = (None, None, {})
        try:
            score, _components, _missing = combine(parts, weights)
        except ValueError:
            continue
        results.append((entity.entity_id, entity.name, score))
    return sorted(results, key=lambda item: item[2], reverse=True)


def _compare(
    baseline: list[tuple[int, str, float]],
    perturbed: list[tuple[int, str, float]],
    *,
    dimension: str,
    description: str,
    top_k: int,
) -> SensitivityResult | None:
    """None when the two rankings do not share the same topics (nothing meaningful to compare)."""
    baseline_ids = [item[0] for item in baseline]
    perturbed_ids = [item[0] for item in perturbed]
    if set(baseline_ids) != set(perturbed_ids) or len(baseline_ids) < 2:
        return None
    order = {topic_id: index for index, topic_id in enumerate(perturbed_ids)}
    perturbed_in_baseline_order = sorted(baseline_ids, key=lambda topic_id: order[topic_id])
    stability = rank_stability(baseline_ids, perturbed_in_baseline_order, k=top_k)
    names = {item[0]: item[1] for item in baseline}
    return SensitivityResult(
        dimension=dimension,
        description=description,
        baseline_top=tuple(names[topic_id] for topic_id in baseline_ids[:top_k]),
        perturbed_top=tuple(names[topic_id] for topic_id in perturbed_ids[:top_k]),
        stability=stability,
    )


def weight_sensitivity(
    topics: Sequence[EntityFeatures],
    config: ScoringConfig,
    *,
    perturbation: float = 0.5,
    top_k: int = DEFAULT_TOP_K,
) -> list[SensitivityResult]:
    """Halve one weight at a time (folding the rest of its share into the others) and measure
    how much the top-K trend ranking moves.

    Raises:
        ScoringConfigError: if no "trend" weight set is configured.
    """
    baseline_weights = config.weights("trend_score")
    baseline = _ranking(topics, baseline_weights)
    results = []
    for component in baseline_weights:
        perturbed_weights = dict(baseline_weights)
        removed = perturbed_weights[component] * perturbation
        perturbed_weights[component] -= removed
        share = removed / (len(perturbed_weights) - 1) if len(perturbed_weights) > 1 else 0.0
        for other in perturbed_weights:
            if other != component:
                perturbed_weights[other] += share
        perturbed = _ranking(topics, perturbed_weights)
        result = _compare(
            baseline,
            perturbed,
            dimension=f"weight:{component}",
            description=f"{component} weight reduced by {perturbation:.0%}, redistributed to the rest",
            top_k=top_k,
        )
        if result is not None:
            results.append(result)
    return results


def normalization_method_sensitivity(
    topics: Sequence[EntityFeatures], config: ScoringConfig, *, top_k: int = DEFAULT_TOP_K
) -> list[SensitivityResult]:
    """Compare the configured normalization method against every alternative.

    Raises:
        ScoringConfigError: if no "trend" weight set is configured.
    """
    weights = config.weights("trend_score")
    baseline_method = "percentile"
    baseline = _ranking(topics, weights, method=baseline_method)
    results = []
    for method in NORMALIZATION_METHODS:
        if method == baseline_method:
            continue
        perturbed = _ranking(topics, weights, method=method)
        result = _compare(
            baseline,
            perturbed,
            dimension=f"normalization:{method}",
            description=f"normalized by {method} instead of {baseline_method}",
            top_k=top_k,
        )
        if result is not None:
            results.append(result)
    return results


def source_removal_sensitivity(
    topics: Sequence[EntityFeatures], config: ScoringConfig, *, top_k: int = DEFAULT_TOP_K
) -> list[SensitivityResult]:
    """Recompute the trend ranking as if one source type had never been collected at all.

    Raises:
        ScoringConfigError: if no "trend" weight set is configured.
    """
    weights = config.weights("trend_score")
    baseline = _ranking(topics, weights)
    present_sources = sorted({source for entity in topics for source in entity.available_sources})
    results = []
    for excluded in present_sources:
        rebuilt = [_without_source(entity, excluded) for entity in topics]
        perturbed = _ranking(rebuilt, weights)
        result = _compare(
            baseline,
            perturbed,
            dimension=f"source_removed:{excluded}",
            description=f"as if no {excluded} records had ever been collected",
            top_k=top_k,
        )
        if result is not None:
            results.append(result)
    return results


def _without_source(entity: EntityFeatures, excluded: str) -> EntityFeatures:
    """A copy of ``entity`` with one source type's contribution removed everywhere it feeds in:
    the composite series (and so velocity, momentum, consistency) and that source's own growth
    component. Uses ``_composite``, the same weighting and reweighting logic ``compute_features``
    itself uses, so a quiet source's weight is shared the same way it would be in production.
    """
    by_source = {
        source: series for source, series in entity.by_source.items() if source != excluded
    }
    months = len(entity.months)
    composite, composite_sources = _composite(by_source, months, DEFAULT_COMPOSITE_WEIGHTS)
    velocity = calculate_velocity(composite) if len(composite) >= 6 else None
    momentum = calculate_momentum(composite)
    consistency = calculate_consistency(composite)
    growth = {source: value for source, value in entity.growth.items() if source != excluded}
    return EntityFeatures(
        entity_type=entity.entity_type,
        entity_id=entity.entity_id,
        name=entity.name,
        months=entity.months,
        by_source=by_source,
        composite=composite,
        composite_sources=composite_sources,
        growth=growth,
        momentum=momentum,
        velocity=velocity,
        consistency=consistency,
        surge=growth_from_series(composite),
        agreement=entity.agreement,
        competition=entity.competition,
        total_records=sum(composite),
        sample_confidence=entity.sample_confidence,
        completeness=entity.completeness,
    )


def window_sensitivity(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime,
    alternate_window_months: int,
    top_k: int = DEFAULT_TOP_K,
) -> list[SensitivityResult]:
    """Compare the configured velocity window against a different one.

    Neither run writes anything to the database: both use ``compute_features(store=False)``.
    Assumes ``activity_aggregates`` is already populated (``run_robustness_checks`` ensures this
    before calling here; a standalone caller should call
    ``cews.features.activity_counts.aggregate_monthly_activity`` first).

    Raises:
        ValueError: if the scoring configuration cannot be read.
    """
    baseline_run = compute_features(session, settings, as_of=as_of, store=False)
    baseline_topics = [e for e in baseline_run.entities if e.entity_type == EntityType.TOPIC.value]
    config = load_scoring_config(settings)
    baseline = _ranking(baseline_topics, config.weights("trend_score"))

    document: dict[str, Any] = {}
    if settings.scoring_config_file.is_file():
        document = yaml.safe_load(settings.scoring_config_file.read_text(encoding="utf-8")) or {}
    document.setdefault("features", {})["velocity_window_months"] = alternate_window_months

    with tempfile.TemporaryDirectory() as tmp:
        temp_path = Path(tmp) / "scoring_weights.yaml"
        temp_path.write_text(yaml.safe_dump(document), encoding="utf-8")
        perturbed_settings = settings.model_copy(update={"scoring_config_file": temp_path})
        perturbed_run = compute_features(session, perturbed_settings, as_of=as_of, store=False)

    perturbed_topics = [
        e for e in perturbed_run.entities if e.entity_type == EntityType.TOPIC.value
    ]
    perturbed = _ranking(perturbed_topics, config.weights("trend_score"))
    result = _compare(
        baseline,
        perturbed,
        dimension="window_months",
        description=(
            f"velocity window changed from {baseline_run.settings.velocity_window_months} to "
            f"{alternate_window_months} months"
        ),
        top_k=top_k,
    )
    return [result] if result is not None else []


def low_sample_confidence_check(
    session: Session, score_date: date | None = None
) -> SensitivityResult | None:
    """Whether confidence is actually lower where the evidence behind a score is thinner.

    Reads stored trend scores directly; unlike the other checks, this does not recompute
    anything, since it is checking a relationship in what was already scored and stored.

    Returns None if fewer than two scored topics are stored at that date to compare.
    """
    from cews.validation.ranking_metrics import spearman_correlation

    query = select(Score).where(Score.score_type == ScoreType.TREND.value)
    if score_date is not None:
        query = query.where(Score.score_date == score_date)
    else:
        latest = select(Score.score_date).order_by(Score.score_date.desc()).limit(1)
        query = query.where(Score.score_date == latest.scalar_subquery())
    rows = list(session.scalars(query))
    if len(rows) < 2:
        return None

    sample_sizes = [float((row.component_json or {}).get("sample_size", 0)) for row in rows]
    confidences = [row.confidence_score for row in rows]
    correlation = spearman_correlation(sample_sizes, confidences)
    ok = correlation.coefficient is not None and correlation.coefficient > 0
    description = (
        f"correlation between sample size and confidence across {len(rows)} scored topic(s): "
        f"{correlation.coefficient}"
        if correlation.coefficient is not None
        else "sample sizes were too uniform to measure a correlation"
    )
    stability = RankStability(spearman=correlation, top_k_overlap=1.0 if ok else 0.0, k=len(rows))
    return SensitivityResult(
        dimension="low_sample_confidence",
        description=description,
        baseline_top=(),
        perturbed_top=(),
        stability=stability,
    )


def run_robustness_checks(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime | None = None,
    top_k: int = DEFAULT_TOP_K,
    alternate_window_months: int = 6,
) -> RobustnessReport:
    """Run every sensitivity dimension and return them together.

    Raises:
        ValueError: if the scoring configuration cannot be read.
    """
    moment = as_of or datetime.now(UTC)
    report = RobustnessReport()
    # compute_features reads activity_aggregates directly and never populates it itself (the
    # same dependency run_topic_ranking_backtest has); this is the same call `cews features`
    # makes, and is idempotent, so it is safe to make again here.
    is_synthetic = count_records_by_origin(session)["synthetic"] > 0
    aggregate_monthly_activity(session, is_synthetic=is_synthetic)
    feature_run = compute_features(session, settings, as_of=moment, store=False)
    topics = [
        entity for entity in feature_run.entities if entity.entity_type == EntityType.TOPIC.value
    ]
    if len(topics) < 2:
        report.warnings.append("fewer than 2 scored topics; robustness checks skipped")
        return report

    config = load_scoring_config(settings)
    report.results.extend(weight_sensitivity(topics, config, top_k=top_k))
    report.results.extend(normalization_method_sensitivity(topics, config, top_k=top_k))
    report.results.extend(source_removal_sensitivity(topics, config, top_k=top_k))
    report.results.extend(
        window_sensitivity(
            session,
            settings,
            as_of=moment,
            alternate_window_months=alternate_window_months,
            top_k=top_k,
        )
    )
    low_sample_result = low_sample_confidence_check(session)
    if low_sample_result is not None:
        report.results.append(low_sample_result)
    else:
        report.warnings.append("no stored trend scores yet; low-sample confidence check skipped")
    return report
