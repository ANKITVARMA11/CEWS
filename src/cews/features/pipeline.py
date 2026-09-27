"""Computing and storing the features that scoring uses.

For every topic and every monitored competitor this builds a 12-month activity series per source
type, derives the features (growth, momentum, velocity, consistency, surge, source agreement,
competition density, sample confidence) and stores each one in ``feature_values`` with both its
raw value and its normalized 0-100 position within a cohort.

A cohort is the set a value is judged against: the same feature, the same entity kind, the same
date. Counts from different sources are never compared directly, which is what makes a topic's
patent growth comparable with its publication growth in the next phase.

Nothing here decides whether a topic is a trend. These are the inputs; the scores are Phase 7.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType, SourceType, TopicStatus
from cews.database.models import FeatureValue, Organization, Topic
from cews.features.activity_counts import monthly_series
from cews.features.competition import CompetitionDensity, calculate_competition_density
from cews.features.consistency import ConsistencyResult, calculate_consistency
from cews.features.growth import GrowthResult, calculate_growth_rate, growth_from_series
from cews.features.momentum import MomentumResult, calculate_momentum
from cews.features.sample_confidence import (
    calculate_data_completeness,
    calculate_sample_confidence,
)
from cews.features.source_agreement import SourceAgreement, calculate_source_agreement
from cews.features.time_windows import month_range, split_series
from cews.features.velocity import VelocityResult, calculate_velocity
from cews.scoring.config import ScoringConfigError, validate_weights
from cews.scoring.normalization import normalize_feature_cohort, renormalize_weights
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

SOURCE_TYPES: tuple[str, ...] = tuple(source.value for source in SourceType)
DEFAULT_COMPOSITE_WEIGHTS: dict[str, float] = {
    "publication": 0.40,
    "clinical_trial": 0.40,
    "announcement": 0.20,
}
QUARTER_MONTHS = 3


@dataclass(frozen=True)
class FeatureSettings:
    """Tunables for the feature pass, from ``config/scoring_weights.yaml`` and ``.env``."""

    alpha: float = 1.0
    momentum_recent_months: int = 3
    momentum_previous_months: int = 3
    velocity_window_months: int = 12
    velocity_min_observations: int = 6
    sample_saturation_k: float = 25.0
    freshness_half_life_days: float = 30.0
    normalization_method: str = "percentile"
    winsor_lower: float = 0.05
    winsor_upper: float = 0.95
    min_sample_size: int = 10
    # How much each source type counts toward the combined activity signal that velocity and
    # momentum are measured on. Sources with no data have their weight shared out, never zeroed.
    composite_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPOSITE_WEIGHTS)
    )

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, for recording what a run used."""
        return {
            "alpha": self.alpha,
            "momentum_windows": f"{self.momentum_recent_months}m vs {self.momentum_previous_months}m",
            "velocity_window_months": self.velocity_window_months,
            "velocity_min_observations": self.velocity_min_observations,
            "sample_saturation_k": self.sample_saturation_k,
            "normalization_method": self.normalization_method,
            "min_sample_size": self.min_sample_size,
            "composite_weights": dict(self.composite_weights),
        }


def load_feature_settings(settings: Settings) -> FeatureSettings:
    """Read the feature tunables, falling back to the defaults for anything not configured.

    Raises:
        ValueError: if the scoring configuration file cannot be read.
    """
    values: dict[str, Any] = {}
    composite = dict(DEFAULT_COMPOSITE_WEIGHTS)
    path = settings.scoring_config_file
    if path.is_file():
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot read {path}: {exc}") from exc
        values = dict(document.get("features") or {})
        configured = (document.get("weight_sets") or {}).get("composite_activity")
        if configured:
            try:
                composite = validate_weights(configured, name="weight_sets.composite_activity")
            except ScoringConfigError as exc:
                raise ValueError(str(exc)) from exc
        unknown = sorted(set(composite) - set(SOURCE_TYPES))
        if unknown:
            raise ValueError(
                f"weight_sets.composite_activity: unknown source type(s) {unknown}; "
                f"expected any of {list(SOURCE_TYPES)}"
            )
    return FeatureSettings(
        alpha=float(values.get("alpha", 1.0)),
        momentum_recent_months=int(values.get("momentum_recent_months", 3)),
        momentum_previous_months=int(values.get("momentum_previous_months", 3)),
        velocity_window_months=int(values.get("velocity_window_months", 12)),
        velocity_min_observations=int(values.get("velocity_min_observations", 6)),
        sample_saturation_k=float(values.get("sample_saturation_k", 25.0)),
        freshness_half_life_days=float(values.get("freshness_half_life_days", 30.0)),
        normalization_method=settings.normalization_method.value,
        winsor_lower=settings.normalization_winsor_lower,
        winsor_upper=settings.normalization_winsor_upper,
        min_sample_size=settings.min_topic_sample_size,
        composite_weights=composite,
    )


@dataclass
class EntityFeatures:
    """Everything computed for one topic or competitor."""

    entity_type: str
    entity_id: int
    name: str
    months: list[date]
    by_source: dict[str, list[float]]
    composite: list[float]
    composite_sources: tuple[str, ...]
    growth: dict[str, GrowthResult]
    momentum: MomentumResult
    velocity: VelocityResult | None
    consistency: ConsistencyResult | None
    surge: GrowthResult
    agreement: SourceAgreement
    competition: CompetitionDensity | None
    total_records: float
    sample_confidence: float
    completeness: float

    @property
    def available_sources(self) -> tuple[str, ...]:
        """Source types that produced any activity for this entity."""
        return tuple(source for source, values in self.by_source.items() if sum(values) > 0)

    def raw_values(self) -> dict[str, float]:
        """The feature values that get normalized against other entities."""
        values: dict[str, float] = {
            "momentum": self.momentum.momentum,
            "recent_surge": self.surge.log_growth,
            "activity_total": self.total_records,
            "source_agreement": self.agreement.agreement,
            "sample_confidence": self.sample_confidence,
            "data_completeness": self.completeness,
        }
        if self.velocity is not None:
            values["velocity"] = self.velocity.slope
        if self.consistency is not None:
            values["consistency"] = self.consistency.consistency
        if self.competition is not None:
            values["competition_density"] = self.competition.density
        for source, series in self.by_source.items():
            values[f"activity_{source}"] = sum(series)
        # Facts the trend rules need, kept as plain numbers so they survive storage.
        values["supporting_sources"] = float(len(self.agreement.supporting))
        if self.consistency is not None:
            values["single_spike"] = 1.0 if self.consistency.single_spike else 0.0
        for source, growth in self.growth.items():
            values[f"growth_{source}"] = growth.log_growth
        return values

    def details(self) -> dict[str, dict[str, Any]]:
        """The explanation behind each value, for the evidence view."""
        details: dict[str, dict[str, Any]] = {
            "momentum": self.momentum.as_dict(),
            "recent_surge": self.surge.as_dict(),
            "source_agreement": self.agreement.as_dict(),
        }
        if self.velocity is not None:
            details["velocity"] = self.velocity.as_dict()
        if self.consistency is not None:
            details["consistency"] = self.consistency.as_dict()
        if self.competition is not None:
            details["competition_density"] = self.competition.as_dict()
        for source, growth in self.growth.items():
            details[f"growth_{source}"] = growth.as_dict()
        return details


@dataclass
class FeatureRun:
    """The outcome of one feature pass."""

    feature_date: date
    months: int
    settings: FeatureSettings
    entities: list[EntityFeatures] = field(default_factory=list)
    values_written: int = 0
    values_updated: int = 0
    values_removed: int = 0
    skipped_no_data: list[str] = field(default_factory=list)
    skipped_low_sample: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        topics = sum(1 for e in self.entities if e.entity_type == EntityType.TOPIC.value)
        return {
            "feature_date": self.feature_date.isoformat(),
            "months": self.months,
            "topics": topics,
            "competitors": len(self.entities) - topics,
            "values_written": self.values_written,
            "values_updated": self.values_updated,
            "values_removed": self.values_removed,
            "skipped_no_data": list(self.skipped_no_data),
            "skipped_low_sample": list(self.skipped_low_sample),
            "settings": self.settings.as_dict(),
            "warnings": list(self.warnings),
        }


def _composite(
    by_source: dict[str, list[float]], months: int, weights: Mapping[str, float]
) -> tuple[list[float], tuple[str, ...]]:
    """The combined activity signal per month, and which sources it was built from.

    Sources are weighted (publications and trials count for more than press releases by
    default). A source with no activity at all in the window has its weight shared across the
    others, so a quiet source does not drag the signal toward zero.
    """
    contributing = [
        source for source in weights if source in by_source and sum(by_source[source]) > 0
    ]
    if not contributing:
        return [0.0] * months, ()
    effective = renormalize_weights(dict(weights), contributing)
    series = [
        sum(effective[source] * by_source[source][index] for source in contributing)
        for index in range(months)
    ]
    return series, tuple(contributing)


def _entity_features(
    entity_type: EntityType,
    entity_id: int,
    name: str,
    months: Sequence[date],
    by_source: dict[str, list[float]],
    options: FeatureSettings,
    competition: CompetitionDensity | None,
) -> EntityFeatures:
    composite, composite_sources = _composite(by_source, len(months), options.composite_weights)
    growth: dict[str, GrowthResult] = {}
    for source, values in by_source.items():
        growth[source] = growth_from_series(
            values,
            options.momentum_recent_months,
            options.momentum_previous_months,
            alpha=options.alpha,
        )
    recent_quarter, previous_quarter = split_series(composite, QUARTER_MONTHS, QUARTER_MONTHS)
    # How much evidence there is counts every source, including those left out of the weighted
    # signal above: a topic carried by patents alone still has evidence behind it.
    all_sources = [
        sum(values[index] for values in by_source.values()) for index in range(len(months))
    ]
    total = sum(all_sources)
    return EntityFeatures(
        entity_type=entity_type.value,
        entity_id=entity_id,
        name=name,
        months=list(months),
        by_source=by_source,
        composite=composite,
        composite_sources=composite_sources,
        growth=growth,
        momentum=calculate_momentum(
            composite,
            recent_months=options.momentum_recent_months,
            previous_months=options.momentum_previous_months,
            alpha=options.alpha,
        ),
        velocity=calculate_velocity(
            composite,
            window_months=options.velocity_window_months,
            min_observations=options.velocity_min_observations,
        ),
        consistency=calculate_consistency(composite),
        surge=calculate_growth_rate(
            sum(recent_quarter), sum(previous_quarter), alpha=options.alpha
        ),
        agreement=calculate_source_agreement(
            {
                source: (growth[source].log_growth if sum(values) > 0 else None)
                for source, values in by_source.items()
            }
        ),
        competition=competition,
        total_records=total,
        sample_confidence=calculate_sample_confidence(total, k=options.sample_saturation_k),
        completeness=(
            calculate_data_completeness([value > 0 for value in all_sources])
            if all_sources
            else 0.0
        ),
    )


def compute_features(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime | date | None = None,
    store: bool = True,
    is_synthetic: bool = False,
) -> FeatureRun:
    """Compute features for every active topic and monitored competitor.

    Args:
        as_of: the date the features describe (default: now). The month containing it is
            excluded, because a part-finished month is not comparable with whole ones.
        store: write the results to ``feature_values``.
        is_synthetic: mark stored rows as demo data.

    Raises:
        ValueError: if the scoring configuration cannot be read.
    """
    options = load_feature_settings(settings)
    moment = as_of or datetime.now(UTC)
    months = month_range(moment, options.velocity_window_months + options.momentum_previous_months)
    months = [month for month in months if month < month_range(moment, 1)[0]]
    run = FeatureRun(
        feature_date=months[-1] if months else month_range(moment, 1)[0],
        months=len(months),
        settings=options,
    )

    topics = list(
        session.scalars(
            select(Topic).where(Topic.active.is_(True), Topic.status == TopicStatus.ACTIVE.value)
        )
    )
    competitors = list(
        session.scalars(
            select(Organization).where(
                (Organization.manually_included.is_(True))
                | (Organization.discovered_automatically.is_(True))
            )
        )
    )
    if not topics and not competitors:
        run.warnings.append("no active topics or monitored competitors; run: cews competitors")
        return run

    topic_activity: dict[int, dict[str, list[float]]] = {}
    for topic in topics:
        topic_activity[topic.id] = monthly_series(session, months, topic_id=topic.id)

    for topic in topics:
        by_source = topic_activity[topic.id]
        density = _topic_competition(session, months, topic.id, competitors)
        features = _entity_features(
            EntityType.TOPIC, topic.id, topic.canonical_name, months, by_source, options, density
        )
        if features.total_records <= 0:
            # Nothing happened here in this window. Ranking an empty topic against real ones
            # would shift everyone else's percentile, so it is reported instead of scored.
            run.skipped_no_data.append(topic.canonical_name)
            continue
        if features.total_records < options.min_sample_size:
            run.skipped_low_sample.append(topic.canonical_name)
        run.entities.append(features)

    for organization in competitors:
        by_source = monthly_series(session, months, organization_id=organization.id)
        features = _entity_features(
            EntityType.COMPETITOR,
            organization.id,
            organization.canonical_name,
            months,
            by_source,
            options,
            None,
        )
        if features.total_records <= 0:
            run.skipped_no_data.append(organization.canonical_name)
            continue
        if features.total_records < options.min_sample_size:
            run.skipped_low_sample.append(organization.canonical_name)
        run.entities.append(features)

    _normalize_and_store(session, run, options, store=store, is_synthetic=is_synthetic)
    LOGGER.info("computed features: %s", run.as_dict())
    return run


def _topic_competition(
    session: Session,
    months: Sequence[date],
    topic_id: int,
    competitors: Sequence[Organization],
) -> CompetitionDensity:
    """How crowded one topic is, from each monitored competitor's activity inside it."""
    activity: dict[str, float] = {}
    for organization in competitors:
        series = monthly_series(session, months, organization_id=organization.id, topic_id=topic_id)
        activity[organization.canonical_name] = sum(sum(values) for values in series.values())
    return calculate_competition_density(activity)


def _normalize_and_store(
    session: Session,
    run: FeatureRun,
    options: FeatureSettings,
    *,
    store: bool,
    is_synthetic: bool,
) -> None:
    """Normalize each feature within its cohort and write the rows."""
    by_kind: dict[str, list[EntityFeatures]] = {}
    for entity in run.entities:
        by_kind.setdefault(entity.entity_type, []).append(entity)

    existing: dict[tuple[date, str, int, str, str], FeatureValue] = {}
    if store:
        existing = {
            (
                row.feature_date,
                row.entity_type,
                row.entity_id,
                row.feature_name,
                row.comparison_cohort,
            ): row
            for row in session.scalars(
                select(FeatureValue).where(FeatureValue.feature_date == run.feature_date)
            )
        }

    for entity_type, entities in by_kind.items():
        names = {name for entity in entities for name in entity.raw_values()}
        for feature_name in sorted(names):
            cohort = f"{entity_type}/{feature_name}/{run.feature_date.isoformat()}"
            values = {
                str(entity.entity_id): entity.raw_values()[feature_name]
                for entity in entities
                if feature_name in entity.raw_values()
            }
            normalized = normalize_feature_cohort(
                values,
                method=options.normalization_method,
                cohort=cohort,
                lower_quantile=options.winsor_lower,
                upper_quantile=options.winsor_upper,
            )
            if not store:
                continue
            for entity in entities:
                result = normalized.get(str(entity.entity_id))
                if result is None:
                    continue
                key = (run.feature_date, entity_type, entity.entity_id, feature_name, cohort)
                row = existing.pop(key, None)
                if row is None:
                    session.add(
                        FeatureValue(
                            feature_date=run.feature_date,
                            entity_type=entity_type,
                            entity_id=entity.entity_id,
                            context_key="",
                            feature_name=feature_name,
                            raw_value=result.raw_value,
                            normalized_value=result.normalized_value,
                            normalization_method=result.method,
                            comparison_cohort=cohort,
                            cohort_size=result.cohort_size,
                            sample_size=int(entity.total_records),
                            is_synthetic=is_synthetic,
                        )
                    )
                    run.values_written += 1
                else:
                    row.raw_value = result.raw_value
                    row.normalized_value = result.normalized_value
                    row.normalization_method = result.method
                    row.cohort_size = result.cohort_size
                    row.sample_size = int(entity.total_records)
                    row.computed_at = datetime.now(UTC)
                    run.values_updated += 1
    if store:
        # Anything left over describes an entity that no longer qualifies (its activity fell to
        # nothing, or the cohort changed). Leaving it would show a stale rank on the dashboard.
        for row in existing.values():
            session.delete(row)
            run.values_removed += 1
        session.flush()
