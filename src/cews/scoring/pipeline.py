"""Running the scoring pass.

Reads the stored features for a date, produces scores, and saves them with their breakdown.

This phase computes the confidence score that every other score is shown beside. The analytical
scores (trend and opportunity for topics, innovation and threat for competitors) are added in the
next two phases and join the same pass, so they are computed from the same inputs, on the same
date, and stored the same way.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import EntityType, ScoreType
from cews.database.models import Organization, Score, Topic
from cews.features.pipeline import load_feature_settings
from cews.scoring.area_threat import area_inputs, collect_area_pairs
from cews.scoring.base import Gate, ScoreResult
from cews.scoring.confidence_score import ConfidenceResult, calculate_confidence_score
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.innovation_score import (
    assign_ranks,
    calculate_innovation_score,
    cohort_sources,
)
from cews.scoring.inputs import EntityInputs, as_evidence, latest_feature_date, load_entity_inputs
from cews.scoring.modifiers import collect_modifiers
from cews.scoring.opportunity_score import calculate_opportunity_score
from cews.scoring.store import StoreSummary, save_scores
from cews.scoring.threat_score import calculate_threat_score
from cews.scoring.trend_score import calculate_trend_score
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoringRun:
    """The outcome of one scoring pass."""

    score_date: date | None
    version: str
    results: list[ScoreResult] = field(default_factory=list)
    stored: StoreSummary = field(default_factory=StoreSummary)
    warnings: list[str] = field(default_factory=list)

    def of_type(self, score_type: str) -> list[ScoreResult]:
        """The results of one score type, highest first."""
        chosen = [result for result in self.results if result.score_type == score_type]
        return sorted(chosen, key=lambda result: result.value, reverse=True)

    def qualified(self, score_type: str) -> list[ScoreResult]:
        """The results of one score type that passed every rule, highest first."""
        return [result for result in self.of_type(score_type) if result.qualified]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.score_type] = counts.get(result.score_type, 0) + 1
        return {
            "score_date": self.score_date.isoformat() if self.score_date else None,
            "scoring_version": self.version,
            "scores": counts,
            "low_confidence": sum(
                1 for result in self.results if result.confidence < LOW_CONFIDENCE_LIMIT
            ),
            "qualified": {
                score_type: len(self.qualified(score_type))
                for score_type in sorted(counts)
                if score_type != ScoreType.CONFIDENCE.value
            },
            "stored": self.stored.as_dict(),
            "warnings": list(self.warnings),
        }


LOW_CONFIDENCE_LIMIT = 60.0


def score_confidence(
    inputs: EntityInputs,
    config: ScoringConfig,
    *,
    as_of: datetime,
    half_life_days: float,
    minimum: float,
) -> tuple[ScoreResult, ConfidenceResult]:
    """The confidence score for one entity.

    Raises:
        ValueError: if a stored feature is outside the 0-1 range it is meant to hold.
    """
    confidence = calculate_confidence_score(
        sample_confidence=inputs.raw_value("sample_confidence", 0.0) or 0.0,
        source_agreement=inputs.raw_value("source_agreement", 0.0) or 0.0,
        data_completeness=inputs.raw_value("data_completeness", 0.0) or 0.0,
        data_freshness=inputs.freshness(as_of=as_of, half_life_days=half_life_days),
        model_stability=inputs.stability(),
        sample_size=inputs.sample_size,
        weights=config.weights("confidence_score"),
    )
    usable = Gate(
        name="usable_confidence",
        passed=confidence.value >= minimum,
        detail=(
            f"confidence {confidence.value:.0f} is below the {minimum:.0f} needed to treat a "
            "score as a finding"
        ),
    )
    notes: list[str] = []
    if "model_stability" in confidence.unavailable:
        notes.append(
            "Stability is unknown because this is the first scored run; its weight was shared "
            "across the other inputs rather than counted as zero."
        )
    result = ScoreResult(
        score_type=ScoreType.CONFIDENCE.value,
        entity_type=inputs.entity_type,
        entity_id=inputs.entity_id,
        entity_name=inputs.name,
        value=confidence.value,
        confidence=confidence.value,
        components=confidence.components,
        unavailable=confidence.unavailable,
        sample_size=inputs.sample_size,
        gates=(usable,),
        notes=notes,
        scoring_version=config.version,
    )
    return result, confidence


def run_scoring(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime | date | None = None,
    store: bool = True,
    is_synthetic: bool = False,
) -> ScoringRun:
    """Score every entity that has features, and save the results.

    Args:
        as_of: score the most recent feature date on or before this point (default: now).
        store: write the scores to the database.
        is_synthetic: mark stored rows as demo data.

    Raises:
        ScoringConfigError: if the scoring configuration does not add up.
    """
    config = load_scoring_config(settings)
    options = load_feature_settings(settings)
    moment = as_of if isinstance(as_of, datetime) else None
    if moment is None:
        moment = (
            datetime(as_of.year, as_of.month, as_of.day, tzinfo=UTC)
            if isinstance(as_of, date)
            else datetime.now(UTC)
        )

    feature_date = latest_feature_date(session, on_or_before=moment.date())
    run = ScoringRun(score_date=feature_date, version=config.version)
    run.warnings.extend(config.warnings)
    if feature_date is None:
        run.warnings.append("no features stored yet; run: cews features")
        return run

    entities = load_entity_inputs(session, feature_date)
    if not entities:
        run.warnings.append(f"no features stored for {feature_date}; run: cews features")
        return run

    evidence: dict[tuple[str, int], Any] = {}
    competitor_inputs: dict[tuple[str, int], tuple[EntityInputs, float]] = {}
    for key, inputs in sorted(entities.items(), key=lambda item: item[0]):
        confidence, _ = score_confidence(
            inputs,
            config,
            as_of=moment,
            half_life_days=options.freshness_half_life_days,
            minimum=config.emerging_trend.min_confidence,
        )
        run.results.append(confidence)
        evidence[key] = as_evidence(inputs)

        if inputs.entity_type != EntityType.TOPIC.value:
            competitor_inputs[key] = (inputs, confidence.value)
            continue
        try:
            trend = calculate_trend_score(inputs, config, confidence=confidence.value)
        except ValueError as exc:  # nothing measurable for this topic
            run.warnings.append(f"{inputs.name}: no trend score ({exc})")
            continue
        run.results.append(trend)
        try:
            run.results.append(
                calculate_opportunity_score(
                    inputs, config, trend=trend, confidence=confidence.value
                )
            )
        except ValueError as exc:
            run.warnings.append(f"{inputs.name}: no opportunity score ({exc})")

    _score_competitors(session, run, config, competitor_inputs, feature_date=feature_date)

    if store:
        run.stored = save_scores(
            session,
            feature_date,
            run.results,
            evidence=evidence,
            is_synthetic=is_synthetic,
        )
    LOGGER.info("scoring pass: %s", run.as_dict())
    return run


def _previous_scores(session: Session, feature_date: date, score_type: str) -> dict[int, float]:
    """The last scores of one type before ``feature_date``, used for rank change."""
    earlier = session.scalar(
        select(func.max(Score.score_date)).where(
            Score.score_type == score_type, Score.score_date < feature_date, Score.context_key == ""
        )
    )
    if earlier is None:
        return {}
    rows = session.scalars(
        select(Score).where(
            Score.score_type == score_type, Score.score_date == earlier, Score.context_key == ""
        )
    )
    return {row.entity_id: float(row.score_value) for row in rows}


def _score_competitors(
    session: Session,
    run: ScoringRun,
    config: ScoringConfig,
    competitor_inputs: Mapping[tuple[str, int], tuple[EntityInputs, float]],
    *,
    feature_date: date,
) -> None:
    """Innovation and threat for every monitored competitor, overall and per therapeutic area."""
    if not competitor_inputs:
        return
    trending = [
        result.entity_id for result in run.of_type(ScoreType.TREND.value) if result.qualified
    ]
    topic_names = {topic.id: topic.canonical_name for topic in session.scalars(select(Topic))}

    # A source that no competitor has data for is switched off, not evidence of inactivity.
    present = cohort_sources([inputs for inputs, _ in competitor_inputs.values()])

    innovation: list[ScoreResult] = []
    for inputs, confidence in competitor_inputs.values():
        try:
            innovation.append(
                calculate_innovation_score(
                    inputs, config, confidence=confidence, available_sources=present
                )
            )
        except ValueError as exc:
            run.warnings.append(f"{inputs.name}: no innovation score ({exc})")
        try:
            modifiers = collect_modifiers(
                session,
                inputs.entity_id,
                as_of=feature_date,
                config=config.threat_modifiers,
                supporting_sources=inputs.supporting_sources,
                trending_topic_ids=trending,
                topic_names=topic_names,
            )
            run.results.append(
                calculate_threat_score(inputs, config, confidence=confidence, modifiers=modifiers)
            )
        except ValueError as exc:
            run.warnings.append(f"{inputs.name}: no threat score ({exc})")

    assign_ranks(innovation, _previous_scores(session, feature_date, ScoreType.INNOVATION.value))
    run.results.extend(innovation)

    competitors = list(
        session.scalars(
            select(Organization).where(
                Organization.id.in_([inputs.entity_id for inputs, _ in competitor_inputs.values()])
            )
        )
    )
    pairs = collect_area_pairs(session, competitors, as_of=feature_date)
    confidence_by_id = {
        inputs.entity_id: confidence for inputs, confidence in competitor_inputs.values()
    }
    areas = {pair.area_key: pair.area_name for pair in pairs}
    for (organization_id, area_key), inputs in area_inputs(pairs, config, feature_date).items():
        try:
            run.results.append(
                calculate_threat_score(
                    inputs,
                    config,
                    confidence=confidence_by_id.get(organization_id, 0.0),
                    context_key=area_key,
                    context_label=areas.get(area_key, area_key),
                )
            )
        except ValueError as exc:
            run.warnings.append(f"{inputs.name} in {area_key}: no threat score ({exc})")


def score_history(
    session: Session, entity_type: str, entity_id: int, score_type: str
) -> list[Score]:
    """Every stored score of one type for one entity, oldest first."""
    return list(
        session.scalars(
            select(Score)
            .where(
                Score.entity_type == entity_type,
                Score.entity_id == entity_id,
                Score.score_type == score_type,
            )
            .order_by(Score.score_date)
        )
    )
