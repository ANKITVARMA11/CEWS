"""The opportunity score: growing topics that are not yet crowded.

    Opportunity = 100 x (0.60 trend + 0.25 (1 - competition density)
                         + 0.15 source agreement)

A topic scores well when it is trending, few organizations are working in it, and independent
sources agree that it is moving. Competition is the inverse of density, so an empty field helps
and a crowded one does not.

**This is not investment advice, and it is not a recommendation.** It is a prioritization signal
that says where an expert might look first, and it is deliberately hard to qualify for: the same
evidence rules as the trend score apply, plus the requirement that competition could actually be
measured. A topic with no competition data is reported as incomplete rather than being scored as
though the field were empty, which would be the most flattering possible reading of missing data.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from cews.constants import EntityType, ScoreType
from cews.scoring.base import Gate, ScoreResult, apply_categories, combine
from cews.scoring.config import ScoringConfig
from cews.scoring.inputs import EntityInputs

LOGGER = logging.getLogger(__name__)

MIN_RECORDS_FOR_SPREAD = 2


def opportunity_gates(
    inputs: EntityInputs,
    trend: ScoreResult,
    confidence: float,
    config: ScoringConfig,
    *,
    competition_measured: bool,
) -> tuple[Gate, ...]:
    """The rules a topic must pass before its opportunity score means anything."""
    rules = config.emerging_trend
    return (
        Gate(
            "minimum_evidence",
            inputs.sample_size >= rules.min_sample_size,
            f"only {inputs.sample_size:.0f} record(s), fewer than the {rules.min_sample_size} required",
        ),
        Gate(
            "confidence_threshold",
            confidence >= rules.min_confidence,
            f"confidence {confidence:.0f} is below the {rules.min_confidence:.0f} required",
        ),
        Gate(
            "independent_sources",
            inputs.supporting_sources >= rules.min_source_types,
            (
                f"only {inputs.supporting_sources} source type(s) show growth; "
                f"{rules.min_source_types} independent sources are required"
            ),
        ),
        Gate(
            "not_one_record",
            inputs.sample_size >= MIN_RECORDS_FOR_SPREAD and not inputs.single_spike,
            "the growth rests on a single record or a single month",
        ),
        Gate(
            "competition_measured",
            competition_measured,
            "competition could not be measured, so an empty-looking field may just be missing data",
        ),
    )


def calculate_opportunity_score(
    inputs: EntityInputs,
    config: ScoringConfig,
    *,
    trend: ScoreResult,
    confidence: float,
) -> ScoreResult:
    """Score how attractive a topic looks for expert review.

    Args:
        inputs: the topic's stored features.
        config: the validated scoring configuration.
        trend: the topic's trend score, which carries most of the weight.
        confidence: the topic's confidence score.

    Raises:
        ValueError: if no component has data at all.
    """
    weights = config.weights("opportunity_score")
    density = inputs.raw_value("competition_density")
    competition_measured = density is not None
    agreement = inputs.raw_value("source_agreement")

    parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {
        "trend": (trend.value, trend.value, {"from": "trend score"}),
        "low_competition": (
            density,
            None if density is None else max(0.0, 100.0 - density),
            {
                "from": "competition density",
                "active_competitors": inputs.raw_value("competition_active", None),
            },
        ),
        "source_agreement": (
            agreement,
            None if agreement is None else agreement * 100.0,
            {"supporting_sources": inputs.supporting_sources},
        ),
    }
    value, components, unavailable = combine(parts, weights)
    gates = opportunity_gates(
        inputs, trend, confidence, config, competition_measured=competition_measured
    )

    notes: list[str] = [
        "An evidence-based prioritization signal for expert review, not a recommendation."
    ]
    if not competition_measured:
        notes.append("Incomplete: competition could not be measured for this topic.")
    result = ScoreResult(
        score_type=ScoreType.OPPORTUNITY.value,
        entity_type=EntityType.TOPIC.value,
        entity_id=inputs.entity_id,
        entity_name=inputs.name,
        value=value,
        confidence=confidence,
        components=components,
        unavailable=unavailable,
        sample_size=inputs.sample_size,
        gates=gates,
        notes=notes,
        scoring_version=config.version,
    )
    return apply_categories(result, config)
