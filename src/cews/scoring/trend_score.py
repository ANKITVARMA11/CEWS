"""The trend score: which research topics are heating up.

    Trend = 0.30 velocity + 0.25 momentum + 0.20 patent growth
            + 0.15 funding growth + 0.10 consistency

Each part is the topic's normalized 0-100 position among the other topics, so a patent count and
a publication count can sit in the same sum. Velocity says which way the year has gone, momentum
whether it is heating up now, consistency whether the growth was sustained or one big month.

A component with no data (no patents anywhere for this topic, say) has its weight shared across
the rest rather than counted as zero, and the gap is recorded.

**A high score is not a finding.** Calling a topic an emerging trend needs the score, enough
confidence, enough evidence, agreement from more than one independent source, and growth that
is not a single isolated spike. Those rules are checked here and travel with the score, so a
number can never be quoted without the reasons it did or did not qualify.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from cews.constants import EntityType, ScoreType
from cews.scoring.base import Gate, ScoreResult, apply_categories, combine
from cews.scoring.config import EmergingTrendRules, ScoringConfig
from cews.scoring.inputs import EntityInputs

LOGGER = logging.getLogger(__name__)

# Which stored feature each weighted component reads, and the source type it depends on.
COMPONENT_FEATURES: dict[str, tuple[str, str | None]] = {
    "velocity": ("velocity", None),
    "momentum": ("momentum", None),
    "patent_growth": ("growth_patent", "patent"),
    "funding_growth": ("growth_funding", "funding"),
    "consistency": ("consistency", None),
}


def _parts(
    inputs: EntityInputs, weights: Mapping[str, float]
) -> dict[str, tuple[float | None, float | None, Mapping[str, Any]]]:
    available = set(inputs.available_sources)
    parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {}
    for component in weights:
        feature, source = COMPONENT_FEATURES.get(component, (component, None))
        normalized = inputs.value(feature)
        raw = inputs.raw_value(feature)
        if source is not None and source not in available:
            # No records of that kind at all, which is different from no growth.
            parts[component] = (raw, None, {"feature": feature, "reason": f"no {source} records"})
            continue
        if normalized is None:
            parts[component] = (
                raw,
                None,
                {"feature": feature, "reason": "not enough months to measure"},
            )
            continue
        parts[component] = (raw, normalized, {"feature": feature})
    return parts


def emerging_trend_gates(
    inputs: EntityInputs, score: float, confidence: float, rules: EmergingTrendRules
) -> tuple[Gate, ...]:
    """Check the rules a topic must pass before it may be called an emerging trend."""
    return (
        Gate(
            "score_threshold",
            score >= rules.min_trend_score,
            f"trend score {score:.0f} is below the {rules.min_trend_score:.0f} required",
        ),
        Gate(
            "confidence_threshold",
            confidence >= rules.min_confidence,
            f"confidence {confidence:.0f} is below the {rules.min_confidence:.0f} required",
        ),
        Gate(
            "minimum_evidence",
            inputs.sample_size >= rules.min_sample_size,
            f"only {inputs.sample_size:.0f} record(s), fewer than the {rules.min_sample_size} required",
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
            "not_a_single_spike",
            not (rules.exclude_single_spike and inputs.single_spike),
            "the growth comes from one isolated month rather than a sustained rise",
        ),
    )


def calculate_trend_score(
    inputs: EntityInputs,
    config: ScoringConfig,
    *,
    confidence: float,
) -> ScoreResult:
    """Score how strongly one topic is trending, and whether it qualifies as a finding.

    Args:
        inputs: the topic's stored features.
        config: the validated scoring configuration.
        confidence: the topic's confidence score, used by the rules.

    Raises:
        ValueError: if no component has data at all.
    """
    weights = config.weights("trend_score")
    value, components, unavailable = combine(_parts(inputs, weights), weights)
    gates = emerging_trend_gates(inputs, value, confidence, config.emerging_trend)

    notes: list[str] = []
    if all(gate.passed for gate in gates):
        notes.append(
            "Qualifies as an emerging trend: it clears the score and confidence thresholds, has "
            f"enough evidence, and {inputs.supporting_sources} independent sources show growth."
        )
    result = ScoreResult(
        score_type=ScoreType.TREND.value,
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
