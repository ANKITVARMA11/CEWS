"""The threat score: which competitors deserve closer attention, and where.

    Threat = 0.50 trial growth + 0.30 patent growth + 0.20 publication growth

Growth, not size. A large company doing what it always does is not news; a smaller one whose
trial activity has doubled is. Trials carry the most weight because starting a trial is the most
expensive and most committing thing on the list.

Capped modifiers add a few points for events growth cannot show on its own: moving into a new
area, trials reaching a later phase, a jump in enrollment, patents in watched topics, several
sources moving together. They are shown separately from the base score, never folded into it
invisibly.

A score is computed for each competitor overall and for each therapeutic area they work in, so
"who is coming after this area" can be answered directly.

Like the other scores, a high number on thin evidence is reported with the rules it fails rather
than presented as a finding: a small company can show the fastest growth simply by starting from
almost nothing.

**Wording matters.** This score says a competitor is worth watching. It is not evidence of a
legal, commercial or scientific threat, and the labels say so: routine monitoring, watch,
elevated activity, high monitoring priority.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from cews.constants import EntityType, ScoreType
from cews.scoring.base import Gate, ScoreComponent, ScoreResult, apply_categories, combine
from cews.scoring.config import ScoringConfig
from cews.scoring.inputs import EntityInputs
from cews.scoring.modifiers import ModifierSet
from cews.scoring.normalization import clamp_score

LOGGER = logging.getLogger(__name__)

# Which stored feature each component reads, and the source type it needs.
COMPONENT_FEATURES: dict[str, tuple[str, str]] = {
    "trial_growth": ("growth_clinical_trial", "clinical_trial"),
    "patent_growth": ("growth_patent", "patent"),
    "publication_growth": ("growth_publication", "publication"),
}
DEFAULT_LABEL = "Routine monitoring"


def _parts(
    inputs: EntityInputs, weights: Mapping[str, float]
) -> dict[str, tuple[float | None, float | None, Mapping[str, Any]]]:
    available = set(inputs.available_sources)
    parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {}
    for component in weights:
        feature, source = COMPONENT_FEATURES[component]
        raw = inputs.raw_value(feature)
        if source not in available:
            parts[component] = (raw, None, {"feature": feature, "reason": f"no {source} records"})
            continue
        parts[component] = (raw, inputs.value(feature), {"feature": feature})
    return parts


def calculate_threat_score(
    inputs: EntityInputs,
    config: ScoringConfig,
    *,
    confidence: float,
    modifiers: ModifierSet | None = None,
    context_key: str = "",
    context_label: str = "",
) -> ScoreResult:
    """Score how closely one competitor should be watched.

    Args:
        inputs: the competitor's stored features (overall, or within one area).
        config: the validated scoring configuration.
        confidence: the competitor's confidence score.
        modifiers: capped adjustments already collected for this competitor.
        context_key: the therapeutic area this score is limited to, if any.
        context_label: a readable name for that area.

    Raises:
        ValueError: if no component has data.
    """
    weights = config.weights("threat_score")
    base, components, unavailable = combine(_parts(inputs, weights), weights)

    adjustments = modifiers or ModifierSet()
    value = clamp_score(base + adjustments.total)
    if adjustments.applied:
        # Carried at zero weight so the base growth measurement stays visible on its own.
        components["modifiers"] = ScoreComponent(
            name="modifiers",
            raw_value=adjustments.total,
            normalized=0.0,
            weight=0.0,
            available=False,
            detail=adjustments.as_dict(),
        )

    rules = config.emerging_trend
    gates = (
        Gate(
            "usable_confidence",
            confidence >= rules.min_confidence,
            f"confidence {confidence:.0f} is below the {rules.min_confidence:.0f} needed to act on this",
        ),
        Gate(
            "minimum_evidence",
            inputs.sample_size >= rules.min_sample_size,
            f"only {inputs.sample_size:.0f} record(s), fewer than the {rules.min_sample_size} required",
        ),
    )
    notes: list[str] = [
        "A monitoring priority based on observed activity, not evidence of a legal, commercial "
        "or scientific threat."
    ]
    if adjustments.applied:
        notes.append(f"Base score {base:.1f}. {adjustments.summary()}")
    if context_label:
        notes.append(f"Limited to activity in {context_label}.")

    name = f"{inputs.name} in {context_label}" if context_label else inputs.name
    result = ScoreResult(
        score_type=ScoreType.THREAT.value,
        entity_type=EntityType.COMPETITOR.value,
        entity_id=inputs.entity_id,
        entity_name=name,
        value=value,
        confidence=confidence,
        components=components,
        unavailable=unavailable,
        sample_size=inputs.sample_size,
        gates=gates,
        context_key=context_key,
        notes=notes,
        scoring_version=config.version,
    )
    result = apply_categories(result, config)
    if not result.category:
        result.category = DEFAULT_LABEL
    return result
