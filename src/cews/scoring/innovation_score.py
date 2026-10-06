"""The innovation score: which competitors are producing the most research output.

    Innovation = 0.40 patents + 0.30 trials + 0.20 publications + 0.10 funding

Unlike the trend score, this measures **levels, not growth**: how much a company is doing
compared with the others being monitored. Patents carry the most weight because a patent is a
costly, deliberate claim on an idea, while a publication is cheaper and often academic.

A competitor with no patents scores low on that component. That is a real finding, not missing
data, so it is scored as the genuine zero it is. A component is only treated as unavailable when
**nobody** has data for it, which means the source itself is off or empty. That distinction
matters: "they file no patents" and "we cannot see patents" deserve opposite treatment.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from cews.constants import EntityType, ScoreType
from cews.scoring.base import ScoreResult, apply_categories, combine
from cews.scoring.config import ScoringConfig
from cews.scoring.inputs import EntityInputs

LOGGER = logging.getLogger(__name__)

# Which stored feature each component reads, and the source type it needs.
COMPONENT_FEATURES: dict[str, tuple[str, str]] = {
    "patent": ("activity_patent", "patent"),
    "clinical_trial": ("activity_clinical_trial", "clinical_trial"),
    "publication": ("activity_publication", "publication"),
    "funding": ("activity_funding", "funding"),
}


def cohort_sources(entities: Sequence[EntityInputs]) -> set[str]:
    """Source types that any competitor in the cohort has activity for.

    Used to tell a source-wide gap ("patents are switched off") apart from one company simply
    not filing patents.
    """
    return {source for entity in entities for source in entity.available_sources}


def calculate_innovation_score(
    inputs: EntityInputs,
    config: ScoringConfig,
    *,
    confidence: float,
    available_sources: set[str] | None = None,
    previous_value: float | None = None,
) -> ScoreResult:
    """Score how much research output one competitor is producing.

    Args:
        inputs: the competitor's stored features.
        config: the validated scoring configuration.
        confidence: the competitor's confidence score.
        available_sources: source types any competitor has data for. When omitted, only this
            competitor's own sources are considered available.
        previous_value: the same score from an earlier run, for the change note.

    Raises:
        ValueError: if no component has data anywhere in the cohort.
    """
    weights = config.weights("innovation_score")
    present = available_sources if available_sources is not None else set(inputs.available_sources)
    parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {}
    for component in weights:
        feature, source = COMPONENT_FEATURES[component]
        raw = inputs.raw_value(feature)
        if source not in present:
            parts[component] = (
                raw,
                None,
                {"feature": feature, "reason": f"no {source} data for any competitor"},
            )
            continue
        normalized = inputs.value(feature)
        parts[component] = (
            raw,
            normalized,
            {"feature": feature, "records": None if raw is None else round(raw, 2)},
        )

    value, components, unavailable = combine(parts, weights)
    notes: list[str] = []
    if previous_value is not None:
        change = value - previous_value
        direction = "up" if change > 0 else "down" if change < 0 else "unchanged"
        notes.append(f"Innovation is {direction} {abs(change):.1f} points since the previous run.")
    result = ScoreResult(
        score_type=ScoreType.INNOVATION.value,
        entity_type=EntityType.COMPETITOR.value,
        entity_id=inputs.entity_id,
        entity_name=inputs.name,
        value=value,
        confidence=confidence,
        components=components,
        unavailable=unavailable,
        sample_size=inputs.sample_size,
        notes=notes,
        scoring_version=config.version,
    )
    return apply_categories(result, config)


def _ranks(values: Mapping[int, float]) -> dict[int, int]:
    """Turn scores into 1-based ranks, highest first, with ties sharing a rank."""
    ranks: dict[int, int] = {}
    last: float | None = None
    position = 0
    for index, (entity_id, value) in enumerate(
        sorted(values.items(), key=lambda item: item[1], reverse=True), start=1
    ):
        if last is None or value < last:
            position, last = index, value
        ranks[entity_id] = position
    return ranks


def assign_ranks(
    results: Sequence[ScoreResult], previous_values: Mapping[int, float] | None = None
) -> None:
    """Add rank, and the change in rank, to each result's notes and breakdown.

    Leadership reads positions before numbers, and a move from 6th to 2nd says more than a score
    of 71. Ties share a rank. ``previous_values`` holds the same scores from an earlier run; the
    earlier ranking is worked out from them, because a rank only means anything relative to the
    same cohort.
    """
    previous = _ranks(previous_values or {})
    ordered = sorted(results, key=lambda result: result.value, reverse=True)
    rank = 0
    last_value: float | None = None
    for index, result in enumerate(ordered, start=1):
        if last_value is None or result.value < last_value:
            rank = index
            last_value = result.value
        detail: dict[str, Any] = {"rank": rank, "of": len(ordered)}
        note = f"Ranked {rank} of {len(ordered)} monitored competitors."
        if previous and result.entity_id in previous:
            moved = previous[result.entity_id] - rank
            detail["rank_change"] = moved
            if moved:
                note += f" Moved {'up' if moved > 0 else 'down'} {abs(moved)} place(s)."
        result.notes.append(note)
        result.context_key = result.context_key or ""
        result.components.setdefault(
            "rank",
            type(next(iter(result.components.values())))(
                name="rank",
                raw_value=float(rank),
                normalized=0.0,
                weight=0.0,
                available=False,
                detail=detail,
            ),
        )
