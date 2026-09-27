"""The shape every CEWS score takes, and how a score explains itself.

Each score is a weighted sum of components that are already on a 0-100 scale. What makes a score
usable on a dashboard is not the number but everything stored beside it: each component's raw
value, its normalized value, the weight applied, what that contributed, which components had no
data, how much evidence there was, and which rules the result passed or failed.

When a component has no data its weight is shared across the components that remain, never
replaced by zero. Missing patent data must not make a topic look inactive; it makes the score
rest on fewer legs, which is recorded and lowers confidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from cews.scoring.config import Category, ScoringConfig
from cews.scoring.normalization import clamp_score, renormalize_weights

NEUTRAL_SCORE = 50.0


@dataclass(frozen=True)
class ScoreComponent:
    """One weighted part of a score."""

    name: str
    raw_value: float | None
    normalized: float
    weight: float
    available: bool = True
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def contribution(self) -> float:
        """How many points this component put into the final score."""
        return self.normalized * self.weight

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the score."""
        payload: dict[str, Any] = {
            "raw_value": round(self.raw_value, 4) if self.raw_value is not None else None,
            "normalized": round(self.normalized, 2),
            "weight": round(self.weight, 4),
            "contribution": round(self.contribution, 3),
            "available": self.available,
        }
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(frozen=True)
class Gate:
    """One rule a result had to satisfy, and whether it did."""

    name: str
    passed: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class ScoreResult:
    """A finished score with everything needed to defend it."""

    score_type: str
    entity_type: str
    entity_id: int
    entity_name: str
    value: float
    confidence: float
    components: dict[str, ScoreComponent] = field(default_factory=dict)
    unavailable: tuple[str, ...] = ()
    sample_size: float = 0.0
    category: str = ""
    gates: tuple[Gate, ...] = ()
    context_key: str = ""
    notes: list[str] = field(default_factory=list)
    scoring_version: str = ""

    @property
    def qualified(self) -> bool:
        """True when every rule attached to this score passed."""
        return all(gate.passed for gate in self.gates)

    @property
    def failed_gates(self) -> tuple[str, ...]:
        """The names of the rules this result did not satisfy."""
        return tuple(gate.name for gate in self.gates if not gate.passed)

    def as_component_json(self) -> dict[str, Any]:
        """The breakdown stored in ``scores.component_json``."""
        return {
            "components": {name: part.as_dict() for name, part in self.components.items()},
            "unavailable": list(self.unavailable),
            "sample_size": round(self.sample_size, 3),
            "category": self.category,
            "gates": [gate.as_dict() for gate in self.gates],
            "qualified": self.qualified,
            "notes": list(self.notes),
            "scoring_version": self.scoring_version,
        }

    def explain(self) -> str:
        """A plain sentence saying how the score was reached and what it rests on."""
        return explain_score(self)


def combine(
    parts: Mapping[str, tuple[float | None, float | None, Mapping[str, Any]]],
    weights: Mapping[str, float],
) -> tuple[float, dict[str, ScoreComponent], tuple[str, ...]]:
    """Weight the available components into a 0-100 score.

    Args:
        parts: component name -> (raw value, normalized 0-100 value, detail). A normalized value
            of None marks a component with no data.
        weights: the configured weight per component.

    Returns:
        The score, the components (including the unavailable ones, at weight 0), and the names
        of the components that had no data.

    Raises:
        ValueError: if no component has data, or a normalized value is outside 0-100.
    """
    available = [
        name
        for name, (_, normalized, _) in parts.items()
        if name in weights and normalized is not None and math.isfinite(normalized)
    ]
    missing = tuple(name for name in weights if name not in available)
    if not available:
        raise ValueError("no component has data; the score cannot be computed")

    effective = renormalize_weights(dict(weights), available)
    components: dict[str, ScoreComponent] = {}
    total = 0.0
    for name in weights:
        raw, normalized, detail = parts.get(name, (None, None, {}))
        if name in effective and normalized is not None:
            if not 0.0 <= normalized <= 100.0:
                raise ValueError(f"component {name!r} is {normalized}, outside the 0-100 scale")
            component = ScoreComponent(name, raw, normalized, effective[name], True, detail)
            total += component.contribution
        else:
            component = ScoreComponent(name, raw, 0.0, 0.0, False, detail)
        components[name] = component
    return clamp_score(total), components, missing


def classify_score(value: float, bands: Sequence[Category]) -> str:
    """The label for a score, or an empty string when no bands are configured."""
    for band in bands:
        if band.contains(value):
            return band.label
    return ""


def apply_categories(result: ScoreResult, config: ScoringConfig) -> ScoreResult:
    """Attach the configured category for this score type."""
    result.category = classify_score(result.value, config.categories_for(result.score_type))
    return result


def explain_score(result: ScoreResult) -> str:
    """Describe a score in one readable passage: what drove it, and what it rests on."""
    ranked = sorted(
        (part for part in result.components.values() if part.available),
        key=lambda part: part.contribution,
        reverse=True,
    )
    if not ranked:
        return f"{result.entity_name}: no component had data, so no score was produced."

    driver_text = ", ".join(
        f"{part.name.replace('_', ' ')} {part.normalized:.0f}/100 "
        f"(weight {part.weight:.2f}, {part.contribution:.1f} points)"
        for part in ranked[:3]
    )
    sentences = [
        f"{result.entity_name} scores {result.value:.1f} for {result.score_type.replace('_', ' ')}"
        + (f" ({result.category})" if result.category else "")
        + f", with confidence {result.confidence:.0f}.",
        f"Largest contributions: {driver_text}.",
        f"Based on {result.sample_size:.0f} record(s).",
    ]
    if result.unavailable:
        sentences.append(
            "No data for "
            + ", ".join(name.replace("_", " ") for name in result.unavailable)
            + "; their weight was shared across the remaining components and confidence reduced."
        )
    if result.failed_gates:
        sentences.append(
            "Not treated as a confirmed finding because it fails: "
            + "; ".join(gate.detail for gate in result.gates if not gate.passed)
            + "."
        )
    sentences.extend(result.notes)
    return " ".join(sentences)
