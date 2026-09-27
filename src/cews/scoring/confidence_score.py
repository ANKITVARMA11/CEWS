"""The confidence score: how much a number on the dashboard can be relied on.

Every analytical score carries one, and the two are always shown together. A trend score of 90
built on six records from a single source is not a finding; the confidence score is what says so.

    Confidence = 100 x (0.35 sample + 0.25 source agreement + 0.20 completeness
                        + 0.10 freshness + 0.10 model stability)

Each input runs from 0 to 1:

* **sample** - how much evidence there is, saturating (``1 - exp(-N / k)``);
* **source agreement** - the share of the sources that have data and point the same way;
* **completeness** - the share of expected months that actually have data;
* **freshness** - how recently the underlying sources were collected, halving with age;
* **model stability** - how much the value moved since the previous run.

Stability is unknown on a first run, because there is nothing to compare against. It is then
treated as unavailable and its weight is shared across the other inputs, rather than scored as
zero, which would make every first run look untrustworthy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cews.scoring.base import ScoreComponent, combine

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "sample": 0.35,
    "source_agreement": 0.25,
    "completeness": 0.20,
    "freshness": 0.10,
    "model_stability": 0.10,
}
LOW_CONFIDENCE = 60.0


@dataclass(frozen=True)
class ConfidenceResult:
    """A confidence score and the inputs behind it."""

    value: float
    components: dict[str, ScoreComponent]
    unavailable: tuple[str, ...]
    sample_size: float

    @property
    def low(self) -> bool:
        """True when this score should be flagged rather than acted on."""
        return self.value < LOW_CONFIDENCE

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored beside the score it describes."""
        return {
            "confidence": round(self.value, 2),
            "components": {name: part.as_dict() for name, part in self.components.items()},
            "unavailable": list(self.unavailable),
            "sample_size": round(self.sample_size, 3),
            "low_confidence": self.low,
        }


def _fraction(name: str, value: float | None) -> float | None:
    """Check an input is a fraction from 0 to 1, or None when it is unknown."""
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {number}")
    return number


def calculate_confidence_score(
    *,
    sample_confidence: float,
    source_agreement: float,
    data_completeness: float,
    data_freshness: float,
    model_stability: float | None = None,
    sample_size: float = 0.0,
    weights: Mapping[str, float] | None = None,
) -> ConfidenceResult:
    """Combine the five inputs into a 0-100 confidence score.

    Args:
        sample_confidence: how much evidence there is (0-1).
        source_agreement: share of available sources pointing the same way (0-1).
        data_completeness: share of expected periods with data (0-1).
        data_freshness: how recently the sources were collected (0-1).
        model_stability: how steady the value is across runs (0-1), or None on a first run.
        sample_size: the record count behind the score, kept for the explanation.
        weights: override the configured weights (they must sum to 1).

    Raises:
        ValueError: if an input is outside 0-1, or the weights do not sum to 1.
    """
    inputs = {
        "sample": _fraction("sample_confidence", sample_confidence),
        "source_agreement": _fraction("source_agreement", source_agreement),
        "completeness": _fraction("data_completeness", data_completeness),
        "freshness": _fraction("data_freshness", data_freshness),
        "model_stability": _fraction("model_stability", model_stability),
    }
    parts: dict[str, tuple[float | None, float | None, Mapping[str, Any]]] = {
        name: (value, None if value is None else value * 100.0, {})
        for name, value in inputs.items()
    }
    value, components, unavailable = combine(parts, dict(weights or CONFIDENCE_WEIGHTS))
    return ConfidenceResult(
        value=value,
        components=components,
        unavailable=unavailable,
        sample_size=float(sample_size),
    )
