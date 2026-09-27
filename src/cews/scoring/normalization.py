"""Putting raw feature values on a comparable 0-100 scale.

Counts from different sources cannot be added together: a topic with 400 publications and 3
trials is not "403 units" of activity. Each value is therefore ranked against a cohort of
comparable values first.

Percentile rank is the project default (``NORMALIZATION_METHOD``). It answers "where does this
value sit among its peers?", is unaffected by outliers, and is easy to explain. The remaining
methods from the scoring specification (winsorized min-max, robust z-score) arrive with the
scoring engine in Phase 6; this module provides what competitor discovery needs today.

Edge cases follow one rule: when a cohort cannot distinguish its members - it is empty, has one
member, or every value is identical - every member scores the neutral **50** rather than 0 or
100, because the data says nothing about who leads.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

NEUTRAL_SCORE = 50.0
METHODS = ("percentile", "winsorized_minmax", "robust_zscore", "minmax")
SCORE_MIN = 0.0
SCORE_MAX = 100.0


def _usable(values: Sequence[float]) -> list[float] | None:
    numbers = [float(v) for v in values]
    if not numbers or any(math.isnan(v) or math.isinf(v) for v in numbers):
        return None
    if len(numbers) == 1 or len(set(numbers)) == 1:
        return None
    return numbers


def percentile_normalize(values: Sequence[float]) -> list[float]:
    """Rank each value against the cohort, as a 0-100 percentile.

    A value scores ``(below + half the ties) / count``, so the middle of a cohort lands near 50
    and ties share a score. Returns 50 for every member of a cohort that cannot rank itself.
    """
    numbers = _usable(values)
    if numbers is None:
        return [NEUTRAL_SCORE] * len(values)
    total = len(numbers)
    ordered = sorted(numbers)
    scores: list[float] = []
    for value in numbers:
        below = sum(1 for other in ordered if other < value)
        ties = sum(1 for other in ordered if other == value)
        scores.append(round((below + 0.5 * ties) / total * 100, 4))
    return scores


def minmax_normalize(values: Sequence[float]) -> list[float]:
    """Scale values linearly so the smallest is 0 and the largest is 100.

    Useful for display, but sensitive to outliers; percentile rank is preferred for scoring.
    """
    numbers = _usable(values)
    if numbers is None:
        return [NEUTRAL_SCORE] * len(values)
    low, high = min(numbers), max(numbers)
    span = high - low
    return [round((value - low) / span * 100, 4) for value in numbers]


def normalize_values(values: Sequence[float], method: str = "percentile") -> list[float]:
    """Normalize with the named method.

    Raises:
        ValueError: for a method this module does not implement yet.
    """
    if method == "percentile":
        return percentile_normalize(values)
    if method == "minmax":
        return minmax_normalize(values)
    if method == "winsorized_minmax":
        return winsorized_minmax_normalize(values)
    if method == "robust_zscore":
        return robust_zscore_normalize(values)
    raise ValueError(f"unknown normalization method {method!r}; use one of: {', '.join(METHODS)}")


def clamp_score(value: float) -> float:
    """Keep a score inside 0-100."""
    return max(SCORE_MIN, min(SCORE_MAX, float(value)))


def renormalize_weights(weights: dict[str, float], available: Sequence[str]) -> dict[str, float]:
    """Spread the weights of unavailable components across the ones that remain.

    A missing source is never treated as zero activity: its weight is redistributed in
    proportion, so the score still ranges 0-100 and the components that do exist keep their
    relative importance. Confidence is lowered separately by the caller.

    Raises:
        ValueError: if none of the components are available or the weights are not positive.
    """
    usable = {name: float(weights[name]) for name in available if name in weights}
    if not usable or sum(usable.values()) <= 0:
        raise ValueError("no weighted components are available")
    total = sum(usable.values())
    return {name: weight / total for name, weight in usable.items()}


@dataclass(frozen=True)
class CohortNormalization:
    """A normalized value together with the cohort and method that produced it."""

    raw_value: float
    normalized_value: float
    method: str
    cohort: str
    cohort_size: int
    neutral: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "raw_value": round(self.raw_value, 4),
            "normalized_value": round(self.normalized_value, 2),
            "method": self.method,
            "cohort": self.cohort,
            "cohort_size": self.cohort_size,
            "neutral": self.neutral,
        }


def winsorized_minmax_normalize(
    values: Sequence[float], lower_quantile: float = 0.05, upper_quantile: float = 0.95
) -> list[float]:
    """Min-max scaling after clipping the extremes, so one outlier cannot flatten the rest.

    Values are clipped at the given quantiles and then scaled to 0-100. A cohort whose clipped
    values are all equal scores the neutral 50.

    Raises:
        ValueError: if the quantiles are out of order or outside 0-1.
    """
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    numbers = _usable(values)
    if numbers is None:
        return [NEUTRAL_SCORE] * len(values)
    ordered = sorted(numbers)
    low = _quantile(ordered, lower_quantile)
    high = _quantile(ordered, upper_quantile)
    if high <= low:
        return [NEUTRAL_SCORE] * len(numbers)
    scores = []
    for value in numbers:
        clipped = min(max(value, low), high)
        scores.append(round((clipped - low) / (high - low) * 100.0, 4))
    return scores


def robust_zscore_normalize(values: Sequence[float], scale: float = 2.0) -> list[float]:
    """Centre on the median and scale by the median absolute deviation, then map onto 0-100.

    Robust to outliers because neither the median nor the MAD moves much when one value is
    extreme. ``scale`` is how many robust deviations reach the ends of the scale. A cohort with
    no spread scores the neutral 50.

    Raises:
        ValueError: if ``scale`` is not positive.
    """
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("scale must be a positive, finite number")
    numbers = _usable(values)
    if numbers is None:
        return [NEUTRAL_SCORE] * len(values)
    median = statistics.median(numbers)
    deviation = statistics.median([abs(value - median) for value in numbers])
    if deviation <= 0:
        return [NEUTRAL_SCORE] * len(numbers)
    scores = []
    for value in numbers:
        robust = 0.6745 * (value - median) / deviation
        scores.append(round(clamp_score(50.0 + 50.0 * robust / scale), 4))
    return scores


def _quantile(ordered: Sequence[float], quantile: float) -> float:
    """Linear-interpolated quantile of an already sorted sequence."""
    if not ordered:
        raise ValueError("cannot take a quantile of an empty sequence")
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def normalize_feature_cohort(
    values: Mapping[str, float],
    *,
    method: str = "percentile",
    cohort: str = "",
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> dict[str, CohortNormalization]:
    """Normalize every member of a cohort, keeping the method and cohort on each result.

    A cohort is the set a value is judged against - the same source type, period and entity kind
    - so counts from different sources never get compared directly. Each result records what it
    was compared with, which is what the explainability view shows.

    Raises:
        ValueError: for an unknown method or invalid quantiles.
    """
    names = list(values)
    raw = [float(values[name]) for name in names]
    if method == "winsorized_minmax":
        scores = winsorized_minmax_normalize(raw, lower_quantile, upper_quantile)
    else:
        scores = normalize_values(raw, method)
    return {
        name: CohortNormalization(
            raw_value=raw[index],
            normalized_value=scores[index],
            method=method,
            cohort=cohort,
            cohort_size=len(names),
            neutral=scores[index] == NEUTRAL_SCORE,
        )
        for index, name in enumerate(names)
    }
