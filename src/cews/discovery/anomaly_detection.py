"""Spotting unusual months, and saying what kind of unusual they are.

Finding an outlier is the easy half. The useful half is deciding what it means, because "patents
tripled last month" can be any of these:

* a **one-off spike** - one busy month, back to normal after (a patent family published at once);
* **persistent momentum** - the level stepped up and stayed up, which is a real change;
* a **seasonal pattern** - it happens every year at this time;
* a **collection gap** - every source went quiet at once, which says more about our pipeline
  than about the field;
* the **start of a trend** - a rise that is still climbing.

Calling all five "an anomaly" would put noise in front of leadership, so each is labelled and the
label travels with the record.

The default test is the robust z-score, ``0.6745 (value - median) / MAD``. It uses the median and
the median absolute deviation instead of the mean and standard deviation, because one enormous
month drags a mean upward far enough to hide itself. A rolling version compares against recent
months rather than the whole history, and a seasonal version compares each month against the
same month in other years.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

LOGGER = logging.getLogger(__name__)

ROBUST_CONSTANT = 0.6745  # makes the robust z-score comparable to an ordinary one
DEFAULT_THRESHOLD = 3.5
DEFAULT_ROLLING_WINDOW = 12
MIN_HISTORY = 6
SEASONS_FOR_SEASONAL = 2
SUSTAINED_MONTHS = 2
RETURN_TOLERANCE = 0.5  # how close to the old level counts as "back to normal"
MIN_SEASONAL_BASELINE = 2.0  # below this a "season" is just small numbers moving around
CONVINCING_COUNT = 10.0  # a jump of two records is arithmetically large and practically small


class AnomalyKind(StrEnum):
    """What an unusual month turned out to be."""

    ONE_TIME_SPIKE = "one_time_spike"
    PERSISTENT_MOMENTUM = "persistent_momentum"
    SEASONAL_PATTERN = "seasonal_pattern"
    COLLECTION_GAP = "collection_gap"
    EMERGING_TREND = "emerging_trend"
    DROP = "drop"


@dataclass(frozen=True)
class Anomaly:
    """One unusual month, with the reason it was flagged and what it looks like."""

    index: int
    period: date | None
    observed: float
    expected: float
    expected_lower: float
    expected_upper: float
    deviation: float
    method: str
    kind: AnomalyKind
    confidence: float
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """ "above" or "below" the expected range."""
        return "above" if self.observed > self.expected else "below"

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the anomaly."""
        return {
            "period": self.period.isoformat() if self.period else None,
            "observed": round(self.observed, 3),
            "expected": round(self.expected, 3),
            "expected_range": [round(self.expected_lower, 3), round(self.expected_upper, 3)],
            "deviation": round(self.deviation, 3),
            "method": self.method,
            "kind": self.kind.value,
            "confidence": round(self.confidence, 2),
            "direction": self.direction,
            "explanation": self.explanation,
            "evidence": dict(self.evidence),
        }


def median_absolute_deviation(values: Sequence[float]) -> float:
    """Median distance from the median; unmoved by a few extreme values."""
    if not values:
        return 0.0
    middle = statistics.median(values)
    return statistics.median([abs(value - middle) for value in values])


def spread_of(history: Sequence[float]) -> float:
    """How much a series moves about, in units a z-score can divide by.

    The median absolute deviation comes first, because one enormous month barely moves it. When
    that is zero the ordinary spread is tried, and when that is zero too the count noise is used
    (about the square root of the level). Without that last step a perfectly steady history
    would have no spread at all, and a collapse from thirty records to one could never be
    flagged - the most obvious anomaly there is would be the one it missed.
    """
    numbers = [float(value) for value in history]
    if not numbers:
        return 0.0
    deviation = median_absolute_deviation(numbers)
    if deviation > 0:
        return deviation / ROBUST_CONSTANT
    spread = statistics.pstdev(numbers) if len(numbers) > 1 else 0.0
    if spread > 0:
        return spread
    return math.sqrt(max(statistics.median(numbers), 1.0))


def robust_zscores(values: Sequence[float]) -> list[float]:
    """Robust z-score of each value against the whole series."""
    numbers = [float(value) for value in values]
    if len(numbers) < 2:
        return [0.0] * len(numbers)
    middle = statistics.median(numbers)
    spread = spread_of(numbers)
    if spread <= 0:
        return [0.0] * len(numbers)
    return [ROBUST_CONSTANT * (value - middle) / (spread * ROBUST_CONSTANT) for value in numbers]


def rolling_robust_zscores(
    values: Sequence[float], window: int = DEFAULT_ROLLING_WINDOW
) -> list[float]:
    """Robust z-score of each month against the ``window`` months before it.

    The first months have nothing to compare against and score zero. Comparing against recent
    history rather than the whole series means a topic that grew steadily for two years does not
    flag every recent month simply for being larger than the distant past.

    Raises:
        ValueError: if the window is below 2.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    numbers = [float(value) for value in values]
    scores = [0.0] * len(numbers)
    for index in range(len(numbers)):
        history = numbers[max(0, index - window) : index]
        if len(history) < MIN_HISTORY:
            continue
        middle = statistics.median(history)
        spread = spread_of(history)
        scores[index] = (numbers[index] - middle) / spread if spread > 0 else 0.0
    return scores


def seasonal_residuals(values: Sequence[float], season: int = 12) -> list[float] | None:
    """Each month minus the average of the same month in other years.

    Returns None without at least two full cycles, because one cycle cannot tell a season from
    an ordinary rise.
    """
    numbers = [float(value) for value in values]
    if season < 2 or len(numbers) < season * SEASONS_FOR_SEASONAL:
        return None
    residuals: list[float] = []
    for index, value in enumerate(numbers):
        same_month = [
            numbers[other]
            for other in range(index % season, len(numbers), season)
            if other != index
        ]
        residuals.append(value - (sum(same_month) / len(same_month)) if same_month else 0.0)
    return residuals


def _classify(
    values: Sequence[float],
    index: int,
    *,
    season: int,
    other_sources_quiet: bool,
) -> tuple[AnomalyKind, str]:
    """Work out what kind of unusual month this is."""
    observed = values[index]
    before = values[max(0, index - season) : index]
    after = values[index + 1 :]
    baseline = statistics.median(before) if before else observed

    if other_sources_quiet:
        return (
            AnomalyKind.COLLECTION_GAP,
            "every source went quiet in the same month, which points at collection rather than "
            "the field",
        )
    if observed < baseline:
        return AnomalyKind.DROP, "activity fell well below its usual level"

    seasonal = seasonal_residuals(values, season)
    if seasonal is not None and baseline >= MIN_SEASONAL_BASELINE:
        # A quiet topic that has always been near zero cannot have a season: without a baseline
        # to compare against, "the same month is also busy" is true of any two small numbers.
        same_month = [
            values[other] for other in range(index % season, len(values), season) if other != index
        ]
        if same_month and statistics.median(same_month) >= max(
            MIN_SEASONAL_BASELINE, baseline * 1.5
        ):
            return (
                AnomalyKind.SEASONAL_PATTERN,
                "the same month in other years is also busy, so this looks seasonal",
            )

    if not after:
        return (
            AnomalyKind.EMERGING_TREND,
            "the most recent month is well above the usual level; whether it lasts is not yet "
            "known",
        )
    sustained = after[:SUSTAINED_MONTHS]
    stepped_up = all(
        value >= baseline + (observed - baseline) * RETURN_TOLERANCE for value in sustained
    )
    if stepped_up:
        return (
            AnomalyKind.PERSISTENT_MOMENTUM,
            "activity stepped up and stayed up, so this is a change in level rather than a spike",
        )
    return (
        AnomalyKind.ONE_TIME_SPIKE,
        "one busy month with activity back to its usual level afterwards, so it is not a trend",
    )


def detect_activity_anomalies(
    values: Sequence[float],
    *,
    periods: Sequence[date] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    window: int = DEFAULT_ROLLING_WINDOW,
    season: int = 12,
    quiet_months: Sequence[int] = (),
    method: str = "rolling_robust_zscore",
) -> list[Anomaly]:
    """Find unusual months in a series and say what kind each one is.

    Args:
        values: monthly activity, oldest first.
        periods: the month each value belongs to, for the record.
        threshold: how many robust deviations count as unusual.
        window: how many months each value is compared against.
        season: the seasonal period, for the seasonal check.
        quiet_months: indexes where every other source also went quiet, which marks a collection
            gap rather than a change in the field.
        method: "rolling_robust_zscore" (default) or "robust_zscore" for the whole series.

    Raises:
        ValueError: for negative or non-finite counts, a threshold below 0, or an unknown method.
    """
    numbers = [float(value) for value in values]
    for value in numbers:
        if not math.isfinite(value) or value < 0:
            raise ValueError("monthly counts must be finite and not negative")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if periods is not None and len(periods) != len(numbers):
        raise ValueError("periods must line up with values")
    if len(numbers) < MIN_HISTORY:
        return []

    if method == "rolling_robust_zscore":
        scores = rolling_robust_zscores(numbers, window)
    elif method == "robust_zscore":
        scores = robust_zscores(numbers)
    else:
        raise ValueError(f"unknown method {method!r}")

    quiet = set(quiet_months)
    found: list[Anomaly] = []
    for index, score in enumerate(scores):
        if abs(score) < threshold:
            continue
        history = numbers[max(0, index - window) : index] or numbers[:index] or [numbers[index]]
        middle = statistics.median(history)
        spread = spread_of(history)
        kind, explanation = _classify(
            numbers, index, season=season, other_sources_quiet=index in quiet
        )
        found.append(
            Anomaly(
                index=index,
                period=periods[index] if periods else None,
                observed=numbers[index],
                expected=middle,
                expected_lower=max(0.0, middle - threshold * spread),
                expected_upper=middle + threshold * spread,
                deviation=score,
                method=method,
                kind=kind,
                # Small counts move a long way in relative terms without meaning much, so the
                # confidence in an anomaly is held back until the numbers are worth noticing.
                confidence=min(100.0, 100.0 * min(1.0, abs(score) / (threshold * 2)))
                * min(1.0, max(numbers[index], middle) / CONVINCING_COUNT),
                explanation=explanation,
                evidence={
                    "months_compared": len(history),
                    "median_of_history": round(middle, 3),
                    "threshold": threshold,
                },
            )
        )
    LOGGER.debug("found %d anomalous month(s)", len(found))
    return found
