"""Growth between two periods.

Raw counts grow unevenly and small numbers are noisy: going from 1 record to 3 is a 200% jump
that means very little. Growth is therefore measured as a smoothed log ratio,

    ln((current + alpha) / (previous + alpha))        alpha defaults to 1

which is symmetric (a halving is the negative of a doubling), finite when a period has no
records, and far less excitable about small counts. The familiar percentage is reported
alongside it for display only; scoring uses the log value.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_ALPHA = 1.0
LOW_SAMPLE_THRESHOLD = 10


@dataclass(frozen=True)
class GrowthResult:
    """Growth between two periods, with everything needed to explain the number."""

    current: float
    previous: float
    log_growth: float
    display_percent: float
    sample_size: float
    alpha: float
    low_sample: bool

    @property
    def direction(self) -> str:
        """ "rising", "falling" or "flat"."""
        if self.log_growth > 0:
            return "rising"
        return "falling" if self.log_growth < 0 else "flat"

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "current": round(self.current, 3),
            "previous": round(self.previous, 3),
            "log_growth": round(self.log_growth, 4),
            "display_percent": round(self.display_percent, 1),
            "sample_size": round(self.sample_size, 3),
            "alpha": self.alpha,
            "low_sample": self.low_sample,
            "direction": self.direction,
        }


def _check(value: float, label: str) -> float:
    if value < 0:
        raise ValueError(f"{label} count cannot be negative")
    if not math.isfinite(value):
        raise ValueError(f"{label} count must be a finite number")
    return float(value)


def calculate_growth_rate(
    current: float,
    previous: float,
    *,
    alpha: float = DEFAULT_ALPHA,
    low_sample_threshold: int = LOW_SAMPLE_THRESHOLD,
) -> GrowthResult:
    """Growth from ``previous`` to ``current``.

    Args:
        current: activity in the recent period.
        previous: activity in the period before it.
        alpha: smoothing constant; larger values damp small-count swings further.
        low_sample_threshold: below this many records in total, the result is flagged as thin
            evidence rather than hidden, so the caller can lower confidence.

    Raises:
        ValueError: for negative or non-finite counts, or a non-positive alpha.
    """
    current = _check(current, "current")
    previous = _check(previous, "previous")
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError("alpha must be a positive, finite number")

    log_growth = math.log((current + alpha) / (previous + alpha))
    display = 100.0 * (current - previous) / max(previous, 1.0)
    sample = current + previous
    return GrowthResult(
        current=current,
        previous=previous,
        log_growth=log_growth,
        display_percent=display,
        sample_size=sample,
        alpha=alpha,
        low_sample=sample < low_sample_threshold,
    )


def calculate_recent_surge(
    current_quarter: float, previous_quarter: float, *, alpha: float = DEFAULT_ALPHA
) -> GrowthResult:
    """A sharp short-term jump: the same ratio applied to consecutive quarters.

    A surge on its own is not a trend. Consistency and velocity say whether it lasted; anomaly
    detection (Phase 8) says whether it was a one-off spike.
    """
    return calculate_growth_rate(current_quarter, previous_quarter, alpha=alpha)


def growth_from_series(
    values: Sequence[float], recent: int = 3, previous: int = 3, *, alpha: float = DEFAULT_ALPHA
) -> GrowthResult:
    """Growth between the last ``recent`` periods and the ``previous`` periods before them.

    Raises:
        ValueError: if either window length is not positive.
    """
    from cews.features.time_windows import split_series

    recent_values, previous_values = split_series(values, recent, previous)
    return calculate_growth_rate(sum(recent_values), sum(previous_values), alpha=alpha)
