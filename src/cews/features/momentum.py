"""Momentum: how much stronger recent activity is than the period just before it.

Momentum uses the same smoothed log ratio as growth, over configurable windows (three months
against the previous three by default). It answers "is this heating up right now?", while
velocity answers "which way has it been going all year?".
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cews.features.growth import DEFAULT_ALPHA, calculate_growth_rate
from cews.features.time_windows import split_series

DEFAULT_RECENT_MONTHS = 3
DEFAULT_PREVIOUS_MONTHS = 3


@dataclass(frozen=True)
class MomentumResult:
    """Recent activity against the preceding window."""

    recent_activity: float
    previous_activity: float
    momentum: float
    recent_months: int
    previous_months: int
    months_available: int
    sample_size: float
    low_sample: bool

    @property
    def direction(self) -> str:
        """ "rising", "falling" or "flat"."""
        if self.momentum > 0:
            return "rising"
        return "falling" if self.momentum < 0 else "flat"

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "recent_activity": round(self.recent_activity, 3),
            "previous_activity": round(self.previous_activity, 3),
            "momentum": round(self.momentum, 4),
            "windows": f"{self.recent_months}m vs {self.previous_months}m",
            "months_available": self.months_available,
            "sample_size": round(self.sample_size, 3),
            "low_sample": self.low_sample,
            "direction": self.direction,
        }


def calculate_momentum(
    monthly: Sequence[float],
    *,
    recent_months: int = DEFAULT_RECENT_MONTHS,
    previous_months: int = DEFAULT_PREVIOUS_MONTHS,
    alpha: float = DEFAULT_ALPHA,
) -> MomentumResult:
    """Momentum from a monthly series (oldest first).

    A series shorter than both windows still produces a result, using the months that exist and
    reporting how many were available, so callers can lower confidence rather than lose the
    signal.

    Raises:
        ValueError: if either window length is not positive, or a count is negative or not
            finite (summing hides a negative, so the series is checked before it is split).
    """
    for value in monthly:
        if value < 0 or not math.isfinite(value):
            raise ValueError("monthly counts must be finite and not negative")
    recent_values, previous_values = split_series(monthly, recent_months, previous_months)
    growth = calculate_growth_rate(sum(recent_values), sum(previous_values), alpha=alpha)
    return MomentumResult(
        recent_activity=growth.current,
        previous_activity=growth.previous,
        momentum=growth.log_growth,
        recent_months=recent_months,
        previous_months=previous_months,
        months_available=len(monthly),
        sample_size=growth.sample_size,
        low_sample=growth.low_sample,
    )
