"""Velocity: the direction and speed of activity over a longer window.

A straight line is fitted to the monthly counts (after a ``log1p`` transform, so the slope reads
as proportional growth per month rather than records per month, and busy topics do not dominate
quiet ones). The slope is the velocity; R-squared says how well a straight line describes the
series at all.

Velocity is deliberately refused when there are too few months: fitting a line through three
points produces a confident-looking number with nothing behind it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_WINDOW_MONTHS = 12
DEFAULT_MIN_OBSERVATIONS = 6


@dataclass(frozen=True)
class VelocityResult:
    """A fitted trend line over monthly activity."""

    slope: float
    intercept: float
    r_squared: float
    observations: int
    window_months: int
    total_activity: float

    @property
    def direction(self) -> str:
        """ "rising", "falling" or "flat"."""
        if self.slope > 0:
            return "rising"
        return "falling" if self.slope < 0 else "flat"

    @property
    def monthly_percent(self) -> float:
        """The slope expressed as an approximate percentage change per month."""
        return (math.exp(self.slope) - 1.0) * 100.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "slope": round(self.slope, 4),
            "monthly_percent": round(self.monthly_percent, 1),
            "r_squared": round(self.r_squared, 4),
            "observations": self.observations,
            "window_months": self.window_months,
            "total_activity": round(self.total_activity, 3),
            "direction": self.direction,
        }


def calculate_velocity(
    monthly: Sequence[float],
    *,
    window_months: int = DEFAULT_WINDOW_MONTHS,
    min_observations: int = DEFAULT_MIN_OBSERVATIONS,
) -> VelocityResult | None:
    """Fit a trend line to a monthly series (oldest first).

    Returns None when fewer than ``min_observations`` months are available, or when every month
    holds the same value (no line can be fitted through a flat series beyond slope zero, which
    is reported as a flat result rather than refused).

    Raises:
        ValueError: for negative or non-finite counts, or a non-positive window.
    """
    if window_months < 1 or min_observations < 2:
        raise ValueError("window_months must be positive and min_observations at least 2")
    values = [float(value) for value in monthly[-window_months:]]
    for value in values:
        if value < 0 or not math.isfinite(value):
            raise ValueError("monthly counts must be finite and not negative")
    if len(values) < min_observations:
        return None

    transformed = [math.log1p(value) for value in values]
    count = len(transformed)
    mean_x = (count - 1) / 2.0
    mean_y = sum(transformed) / count
    variance_x = sum((index - mean_x) ** 2 for index in range(count))
    covariance = sum((index - mean_x) * (y - mean_y) for index, y in enumerate(transformed))
    slope = covariance / variance_x if variance_x else 0.0
    intercept = mean_y - slope * mean_x

    residual = sum((y - (intercept + slope * index)) ** 2 for index, y in enumerate(transformed))
    total = sum((y - mean_y) ** 2 for y in transformed)
    r_squared = 1.0 - residual / total if total > 0 else 1.0
    return VelocityResult(
        slope=slope,
        intercept=intercept,
        r_squared=max(0.0, min(1.0, r_squared)),
        observations=count,
        window_months=window_months,
        total_activity=sum(values),
    )
