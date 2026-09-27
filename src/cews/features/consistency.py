"""Consistency: whether growth was sustained or came from a single jump.

Consistency is the share of month-to-month changes that were increases, from 0 (falling every
month) to 1 (rising every month). A topic that jumped once and flattened scores low even though
its growth looks large, which is what separates a real trend from a one-off spike.

Volatility (the spread of the month-to-month changes) is reported alongside it, because steady
growth and a see-saw can share the same average.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MIN_COMPARISONS = 2


@dataclass(frozen=True)
class ConsistencyResult:
    """How steady a series was, month to month."""

    consistency: float
    increases: int
    decreases: int
    unchanged: int
    comparisons: int
    volatility: float
    largest_share: float

    @property
    def single_spike(self) -> bool:
        """True when most of the activity landed in one month and growth was not sustained."""
        return self.largest_share >= 0.5 and self.consistency < 0.5

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "consistency": round(self.consistency, 4),
            "increases": self.increases,
            "decreases": self.decreases,
            "unchanged": self.unchanged,
            "comparisons": self.comparisons,
            "volatility": round(self.volatility, 4),
            "largest_month_share": round(self.largest_share, 4),
            "single_spike": self.single_spike,
        }


def calculate_consistency(monthly: Sequence[float]) -> ConsistencyResult | None:
    """Consistency of a monthly series (oldest first).

    Returns None when there are fewer than three months, which is too few to say whether
    anything was sustained.

    Raises:
        ValueError: for negative or non-finite counts.
    """
    values = [float(value) for value in monthly]
    for value in values:
        if value < 0 or not math.isfinite(value):
            raise ValueError("monthly counts must be finite and not negative")
    changes = [later - earlier for earlier, later in zip(values, values[1:], strict=False)]
    if len(changes) < MIN_COMPARISONS:
        return None

    increases = sum(1 for change in changes if change > 0)
    decreases = sum(1 for change in changes if change < 0)
    total = sum(values)
    return ConsistencyResult(
        consistency=increases / len(changes),
        increases=increases,
        decreases=decreases,
        unchanged=len(changes) - increases - decreases,
        comparisons=len(changes),
        volatility=statistics.pstdev(changes) if len(changes) > 1 else 0.0,
        largest_share=(max(values) / total) if total > 0 else 0.0,
    )
