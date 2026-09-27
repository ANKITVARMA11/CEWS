"""Baseline forecasts, and the shape every forecast takes.

Two deliberately unambitious models:

* **naive** - next month looks like this month;
* **seasonal naive** - next month looks like the same month last year.

They exist to be beaten. A clever model that cannot beat "next month looks like this month" is
not adding anything, and on short or noisy histories that is often the honest answer, so the
baselines are also what CEWS falls back to when there is too little history to fit anything else.

Every forecast carries an interval, because a single predicted number invites more confidence
than monthly counts deserve. Intervals come from how wrong the same model was on the history it
was fitted to, and are clipped at zero: activity counts cannot be negative.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

NAIVE = "naive"
SEASONAL_NAIVE = "seasonal_naive"
DEFAULT_SEASON = 12
CONFIDENCE_MULTIPLIER = 1.96  # about 95% for roughly normal residuals
MIN_HISTORY = 3


@dataclass(frozen=True)
class ForecastResult:
    """A forecast, its interval, and how it was produced."""

    model: str
    predictions: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    training_months: int
    season: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @property
    def horizon(self) -> int:
        """How many periods ahead this forecast reaches."""
        return len(self.predictions)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "model": self.model,
            "predictions": [round(value, 3) for value in self.predictions],
            "lower": [round(value, 3) for value in self.lower],
            "upper": [round(value, 3) for value in self.upper],
            "training_months": self.training_months,
            "season": self.season,
            "parameters": dict(self.parameters),
            "warnings": list(self.warnings),
        }


def check_series(series: Sequence[float], *, minimum: int = MIN_HISTORY) -> list[float]:
    """Validate a monthly series and return it as floats.

    Raises:
        ValueError: for negative or non-finite counts, or too little history.
    """
    values = [float(value) for value in series]
    for value in values:
        if not math.isfinite(value) or value < 0:
            raise ValueError("monthly counts must be finite and not negative")
    if len(values) < minimum:
        raise ValueError(f"need at least {minimum} months of history, got {len(values)}")
    return values


def count_noise(values: Sequence[float]) -> float:
    """The uncertainty a count of this size carries by itself.

    Monthly counts behave roughly like a Poisson process, where the spread is about the square
    root of the level. A smooth history would otherwise produce a zero-width interval, claiming
    a certainty that counting twenty things a month never gives.
    """
    recent = [value for value in values[-6:] if math.isfinite(value)]
    level = sum(recent) / len(recent) if recent else 0.0
    return math.sqrt(max(level, 0.0))


def interval(
    predictions: Sequence[float],
    residuals: Sequence[float],
    *,
    multiplier: float = CONFIDENCE_MULTIPLIER,
    minimum_spread: float = 0.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Bounds around a forecast, widening with the model's own past errors.

    The interval grows with the square root of how far ahead the period is, because errors
    accumulate. It never narrows below the noise a count of that size carries anyway. Lower
    bounds are clipped at zero.
    """
    usable = [float(value) for value in residuals if math.isfinite(value)]
    spread = max(statistics.pstdev(usable) if len(usable) > 1 else 0.0, minimum_spread)
    lower: list[float] = []
    upper: list[float] = []
    for step, value in enumerate(predictions, start=1):
        margin = multiplier * spread * math.sqrt(step)
        lower.append(max(0.0, value - margin))
        upper.append(max(0.0, value + margin))
    return tuple(lower), tuple(upper)


def build_naive_forecast(series: Sequence[float], horizon: int) -> ForecastResult:
    """Repeat the last observed month.

    Raises:
        ValueError: for an invalid series or a horizon below 1.
    """
    values = check_series(series, minimum=2)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    last = values[-1]
    predictions = tuple([last] * horizon)
    residuals = [values[index] - values[index - 1] for index in range(1, len(values))]
    lower, upper = interval(predictions, residuals, minimum_spread=count_noise(values))
    return ForecastResult(
        model=NAIVE,
        predictions=predictions,
        lower=lower,
        upper=upper,
        training_months=len(values),
        parameters={"last_value": round(last, 3)},
    )


def build_seasonal_naive_forecast(
    series: Sequence[float], horizon: int, *, season: int = DEFAULT_SEASON
) -> ForecastResult:
    """Repeat the same month from one season ago.

    Raises:
        ValueError: for an invalid series, a horizon below 1, or a season below 2.
        ValueError: if the history is shorter than one full season, which would make the model
            invent a seasonal pattern it has never seen.
    """
    if season < 2:
        raise ValueError("season must be at least 2")
    values = check_series(series, minimum=season)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    predictions = tuple(values[-season + (step - 1) % season] for step in range(1, horizon + 1))
    residuals = [values[index] - values[index - season] for index in range(season, len(values))]
    lower, upper = interval(predictions, residuals, minimum_spread=count_noise(values))
    return ForecastResult(
        model=SEASONAL_NAIVE,
        predictions=predictions,
        lower=lower,
        upper=upper,
        training_months=len(values),
        season=season,
        parameters={"season": season},
    )
