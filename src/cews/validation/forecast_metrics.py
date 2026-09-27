"""How wrong a forecast was.

Five measures, because no single number describes a forecast fairly:

* **MAE** - average error in records. Easy to explain, but not comparable between a busy topic
  and a quiet one.
* **RMSE** - punishes large misses harder than small ones.
* **MASE** - the error divided by what a naive "next month looks like this month" forecast would
  have scored. Below 1 means the model beat that; above 1 means it did not. This is the measure
  model selection uses, because it is comparable across topics of any size **and** it survives
  months with no activity.
* **SMAPE** - a percentage that stays finite when the truth is zero, unlike plain MAPE.
* **Directional accuracy** - how often the forecast got the direction of change right, which is
  usually what a reader actually wants to know.

MAPE is deliberately absent: activity counts are frequently zero, and dividing by zero makes it
either infinite or quietly misleading.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

EPSILON = 1e-9


@dataclass(frozen=True)
class ForecastMetrics:
    """Error measures for one set of predictions."""

    mae: float
    rmse: float
    mase: float | None
    smape: float
    directional_accuracy: float | None
    observations: int

    @property
    def beats_naive(self) -> bool | None:
        """True when the model did better than repeating the last observed value."""
        return None if self.mase is None else self.mase < 1.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the forecast."""
        return {
            "mae": round(self.mae, 4),
            "rmse": round(self.rmse, 4),
            "mase": None if self.mase is None else round(self.mase, 4),
            "smape": round(self.smape, 2),
            "directional_accuracy": (
                None if self.directional_accuracy is None else round(self.directional_accuracy, 4)
            ),
            "observations": self.observations,
            "beats_naive": self.beats_naive,
        }


def _check(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must be the same length")
    if not actual:
        raise ValueError("cannot measure error without observations")
    for series, label in ((actual, "actual"), (predicted, "predicted")):
        for value in series:
            if not math.isfinite(value):
                raise ValueError(f"{label} values must be finite numbers")


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Average absolute difference, in records."""
    _check(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def root_mean_squared_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Square root of the average squared difference; large misses count for more."""
    _check(actual, predicted)
    return math.sqrt(
        sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual)
    )


def symmetric_mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric percentage error, finite even when the truth is zero (0 to 200)."""
    _check(actual, predicted)
    total = 0.0
    for a, p in zip(actual, predicted, strict=True):
        denominator = (abs(a) + abs(p)) / 2.0
        total += 0.0 if denominator < EPSILON else abs(a - p) / denominator
    return 100.0 * total / len(actual)


def mean_absolute_scaled_error(
    actual: Sequence[float], predicted: Sequence[float], training: Sequence[float], season: int = 1
) -> float | None:
    """Error relative to a naive forecast fitted on the training history.

    Returns None when the training history never changes from one period to the next, because
    there is then no naive error to scale against (a flat series is not a useful yardstick).

    Raises:
        ValueError: for mismatched lengths, non-finite values, or a season below 1.
    """
    _check(actual, predicted)
    if season < 1:
        raise ValueError("season must be at least 1")
    if len(training) <= season:
        return None
    naive_errors = [
        abs(training[index] - training[index - season]) for index in range(season, len(training))
    ]
    scale = sum(naive_errors) / len(naive_errors)
    if scale < EPSILON:
        return None
    return mean_absolute_error(actual, predicted) / scale


def directional_accuracy(
    actual: Sequence[float], predicted: Sequence[float], last_observed: float
) -> float | None:
    """How often the forecast got the direction of change right, from 0 to 1.

    Periods where the truth did not move are skipped: there is no direction to get right.
    Returns None when nothing moved at all.
    """
    _check(actual, predicted)
    hits = 0
    counted = 0
    previous = last_observed
    for a, p in zip(actual, predicted, strict=True):
        actual_direction = (a > previous) - (a < previous)
        if actual_direction != 0:
            counted += 1
            predicted_direction = (p > previous) - (p < previous)
            hits += int(actual_direction == predicted_direction)
        previous = a
    return None if counted == 0 else hits / counted


def calculate_forecast_metrics(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    training: Sequence[float] = (),
    season: int = 1,
    last_observed: float | None = None,
) -> ForecastMetrics:
    """Every measure for one set of predictions.

    Args:
        actual: what happened.
        predicted: what was forecast.
        training: the history the model was fitted on, used to scale MASE.
        season: the seasonal period for the naive yardstick (1 for month-on-month).
        last_observed: the value before the forecast window, for directional accuracy.

    Raises:
        ValueError: for mismatched lengths or non-finite values.
    """
    return ForecastMetrics(
        mae=mean_absolute_error(actual, predicted),
        rmse=root_mean_squared_error(actual, predicted),
        mase=mean_absolute_scaled_error(actual, predicted, training, season) if training else None,
        smape=symmetric_mape(actual, predicted),
        directional_accuracy=(
            None
            if last_observed is None
            else directional_accuracy(actual, predicted, last_observed)
        ),
        observations=len(actual),
    )
