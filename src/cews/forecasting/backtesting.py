"""Rolling-origin backtesting: how a model would have done, using only what it knew at the time.

The series is replayed. At each origin the model sees only the months before it, forecasts the
next few, and is scored against what actually happened. The origin then moves forward and it
repeats. Averaging those errors says how the model behaves on data it has never seen, which is
the only honest way to compare models.

**No future data leaks backwards.** Every fit is given a prefix of the series and nothing else,
which is what separates a backtest from fitting a curve to the whole history and admiring it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from cews.forecasting.baseline import ForecastResult
from cews.validation.forecast_metrics import ForecastMetrics, calculate_forecast_metrics

LOGGER = logging.getLogger(__name__)

# A model builder: (history, horizon) -> forecast, or None when it cannot be fitted.
ModelBuilder = Callable[[Sequence[float], int], ForecastResult | None]

DEFAULT_MIN_TRAIN = 12
DEFAULT_HORIZON = 3
MIN_ORIGINS = 2


@dataclass(frozen=True)
class BacktestFold:
    """One replay: what the model knew, what it said, and what happened."""

    origin: int
    training_months: int
    actual: tuple[float, ...]
    predicted: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "origin": self.origin,
            "training_months": self.training_months,
            "actual": [round(value, 3) for value in self.actual],
            "predicted": [round(value, 3) for value in self.predicted],
        }


@dataclass
class BacktestResult:
    """How one model did across every replay."""

    model: str
    metrics: ForecastMetrics | None
    folds: list[BacktestFold] = field(default_factory=list)
    failures: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when the model produced enough replays to be judged."""
        return self.metrics is not None and len(self.folds) >= MIN_ORIGINS

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the forecast."""
        return {
            "model": self.model,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "folds": len(self.folds),
            "failures": self.failures,
            "usable": self.usable,
            "warnings": list(self.warnings),
        }


def rolling_origin_backtest(
    series: Sequence[float],
    build: ModelBuilder,
    *,
    model: str,
    horizon: int = DEFAULT_HORIZON,
    minimum_training: int = DEFAULT_MIN_TRAIN,
    step: int = 1,
    season: int = 1,
) -> BacktestResult:
    """Replay a model over a series and measure how wrong it was.

    Args:
        series: the full monthly history, oldest first.
        build: makes a forecast from a prefix of the history.
        model: the model's name, for the result.
        horizon: how many months each replay forecasts.
        minimum_training: months the model must see before the first replay.
        step: how far the origin moves each time.
        season: the seasonal period used to scale MASE.

    Raises:
        ValueError: for a horizon, training length or step below 1.
    """
    if horizon < 1 or minimum_training < 1 or step < 1:
        raise ValueError("horizon, minimum_training and step must be positive")
    values = [float(value) for value in series]
    result = BacktestResult(model=model, metrics=None)

    if len(values) < minimum_training + horizon:
        result.warnings.append(
            f"{len(values)} month(s) of history is too short to replay {model} "
            f"({minimum_training + horizon} needed)"
        )
        return result

    actuals: list[float] = []
    predictions: list[float] = []
    last_observed: float | None = None
    for origin in range(minimum_training, len(values) - horizon + 1, step):
        history = values[:origin]  # everything the model is allowed to know
        future = values[origin : origin + horizon]
        try:
            forecast = build(history, horizon)
        except Exception as exc:  # a model that cannot fit this prefix is not fatal
            LOGGER.debug("%s failed at origin %d: %s", model, origin, exc)
            result.failures += 1
            continue
        if forecast is None or forecast.horizon != horizon:
            result.failures += 1
            continue
        result.folds.append(
            BacktestFold(
                origin=origin,
                training_months=len(history),
                actual=tuple(future),
                predicted=tuple(forecast.predictions),
            )
        )
        actuals.extend(future)
        predictions.extend(forecast.predictions)
        if last_observed is None:
            last_observed = history[-1]

    if len(result.folds) < MIN_ORIGINS:
        result.warnings.append(
            f"{model} could only be replayed {len(result.folds)} time(s); at least "
            f"{MIN_ORIGINS} are needed to judge it"
        )
        return result

    result.metrics = calculate_forecast_metrics(
        actuals,
        predictions,
        training=values[:minimum_training],
        season=season,
        last_observed=last_observed,
    )
    return result
