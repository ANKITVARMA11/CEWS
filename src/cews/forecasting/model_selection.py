"""Choosing a forecasting model by how it would have performed, not by how well it fits.

Every candidate is replayed over the history (see :mod:`cews.forecasting.backtesting`) and the
one with the lowest MASE wins. MASE compares each model against "next month looks like this
month", so the winner has to earn its place: if nothing beats that baseline, the baseline is
what gets used and the reason is recorded.

Deliberately, the simplest model wins ties. A complicated model that is no better is worse,
because it is harder to explain and more likely to be fitting noise.

With too little history nothing is selected at all. A forecast from six months of data would
look identical to one built on six years, and nothing on the dashboard would say otherwise.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cews.forecasting.arima import ARIMA, build_arima_forecast
from cews.forecasting.backtesting import (
    BacktestResult,
    ModelBuilder,
    rolling_origin_backtest,
)
from cews.forecasting.baseline import (
    NAIVE,
    SEASONAL_NAIVE,
    ForecastResult,
    build_naive_forecast,
    build_seasonal_naive_forecast,
)
from cews.forecasting.exponential_smoothing import (
    HOLT,
    HOLT_WINTERS,
    build_exponential_smoothing_forecast,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_SEASON = 12
DEFAULT_HORIZON = 6
MIN_HISTORY_MONTHS = 15  # a year of training plus a few months to score against
TIE_TOLERANCE = 1e-9
# Simplest first: ties are resolved in this order.
MODEL_ORDER: tuple[str, ...] = (NAIVE, SEASONAL_NAIVE, HOLT, HOLT_WINTERS, ARIMA)


def candidate_builders(season: int = DEFAULT_SEASON) -> dict[str, ModelBuilder]:
    """Every model worth trying, keyed by name.

    Models that need more history than is available, or a library that is not installed, simply
    return None when asked and drop out of the comparison.
    """
    return {
        NAIVE: lambda history, horizon: build_naive_forecast(history, horizon),
        SEASONAL_NAIVE: lambda history, horizon: (
            build_seasonal_naive_forecast(history, horizon, season=season)
            if len(history) >= season
            else None
        ),
        HOLT: lambda history, horizon: build_exponential_smoothing_forecast(history, horizon),
        HOLT_WINTERS: lambda history, horizon: build_exponential_smoothing_forecast(
            history, horizon, season=season
        ),
        ARIMA: lambda history, horizon: build_arima_forecast(history, horizon),
    }


@dataclass
class ModelSelection:
    """The chosen model, the forecast it produced, and how every candidate scored."""

    model: str
    forecast: ForecastResult | None
    backtests: dict[str, BacktestResult] = field(default_factory=dict)
    reason: str = ""
    metric_name: str = "mase"
    warnings: list[str] = field(default_factory=list)

    @property
    def metric(self) -> float | None:
        """The winning model's score on whichever measure was used to choose it."""
        result = self.backtests.get(self.model)
        if result is None or result.metrics is None:
            return None
        return result.metrics.mase if self.metric_name == "mase" else result.metrics.mae

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the forecast."""
        return {
            "selected": self.model,
            "reason": self.reason,
            "metric_name": self.metric_name,
            "metric": None if self.metric is None else round(self.metric, 4),
            "candidates": {
                name: result.as_dict() for name, result in sorted(self.backtests.items())
            },
            "forecast": self.forecast.as_dict() if self.forecast else None,
            "warnings": list(self.warnings),
        }


def select_best_model(
    series: Sequence[float],
    *,
    horizon: int = DEFAULT_HORIZON,
    season: int = DEFAULT_SEASON,
    backtest_horizon: int = 3,
    minimum_training: int = 12,
    minimum_history: int = MIN_HISTORY_MONTHS,
    builders: dict[str, ModelBuilder] | None = None,
) -> ModelSelection:
    """Pick the model that did best on unseen history, and forecast with it.

    Args:
        series: the monthly history, oldest first.
        horizon: how many months the final forecast covers.
        season: the seasonal period, for the seasonal models and MASE.
        backtest_horizon: how far ahead each replay forecasts.
        minimum_training: months a model must see before the first replay.
        minimum_history: months required before forecasting at all.
        builders: override the candidate models.

    Raises:
        ValueError: for an invalid series or a horizon below 1.
    """
    values = [float(value) for value in series]
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    selection = ModelSelection(model="", forecast=None)
    if len(values) < minimum_history:
        selection.reason = (
            f"only {len(values)} month(s) of history; at least {minimum_history} are needed "
            "before a forecast means anything"
        )
        selection.warnings.append(selection.reason)
        return selection

    candidates = builders if builders is not None else candidate_builders(season)
    for name, build in candidates.items():
        selection.backtests[name] = rolling_origin_backtest(
            values,
            build,
            model=name,
            horizon=backtest_horizon,
            minimum_training=minimum_training,
            season=1,
        )

    usable = [
        (name, result.metrics)
        for name, result in selection.backtests.items()
        if result.usable and result.metrics is not None
    ]
    scored = [(metrics.mase, name) for name, metrics in usable if metrics.mase is not None]
    order = {name: index for index, name in enumerate(MODEL_ORDER)}

    if scored:
        selection.metric_name = "mase"
        best_metric = min(metric for metric, _ in scored)
        # Compared against the naive model's own replay, not against MASE 1: MASE scales by a
        # one-step-ahead naive forecast, so at a three-month horizon a score above 1 is ordinary
        # and says nothing about whether this model beat the baseline.
        naive_result = selection.backtests.get(NAIVE)
        naive_metric = (
            naive_result.metrics.mase
            if naive_result and naive_result.metrics and naive_result.usable
            else None
        )
        if naive_metric is None:
            comparison = "the naive baseline could not be replayed for comparison"
        elif best_metric < naive_metric - TIE_TOLERANCE:
            comparison = f"better than the naive baseline's {naive_metric:.2f}"
        else:
            comparison = f"no better than the naive baseline's {naive_metric:.2f}"
        summary = f"lowest backtest error (MASE {best_metric:.2f}, {comparison})"
        winners = [name for metric, name in scored if metric <= best_metric + TIE_TOLERANCE]
    elif usable:
        # MASE divides by the error a naive forecast would have made. On a history that never
        # changes from month to month that error is zero, so MASE says nothing and the plain
        # average miss is used instead.
        selection.metric_name = "mae"
        best_metric = min(metrics.mae for _, metrics in usable)
        summary = (
            f"lowest average miss (MAE {best_metric:.2f}); this history barely changes from "
            "month to month, so there is no naive error to measure against"
        )
        winners = [name for name, metrics in usable if metrics.mae <= best_metric + TIE_TOLERANCE]
        selection.warnings.append(
            "models were compared by average miss because the history is too flat to scale "
            "against a naive forecast"
        )
    else:
        selection.model = NAIVE
        selection.reason = "no model could be replayed on this history; using the naive baseline"
        selection.warnings.append(selection.reason)
        winners = []

    if winners:
        # Simplest model within a whisker of the best, so a tie never buys complexity.
        chosen = min(winners, key=lambda name: order.get(name, len(order)))
        selection.model = chosen
        selection.reason = f"{chosen} had the {summary}"

    chosen_builder = candidates.get(selection.model)
    forecast = chosen_builder(values, horizon) if chosen_builder is not None else None
    if forecast is None:  # the winner cannot fit the full history after all
        forecast = build_naive_forecast(values, horizon)
        selection.warnings.append(
            f"{selection.model} could not be fitted to the full history; "
            "the naive baseline was used instead"
        )
        selection.model = NAIVE
    selection.forecast = forecast
    LOGGER.debug("selected %s: %s", selection.model, selection.reason)
    return selection
