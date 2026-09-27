"""Unit tests for the statistical models and choosing between them."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from cews.forecasting.arima import ARIMA, build_arima_forecast
from cews.forecasting.baseline import NAIVE, SEASONAL_NAIVE, ForecastResult, build_naive_forecast
from cews.forecasting.exponential_smoothing import (
    HOLT,
    HOLT_WINTERS,
    build_exponential_smoothing_forecast,
)
from cews.forecasting.model_selection import (
    MIN_HISTORY_MONTHS,
    candidate_builders,
    select_best_model,
)

pytestmark = pytest.mark.unit

RISING = [2.0 + index for index in range(30)]
SEASONAL = [round(12 + 7 * math.sin(index * math.pi / 6)) for index in range(30)]
FLAT = [8.0] * 30


# --------------------------------------------------------------------------------------
# The models themselves
# --------------------------------------------------------------------------------------
def test_holt_follows_a_trend_instead_of_flattening() -> None:
    """The naive baseline repeats the last month; Holt keeps climbing."""
    holt = build_exponential_smoothing_forecast(RISING, 3)
    assert holt is not None and holt.model == HOLT
    assert holt.predictions[0] > RISING[-1]
    assert holt.predictions[2] > holt.predictions[0]


def test_holt_winters_reproduces_a_yearly_shape() -> None:
    forecast = build_exponential_smoothing_forecast(SEASONAL, 12, season=12)
    assert forecast is not None and forecast.model == HOLT_WINTERS
    assert forecast.season == 12
    assert max(forecast.predictions) > min(forecast.predictions)  # it varies through the year


def test_holt_winters_refuses_without_two_full_cycles() -> None:
    """One cycle cannot tell a season from ordinary movement."""
    assert build_exponential_smoothing_forecast(SEASONAL[:18], 3, season=12) is None


def test_arima_fits_a_trending_series() -> None:
    forecast = build_arima_forecast(RISING, 3)
    assert forecast is not None and forecast.model == ARIMA
    assert forecast.predictions[0] > RISING[-1]
    assert "order" in forecast.parameters and "aic" in forecast.parameters


def test_the_statistical_models_need_enough_history() -> None:
    assert build_exponential_smoothing_forecast(RISING[:4], 3) is None
    assert build_arima_forecast(RISING[:6], 3) is None


def test_forecasts_are_never_negative() -> None:
    """Counts cannot go below zero, however steep the decline."""
    falling = [max(0.0, 30.0 - 2 * index) for index in range(30)]
    for forecast in (
        build_exponential_smoothing_forecast(falling, 6),
        build_arima_forecast(falling, 6),
    ):
        assert forecast is not None
        assert all(value >= 0.0 for value in forecast.predictions)
        assert all(value >= 0.0 for value in forecast.lower)


def test_a_missing_library_is_an_ordinary_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without statsmodels these models bow out; the pass does not fail."""
    import cews.forecasting.arima as arima_module
    import cews.forecasting.exponential_smoothing as smoothing_module

    monkeypatch.setattr(smoothing_module, "statsmodels_available", lambda: False)
    monkeypatch.setattr(arima_module, "statsmodels_available", lambda: False)
    assert build_exponential_smoothing_forecast(RISING, 3) is None
    assert build_arima_forecast(RISING, 3) is None
    selection = select_best_model(RISING, horizon=3)
    assert selection.model in (NAIVE, SEASONAL_NAIVE)  # the baselines still work


@pytest.mark.parametrize("horizon", [0, -2])
def test_models_refuse_a_horizon_that_looks_backwards(horizon: int) -> None:
    for build in (build_exponential_smoothing_forecast, build_arima_forecast):
        with pytest.raises(ValueError, match="horizon"):
            build(RISING, horizon)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
def test_a_trend_is_forecast_by_a_model_that_can_follow_one() -> None:
    selection = select_best_model(RISING, horizon=3)
    assert selection.model in (HOLT, HOLT_WINTERS, ARIMA)
    assert selection.metric is not None and selection.metric < 1.0
    # the comparison is against the naive model's own replay, not against a fixed MASE of 1
    assert "better than the naive baseline" in selection.reason
    naive = selection.backtests[NAIVE]
    assert naive.metrics is not None and naive.metrics.mase is not None
    assert selection.metric < naive.metrics.mase


def test_a_seasonal_series_is_forecast_seasonally() -> None:
    selection = select_best_model(SEASONAL, horizon=6)
    assert selection.model in (SEASONAL_NAIVE, HOLT_WINTERS)
    assert selection.forecast is not None and selection.forecast.horizon == 6


def test_every_candidate_is_scored_and_kept() -> None:
    """The comparison is part of the record, not just the winner."""
    selection = select_best_model(RISING, horizon=3)
    assert set(selection.backtests) == set(candidate_builders())
    payload = selection.as_dict()
    assert set(payload["candidates"]) == set(candidate_builders())
    assert payload["selected"] == selection.model


def test_the_simplest_model_wins_a_tie() -> None:
    """A complicated model that is no better is worse: harder to explain, easier to overfit."""

    def clone(_history: Sequence[float], horizon: int) -> ForecastResult:
        return build_naive_forecast(RISING, horizon)

    builders = {NAIVE: clone, ARIMA: clone, HOLT: clone}
    selection = select_best_model(RISING, horizon=3, builders=builders)
    assert selection.model == NAIVE


def test_a_flat_history_is_compared_by_average_miss() -> None:
    """MASE divides by the naive error, which is zero when nothing ever changes."""
    selection = select_best_model(FLAT, horizon=3)
    assert selection.metric_name == "mae"
    assert selection.model and selection.forecast is not None
    assert "no naive error to measure against" in selection.reason
    assert any("too flat" in warning for warning in selection.warnings)


def test_too_little_history_produces_no_forecast_at_all() -> None:
    """A forecast from six months would look identical to one from six years."""
    selection = select_best_model([5.0] * 6, horizon=3)
    assert selection.forecast is None and selection.model == ""
    assert "at least" in selection.reason and selection.warnings


def test_the_history_threshold_is_the_documented_one() -> None:
    assert select_best_model([5.0] * (MIN_HISTORY_MONTHS - 1), horizon=3).forecast is None
    assert select_best_model(RISING[:MIN_HISTORY_MONTHS], horizon=3).forecast is not None


def test_when_no_model_can_be_replayed_the_baseline_is_used() -> None:
    def useless(_history: Sequence[float], _horizon: int) -> ForecastResult | None:
        return None

    selection = select_best_model(RISING, horizon=3, builders={HOLT: useless, ARIMA: useless})
    assert selection.model == NAIVE
    assert selection.forecast is not None  # a forecast is still produced
    assert "naive baseline" in selection.reason


def test_the_selection_explains_itself() -> None:
    payload = select_best_model(RISING, horizon=3).as_dict()
    assert payload["metric_name"] in ("mase", "mae")
    assert payload["reason"] and payload["forecast"] is not None
    import json

    json.dumps(payload)


@pytest.mark.parametrize("horizon", [0, -1])
def test_selection_refuses_a_horizon_that_looks_backwards(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        select_best_model(RISING, horizon=horizon)
