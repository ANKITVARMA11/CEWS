"""Unit tests for the baseline forecasts and their prediction intervals."""

from __future__ import annotations

import json
import math

import pytest

from cews.forecasting.baseline import (
    CONFIDENCE_MULTIPLIER,
    NAIVE,
    SEASONAL_NAIVE,
    build_naive_forecast,
    build_seasonal_naive_forecast,
    check_series,
    count_noise,
    interval,
)

pytestmark = pytest.mark.unit

RISING = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
ALTERNATING = [5.0, 9.0] * 6


# --------------------------------------------------------------------------------------
# Naive
# --------------------------------------------------------------------------------------
def test_naive_repeats_the_last_month() -> None:
    result = build_naive_forecast(RISING, 3)
    assert result.model == NAIVE
    assert result.predictions == (13.0, 13.0, 13.0)
    assert result.training_months == 12
    assert result.horizon == 3


def test_naive_works_from_two_months() -> None:
    assert build_naive_forecast([4.0, 7.0], 1).predictions == (7.0,)


def test_a_quiet_series_forecasts_quiet() -> None:
    result = build_naive_forecast([0.0] * 12, 2)
    assert result.predictions == (0.0, 0.0)
    assert result.lower == (0.0, 0.0)  # nothing has ever happened here


# --------------------------------------------------------------------------------------
# Seasonal naive
# --------------------------------------------------------------------------------------
def test_seasonal_naive_repeats_the_same_month_last_year() -> None:
    result = build_seasonal_naive_forecast(list(range(1, 13)), 3)
    assert result.model == SEASONAL_NAIVE
    assert result.predictions == (1.0, 2.0, 3.0)
    assert result.season == 12


def test_seasonal_naive_wraps_past_a_full_cycle() -> None:
    """Forecasting further than one season repeats the cycle rather than running out."""
    result = build_seasonal_naive_forecast(list(range(1, 13)), 14)
    assert result.predictions[:12] == tuple(float(value) for value in range(1, 13))
    assert result.predictions[12:] == (1.0, 2.0)


def test_seasonal_naive_reproduces_a_short_cycle() -> None:
    result = build_seasonal_naive_forecast(ALTERNATING, 4, season=2)
    assert result.predictions == (5.0, 9.0, 5.0, 9.0)


def test_seasonal_naive_refuses_less_than_one_cycle() -> None:
    """With under a year of history a yearly pattern would be invented, not observed."""
    with pytest.raises(ValueError, match="at least 12 months"):
        build_seasonal_naive_forecast(RISING[:6], 3)


# --------------------------------------------------------------------------------------
# Intervals
# --------------------------------------------------------------------------------------
def test_intervals_widen_further_ahead() -> None:
    result = build_naive_forecast(RISING, 4)
    widths = [upper - lower for lower, upper in zip(result.lower, result.upper, strict=True)]
    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def test_a_noisy_history_gives_a_wider_interval() -> None:
    calm = build_naive_forecast([10.0] * 12, 3)
    noisy = build_naive_forecast([10.0, 1.0] * 6, 3)
    assert (noisy.upper[0] - noisy.lower[0]) > (calm.upper[0] - calm.lower[0])


def test_a_smooth_history_still_admits_uncertainty() -> None:
    """A perfectly smooth series would otherwise claim a certainty counting never gives."""
    result = build_naive_forecast(RISING, 1)
    assert result.upper[0] > result.predictions[0] > result.lower[0]


def test_counts_cannot_be_forecast_negative() -> None:
    result = build_naive_forecast([1.0, 20.0, 1.0, 20.0, 1.0, 20.0], 3)
    assert all(value >= 0.0 for value in result.lower)


def test_count_noise_follows_the_level() -> None:
    """Counting 100 things a month is noisier in absolute terms than counting 4."""
    assert count_noise([100.0] * 6) > count_noise([4.0] * 6)
    assert count_noise([0.0] * 6) == 0.0
    assert count_noise([]) == 0.0


def test_interval_uses_the_residual_spread() -> None:
    lower, upper = interval([10.0], [1.0, -1.0, 1.0, -1.0])
    assert upper[0] == pytest.approx(10.0 + CONFIDENCE_MULTIPLIER)
    assert lower[0] == pytest.approx(10.0 - CONFIDENCE_MULTIPLIER)


def test_interval_never_narrows_below_the_floor() -> None:
    lower, upper = interval([10.0], [0.0, 0.0], minimum_spread=3.0)
    assert upper[0] - lower[0] == pytest.approx(2 * CONFIDENCE_MULTIPLIER * 3.0)


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("series", [[1.0, -2.0], [1.0, math.inf], [1.0, math.nan]])
def test_invalid_counts_are_refused(series: list[float]) -> None:
    with pytest.raises(ValueError, match="finite and not negative"):
        build_naive_forecast(series, 2)


@pytest.mark.parametrize("horizon", [0, -3])
def test_a_forecast_must_look_forward(horizon: int) -> None:
    with pytest.raises(ValueError, match="horizon"):
        build_naive_forecast(RISING, horizon)


def test_too_little_history_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 months"):
        build_naive_forecast([5.0], 1)


def test_a_season_of_one_is_meaningless() -> None:
    with pytest.raises(ValueError, match="season must be at least 2"):
        build_seasonal_naive_forecast(RISING, 2, season=1)


def test_check_series_returns_floats() -> None:
    assert check_series([1, 2, 3], minimum=3) == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------------------
# The result carries its own explanation
# --------------------------------------------------------------------------------------
def test_the_result_says_how_it_was_made() -> None:
    payload = build_seasonal_naive_forecast(list(range(1, 13)), 2).as_dict()
    json.dumps(payload)
    assert payload["model"] == SEASONAL_NAIVE
    assert payload["training_months"] == 12 and payload["season"] == 12
    assert len(payload["predictions"]) == len(payload["lower"]) == len(payload["upper"]) == 2
