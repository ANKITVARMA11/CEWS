"""Unit tests for the forecast error measures."""

from __future__ import annotations

import math

import pytest

from cews.validation.forecast_metrics import (
    calculate_forecast_metrics,
    directional_accuracy,
    mean_absolute_error,
    mean_absolute_scaled_error,
    root_mean_squared_error,
    symmetric_mape,
)

pytestmark = pytest.mark.unit

ACTUAL = [10.0, 12.0, 14.0, 16.0]


def test_a_perfect_forecast_scores_zero() -> None:
    assert mean_absolute_error(ACTUAL, ACTUAL) == 0.0
    assert root_mean_squared_error(ACTUAL, ACTUAL) == 0.0
    assert symmetric_mape(ACTUAL, ACTUAL) == 0.0


def test_mean_absolute_error_is_the_average_miss() -> None:
    assert mean_absolute_error([10, 10], [8, 14]) == pytest.approx(3.0)


def test_rmse_punishes_one_large_miss_harder() -> None:
    """Two misses of 3 versus one of 6: the same total, a worse RMSE."""
    spread_out = root_mean_squared_error([10, 10], [7, 13])
    concentrated = root_mean_squared_error([10, 10], [10, 16])
    assert concentrated > spread_out
    assert mean_absolute_error([10, 10], [7, 13]) == mean_absolute_error([10, 10], [10, 16])


def test_smape_survives_a_zero_month() -> None:
    """A percentage error must stay finite when nothing happened, which MAPE does not."""
    value = symmetric_mape([0.0, 5.0], [2.0, 5.0])
    assert math.isfinite(value) and 0.0 < value <= 200.0


def test_smape_of_two_zeros_is_zero_not_undefined() -> None:
    assert symmetric_mape([0.0, 0.0], [0.0, 0.0]) == 0.0


# --------------------------------------------------------------------------------------
# MASE: the measure model selection uses
# --------------------------------------------------------------------------------------
def test_mase_compares_against_the_naive_forecast() -> None:
    """The naive error here is 1 per step, so an average miss of 2 scores 2."""
    training = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert mean_absolute_scaled_error([6.0, 7.0], [8.0, 9.0], training) == pytest.approx(2.0)


def test_below_one_means_better_than_naive() -> None:
    training = [1.0, 2.0, 3.0, 4.0, 5.0]
    good = mean_absolute_scaled_error([6.0, 7.0], [6.2, 7.1], training)
    assert good is not None and good < 1.0


def test_mase_is_undefined_against_a_flat_history() -> None:
    """A history that never moves gives no yardstick to scale against."""
    assert mean_absolute_scaled_error([5.0], [6.0], [3.0, 3.0, 3.0]) is None


def test_mase_needs_more_history_than_the_season() -> None:
    assert mean_absolute_scaled_error([5.0], [6.0], [3.0], season=1) is None
    assert mean_absolute_scaled_error([5.0], [6.0], [1.0, 4.0, 2.0], season=12) is None


def test_mase_uses_the_seasonal_yardstick_when_asked() -> None:
    training = [1.0, 9.0, 1.0, 9.0, 1.0, 9.0]  # alternates, so month-on-month error is large
    monthly = mean_absolute_scaled_error([1.0], [3.0], training, season=1)
    seasonal = mean_absolute_scaled_error([1.0], [3.0], training, season=2)
    assert monthly is not None and seasonal is None  # a perfect season leaves no naive error


# --------------------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------------------
def test_directional_accuracy_counts_right_directions() -> None:
    assert directional_accuracy([12.0, 14.0], [11.0, 13.0], last_observed=10.0) == 1.0
    assert directional_accuracy([12.0, 14.0], [9.0, 8.0], last_observed=10.0) == 0.0


def test_months_that_did_not_move_are_skipped() -> None:
    """There is no direction to get right when the truth stayed put."""
    assert directional_accuracy([10.0, 12.0], [9.0, 13.0], last_observed=10.0) == 1.0


def test_directional_accuracy_is_none_when_nothing_moved() -> None:
    assert directional_accuracy([10.0, 10.0], [8.0, 12.0], last_observed=10.0) is None


# --------------------------------------------------------------------------------------
# The combined result
# --------------------------------------------------------------------------------------
def test_every_measure_is_reported_together() -> None:
    metrics = calculate_forecast_metrics(
        ACTUAL, [11.0, 11.0, 15.0, 15.0], training=[2.0, 4.0, 6.0, 8.0], last_observed=8.0
    )
    payload = metrics.as_dict()
    assert set(payload) == {
        "mae",
        "rmse",
        "mase",
        "smape",
        "directional_accuracy",
        "observations",
        "beats_naive",
    }
    assert payload["observations"] == 4


def test_beating_the_naive_baseline_is_stated_plainly() -> None:
    training = [1.0, 2.0, 3.0, 4.0, 5.0]
    good = calculate_forecast_metrics([6.0], [6.1], training=training)
    bad = calculate_forecast_metrics([6.0], [16.0], training=training)
    assert good.beats_naive is True and bad.beats_naive is False


def test_without_training_history_mase_is_unknown_not_zero() -> None:
    metrics = calculate_forecast_metrics(ACTUAL, ACTUAL)
    assert metrics.mase is None and metrics.beats_naive is None


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([], []), ([1.0], [1.0, 2.0]), ([math.nan], [1.0]), ([1.0], [math.inf])],
)
def test_invalid_input_is_refused(actual: list[float], predicted: list[float]) -> None:
    with pytest.raises(ValueError):
        mean_absolute_error(actual, predicted)


def test_a_negative_season_is_refused() -> None:
    with pytest.raises(ValueError, match="season"):
        mean_absolute_scaled_error([1.0], [1.0], [1.0, 2.0, 3.0], season=0)
