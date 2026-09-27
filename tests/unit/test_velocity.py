"""Unit tests for velocity (the fitted trend line)."""

from __future__ import annotations

import math

import pytest

from cews.features.velocity import calculate_velocity

pytestmark = pytest.mark.unit

STEADY_RISE = [2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]


def test_a_rising_series_has_a_positive_slope() -> None:
    result = calculate_velocity(STEADY_RISE)
    assert result is not None
    assert result.slope > 0 and result.direction == "rising"
    assert result.r_squared > 0.9  # a straight line describes it well
    assert result.observations == 12


def test_a_falling_series_mirrors_it() -> None:
    rising = calculate_velocity(STEADY_RISE)
    falling = calculate_velocity(list(reversed(STEADY_RISE)))
    assert rising is not None and falling is not None
    assert falling.slope == pytest.approx(-rising.slope)
    assert falling.direction == "falling"


def test_a_flat_series_has_no_slope() -> None:
    result = calculate_velocity([5] * 12)
    assert result is not None
    assert result.slope == 0.0 and result.direction == "flat"


def test_doubling_every_month_is_a_constant_slope() -> None:
    """The log transform means proportional growth reads as a straight line."""
    result = calculate_velocity([1, 2, 4, 8, 16, 32, 64, 128])
    assert result is not None
    assert result.r_squared > 0.99
    assert result.monthly_percent > 50  # roughly a doubling each month


def test_one_spike_is_not_a_trend() -> None:
    """A single jump should not look like sustained growth."""
    spike = [2, 2, 2, 2, 2, 30, 2, 2, 2, 2, 2, 2]
    result = calculate_velocity(spike)
    assert result is not None
    assert abs(result.slope) < 0.05
    assert result.r_squared < 0.2  # a line explains almost nothing about this series


def test_too_few_months_returns_nothing() -> None:
    """Fitting a line through three points would look confident and mean nothing."""
    assert calculate_velocity([1, 2, 3]) is None
    assert calculate_velocity([1, 2, 3, 4, 5], min_observations=6) is None
    assert calculate_velocity([1, 2, 3, 4, 5, 6], min_observations=6) is not None


def test_only_the_window_is_used() -> None:
    long_series = [100] * 12 + [1, 2, 3, 4, 5, 6]
    result = calculate_velocity(long_series, window_months=6)
    assert result is not None
    assert result.observations == 6 and result.slope > 0


def test_empty_months_count_as_zero_not_missing() -> None:
    result = calculate_velocity([0, 0, 0, 1, 2, 4, 8, 12])
    assert result is not None and result.slope > 0


@pytest.mark.parametrize(
    ("series", "window", "minimum"),
    [([1, -2, 3, 4, 5, 6], 12, 6), ([1, math.inf, 3, 4, 5, 6], 12, 6)],
)
def test_invalid_counts_are_rejected(series: list[float], window: int, minimum: int) -> None:
    with pytest.raises(ValueError):
        calculate_velocity(series, window_months=window, min_observations=minimum)


@pytest.mark.parametrize(("window", "minimum"), [(0, 6), (-1, 6), (12, 1)])
def test_invalid_settings_are_rejected(window: int, minimum: int) -> None:
    with pytest.raises(ValueError):
        calculate_velocity(STEADY_RISE, window_months=window, min_observations=minimum)


def test_result_is_json_friendly() -> None:
    import json

    result = calculate_velocity(STEADY_RISE)
    assert result is not None
    json.dumps(result.as_dict())
