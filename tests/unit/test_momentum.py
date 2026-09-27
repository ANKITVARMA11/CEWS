"""Unit tests for momentum."""

from __future__ import annotations

import math

import pytest

from cews.features.momentum import calculate_momentum

pytestmark = pytest.mark.unit


def test_momentum_compares_the_last_two_windows() -> None:
    result = calculate_momentum([1, 1, 1, 4, 5, 6])
    assert (result.previous_activity, result.recent_activity) == (3.0, 15.0)
    assert result.momentum == pytest.approx(math.log(16 / 4))
    assert result.direction == "rising"


def test_windows_are_configurable() -> None:
    series = [0, 0, 10, 10, 2, 2]
    result = calculate_momentum(series, recent_months=2, previous_months=2)
    assert (result.recent_activity, result.previous_activity) == (4.0, 20.0)
    assert result.direction == "falling"
    assert result.as_dict()["windows"] == "2m vs 2m"


def test_a_quiet_period_is_flat_not_an_error() -> None:
    result = calculate_momentum([0, 0, 0, 0, 0, 0])
    assert result.momentum == 0.0 and result.direction == "flat"
    assert result.low_sample is True


def test_short_series_reports_what_it_had() -> None:
    result = calculate_momentum([4, 5])
    assert result.months_available == 2
    assert result.recent_activity == 9.0 and result.previous_activity == 0.0


def test_empty_series() -> None:
    result = calculate_momentum([])
    assert result.momentum == 0.0 and result.months_available == 0


@pytest.mark.parametrize("recent,previous", [(0, 3), (3, 0), (-1, 3)])
def test_invalid_windows_are_rejected(recent: int, previous: int) -> None:
    with pytest.raises(ValueError):
        calculate_momentum([1, 2, 3], recent_months=recent, previous_months=previous)


def test_negative_counts_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_momentum([1, -2, 3])
