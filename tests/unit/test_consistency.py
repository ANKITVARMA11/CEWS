"""Unit tests for consistency (sustained growth against a one-off jump)."""

from __future__ import annotations

import math

import pytest

from cews.features.consistency import calculate_consistency

pytestmark = pytest.mark.unit


def test_growth_every_month_is_fully_consistent() -> None:
    result = calculate_consistency([1, 2, 3, 4, 5])
    assert result is not None
    assert result.consistency == 1.0
    assert (result.increases, result.decreases, result.unchanged) == (4, 0, 0)


def test_decline_every_month_scores_zero() -> None:
    result = calculate_consistency([5, 4, 3, 2, 1])
    assert result is not None
    assert result.consistency == 0.0 and result.decreases == 4


def test_a_flat_series_is_neither_up_nor_down() -> None:
    result = calculate_consistency([3, 3, 3, 3])
    assert result is not None
    assert result.consistency == 0.0 and result.unchanged == 3
    assert result.volatility == 0.0


def test_half_up_half_down() -> None:
    result = calculate_consistency([1, 2, 1, 2, 1])
    assert result is not None
    assert result.consistency == pytest.approx(0.5)


def test_a_single_spike_is_identified() -> None:
    """The case that must not be called a trend: one big month, nothing sustained."""
    result = calculate_consistency([1, 1, 1, 40, 1, 1, 1])
    assert result is not None
    assert result.single_spike is True
    assert result.consistency < 0.5
    assert result.largest_share > 0.5


def test_steady_growth_is_not_a_spike() -> None:
    result = calculate_consistency([2, 3, 4, 5, 6, 7, 8])
    assert result is not None
    assert result.single_spike is False and result.consistency == 1.0


def test_volatility_separates_steady_from_see_saw() -> None:
    steady = calculate_consistency([1, 2, 3, 4, 5, 6])
    see_saw = calculate_consistency([1, 6, 1, 6, 1, 6])
    assert steady is not None and see_saw is not None
    assert see_saw.volatility > steady.volatility
    assert see_saw.consistency < 1.0


def test_too_few_months_returns_nothing() -> None:
    assert calculate_consistency([5]) is None
    assert calculate_consistency([5, 6]) is None
    assert calculate_consistency([5, 6, 7]) is not None


def test_all_zero_months() -> None:
    result = calculate_consistency([0, 0, 0, 0])
    assert result is not None
    assert result.consistency == 0.0 and result.largest_share == 0.0
    assert result.single_spike is False


@pytest.mark.parametrize("series", [[1, -1, 2], [1, math.nan, 2], [1, math.inf, 2]])
def test_invalid_counts_are_rejected(series: list[float]) -> None:
    with pytest.raises(ValueError):
        calculate_consistency(series)


def test_result_is_json_friendly() -> None:
    import json

    result = calculate_consistency([1, 2, 3, 4])
    assert result is not None
    json.dumps(result.as_dict())
