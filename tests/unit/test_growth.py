"""Unit tests for growth and the recent surge."""

from __future__ import annotations

import math

import pytest

from cews.features.growth import calculate_growth_rate, calculate_recent_surge, growth_from_series

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("current", "previous", "log_growth", "display"),
    [
        (10, 5, math.log(11 / 6), 100.0),  # doubling
        (5, 10, math.log(6 / 11), -50.0),  # halving
        (7, 7, 0.0, 0.0),  # unchanged
        (0, 0, 0.0, 0.0),  # nothing either period
        (3, 0, math.log(4.0), 300.0),  # from nothing: finite, not infinite
        (0, 3, math.log(0.25), -100.0),  # to nothing
        (1, 0, math.log(2.0), 100.0),
    ],
)
def test_hand_calculated_growth(
    current: float, previous: float, log_growth: float, display: float
) -> None:
    result = calculate_growth_rate(current, previous)
    assert result.log_growth == pytest.approx(log_growth)
    assert result.display_percent == pytest.approx(display)


def test_growth_is_symmetric() -> None:
    """A halving is exactly the negative of a doubling, which a percentage never is."""
    up = calculate_growth_rate(20, 10).log_growth
    down = calculate_growth_rate(10, 20).log_growth
    assert up == pytest.approx(-down)


def test_smoothing_damps_small_counts() -> None:
    """1 -> 3 is a 200% jump, but on tiny numbers it should not look like 100 -> 300."""
    small = calculate_growth_rate(3, 1)
    large = calculate_growth_rate(300, 100)
    assert small.display_percent == large.display_percent == 200.0
    assert small.log_growth < large.log_growth


def test_alpha_controls_the_damping() -> None:
    gentle = calculate_growth_rate(3, 1, alpha=1.0).log_growth
    heavy = calculate_growth_rate(3, 1, alpha=10.0).log_growth
    assert heavy < gentle  # a larger alpha pulls the ratio toward "no change"


def test_thin_evidence_is_flagged() -> None:
    assert calculate_growth_rate(3, 2).low_sample is True
    assert calculate_growth_rate(30, 20).low_sample is False
    assert calculate_growth_rate(3, 2, low_sample_threshold=2).low_sample is False


@pytest.mark.parametrize(
    ("current", "previous", "direction"),
    [(5, 1, "rising"), (1, 5, "falling"), (4, 4, "flat"), (0, 0, "flat")],
)
def test_direction(current: float, previous: float, direction: str) -> None:
    assert calculate_growth_rate(current, previous).direction == direction


@pytest.mark.parametrize(
    ("current", "previous", "alpha"),
    [(-1, 5, 1.0), (5, -1, 1.0), (math.inf, 1, 1.0), (1, math.nan, 1.0), (5, 5, 0.0), (5, 5, -2.0)],
)
def test_invalid_input_is_rejected(current: float, previous: float, alpha: float) -> None:
    with pytest.raises(ValueError):
        calculate_growth_rate(current, previous, alpha=alpha)


def test_growth_from_a_series_uses_the_last_windows() -> None:
    series = [1, 1, 1, 4, 5, 6]  # previous three sum to 3, recent three to 15
    result = growth_from_series(series, 3, 3)
    assert (result.previous, result.current) == (3.0, 15.0)
    assert result.log_growth == pytest.approx(math.log(16 / 4))


def test_a_short_series_still_produces_a_result() -> None:
    result = growth_from_series([2, 6], 3, 3)
    assert (result.previous, result.current) == (0.0, 8.0)


def test_surge_compares_quarters() -> None:
    surge = calculate_recent_surge(30, 10)
    assert surge.log_growth == pytest.approx(math.log(31 / 11))
    assert surge.as_dict()["direction"] == "rising"


def test_result_is_json_friendly() -> None:
    import json

    json.dumps(calculate_growth_rate(9, 4).as_dict())
