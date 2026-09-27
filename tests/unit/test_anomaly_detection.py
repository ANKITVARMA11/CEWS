"""Unit tests for anomaly detection, and for telling one kind of unusual from another."""

from __future__ import annotations

import json
import math
from datetime import date

import pytest

from cews.discovery.anomaly_detection import (
    AnomalyKind,
    detect_activity_anomalies,
    median_absolute_deviation,
    robust_zscores,
    rolling_robust_zscores,
    seasonal_residuals,
    spread_of,
)
from cews.features.time_windows import month_range

pytestmark = pytest.mark.unit

QUIET = [5.0, 6.0, 5.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 6.0, 5.0]


def kinds(series: list[float], **kwargs: object) -> list[str]:
    return [found.kind.value for found in detect_activity_anomalies(series, **kwargs)]


# --------------------------------------------------------------------------------------
# What kind of unusual is it?
# --------------------------------------------------------------------------------------
def test_one_busy_month_is_a_spike_not_a_trend() -> None:
    """The distinction the whole module exists for."""
    found = detect_activity_anomalies(QUIET + [40.0, 6.0, 5.0, 6.0])
    assert found and found[0].kind is AnomalyKind.ONE_TIME_SPIKE
    assert "back to its usual level" in found[0].explanation
    assert found[0].confidence > 80


def test_a_level_that_steps_up_and_stays_is_momentum() -> None:
    found = detect_activity_anomalies(QUIET + [30.0, 28.0, 31.0, 29.0])
    assert found[0].kind is AnomalyKind.PERSISTENT_MOMENTUM
    assert "stayed up" in found[0].explanation


def test_a_rise_still_in_progress_is_not_yet_judged() -> None:
    """The most recent month has no "afterwards" to look at."""
    found = detect_activity_anomalies(QUIET + [8.0, 12.0, 18.0, 40.0])
    assert found[-1].kind is AnomalyKind.EMERGING_TREND
    assert "not yet known" in found[-1].explanation


def test_every_source_going_quiet_is_a_collection_gap() -> None:
    """More likely our pipeline than the whole field pausing at once."""
    found = detect_activity_anomalies(QUIET + [0.0, 6.0, 5.0, 6.0], quiet_months=[12])
    assert found[0].kind is AnomalyKind.COLLECTION_GAP
    assert "collection rather than" in found[0].explanation


def test_a_collapse_is_reported_as_a_drop() -> None:
    found = detect_activity_anomalies([30.0] * 12 + [1.0, 29.0, 30.0, 31.0])
    assert found and found[0].kind is AnomalyKind.DROP
    assert found[0].direction == "below"


def test_a_steady_history_can_still_produce_an_anomaly() -> None:
    """A history with no variation at all has no spread to divide by; count noise is used
    instead, so the most obvious anomaly there is does not go unnoticed."""
    assert spread_of([30.0] * 12) == pytest.approx(math.sqrt(30.0))
    assert spread_of([]) == 0.0
    assert spread_of([0.0] * 5) == 1.0  # never zero, so a first record can register


def test_a_busy_month_every_year_is_seasonal() -> None:
    year = [4.0, 4.0, 20.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    found = detect_activity_anomalies(year * 3, season=12)
    assert found and any(item.kind is AnomalyKind.SEASONAL_PATTERN for item in found)


def test_small_numbers_cannot_look_seasonal() -> None:
    """Two small numbers in the same month of different years is not a season."""
    year = [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert AnomalyKind.SEASONAL_PATTERN.value not in kinds(year * 3, season=12)


# --------------------------------------------------------------------------------------
# What is not an anomaly
# --------------------------------------------------------------------------------------
def test_ordinary_variation_is_left_alone() -> None:
    assert kinds(QUIET + [6.0, 5.0, 7.0, 6.0]) == []


def test_steady_growth_is_not_an_anomaly() -> None:
    """A topic that grows every month is a trend, and the trend score already says so."""
    assert kinds([float(5 + index) for index in range(16)]) == []


def test_a_short_history_is_not_judged() -> None:
    assert kinds([1.0, 50.0, 1.0]) == []


def test_a_jump_in_tiny_counts_is_flagged_but_barely_believed() -> None:
    """Three records against nothing is a large ratio and a small fact."""
    found = detect_activity_anomalies(
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 1.0]
    )
    assert found and found[0].confidence < 40


def test_confidence_rises_with_the_size_of_the_numbers() -> None:
    small = detect_activity_anomalies(QUIET + [20.0, 6.0, 5.0, 6.0])[0]
    large = detect_activity_anomalies([value * 10 for value in QUIET] + [200.0, 60.0, 50.0, 60.0])[
        0
    ]
    assert large.confidence >= small.confidence


# --------------------------------------------------------------------------------------
# The underlying measures
# --------------------------------------------------------------------------------------
def test_the_median_absolute_deviation_barely_moves_for_an_outlier() -> None:
    """Why the median is used and not the mean: one enormous month must not hide itself."""
    import statistics

    ordinary = [5.0, 6.0, 5.0, 6.0, 5.0, 6.0]
    with_outlier = [*ordinary, 500.0]
    mad_shift = median_absolute_deviation(with_outlier) - median_absolute_deviation(ordinary)
    stdev_shift = statistics.pstdev(with_outlier) - statistics.pstdev(ordinary)
    assert mad_shift < 1.0 < stdev_shift  # the standard deviation explodes; the MAD does not


def test_a_robust_score_flags_the_outlier_that_would_hide_itself() -> None:
    scores = robust_zscores([5.0, 6.0, 5.0, 6.0, 5.0, 6.0, 60.0])
    assert scores[-1] > 5.0  # the mean would have been dragged up toward the outlier


def test_the_rolling_score_compares_against_recent_months() -> None:
    """A topic that grew for years should not flag every month for beating the distant past."""
    growing = [float(index) for index in range(1, 25)]
    assert max(abs(score) for score in rolling_robust_zscores(growing)) < 3.5


def test_the_first_months_have_nothing_to_compare_against() -> None:
    assert rolling_robust_zscores([5.0] * 10)[:6] == [0.0] * 6


def test_seasonal_residuals_need_two_full_cycles() -> None:
    assert seasonal_residuals([1.0] * 12, season=12) is None
    assert seasonal_residuals([1.0] * 24, season=12) is not None
    assert seasonal_residuals([1.0] * 24, season=1) is None


def test_a_window_below_two_is_refused() -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_robust_zscores([1.0, 2.0, 3.0], window=1)


# --------------------------------------------------------------------------------------
# The record it leaves
# --------------------------------------------------------------------------------------
def test_an_anomaly_records_what_it_was_compared_against() -> None:
    months = month_range(date(2026, 8, 1), 16)
    found = detect_activity_anomalies(QUIET + [40.0, 6.0, 5.0, 6.0], periods=months)[0]
    assert found.period == months[12]
    assert found.expected_lower <= found.expected <= found.expected_upper
    assert found.observed > found.expected_upper  # outside the range it was judged against
    payload = found.as_dict()
    json.dumps(payload)
    assert payload["evidence"]["months_compared"] > 0
    assert payload["method"] == "rolling_robust_zscore"
    assert payload["explanation"]


def test_the_whole_series_method_can_be_chosen() -> None:
    found = detect_activity_anomalies(QUIET + [40.0, 6.0, 5.0, 6.0], method="robust_zscore")
    assert found and found[0].method == "robust_zscore"


def test_a_higher_threshold_flags_less() -> None:
    series = QUIET + [25.0, 6.0, 5.0, 6.0]
    assert len(kinds(series, threshold=3.0)) >= len(kinds(series, threshold=8.0))


@pytest.mark.parametrize(
    ("series", "kwargs", "message"),
    [
        ([1.0, -2.0] * 6, {}, "finite and not negative"),
        ([1.0, math.inf] * 6, {}, "finite and not negative"),
        ([1.0] * 12, {"threshold": 0}, "threshold"),
        ([1.0] * 12, {"method": "vibes"}, "unknown method"),
    ],
)
def test_invalid_input_is_refused(
    series: list[float], kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        detect_activity_anomalies(series, **kwargs)


def test_periods_must_line_up_with_the_series() -> None:
    with pytest.raises(ValueError, match="line up"):
        detect_activity_anomalies([1.0] * 12, periods=month_range(date(2026, 8, 1), 3))
