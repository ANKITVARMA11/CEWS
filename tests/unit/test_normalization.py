"""Unit tests for feature normalization: the four methods and cohort handling."""

from __future__ import annotations

import math

import pytest

from cews.constants import NormalizationMethod
from cews.scoring.normalization import (
    METHODS,
    NEUTRAL_SCORE,
    CohortNormalization,
    minmax_normalize,
    normalize_feature_cohort,
    normalize_values,
    percentile_normalize,
    robust_zscore_normalize,
    winsorized_minmax_normalize,
)

pytestmark = pytest.mark.unit

RISING = [1.0, 2.0, 3.0, 4.0, 5.0]
WITH_OUTLIER = [1.0, 2.0, 3.0, 4.0, 1000.0]


def test_every_configurable_method_is_implemented() -> None:
    """A method allowed in .env must not fail at runtime."""
    assert set(METHODS) == {member.value for member in NormalizationMethod}
    for method in METHODS:
        assert len(normalize_values(RISING, method)) == len(RISING)


def test_unknown_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown normalization method"):
        normalize_values(RISING, "astrology")


# --------------------------------------------------------------------------------------
# Percentile (the default)
# --------------------------------------------------------------------------------------
def test_percentile_ranks_within_the_cohort() -> None:
    assert percentile_normalize([1, 2, 3]) == pytest.approx([16.6667, 50.0, 83.3333], abs=0.001)


def test_percentile_ties_share_a_score() -> None:
    assert percentile_normalize([5, 5, 9]) == pytest.approx([33.3333, 33.3333, 83.3333], abs=0.001)


def test_percentile_ignores_how_far_apart_values_are() -> None:
    """Rank, not distance: one extreme value cannot flatten the rest."""
    assert percentile_normalize(RISING) == percentile_normalize(WITH_OUTLIER)


# --------------------------------------------------------------------------------------
# The other methods
# --------------------------------------------------------------------------------------
def test_minmax_stretches_to_the_ends() -> None:
    assert minmax_normalize(RISING) == pytest.approx([0.0, 25.0, 50.0, 75.0, 100.0])


def test_minmax_is_dominated_by_an_outlier() -> None:
    """Shown for contrast: this is why percentile is the default."""
    scores = minmax_normalize(WITH_OUTLIER)
    assert scores[-1] == 100.0
    assert max(scores[:-1]) < 1.0  # everything real is squashed together


def test_winsorizing_clips_the_extremes_first() -> None:
    scores = winsorized_minmax_normalize(WITH_OUTLIER, 0.0, 0.75)
    assert scores[-1] == 100.0
    assert scores[1] > 30.0  # the ordinary values keep a usable spread


def test_robust_zscore_centres_on_the_median() -> None:
    scores = robust_zscore_normalize([10, 11, 12, 13, 14])
    assert scores[2] == pytest.approx(NEUTRAL_SCORE)  # the median sits at the middle
    assert scores[0] < NEUTRAL_SCORE < scores[-1]
    assert all(0.0 <= score <= 100.0 for score in scores)


def test_robust_zscore_survives_an_outlier() -> None:
    scores = robust_zscore_normalize(WITH_OUTLIER)
    assert scores[-1] == 100.0
    assert scores[0] < scores[1] < scores[2]  # the rest still separate


# --------------------------------------------------------------------------------------
# Awkward cohorts
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_a_cohort_where_everyone_is_equal_scores_neutral(method: str) -> None:
    """Nothing can be said about rank, so nobody is favoured."""
    assert normalize_values([7.0, 7.0, 7.0], method) == [NEUTRAL_SCORE] * 3


@pytest.mark.parametrize("method", METHODS)
def test_a_single_member_cohort_scores_neutral(method: str) -> None:
    assert normalize_values([3.0], method) == [NEUTRAL_SCORE]


@pytest.mark.parametrize("method", METHODS)
def test_an_empty_cohort_returns_nothing(method: str) -> None:
    assert normalize_values([], method) == []


@pytest.mark.parametrize("method", METHODS)
def test_non_finite_values_fall_back_to_neutral(method: str) -> None:
    assert normalize_values([1.0, math.inf, 3.0], method) == [NEUTRAL_SCORE] * 3
    assert normalize_values([1.0, math.nan, 3.0], method) == [NEUTRAL_SCORE] * 3


@pytest.mark.parametrize("method", METHODS)
def test_negative_values_are_ranked_like_any_other(method: str) -> None:
    """Growth can be negative; normalization must not treat that as an error."""
    scores = normalize_values([-2.0, -1.0, 0.0, 1.0], method)
    assert len(scores) == 4
    assert scores[0] <= scores[1] <= scores[2] <= scores[3]
    assert all(0.0 <= score <= 100.0 for score in scores)


@pytest.mark.parametrize("method", METHODS)
def test_scores_never_leave_the_scale(method: str) -> None:
    for values in ([0.0001, 1e9], [-1e9, 0.0, 1e9], list(range(50))):
        assert all(0.0 <= score <= 100.0 for score in normalize_values(values, method))


def test_winsor_quantiles_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="quantiles"):
        winsorized_minmax_normalize(RISING, 0.9, 0.1)
    with pytest.raises(ValueError, match="quantiles"):
        winsorized_minmax_normalize(RISING, -0.1, 0.9)


def test_robust_scale_must_be_positive() -> None:
    with pytest.raises(ValueError, match="scale"):
        robust_zscore_normalize(RISING, scale=0)


# --------------------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------------------
def test_cohort_normalization_records_how_each_value_was_judged() -> None:
    results = normalize_feature_cohort(
        {"a": 1.0, "b": 5.0, "c": 9.0}, method="percentile", cohort="topic/velocity/2026-08-01"
    )
    assert set(results) == {"a", "b", "c"}
    middle = results["b"]
    assert isinstance(middle, CohortNormalization)
    assert middle.raw_value == 5.0 and middle.normalized_value == NEUTRAL_SCORE
    assert middle.cohort == "topic/velocity/2026-08-01" and middle.cohort_size == 3
    assert middle.method == "percentile"
    assert results["c"].normalized_value > results["a"].normalized_value


def test_cohort_normalization_marks_neutral_results() -> None:
    results = normalize_feature_cohort({"a": 2.0, "b": 2.0}, cohort="x")
    assert all(result.neutral for result in results.values())


def test_cohort_normalization_is_json_friendly() -> None:
    import json

    results = normalize_feature_cohort({"a": 1.0, "b": 2.0}, cohort="x")
    json.dumps({name: result.as_dict() for name, result in results.items()})


def test_empty_cohort() -> None:
    assert normalize_feature_cohort({}, cohort="x") == {}
