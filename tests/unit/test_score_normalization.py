"""Unit tests for putting raw values on a comparable 0-100 scale."""

from __future__ import annotations

import math

import pytest

from cews.scoring.normalization import (
    NEUTRAL_SCORE,
    clamp_score,
    minmax_normalize,
    normalize_values,
    percentile_normalize,
    renormalize_weights,
)

pytestmark = pytest.mark.unit


def test_percentile_ranks_values_against_their_peers() -> None:
    assert percentile_normalize([1, 2, 3, 4]) == [12.5, 37.5, 62.5, 87.5]


def test_percentile_gives_tied_values_the_same_score() -> None:
    scores = percentile_normalize([5, 5, 10])
    assert scores[0] == scores[1] < scores[2]


def test_percentile_ignores_how_extreme_an_outlier_is() -> None:
    assert percentile_normalize([1, 2, 3, 1000]) == percentile_normalize([1, 2, 3, 4])


def test_percentile_scores_stay_inside_the_scale() -> None:
    scores = percentile_normalize(list(range(50)))
    assert all(0 <= score <= 100 for score in scores)


@pytest.mark.parametrize(
    "values", [[], [7], [5, 5, 5], [0, 0], [float("nan"), 1], [float("inf"), 2]]
)
def test_cohorts_that_cannot_rank_themselves_score_neutral(values: list[float]) -> None:
    scores = percentile_normalize(values)
    assert scores == [NEUTRAL_SCORE] * len(values)


def test_negative_values_are_ranked_like_any_other() -> None:
    assert percentile_normalize([-5, 0, 5]) == [16.6667, 50.0, 83.3333]


def test_minmax_stretches_to_the_full_range() -> None:
    assert minmax_normalize([0, 5, 10]) == [0.0, 50.0, 100.0]
    assert minmax_normalize([2, 2]) == [NEUTRAL_SCORE, NEUTRAL_SCORE]


def test_normalize_values_dispatches_and_rejects_unknown_methods() -> None:
    assert normalize_values([1, 2], "percentile") == percentile_normalize([1, 2])
    assert normalize_values([1, 2], "minmax") == minmax_normalize([1, 2])
    # winsorized_minmax and robust_zscore arrived with the feature pass; the full behaviour of
    # all four methods is covered in tests/unit/test_normalization.py.
    assert len(normalize_values([1, 2, 3], "winsorized_minmax")) == 3
    assert len(normalize_values([1, 2, 3], "robust_zscore")) == 3
    with pytest.raises(ValueError, match="unknown normalization method"):
        normalize_values([1, 2], "astrology")


@pytest.mark.parametrize(
    ("value", "expected"), [(-5, 0.0), (0, 0.0), (42.5, 42.5), (100, 100.0), (250, 100.0)]
)
def test_clamp_score(value: float, expected: float) -> None:
    assert clamp_score(value) == expected


def test_missing_components_share_their_weight_proportionally() -> None:
    weights = {
        "trial": 0.3,
        "patent": 0.25,
        "publication": 0.2,
        "funding": 0.15,
        "announcement": 0.1,
    }
    shared = renormalize_weights(weights, ["trial", "publication", "announcement"])
    assert math.isclose(sum(shared.values()), 1.0)
    # the available components keep their relative importance
    assert math.isclose(shared["trial"] / shared["publication"], 0.3 / 0.2)


def test_all_components_available_leaves_the_weights_alone() -> None:
    weights = {"a": 0.6, "b": 0.4}
    assert renormalize_weights(weights, ["a", "b"]) == pytest.approx(weights)


def test_a_single_available_component_takes_the_whole_weight() -> None:
    assert renormalize_weights({"a": 0.6, "b": 0.4}, ["a"]) == {"a": 1.0}


@pytest.mark.parametrize(
    ("weights", "available"),
    [({"a": 0.5}, []), ({"a": 0.5}, ["missing"]), ({"a": 0.0}, ["a"])],
)
def test_nothing_available_is_an_error(weights: dict[str, float], available: list[str]) -> None:
    with pytest.raises(ValueError, match="no weighted components"):
        renormalize_weights(weights, available)
