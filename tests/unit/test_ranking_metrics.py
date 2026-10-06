"""Unit tests for the ranking metrics: Precision@K, Recall@K, NDCG, Spearman, Kendall, stability."""

from __future__ import annotations

import pytest

from cews.validation.ranking_metrics import (
    kendall_correlation,
    ndcg_at_k,
    precision_at_k,
    rank_stability,
    recall_at_k,
    spearman_correlation,
)

pytestmark = pytest.mark.unit

RANKING = ["A", "B", "C", "D", "E"]


# --------------------------------------------------------------------------------------
# Precision@K
# --------------------------------------------------------------------------------------
def test_precision_counts_hits_in_the_top_k() -> None:
    """A, C are relevant; F is relevant but never even ranked."""
    assert precision_at_k(RANKING, {"A", "C", "F"}, 3) == pytest.approx(2 / 3)


def test_precision_at_k_larger_than_the_ranking_uses_everything_available() -> None:
    assert precision_at_k(RANKING, {"A"}, 100) == pytest.approx(1 / 5)


def test_precision_at_zero_is_zero_not_an_error() -> None:
    assert precision_at_k(RANKING, {"A"}, 0) == 0.0


def test_precision_on_an_empty_ranking_is_zero() -> None:
    assert precision_at_k([], {"A"}, 3) == 0.0


def test_precision_with_nothing_relevant_at_all_is_zero() -> None:
    assert precision_at_k(RANKING, set(), 3) == 0.0


def test_a_negative_k_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        precision_at_k(RANKING, {"A"}, -1)


# --------------------------------------------------------------------------------------
# Recall@K
# --------------------------------------------------------------------------------------
def test_recall_counts_how_much_of_what_mattered_was_found() -> None:
    assert recall_at_k(RANKING, {"A", "C", "F"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k_covering_everything_is_perfect() -> None:
    assert recall_at_k(RANKING, {"A", "E"}, 5) == 1.0


def test_recall_with_nothing_relevant_is_a_perfect_score_not_undefined() -> None:
    """There is nothing to have missed."""
    assert recall_at_k(RANKING, set(), 3) == 1.0


def test_recall_at_zero_finds_nothing() -> None:
    assert recall_at_k(RANKING, {"A"}, 0) == 0.0


def test_a_negative_k_is_refused_for_recall() -> None:
    with pytest.raises(ValueError, match="negative"):
        recall_at_k(RANKING, {"A"}, -1)


# --------------------------------------------------------------------------------------
# NDCG@K
# --------------------------------------------------------------------------------------
def test_ndcg_is_perfect_for_the_ideal_order() -> None:
    relevance = {"A": 3.0, "B": 2.0, "C": 1.0}
    assert ndcg_at_k(["A", "B", "C"], relevance, 3) == pytest.approx(1.0)


def test_ndcg_penalizes_a_reversed_order() -> None:
    relevance = {"A": 3.0, "B": 2.0, "C": 1.0}
    assert ndcg_at_k(["C", "B", "A"], relevance, 3) < 1.0


def test_ndcg_with_no_positive_relevance_is_zero() -> None:
    assert ndcg_at_k(["A", "B"], {"A": 0.0, "B": 0.0}, 2) == 0.0


def test_ndcg_treats_an_unlisted_item_as_zero_relevance() -> None:
    """An item with no entry in the relevance map is not excluded, just worth nothing."""
    relevance = {"A": 3.0}
    with_extra = ndcg_at_k(["A", "Z"], relevance, 2)
    without_extra = ndcg_at_k(["A"], relevance, 1)
    assert with_extra == pytest.approx(without_extra)


def test_ndcg_rejects_negative_relevance() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        ndcg_at_k(["A"], {"A": -1.0}, 1)


def test_ndcg_rejects_a_negative_k() -> None:
    with pytest.raises(ValueError, match="not be negative"):
        ndcg_at_k(["A"], {"A": 1.0}, -1)


def test_ndcg_at_zero_is_zero() -> None:
    assert ndcg_at_k(["A"], {"A": 1.0}, 0) == 0.0


# --------------------------------------------------------------------------------------
# Spearman and Kendall
# --------------------------------------------------------------------------------------
def test_spearman_of_identical_order_is_one() -> None:
    result = spearman_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result.coefficient == pytest.approx(1.0)


def test_spearman_of_fully_reversed_order_is_minus_one() -> None:
    result = spearman_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert result.coefficient == pytest.approx(-1.0)


def test_spearman_only_cares_about_order_not_magnitude() -> None:
    """A model that overshoots by a fixed amount but never mis-orders scores a perfect 1.0."""
    result = spearman_correlation([1, 2, 3], [10, 20, 30])
    assert result.coefficient == pytest.approx(1.0)


def test_spearman_on_constant_input_is_none_not_zero() -> None:
    """Undefined and 'no correlation' are different answers; only one of them is honest here."""
    result = spearman_correlation([1, 1, 1], [1, 2, 3])
    assert result.coefficient is None and result.p_value is None


def test_spearman_requires_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        spearman_correlation([1, 2, 3], [1, 2])


def test_spearman_requires_at_least_two_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        spearman_correlation([1], [1])


def test_kendall_of_identical_order_is_one() -> None:
    result = kendall_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert result.coefficient == pytest.approx(1.0)


def test_kendall_is_more_conservative_than_spearman_on_one_swap() -> None:
    """One adjacent swap moves Kendall's pairwise count less than it moves Spearman's distance."""
    predicted = [1, 2, 3, 4, 5]
    actual = [1, 2, 3, 5, 4]  # one adjacent pair swapped
    spearman = spearman_correlation(predicted, actual).coefficient
    kendall = kendall_correlation(predicted, actual).coefficient
    assert kendall is not None and spearman is not None
    assert kendall < spearman


def test_kendall_on_constant_input_is_none() -> None:
    result = kendall_correlation([1, 1, 1], [1, 2, 3])
    assert result.coefficient is None


def test_no_warning_leaks_out_on_constant_input(recwarn: pytest.WarningsRecorder) -> None:
    spearman_correlation([1, 1, 1], [1, 2, 3])
    kendall_correlation([1, 1, 1], [1, 2, 3])
    assert len(recwarn) == 0


# --------------------------------------------------------------------------------------
# Rank stability
# --------------------------------------------------------------------------------------
def test_stability_of_an_unchanged_ranking_is_perfect() -> None:
    result = rank_stability(["A", "B", "C", "D"], ["A", "B", "C", "D"], k=2)
    assert result.spearman.coefficient == pytest.approx(1.0)
    assert result.top_k_overlap == 1.0


def test_stability_of_a_fully_reversed_ranking() -> None:
    result = rank_stability(["A", "B", "C", "D"], ["D", "C", "B", "A"], k=2)
    assert result.spearman.coefficient == pytest.approx(-1.0)
    assert result.top_k_overlap == 0.0


def test_top_k_overlap_can_be_partial() -> None:
    """Rank order changed, but the same three names are still in the top 3."""
    result = rank_stability(["A", "B", "C", "D"], ["C", "A", "B", "D"], k=3)
    assert result.top_k_overlap == 1.0
    assert result.spearman.coefficient is not None and result.spearman.coefficient < 1.0


def test_stability_requires_the_same_items_in_both_rankings() -> None:
    with pytest.raises(ValueError, match="exactly the same items"):
        rank_stability(["A", "B"], ["A", "C"])


def test_stability_with_two_items_still_works(recwarn: pytest.WarningsRecorder) -> None:
    result = rank_stability(["A", "B"], ["B", "A"], k=1)
    assert result.spearman.coefficient == pytest.approx(-1.0)
    assert len(recwarn) == 0


def test_stability_with_a_single_item_has_no_correlation_but_full_overlap() -> None:
    """One item can't be reordered, so a correlation coefficient means nothing here."""
    result = rank_stability(["A"], ["A"], k=1)
    assert result.spearman.coefficient is None
    assert result.top_k_overlap == 1.0


def test_a_negative_k_is_refused_for_stability() -> None:
    with pytest.raises(ValueError, match="negative"):
        rank_stability(["A", "B"], ["A", "B"], k=-1)


def test_every_result_is_json_friendly() -> None:
    import json

    json.dumps(spearman_correlation([1, 2, 3], [1, 2, 3]).as_dict())
    json.dumps(kendall_correlation([1, 2, 3], [1, 2, 3]).as_dict())
    json.dumps(rank_stability(["A", "B"], ["A", "B"]).as_dict())
