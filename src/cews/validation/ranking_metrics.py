"""How good a ranking actually is, measured against what later turned out to be true.

Every score CEWS produces is, underneath, a ranking: which topics matter most, which
competitors deserve the closest look. These metrics answer the question a ranking can be judged
on that a single score cannot - "if we had acted on this list, how much of what mattered would
we have caught, and how far off was the order?"

Rank correlation (Spearman, Kendall) is delegated to ``scipy.stats``, which is already a project
dependency; everything specific to this project's own notion of "relevant" (Precision@K,
Recall@K, NDCG, rank stability) is implemented directly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from scipy import stats

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class CorrelationResult:
    """A rank correlation coefficient, with the significance scipy computed alongside it.

    ``coefficient`` is ``None`` when it cannot be computed at all (every value tied), which is a
    different, more honest answer than a coefficient of 0.
    """

    coefficient: float | None
    p_value: float | None
    method: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "coefficient": None if self.coefficient is None else round(self.coefficient, 4),
            "p_value": None if self.p_value is None else round(self.p_value, 4),
            "method": self.method,
        }


def _check_equal_length(predicted: Sequence[float], actual: Sequence[float]) -> None:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    if len(predicted) < 2:
        raise ValueError("need at least 2 items to compute a rank correlation")


def precision_at_k(ranking: Sequence[T], relevant: set[T], k: int) -> float:
    """Of the top ``k`` ranked items, what share are in ``relevant``.

    Returns 0.0 for an empty ranking or ``k=0`` rather than dividing by zero; an empty top-k has
    nothing right in it, which 0.0 states plainly.

    Raises:
        ValueError: if ``k`` is negative.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    top_k = list(ranking)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(ranking: Sequence[T], relevant: set[T], k: int) -> float:
    """Of everything that was actually relevant, what share appears in the top ``k``.

    Returns 1.0 when there was nothing relevant to find at all - recall cannot be said to have
    failed when there was nothing to recall.

    Raises:
        ValueError: if ``k`` is negative.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    if not relevant:
        return 1.0
    top_k = set(list(ranking)[:k])
    return len(top_k & relevant) / len(relevant)


def ndcg_at_k(ranking: Sequence[T], relevance: Mapping[T, float], k: int) -> float:
    """Normalized discounted cumulative gain at ``k``: rewards relevant items more the higher
    they are ranked, and rewards a *graded* notion of relevance (not just "relevant or not").

    Items with no entry in ``relevance`` are treated as having relevance 0, not excluded.

    Returns 0.0 when nothing in ``relevance`` has a positive value (there is nothing to gain, so
    no ranking can be scored against it) or when ``k`` is 0.

    Raises:
        ValueError: if ``k`` is negative, or any relevance value is negative.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    if any(value < 0 for value in relevance.values()):
        raise ValueError("relevance values must not be negative")
    if k == 0 or not any(value > 0 for value in relevance.values()):
        return 0.0

    top_k = list(ranking)[:k]
    dcg = sum(
        relevance.get(item, 0.0) / math.log2(position + 2)  # position is 0-based; +2 so log2(2)=1
        for position, item in enumerate(top_k)
    )
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(value / math.log2(position + 2) for position, value in enumerate(ideal_order))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def spearman_correlation(predicted: Sequence[float], actual: Sequence[float]) -> CorrelationResult:
    """Spearman's rank correlation between a predicted ordering and what actually happened.

    Works on the values' *ranks*, so it only asks whether the order was right, not by how much -
    a model that always overshoots by 10% but never gets the order wrong scores a perfect 1.0.

    Raises:
        ValueError: if the two sequences differ in length or have fewer than 2 items.
    """
    _check_equal_length(predicted, actual)
    with warnings.catch_warnings():
        # scipy warns when an input is constant; that case is already handled below by
        # returning None rather than a meaningless coefficient, so the warning is expected.
        warnings.simplefilter("ignore")
        coefficient, p_value = stats.spearmanr(predicted, actual)
    valid = coefficient is not None and math.isfinite(coefficient)
    return CorrelationResult(
        coefficient=float(coefficient) if valid else None,
        p_value=float(p_value) if valid else None,
        method="spearman",
    )


def kendall_correlation(predicted: Sequence[float], actual: Sequence[float]) -> CorrelationResult:
    """Kendall's tau: the share of pairs the predicted ranking put in the right relative order.

    More conservative than Spearman and less sensitive to a single badly-ranked outlier, since it
    counts pairwise agreements rather than distances between rank positions.

    Raises:
        ValueError: if the two sequences differ in length or have fewer than 2 items.
    """
    _check_equal_length(predicted, actual)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # see spearman_correlation: constant input is handled below
        result = stats.kendalltau(predicted, actual)
    coefficient, p_value = result.statistic, result.pvalue
    valid = coefficient is not None and math.isfinite(coefficient)
    return CorrelationResult(
        coefficient=float(coefficient) if valid else None,
        p_value=float(p_value) if valid else None,
        method="kendall",
    )


@dataclass(frozen=True)
class RankStability:
    """How much a ranking has changed between two runs over the same set of items."""

    spearman: CorrelationResult
    top_k_overlap: float
    k: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "spearman": self.spearman.as_dict(),
            "top_k_overlap": round(self.top_k_overlap, 4),
            "k": self.k,
        }


def rank_stability(earlier: Sequence[T], later: Sequence[T], *, k: int = 10) -> RankStability:
    """Compare two rankings of the same items taken at different times.

    ``top_k_overlap`` is the plainer, more presentable number: what share of last time's top
    ``k`` are still in this time's top ``k``. ``spearman`` gives the fuller picture across every
    ranked item, not just the top.

    Raises:
        ValueError: if the two rankings do not contain exactly the same items, or ``k`` is
            negative.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    if set(earlier) != set(later):
        raise ValueError("both rankings must contain exactly the same items to compare positions")
    earlier_rank = {item: position for position, item in enumerate(earlier)}
    later_rank = {item: position for position, item in enumerate(later)}
    items = list(earlier_rank)
    correlation = (
        spearman_correlation(
            [earlier_rank[item] for item in items], [later_rank[item] for item in items]
        )
        if len(items) >= 2
        else CorrelationResult(None, None, "spearman")
    )
    earlier_top = set(list(earlier)[:k])
    later_top = set(list(later)[:k])
    overlap = len(earlier_top & later_top) / len(earlier_top) if earlier_top else 1.0
    return RankStability(spearman=correlation, top_k_overlap=overlap, k=k)
