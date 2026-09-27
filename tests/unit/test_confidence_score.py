"""Unit tests for the confidence score and the shared score machinery."""

from __future__ import annotations

import json

import pytest

from cews.scoring.base import (
    Gate,
    ScoreComponent,
    ScoreResult,
    classify_score,
    combine,
    explain_score,
)
from cews.scoring.confidence_score import (
    CONFIDENCE_WEIGHTS,
    calculate_confidence_score,
)
from cews.scoring.config import Category

pytestmark = pytest.mark.unit

STRONG = {
    "sample_confidence": 1.0,
    "source_agreement": 1.0,
    "data_completeness": 1.0,
    "data_freshness": 1.0,
    "model_stability": 1.0,
}


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------
def test_everything_perfect_scores_one_hundred() -> None:
    assert calculate_confidence_score(**STRONG).value == pytest.approx(100.0)


def test_nothing_known_scores_zero() -> None:
    result = calculate_confidence_score(
        sample_confidence=0.0,
        source_agreement=0.0,
        data_completeness=0.0,
        data_freshness=0.0,
        model_stability=0.0,
    )
    assert result.value == 0.0 and result.low is True


def test_hand_calculated_mixture() -> None:
    """0.35(0.8) + 0.25(0.5) + 0.20(1.0) + 0.10(0.5) + 0.10(0.0) = 0.655 -> 65.5"""
    result = calculate_confidence_score(
        sample_confidence=0.8,
        source_agreement=0.5,
        data_completeness=1.0,
        data_freshness=0.5,
        model_stability=0.0,
    )
    assert result.value == pytest.approx(65.5)
    assert sum(part.contribution for part in result.components.values()) == pytest.approx(65.5)


def test_evidence_volume_matters_most() -> None:
    """Sample carries the largest weight, so it moves the score more than freshness."""
    baseline = calculate_confidence_score(
        sample_confidence=0.5,
        source_agreement=0.5,
        data_completeness=0.5,
        data_freshness=0.5,
        model_stability=0.5,
    ).value
    more_evidence = calculate_confidence_score(
        **{
            **STRONG,
            "source_agreement": 0.5,
            "data_completeness": 0.5,
            "data_freshness": 0.5,
            "model_stability": 0.5,
        }
    ).value
    fresher = calculate_confidence_score(
        sample_confidence=0.5,
        source_agreement=0.5,
        data_completeness=0.5,
        data_freshness=1.0,
        model_stability=0.5,
    ).value
    assert more_evidence - baseline > fresher - baseline


def test_weights_match_the_specification() -> None:
    assert CONFIDENCE_WEIGHTS == {
        "sample": 0.35,
        "source_agreement": 0.25,
        "completeness": 0.20,
        "freshness": 0.10,
        "model_stability": 0.10,
    }
    assert sum(CONFIDENCE_WEIGHTS.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# A first run has no history
# --------------------------------------------------------------------------------------
def test_unknown_stability_is_shared_out_not_scored_zero() -> None:
    """Otherwise every first run would look untrustworthy."""
    unknown = calculate_confidence_score(**{**STRONG, "model_stability": None})
    as_zero = calculate_confidence_score(**{**STRONG, "model_stability": 0.0})
    assert unknown.value == pytest.approx(100.0)
    assert as_zero.value == pytest.approx(90.0)
    assert unknown.unavailable == ("model_stability",)
    assert unknown.components["model_stability"].available is False


def test_remaining_weights_still_sum_to_one_without_stability() -> None:
    result = calculate_confidence_score(**{**STRONG, "model_stability": None})
    used = [part.weight for part in result.components.values() if part.available]
    assert sum(used) == pytest.approx(1.0)
    assert result.components["sample"].weight == pytest.approx(0.35 / 0.9)


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [-0.1, 1.5, float("nan"), float("inf")])
def test_inputs_outside_zero_to_one_are_refused(value: float) -> None:
    with pytest.raises(ValueError):
        calculate_confidence_score(**{**STRONG, "sample_confidence": value})


def test_low_confidence_flag_follows_the_threshold() -> None:
    """The threshold itself counts as usable; anything below it is flagged."""
    assert calculate_confidence_score(**{k: 0.59 for k in STRONG}).low is True
    assert calculate_confidence_score(**{k: 0.60 for k in STRONG}).value == pytest.approx(60.0)
    assert calculate_confidence_score(**{k: 0.60 for k in STRONG}).low is False
    assert calculate_confidence_score(**{k: 0.9 for k in STRONG}).low is False


def test_result_is_json_friendly() -> None:
    json.dumps(calculate_confidence_score(**STRONG).as_dict())


# --------------------------------------------------------------------------------------
# combine(), classify_score() and explanations
# --------------------------------------------------------------------------------------
def test_combine_weights_available_components_only() -> None:
    score, components, missing = combine(
        {"a": (1.0, 80.0, {}), "b": (2.0, 40.0, {}), "c": (None, None, {})},
        {"a": 0.5, "b": 0.3, "c": 0.2},
    )
    assert missing == ("c",)
    assert components["c"].available is False and components["c"].weight == 0.0
    # 0.5 and 0.3 rescale to 0.625 and 0.375: 80(0.625) + 40(0.375) = 65
    assert score == pytest.approx(65.0)


def test_combine_refuses_when_nothing_has_data() -> None:
    with pytest.raises(ValueError, match="no component has data"):
        combine({"a": (None, None, {})}, {"a": 1.0})


def test_combine_refuses_a_component_off_the_scale() -> None:
    with pytest.raises(ValueError, match="outside the 0-100 scale"):
        combine({"a": (1.0, 140.0, {})}, {"a": 1.0})


def test_classify_score() -> None:
    bands = (Category("Low", 0, 39), Category("Watchlist", 40, 59), Category("High", 60, 100))
    assert classify_score(10, bands) == "Low"
    assert classify_score(59, bands) == "Watchlist"
    assert classify_score(100, bands) == "High"
    assert classify_score(50, ()) == ""


def test_explanation_names_the_drivers_and_the_gaps() -> None:
    result = ScoreResult(
        score_type="trend",
        entity_type="topic",
        entity_id=1,
        entity_name="CRISPR gene editing",
        value=72.5,
        confidence=88.0,
        components={
            "velocity": ScoreComponent("velocity", 0.12, 90.0, 0.35),
            "momentum": ScoreComponent("momentum", 0.30, 60.0, 0.30),
            "patent_growth": ScoreComponent("patent_growth", None, 0.0, 0.0, available=False),
        },
        unavailable=("patent_growth",),
        sample_size=442,
        category="Emerging",
        gates=(Gate("two_sources", False, "only one source supports this"),),
    )
    text = explain_score(result)
    assert "CRISPR gene editing scores 72.5" in text and "Emerging" in text
    assert "velocity" in text and "442 record(s)" in text
    assert "patent growth" in text and "shared across" in text
    assert "only one source supports this" in text
    assert result.qualified is False and result.failed_gates == ("two_sources",)


def test_explanation_when_nothing_could_be_scored() -> None:
    empty = ScoreResult("trend", "topic", 1, "Quiet topic", 0.0, 0.0)
    assert "no component had data" in explain_score(empty)


def test_component_json_round_trips() -> None:
    result = ScoreResult(
        "confidence",
        "topic",
        1,
        "A topic",
        55.0,
        55.0,
        components={"sample": ScoreComponent("sample", 0.2, 20.0, 1.0)},
        sample_size=6,
        gates=(Gate("usable_confidence", False, "below the threshold"),),
    )
    payload = result.as_component_json()
    json.dumps(payload)
    assert payload["qualified"] is False
    assert payload["components"]["sample"]["contribution"] == 20.0
    assert payload["gates"][0]["passed"] is False
