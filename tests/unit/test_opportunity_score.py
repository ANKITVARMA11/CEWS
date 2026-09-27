"""Unit tests for the opportunity score."""

from __future__ import annotations

import pytest

from cews.constants import EntityType
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.inputs import EntityInputs
from cews.scoring.opportunity_score import calculate_opportunity_score
from cews.scoring.trend_score import calculate_trend_score
from cews.settings import load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

TREND_FEATURES = {
    "velocity": 80.0,
    "momentum": 80.0,
    "growth_patent": 80.0,
    "growth_funding": 80.0,
    "consistency": 80.0,
}


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    return load_scoring_config(settings)


def make_inputs(
    *,
    density: float | None = 20.0,
    agreement: float = 0.8,
    supporting: int = 3,
    sample: float = 200.0,
    spike: bool = False,
) -> EntityInputs:
    raw: dict[str, float] = {
        f"activity_{source}": 10.0
        for source in ("publication", "clinical_trial", "patent", "funding")
    }
    raw["supporting_sources"] = float(supporting)
    raw["single_spike"] = 1.0 if spike else 0.0
    raw["source_agreement"] = agreement
    if density is not None:
        raw["competition_density"] = density
    return EntityInputs(
        entity_type=EntityType.TOPIC.value,
        entity_id=1,
        name="A topic",
        raw=raw,
        normalized=dict(TREND_FEATURES),
        sample_size=sample,
    )


def score(config: ScoringConfig, inputs: EntityInputs, confidence: float = 90.0):
    trend = calculate_trend_score(inputs, config, confidence=confidence)
    return calculate_opportunity_score(inputs, config, trend=trend, confidence=confidence)


def test_score_matches_a_hand_calculation(config: ScoringConfig) -> None:
    """Trend 80, density 20 (so 80 open), agreement 0.8: 0.60(80) + 0.25(80) + 0.15(80) = 80"""
    result = score(config, make_inputs())
    assert result.value == pytest.approx(80.0)
    assert sum(part.contribution for part in result.components.values()) == pytest.approx(80.0)


def test_an_empty_field_beats_a_crowded_one(config: ScoringConfig) -> None:
    open_field = score(config, make_inputs(density=5.0)).value
    crowded = score(config, make_inputs(density=95.0)).value
    assert open_field > crowded
    assert open_field - crowded == pytest.approx(0.25 * 90.0, abs=0.01)


def test_trend_carries_most_of_the_weight(config: ScoringConfig) -> None:
    assert score(config, make_inputs()).components["trend"].weight == pytest.approx(0.60)
    assert score(config, make_inputs()).components["low_competition"].weight == pytest.approx(0.25)


def test_agreement_lifts_the_score(config: ScoringConfig) -> None:
    alone = score(config, make_inputs(agreement=0.2)).value
    corroborated = score(config, make_inputs(agreement=1.0)).value
    assert corroborated > alone


def test_unmeasured_competition_is_reported_not_assumed_empty(config: ScoringConfig) -> None:
    """Treating missing data as an empty field would be the most flattering possible reading."""
    result = score(config, make_inputs(density=None))
    assert "low_competition" in result.unavailable
    assert result.qualified is False
    assert "competition_measured" in result.failed_gates
    assert "Incomplete" in " ".join(result.notes)


@pytest.mark.parametrize(
    ("kwargs", "confidence", "failed"),
    [
        ({"sample": 4}, 90.0, "minimum_evidence"),
        ({"supporting": 1}, 90.0, "independent_sources"),
        ({"spike": True}, 90.0, "not_one_record"),
        ({}, 40.0, "confidence_threshold"),
    ],
)
def test_each_eligibility_rule(
    config: ScoringConfig, kwargs: dict[str, object], confidence: float, failed: str
) -> None:
    result = score(config, make_inputs(**kwargs), confidence=confidence)
    assert failed in result.failed_gates
    assert result.qualified is False


def test_a_good_topic_qualifies(config: ScoringConfig) -> None:
    result = score(config, make_inputs())
    assert result.qualified is True
    assert result.category == "High-priority opportunity for expert review"


def test_it_never_calls_itself_a_recommendation(config: ScoringConfig) -> None:
    result = score(config, make_inputs())
    joined = " ".join(result.notes)
    assert "not a recommendation" in joined
    assert "prioritization signal" in joined


def test_scores_stay_on_the_scale(config: ScoringConfig) -> None:
    for density in (0.0, 50.0, 100.0):
        for agreement in (0.0, 1.0):
            result = score(config, make_inputs(density=density, agreement=agreement))
            assert 0.0 <= result.value <= 100.0


def test_the_result_records_its_identity(config: ScoringConfig) -> None:
    result = score(config, make_inputs())
    assert result.score_type == "opportunity" and result.entity_type == "topic"
    assert result.scoring_version == config.version
