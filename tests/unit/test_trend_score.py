"""Unit tests for the trend score and the rules that decide what counts as a finding."""

from __future__ import annotations

import pytest

from cews.constants import EntityType
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.inputs import EntityInputs
from cews.scoring.trend_score import COMPONENT_FEATURES, calculate_trend_score
from cews.settings import load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

STRONG_FEATURES = {
    "velocity": 80.0,
    "momentum": 60.0,
    "growth_patent": 50.0,
    "growth_funding": 40.0,
    "consistency": 90.0,
}


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    return load_scoring_config(settings)


def make_inputs(
    *,
    normalized: dict[str, float] | None = None,
    sources: tuple[str, ...] = ("publication", "clinical_trial", "patent", "funding"),
    supporting: int = 3,
    sample: float = 200.0,
    spike: bool = False,
    name: str = "A topic",
) -> EntityInputs:
    raw: dict[str, float] = {f"activity_{source}": 10.0 for source in sources}
    raw["supporting_sources"] = float(supporting)
    raw["single_spike"] = 1.0 if spike else 0.0
    return EntityInputs(
        entity_type=EntityType.TOPIC.value,
        entity_id=1,
        name=name,
        raw=raw,
        normalized=dict(normalized if normalized is not None else STRONG_FEATURES),
        sample_size=sample,
    )


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------
def test_score_matches_a_hand_calculation(config: ScoringConfig) -> None:
    """0.30(80) + 0.25(60) + 0.20(50) + 0.15(40) + 0.10(90) = 24 + 15 + 10 + 6 + 9 = 64"""
    result = calculate_trend_score(make_inputs(), config, confidence=90.0)
    assert result.value == pytest.approx(64.0)
    assert sum(part.contribution for part in result.components.values()) == pytest.approx(64.0)


def test_components_use_the_configured_weights(config: ScoringConfig) -> None:
    result = calculate_trend_score(make_inputs(), config, confidence=90.0)
    assert result.components["velocity"].weight == pytest.approx(0.30)
    assert result.components["consistency"].weight == pytest.approx(0.10)
    assert set(result.components) == set(COMPONENT_FEATURES)


def test_a_flat_topic_scores_in_the_middle(config: ScoringConfig) -> None:
    middling = dict.fromkeys(STRONG_FEATURES, 50.0)
    assert calculate_trend_score(
        make_inputs(normalized=middling), config, confidence=90.0
    ).value == pytest.approx(50.0)


def test_scores_stay_on_the_scale(config: ScoringConfig) -> None:
    for level in (0.0, 100.0):
        result = calculate_trend_score(
            make_inputs(normalized=dict.fromkeys(STRONG_FEATURES, level)), config, confidence=50.0
        )
        assert 0.0 <= result.value <= 100.0


# --------------------------------------------------------------------------------------
# Missing sources
# --------------------------------------------------------------------------------------
def test_a_missing_source_shares_its_weight_instead_of_scoring_zero(
    config: ScoringConfig,
) -> None:
    """With no patents anywhere, patent growth is unknown, not zero."""
    without_patents = make_inputs(sources=("publication", "clinical_trial", "funding"))
    result = calculate_trend_score(without_patents, config, confidence=90.0)
    assert result.unavailable == ("patent_growth",)
    assert result.components["patent_growth"].available is False
    used = [part.weight for part in result.components.values() if part.available]
    assert sum(used) == pytest.approx(1.0)
    # the remaining components are worth more, so the score is not dragged down
    assert result.value > 64.0


def test_the_reason_a_component_is_missing_is_recorded(config: ScoringConfig) -> None:
    result = calculate_trend_score(make_inputs(sources=("publication",)), config, confidence=90.0)
    assert "no patent records" in str(result.components["patent_growth"].detail)
    assert "shared across" in result.explain()


def test_velocity_that_could_not_be_measured_is_unavailable(config: ScoringConfig) -> None:
    """Too few months to fit a line is a gap, not a zero."""
    without_velocity = {k: v for k, v in STRONG_FEATURES.items() if k != "velocity"}
    result = calculate_trend_score(
        make_inputs(normalized=without_velocity), config, confidence=90.0
    )
    assert "velocity" in result.unavailable
    assert "not enough months" in str(result.components["velocity"].detail)


def test_a_topic_with_nothing_measurable_is_refused(config: ScoringConfig) -> None:
    with pytest.raises(ValueError, match="no component has data"):
        calculate_trend_score(make_inputs(normalized={}, sources=()), config, confidence=90.0)


# --------------------------------------------------------------------------------------
# The rules
# --------------------------------------------------------------------------------------
def test_a_strong_well_evidenced_topic_qualifies(config: ScoringConfig) -> None:
    high = dict.fromkeys(STRONG_FEATURES, 85.0)
    result = calculate_trend_score(make_inputs(normalized=high), config, confidence=90.0)
    assert result.qualified is True
    assert result.category == "High priority"
    assert "Qualifies as an emerging trend" in result.explain()


def test_a_high_score_on_thin_evidence_does_not_qualify(config: ScoringConfig) -> None:
    """The case the whole design exists for."""
    high = dict.fromkeys(STRONG_FEATURES, 95.0)
    result = calculate_trend_score(
        make_inputs(normalized=high, sample=6, supporting=1, spike=True, sources=("publication",)),
        config,
        confidence=52.0,
    )
    assert result.value > 90.0  # the number looks spectacular
    assert result.qualified is False  # and it is still not a finding
    assert set(result.failed_gates) == {
        "confidence_threshold",
        "minimum_evidence",
        "independent_sources",
        "not_a_single_spike",
    }


@pytest.mark.parametrize(
    ("kwargs", "confidence", "failed"),
    [
        ({"sample": 3}, 90.0, "minimum_evidence"),
        ({"supporting": 1}, 90.0, "independent_sources"),
        ({"spike": True}, 90.0, "not_a_single_spike"),
        ({}, 40.0, "confidence_threshold"),
    ],
)
def test_each_rule_can_fail_on_its_own(
    config: ScoringConfig, kwargs: dict[str, object], confidence: float, failed: str
) -> None:
    high = dict.fromkeys(STRONG_FEATURES, 85.0)
    result = calculate_trend_score(
        make_inputs(normalized=high, **kwargs), config, confidence=confidence
    )
    assert result.failed_gates == (failed,)
    assert result.qualified is False


def test_a_middling_score_fails_the_threshold(config: ScoringConfig) -> None:
    """Well evidenced but simply not moving enough to be called a trend."""
    flat = dict.fromkeys(STRONG_FEATURES, 45.0)
    result = calculate_trend_score(make_inputs(normalized=flat), config, confidence=90.0)
    assert result.value < config.emerging_trend.min_trend_score
    assert result.failed_gates == ("score_threshold",)


def test_every_rule_explains_itself(config: ScoringConfig) -> None:
    result = calculate_trend_score(make_inputs(sample=2, supporting=1), config, confidence=30.0)
    text = result.explain()
    for gate in result.gates:
        if not gate.passed:
            assert gate.detail in text


def test_categories_come_from_the_configuration(config: ScoringConfig) -> None:
    for level, expected in (
        (10.0, "Low activity or declining"),
        (50.0, "Watchlist"),
        (70.0, "Emerging"),
        (95.0, "High priority"),
    ):
        result = calculate_trend_score(
            make_inputs(normalized=dict.fromkeys(STRONG_FEATURES, level)), config, confidence=90.0
        )
        assert result.category == expected


def test_the_result_records_its_identity(config: ScoringConfig) -> None:
    result = calculate_trend_score(make_inputs(name="CRISPR"), config, confidence=90.0)
    assert result.score_type == "trend" and result.entity_type == "topic"
    assert result.entity_name == "CRISPR" and result.scoring_version == config.version
    assert result.sample_size == 200.0
