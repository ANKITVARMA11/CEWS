"""Unit tests for the threat score, its capped modifiers and its wording."""

from __future__ import annotations

import pytest

from cews.constants import EntityType
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.inputs import EntityInputs
from cews.scoring.modifiers import Modifier, ModifierSet
from cews.scoring.threat_score import calculate_threat_score
from cews.settings import load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

SOURCES = ("clinical_trial", "patent", "publication")


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    return load_scoring_config(settings)


def make_inputs(
    *,
    trial: float = 80.0,
    patent: float = 60.0,
    publication: float = 40.0,
    sources: tuple[str, ...] = SOURCES,
    sample: float = 150.0,
    name: str = "Example Pharma",
) -> EntityInputs:
    return EntityInputs(
        entity_type=EntityType.COMPETITOR.value,
        entity_id=1,
        name=name,
        raw={f"activity_{source}": 10.0 for source in sources},
        normalized={
            "growth_clinical_trial": trial,
            "growth_patent": patent,
            "growth_publication": publication,
        },
        sample_size=sample,
    )


def modifier_set(*items: tuple[str, float, str]) -> ModifierSet:
    applied = [Modifier(name, points, detail) for name, points, detail in items]
    return ModifierSet(applied=applied, total=sum(item.points for item in applied))


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------
def test_score_matches_a_hand_calculation(config: ScoringConfig) -> None:
    """0.50(80) + 0.30(60) + 0.20(40) = 40 + 18 + 8 = 66"""
    result = calculate_threat_score(make_inputs(), config, confidence=90.0)
    assert result.value == pytest.approx(66.0)


def test_trials_carry_the_most_weight(config: ScoringConfig) -> None:
    result = calculate_threat_score(make_inputs(), config, confidence=90.0)
    assert result.components["trial_growth"].weight == pytest.approx(0.50)
    assert result.components["publication_growth"].weight == pytest.approx(0.20)


def test_it_measures_growth_not_size(config: ScoringConfig) -> None:
    """A large company doing what it always does is not news."""
    steady = calculate_threat_score(
        make_inputs(trial=10.0, patent=10.0, publication=10.0), config, confidence=90.0
    )
    accelerating = calculate_threat_score(
        make_inputs(trial=95.0, patent=95.0, publication=95.0), config, confidence=90.0
    )
    assert accelerating.value > steady.value


def test_a_missing_source_shares_its_weight(config: ScoringConfig) -> None:
    result = calculate_threat_score(
        make_inputs(sources=("clinical_trial", "publication")), config, confidence=90.0
    )
    assert result.unavailable == ("patent_growth",)
    used = [part.weight for part in result.components.values() if part.available]
    assert sum(used) == pytest.approx(1.0)


def test_a_competitor_with_no_growth_data_cannot_be_scored(config: ScoringConfig) -> None:
    with pytest.raises(ValueError, match="no component has data"):
        calculate_threat_score(make_inputs(sources=()), config, confidence=90.0)


# --------------------------------------------------------------------------------------
# Modifiers
# --------------------------------------------------------------------------------------
def test_modifiers_add_points_on_top_of_the_base(config: ScoringConfig) -> None:
    modifiers = modifier_set(("new_therapeutic_area_entry", 5.0, "first activity in Neurology"))
    result = calculate_threat_score(make_inputs(), config, confidence=90.0, modifiers=modifiers)
    assert result.value == pytest.approx(71.0)  # 66 base + 5
    assert "Base score 66.0" in " ".join(result.notes)


def test_modifiers_are_shown_separately_not_folded_in(config: ScoringConfig) -> None:
    modifiers = modifier_set(("phase_progression", 5.0, "trials reached PHASE3"))
    result = calculate_threat_score(make_inputs(), config, confidence=90.0, modifiers=modifiers)
    detail = result.components["modifiers"].detail
    assert detail["total_points"] == 5.0
    assert detail["applied"][0]["name"] == "phase_progression"
    assert result.components["modifiers"].weight == 0.0  # not part of the weighted sum
    assert "trials reached PHASE3" in " ".join(result.notes)


def test_modifiers_cannot_push_a_score_past_the_scale(config: ScoringConfig) -> None:
    modifiers = modifier_set(("a", 10.0, "one"), ("b", 10.0, "two"))
    result = calculate_threat_score(
        make_inputs(trial=100.0, patent=100.0, publication=100.0),
        config,
        confidence=90.0,
        modifiers=modifiers,
    )
    assert result.value == 100.0


def test_no_modifiers_leaves_the_base_score_alone(config: ScoringConfig) -> None:
    result = calculate_threat_score(make_inputs(), config, confidence=90.0, modifiers=ModifierSet())
    assert result.value == pytest.approx(66.0)
    assert "modifiers" not in result.components


# --------------------------------------------------------------------------------------
# Evidence rules and wording
# --------------------------------------------------------------------------------------
def test_thin_evidence_is_flagged(config: ScoringConfig) -> None:
    """A small company can show the fastest growth simply by starting from almost nothing."""
    result = calculate_threat_score(
        make_inputs(trial=95.0, patent=95.0, publication=95.0, sample=4),
        config,
        confidence=45.0,
    )
    assert result.value > 90.0
    assert result.qualified is False
    assert set(result.failed_gates) == {"usable_confidence", "minimum_evidence"}


def test_a_well_evidenced_competitor_passes(config: ScoringConfig) -> None:
    result = calculate_threat_score(make_inputs(), config, confidence=90.0)
    assert result.qualified is True


def test_it_never_claims_an_actual_threat(config: ScoringConfig) -> None:
    result = calculate_threat_score(make_inputs(), config, confidence=90.0)
    joined = " ".join(result.notes)
    assert "not evidence of a legal, commercial or scientific threat" in joined
    assert result.category in (
        "Routine activity",
        "Worth watching",
        "Elevated competitive activity",
        "High monitoring priority",
        "Potential strategic threat requiring review",
    )


def test_every_score_gets_a_label(config: ScoringConfig) -> None:
    for level in (5.0, 45.0, 65.0, 85.0):
        result = calculate_threat_score(
            make_inputs(trial=level, patent=level, publication=level), config, confidence=90.0
        )
        assert result.category


# --------------------------------------------------------------------------------------
# Per therapeutic area
# --------------------------------------------------------------------------------------
def test_an_area_score_records_its_context(config: ScoringConfig) -> None:
    result = calculate_threat_score(
        make_inputs(),
        config,
        confidence=90.0,
        context_key="area:oncology",
        context_label="Oncology",
    )
    assert result.context_key == "area:oncology"
    assert result.entity_name == "Example Pharma in Oncology"
    assert "Limited to activity in Oncology" in " ".join(result.notes)


def test_area_and_overall_scores_stay_separate(config: ScoringConfig) -> None:
    overall = calculate_threat_score(make_inputs(), config, confidence=90.0)
    in_area = calculate_threat_score(
        make_inputs(trial=20.0),
        config,
        confidence=90.0,
        context_key="area:oncology",
        context_label="Oncology",
    )
    assert overall.context_key == "" and in_area.context_key == "area:oncology"
    assert overall.value != in_area.value
