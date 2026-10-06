"""Unit tests for the innovation score and competitor ranking."""

from __future__ import annotations

import pytest

from cews.constants import EntityType
from cews.scoring.base import ScoreResult
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.innovation_score import (
    assign_ranks,
    calculate_innovation_score,
    cohort_sources,
)
from cews.scoring.inputs import EntityInputs
from cews.settings import load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

ALL_SOURCES = {"patent", "clinical_trial", "publication", "funding"}


@pytest.fixture(scope="module")
def config() -> ScoringConfig:
    settings = load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})
    return load_scoring_config(settings)


def make_inputs(
    *,
    patent: float = 80.0,
    trial: float = 60.0,
    publication: float = 40.0,
    funding: float = 20.0,
    activity: dict[str, float] | None = None,
    entity_id: int = 1,
    name: str = "Example Pharma",
    sample: float = 150.0,
) -> EntityInputs:
    raw = {f"activity_{source}": 10.0 for source in ALL_SOURCES}
    if activity is not None:
        raw = {f"activity_{source}": value for source, value in activity.items()}
    return EntityInputs(
        entity_type=EntityType.COMPETITOR.value,
        entity_id=entity_id,
        name=name,
        raw=raw,
        normalized={
            "activity_patent": patent,
            "activity_clinical_trial": trial,
            "activity_publication": publication,
            "activity_funding": funding,
        },
        sample_size=sample,
    )


# --------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------
def test_score_matches_a_hand_calculation(config: ScoringConfig) -> None:
    """0.40(80) + 0.30(60) + 0.20(40) + 0.10(20) = 32 + 18 + 8 + 2 = 60"""
    result = calculate_innovation_score(
        make_inputs(), config, confidence=90.0, available_sources=ALL_SOURCES
    )
    assert result.value == pytest.approx(60.0)
    assert sum(part.contribution for part in result.components.values()) == pytest.approx(60.0)


@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        (10.0, "Limited recent innovation"),
        (50.0, "Moderate innovation activity"),
        (70.0, "Strong innovation activity"),
        (95.0, "Innovation leader"),
    ],
)
def test_the_configured_label_is_applied(
    config: ScoringConfig, strength: float, expected: str
) -> None:
    """Regression: innovation scores were the only kind that never received their label."""
    result = calculate_innovation_score(
        make_inputs(patent=strength, trial=strength, publication=strength, funding=strength),
        config,
        confidence=90.0,
        available_sources=ALL_SOURCES,
    )
    assert result.category == expected


def test_patents_carry_the_most_weight(config: ScoringConfig) -> None:
    result = calculate_innovation_score(
        make_inputs(), config, confidence=90.0, available_sources=ALL_SOURCES
    )
    assert result.components["patent"].weight == pytest.approx(0.40)
    assert result.components["funding"].weight == pytest.approx(0.10)


def test_it_measures_output_not_growth(config: ScoringConfig) -> None:
    """A company doing a lot scores high even if it is not accelerating."""
    busy = calculate_innovation_score(
        make_inputs(patent=95.0, trial=95.0, publication=95.0, funding=95.0),
        config,
        confidence=90.0,
        available_sources=ALL_SOURCES,
    )
    assert busy.value == pytest.approx(95.0)


# --------------------------------------------------------------------------------------
# A real zero is not a missing source
# --------------------------------------------------------------------------------------
def test_a_company_that_files_no_patents_scores_low_not_unavailable(
    config: ScoringConfig,
) -> None:
    """Others file patents, so this is a finding about the company, not a gap in the data."""
    no_patents = make_inputs(
        patent=0.0,
        activity={"patent": 0.0, "clinical_trial": 10.0, "publication": 10.0, "funding": 10.0},
    )
    result = calculate_innovation_score(
        no_patents, config, confidence=90.0, available_sources=ALL_SOURCES
    )
    assert result.unavailable == ()
    assert result.components["patent"].available is True
    assert result.components["patent"].normalized == 0.0
    assert result.value == pytest.approx(28.0)  # 0 + 18 + 8 + 2


def test_a_source_nobody_has_is_treated_as_missing(config: ScoringConfig) -> None:
    """Patents switched off for everyone must not push every competitor down."""
    result = calculate_innovation_score(
        make_inputs(),
        config,
        confidence=90.0,
        available_sources={"clinical_trial", "publication", "funding"},
    )
    assert result.unavailable == ("patent",)
    assert "no patent data for any competitor" in str(result.components["patent"].detail)
    used = [part.weight for part in result.components.values() if part.available]
    assert sum(used) == pytest.approx(1.0)
    # 60, 40 and 20 reweighted across 0.30/0.20/0.10 -> 46.67
    assert result.value == pytest.approx(46.667, abs=0.01)


def test_cohort_sources_collects_what_anyone_has() -> None:
    quiet = make_inputs(activity={"publication": 5.0})
    busy = make_inputs(activity={"patent": 5.0, "clinical_trial": 5.0})
    assert cohort_sources([quiet, busy]) == {"publication", "patent", "clinical_trial"}


def test_without_a_cohort_only_the_companys_own_sources_count(config: ScoringConfig) -> None:
    result = calculate_innovation_score(
        make_inputs(activity={"publication": 5.0}), config, confidence=90.0
    )
    assert set(result.unavailable) == {"patent", "clinical_trial", "funding"}


def test_a_company_with_no_activity_at_all_cannot_be_scored(config: ScoringConfig) -> None:
    with pytest.raises(ValueError, match="no component has data"):
        calculate_innovation_score(
            make_inputs(activity={}), config, confidence=90.0, available_sources=set()
        )


# --------------------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------------------
def score(config: ScoringConfig, value: float, entity_id: int, name: str) -> ScoreResult:
    return calculate_innovation_score(
        make_inputs(
            patent=value,
            trial=value,
            publication=value,
            funding=value,
            entity_id=entity_id,
            name=name,
        ),
        config,
        confidence=90.0,
        available_sources=ALL_SOURCES,
    )


def test_ranks_are_assigned_highest_first(config: ScoringConfig) -> None:
    results = [score(config, 90, 1, "A"), score(config, 50, 2, "B"), score(config, 70, 3, "C")]
    assign_ranks(results)
    ranks = {result.entity_name: result.components["rank"].raw_value for result in results}
    assert ranks == {"A": 1.0, "C": 2.0, "B": 3.0}
    assert "Ranked 1 of 3" in " ".join(results[0].notes)


def test_ties_share_a_rank(config: ScoringConfig) -> None:
    results = [score(config, 70, 1, "A"), score(config, 70, 2, "B"), score(config, 10, 3, "C")]
    assign_ranks(results)
    ranks = {r.entity_name: r.components["rank"].raw_value for r in results}
    assert ranks["A"] == ranks["B"] == 1.0
    assert ranks["C"] == 3.0


def test_rank_change_is_reported(config: ScoringConfig) -> None:
    """Leadership reads positions before numbers."""
    results = [score(config, 90, 1, "A"), score(config, 50, 2, "B")]
    assign_ranks(results, previous_values={1: 10.0, 2: 80.0})  # A was 2nd, B was 1st
    moved = {r.entity_name: r.components["rank"].detail.get("rank_change") for r in results}
    assert moved == {"A": 1, "B": -1}
    assert "Moved up 1 place(s)" in " ".join(results[0].notes)


def test_rank_is_recorded_without_affecting_the_score(config: ScoringConfig) -> None:
    results = [score(config, 90, 1, "A")]
    before = results[0].value
    assign_ranks(results)
    assert results[0].value == before
    assert results[0].components["rank"].weight == 0.0
    assert results[0].components["rank"].available is False


def test_change_since_the_previous_run_is_noted(config: ScoringConfig) -> None:
    result = calculate_innovation_score(
        make_inputs(),
        config,
        confidence=90.0,
        available_sources=ALL_SOURCES,
        previous_value=50.0,
    )
    assert "up 10.0 points" in " ".join(result.notes)
