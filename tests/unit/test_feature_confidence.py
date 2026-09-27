"""Unit tests for competition density, source agreement and the confidence inputs."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from cews.features.competition import calculate_competition_density
from cews.features.sample_confidence import (
    calculate_data_completeness,
    calculate_data_freshness,
    calculate_sample_confidence,
)
from cews.features.source_agreement import calculate_source_agreement

pytestmark = pytest.mark.unit
NOW = datetime(2026, 9, 1, tzinfo=UTC)


# --------------------------------------------------------------------------------------
# Competition density
# --------------------------------------------------------------------------------------
def test_more_players_means_a_busier_topic() -> None:
    quiet = calculate_competition_density({"A": 10, "B": 9})
    busy = calculate_competition_density({f"C{index}": 10 for index in range(10)})
    assert busy.density > quiet.density
    assert busy.active_competitors == 10


def test_one_dominant_player_is_not_a_crowded_field() -> None:
    shared = calculate_competition_density({"A": 10, "B": 10, "C": 10})
    dominated = calculate_competition_density({"A": 96, "B": 2, "C": 2})
    assert dominated.concentration > shared.concentration
    assert dominated.density < shared.density
    assert dominated.leader == "A" and dominated.leader_share == pytest.approx(0.96)


def test_passing_mentions_do_not_make_a_field_look_crowded() -> None:
    result = calculate_competition_density({"A": 10, "B": 0.4, "C": 0.2}, min_activity=1.0)
    assert result.active_competitors == 1 and result.counted_competitors == 3


def test_an_empty_topic_has_no_competition() -> None:
    result = calculate_competition_density({})
    assert result.active_competitors == 0 and result.density == 0.0
    assert result.leader is None


def test_all_zero_activity() -> None:
    result = calculate_competition_density({"A": 0.0, "B": 0.0})
    assert result.active_competitors == 0 and result.density == 0.0


def test_density_stays_on_the_scale() -> None:
    for count in (1, 3, 8, 40):
        result = calculate_competition_density({f"C{index}": 5 for index in range(count)})
        assert 0.0 <= result.density <= 100.0


@pytest.mark.parametrize("activity", [{"A": -1.0}, {"A": math.inf}])
def test_invalid_activity_is_rejected(activity: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        calculate_competition_density(activity)


def test_invalid_saturation_is_rejected() -> None:
    with pytest.raises(ValueError, match="saturation"):
        calculate_competition_density({"A": 1.0}, saturation=0)


# --------------------------------------------------------------------------------------
# Source agreement
# --------------------------------------------------------------------------------------
def test_agreement_counts_only_sources_with_data() -> None:
    result = calculate_source_agreement(
        {"trial": 0.4, "patent": 0.2, "publication": -0.1, "funding": None, "announcement": 0.0}
    )
    assert result.supporting == ("patent", "trial")
    assert result.contradicting == ("publication",)
    assert result.flat == ("announcement",)
    assert result.unavailable == ("funding",)
    assert result.agreement == pytest.approx(0.5)  # 2 of the 4 sources with data
    assert result.available_count == 4


def test_a_disabled_source_neither_helps_nor_hurts() -> None:
    both_growing = calculate_source_agreement({"trial": 0.5, "patent": 0.5})
    with_missing = calculate_source_agreement({"trial": 0.5, "patent": 0.5, "funding": None})
    assert both_growing.agreement == with_missing.agreement == 1.0


def test_a_single_source_cannot_corroborate_itself() -> None:
    result = calculate_source_agreement({"publication": 2.0})
    assert result.agreement == 1.0
    assert result.multi_source is False  # the flag the trend rules gate on


def test_two_sources_agreeing_is_the_bar() -> None:
    assert calculate_source_agreement({"trial": 0.1, "patent": 0.1}).multi_source is True


def test_no_data_at_all() -> None:
    result = calculate_source_agreement({"trial": None, "patent": None})
    assert result.agreement == 0.0 and result.available_count == 0


def test_non_finite_growth_is_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_source_agreement({"trial": math.inf})


# --------------------------------------------------------------------------------------
# Confidence inputs
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("records", "expected"), [(0, 0.0), (2, 0.077), (10, 0.33), (25, 0.632), (100, 0.982)]
)
def test_sample_confidence_saturates(records: float, expected: float) -> None:
    assert calculate_sample_confidence(records) == pytest.approx(expected, abs=0.001)


def test_sample_confidence_rises_with_evidence() -> None:
    values = [calculate_sample_confidence(n) for n in (1, 5, 20, 50, 200)]
    assert values == sorted(values)
    assert all(0.0 <= value < 1.0 for value in values)


def test_saturation_constant_moves_the_curve() -> None:
    strict = calculate_sample_confidence(25, k=100)
    lenient = calculate_sample_confidence(25, k=5)
    assert strict < lenient


@pytest.mark.parametrize(("records", "k"), [(-1, 25), (math.inf, 25), (10, 0), (10, -5)])
def test_invalid_sample_confidence_input(records: float, k: float) -> None:
    with pytest.raises(ValueError):
        calculate_sample_confidence(records, k=k)


@pytest.mark.parametrize(("age_days", "expected"), [(0, 1.0), (30, 0.5), (60, 0.25), (90, 0.125)])
def test_freshness_halves_with_age(age_days: int, expected: float) -> None:
    stamp = NOW - timedelta(days=age_days)
    assert calculate_data_freshness(stamp, as_of=NOW) == pytest.approx(expected, abs=0.001)


def test_never_collected_is_not_fresh() -> None:
    assert calculate_data_freshness(None) == 0.0


def test_a_future_timestamp_is_treated_as_current() -> None:
    assert calculate_data_freshness(NOW + timedelta(days=1), as_of=NOW) == 1.0


def test_freshness_needs_timezone_aware_times() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate_data_freshness(datetime(2026, 8, 1), as_of=NOW)


def test_invalid_half_life_is_rejected() -> None:
    with pytest.raises(ValueError, match="half_life_days"):
        calculate_data_freshness(NOW, as_of=NOW, half_life_days=0)


@pytest.mark.parametrize(
    ("present", "expected"),
    [([True, True, False, True], 0.75), ([True], 1.0), ([False, False], 0.0)],
)
def test_completeness(present: list[bool], expected: float) -> None:
    assert calculate_data_completeness(present) == pytest.approx(expected)


def test_completeness_accepts_named_inputs() -> None:
    assert calculate_data_completeness({"title": True, "abstract": False}) == pytest.approx(0.5)


def test_completeness_needs_something_to_count() -> None:
    with pytest.raises(ValueError, match="at least one"):
        calculate_data_completeness([])
