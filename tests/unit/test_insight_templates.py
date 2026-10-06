"""Unit tests for the insight text templates.

Pure functions, no database: given the same numbers, they must always produce the same three
sentences, and the three parts (fact, interpretation, recommendation) must stay genuinely
separate rather than repeating each other.
"""

from __future__ import annotations

import pytest

from cews.insights.templates import (
    competitor_movement_text,
    emerging_trend_text,
    new_market_entry_text,
    opportunity_text,
    patent_surge_text,
)

pytestmark = pytest.mark.unit


def test_emerging_trend_text_states_the_fact_and_the_judgement_separately() -> None:
    text = emerging_trend_text(
        "CRISPR gene editing", score=78.0, confidence=90.0, supporting_sources=3, sample_size=442
    )
    assert "CRISPR gene editing" in text.title
    assert "78" in text.observed_fact and "442" in text.observed_fact and "3" in text.observed_fact
    assert "genuine trend" in text.interpretation
    assert "90" in text.interpretation
    assert "expert" in text.recommended_review.lower()
    # the three parts are not just copies of each other
    assert text.observed_fact != text.interpretation != text.recommended_review


def test_competitor_movement_text_describes_moving_up() -> None:
    text = competitor_movement_text(
        "Halcyra Biosciences", previous_rank=6, current_rank=2, score=71.0
    )
    assert "up" in text.title
    assert "6" in text.observed_fact and "2" in text.observed_fact
    assert "increased" in text.interpretation


def test_competitor_movement_text_describes_moving_down() -> None:
    text = competitor_movement_text(
        "Halcyra Biosciences", previous_rank=2, current_rank=6, score=40.0
    )
    assert "down" in text.title
    assert "decreased" in text.interpretation


def test_new_market_entry_text_names_the_area_and_the_silence() -> None:
    text = new_market_entry_text(
        "Zentavia Pharma", area_name="Neurodegeneration", records_in_window=15, quiet_months=18
    )
    assert "Neurodegeneration" in text.title
    assert "15" in text.observed_fact and "18" in text.observed_fact
    assert "first move" in text.interpretation


def test_patent_surge_text_names_the_growth_and_the_kind_of_entity() -> None:
    topic_text = patent_surge_text(
        "CRISPR gene editing", "topic", growth_percent=39.0, sample_size=8
    )
    assert "topic CRISPR gene editing" in topic_text.observed_fact
    assert "39" in topic_text.observed_fact and "8" in topic_text.observed_fact

    competitor_text = patent_surge_text(
        "Orvexa Bio", "competitor", growth_percent=66.0, sample_size=8
    )
    assert "competitor Orvexa Bio" in competitor_text.observed_fact


def test_opportunity_text_is_never_worded_as_a_recommendation() -> None:
    text = opportunity_text(
        "AI-assisted drug discovery",
        score=73.0,
        confidence=94.0,
        competition_note="the field is open",
    )
    assert "not an investment, commercial or scientific recommendation" in text.interpretation
    assert "the field is open" in text.observed_fact


@pytest.mark.parametrize(
    "build",
    [
        lambda: emerging_trend_text(
            "X", score=1, confidence=1, supporting_sources=1, sample_size=1
        ),
        lambda: competitor_movement_text("X", previous_rank=1, current_rank=2, score=1),
        lambda: new_market_entry_text("X", area_name="Y", records_in_window=1, quiet_months=1),
        lambda: patent_surge_text("X", "topic", growth_percent=1, sample_size=1),
        lambda: opportunity_text("X", score=1, confidence=1, competition_note="Z"),
    ],
)
def test_every_template_fills_all_three_fields(build: object) -> None:
    text = build()  # type: ignore[operator]
    assert text.title and text.observed_fact and text.interpretation and text.recommended_review


def test_templates_are_deterministic() -> None:
    args = {"score": 60.0, "confidence": 80.0, "supporting_sources": 2, "sample_size": 50}
    first = emerging_trend_text("Oncology", **args)
    second = emerging_trend_text("Oncology", **args)
    assert first == second
