"""Unit tests for keyword-based topic assignment."""

from __future__ import annotations

from typing import Any

import pytest

from cews.normalization.topics import TopicMatcher, load_taxonomy
from support import TAXONOMY_FILE

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def matcher() -> TopicMatcher:
    return TopicMatcher(load_taxonomy(TAXONOMY_FILE))


def keys(matches: list[Any]) -> set[str]:
    return {match.topic_key for match in matches}


def test_a_topic_is_found_in_the_title(matcher: TopicMatcher) -> None:
    matches = matcher.match({"title": "CRISPR base editing in primary cells"})
    assert keys(matches) == {"crispr_gene_editing"}  # the area has no term in this text
    assert matches[0].matched_fields == ("title",)
    assert set(matches[0].matched_terms) >= {"CRISPR", "base editing"}


def test_several_topics_can_be_assigned(matcher: TopicMatcher) -> None:
    matches = matcher.match(
        {"title": "mRNA vaccines", "abstract": "We combine PD-1 blockade with CAR-T cells."}
    )
    assert keys(matches) >= {"mrna_therapeutics", "immune_checkpoint"}


def test_a_title_match_outranks_an_abstract_match(matcher: TopicMatcher) -> None:
    in_title = matcher.match({"title": "CRISPR study"})[0]
    in_abstract = matcher.match({"abstract": "CRISPR study"})[0]
    assert in_title.confidence > in_abstract.confidence


def test_more_matching_terms_raise_confidence(matcher: TopicMatcher) -> None:
    one = matcher.match({"abstract": "CRISPR"})[0]
    several = matcher.match({"abstract": "CRISPR, Cas9 and prime editing"})[0]
    assert several.confidence > one.confidence
    assert several.confidence <= 0.95


def test_structured_metadata_counts_as_strong_evidence(matcher: TopicMatcher) -> None:
    matches = matcher.match({"title": "A trial", "mesh_terms": ["amyloid beta"]})
    assert "neurodegeneration" in keys(matches)
    assert matches[0].confidence >= 0.8


def test_separators_and_plurals_are_tolerated(matcher: TopicMatcher) -> None:
    for text in ("CAR-T therapy", "CAR T cells", "mRNA vaccines", "mRNA vaccine"):
        assert matcher.match({"title": text}), text


def test_terms_inside_other_words_do_not_match(matcher: TopicMatcher) -> None:
    assert matcher.match({"title": "SCARTER protocol and AAVX carrier"}) == []


def test_unrelated_text_matches_nothing(matcher: TopicMatcher) -> None:
    assert matcher.match({"title": "Supply chain logistics", "abstract": "Nothing here."}) == []


def test_empty_input_is_handled(matcher: TopicMatcher) -> None:
    assert matcher.match({}) == []
    assert matcher.match({"title": None, "abstract": None}) == []
    assert matcher.match({"keywords": []}) == []


def test_a_topic_missing_from_the_taxonomy_is_not_invented(matcher: TopicMatcher) -> None:
    # PROTAC / targeted protein degradation is deliberately absent; AI discovery covers it later.
    assert (
        matcher.match({"title": "PROTAC degrader", "abstract": "targeted protein degradation"})
        == []
    )


def test_results_are_sorted_and_deterministic(matcher: TopicMatcher) -> None:
    fields = {"title": "CRISPR and mRNA vaccine", "abstract": "PD-1"}
    first = matcher.match(fields)
    assert [m.confidence for m in first] == sorted((m.confidence for m in first), reverse=True)
    assert [m.topic_key for m in first] == [m.topic_key for m in matcher.match(fields)]


def test_the_confidence_floor_can_be_raised(matcher: TopicMatcher) -> None:
    strict = TopicMatcher(load_taxonomy(TAXONOMY_FILE), min_confidence=0.9)
    assert strict.match({"abstract": "CRISPR"}) == []
    assert strict.match({"title": "CRISPR, Cas9 and prime editing"})


def test_list_fields_are_searched(matcher: TopicMatcher) -> None:
    assert "immune_checkpoint" in keys(
        matcher.match({"conditions": ["Melanoma"], "keywords": ["PD-L1"]})
    )


def test_therapeutic_areas_are_matchable_topics(matcher: TopicMatcher) -> None:
    # "CAR-T" is a monitored area with no narrower topic of its own.
    assert "area:car_t" in keys(matcher.match({"title": "CAR-T therapy outcomes"}))
    assert "area:oncology" in keys(matcher.match({"abstract": "a solid tumor cohort"}))
