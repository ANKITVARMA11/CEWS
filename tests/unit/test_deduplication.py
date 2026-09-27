"""Unit tests for record hashing and cross-source duplicate detection."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from cews.normalization.deduplication import (
    create_record_hash,
    duplicate_keys,
    find_duplicates,
    normalize_doi,
    normalize_title,
)

pytestmark = pytest.mark.unit

BASE = {"source": "s", "source_record_id": "1", "title": "A title", "abstract": "Body"}


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------
def test_hash_is_stable_and_ignores_formatting() -> None:
    first = create_record_hash(**{**BASE, "title": "A  title"}, payload={"x": 1, "y": 2})
    second = create_record_hash(**BASE, payload={"y": 2, "x": 1})
    assert first == second and len(first) == 64


def test_hash_changes_with_content() -> None:
    assert create_record_hash(**BASE) != create_record_hash(**{**BASE, "abstract": "Other"})


def test_hash_requires_identity() -> None:
    with pytest.raises(ValueError, match="required"):
        create_record_hash(**{**BASE, "source": ""})


def test_hash_handles_dates_and_rejects_junk() -> None:
    assert create_record_hash(**BASE, payload={"d": date(2026, 1, 1)})
    with pytest.raises(TypeError):
        create_record_hash(**BASE, payload={"bad": object()})


# --------------------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("10.1000/abc", "10.1000/abc"),
        ("https://doi.org/10.1000/ABC", "10.1000/abc"),
        ("doi:10.1000/abc ", "10.1000/abc"),
        ("  10.1000/abc.  ", "10.1000/abc"),
        ("nonsense", None),
        ("10.1000", None),
        (None, None),
        (42, None),
    ],
)
def test_normalize_doi(value: Any, expected: str | None) -> None:
    assert normalize_doi(value) == expected


def test_normalize_title_needs_enough_words() -> None:
    assert normalize_title("Targeting KRAS with FX-101!") == "targeting kras with fx 101"
    assert normalize_title("Short title") is None
    assert normalize_title(None) is None


def test_keys_are_ordered_by_reliability() -> None:
    keys = duplicate_keys(
        title="Targeting KRAS with FX-101",
        published_at=datetime(2026, 5, 1, tzinfo=UTC),
        payload={"doi": "10.1/x", "pmid": "99000001"},
    )
    assert [kind for kind, _ in keys] == ["doi", "pmid", "title"]
    assert keys[2][1].startswith("2026:")


def test_a_pubmed_record_id_counts_as_its_pmid() -> None:
    keys = duplicate_keys(
        title=None, published_at=None, payload={}, source_record_id="99000001", source="pubmed"
    )
    assert keys == [("pmid", "99000001")]


# --------------------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------------------
def _record(
    record_id: int,
    source: str,
    title: str,
    *,
    doi: str | None = None,
    pmid: str | None = None,
    year: int = 2026,
    record_type: str = "publication",
) -> Any:
    return SimpleNamespace(
        id=record_id,
        source=source,
        # for PubMed the record id is the PMID, as the real adapter stores it
        source_record_id=pmid if (source == "pubmed" and pmid) else str(record_id),
        record_type=record_type,
        title=title,
        published_at=datetime(year, 1, 1, tzinfo=UTC),
        raw_payload_json={"doi": doi, "pmid": pmid},
    )


def test_the_same_doi_in_two_sources_is_one_work() -> None:
    links = find_duplicates(
        [
            _record(1, "pubmed", "Targeting KRAS with FX-101", doi="10.1/x"),
            _record(2, "europe_pmc", "Targeting KRAS with FX-101 (preprint)", doi="10.1/X"),
        ]
    )
    assert [(link.record_id, link.canonical_id, link.reason) for link in links] == [(2, 1, "doi")]


def test_the_first_record_seen_is_the_canonical_one() -> None:
    links = find_duplicates(
        [
            _record(7, "europe_pmc", "A paper", doi="10.1/x"),
            _record(3, "pubmed", "A paper", doi="10.1/x"),
        ]
    )
    assert links[0].canonical_id == 3


def test_a_shared_pmid_links_records() -> None:
    links = find_duplicates(
        [
            _record(1, "pubmed", "Some longer title here", pmid="99000001"),
            _record(2, "europe_pmc", "Some longer title here", pmid="99000001"),
        ]
    )
    assert links[0].reason == "pmid"


def test_titles_only_link_across_different_sources() -> None:
    same_source = find_duplicates(
        [
            _record(1, "pubmed", "A common experimental protocol paper"),
            _record(2, "pubmed", "A common experimental protocol paper"),
        ]
    )
    assert same_source == []  # two distinct records that happen to share a title
    cross_source = find_duplicates(
        [
            _record(1, "pubmed", "A common experimental protocol paper"),
            _record(2, "europe_pmc", "A common experimental protocol paper"),
        ]
    )
    assert [link.reason for link in cross_source] == ["title"]


def test_the_same_title_in_another_year_is_a_different_work() -> None:
    assert (
        find_duplicates(
            [
                _record(1, "pubmed", "Annual report of the fixture consortium", year=2024),
                _record(2, "europe_pmc", "Annual report of the fixture consortium", year=2026),
            ]
        )
        == []
    )


def test_record_types_are_never_mixed() -> None:
    assert (
        find_duplicates(
            [
                _record(1, "pubmed", "A shared long title", doi="10.1/x"),
                _record(2, "patents", "A shared long title", doi="10.1/x", record_type="patent"),
            ]
        )
        == []
    )


def test_three_copies_all_point_at_the_first() -> None:
    links = find_duplicates(
        [
            _record(1, "pubmed", "A paper with a long enough title", doi="10.1/x"),
            _record(2, "europe_pmc", "A paper with a long enough title", doi="10.1/x"),
            _record(3, "openalex", "A paper with a long enough title", doi="10.1/x"),
        ]
    )
    assert {link.canonical_id for link in links} == {1}
    assert {link.record_id for link in links} == {2, 3}


def test_unrelated_records_are_left_alone() -> None:
    assert (
        find_duplicates(
            [_record(1, "pubmed", "One title here"), _record(2, "pubmed", "Another title here")]
        )
        == []
    )
    assert find_duplicates([]) == []
