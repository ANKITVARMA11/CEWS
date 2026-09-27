"""Unit tests for organization name normalization, classification and affiliations."""

from __future__ import annotations

import pytest

from cews.constants import MAX_ORG_NAME_LENGTH, OrganizationType
from cews.normalization.organizations import (
    classify_organization_type,
    normalize_organization_name,
    shares_distinguishing_words,
    split_affiliation,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Zentavia Pharma", "zentavia pharmaceuticals"),
        ("Zentavia Pharma, Inc.", "zentavia pharmaceuticals"),
        ("ZENTAVIA PHARMA INC", "zentavia pharmaceuticals"),
        ("Zentavia Pharmaceuticals Ltd", "zentavia pharmaceuticals"),
        ("Lumaris Tx", "lumaris therapeutics"),
        ("Varethyn Labs", "varethyn laboratories"),
        ("Nexoria Gen.", "nexoria genetics"),
        ("Procter & Gamble", "procter and gamble"),
        ("Société Générale de Biologie", "societe generale de biologie"),
        ("  The   Acme   Company  ", "acme"),
    ],
)
def test_spelling_variants_collapse_to_one_key(name: str, expected: str) -> None:
    assert normalize_organization_name(name).expanded == expected


def test_original_spelling_is_preserved() -> None:
    parsed = normalize_organization_name("  ZENTAVIA PHARMA INC  ")
    assert parsed.original == "ZENTAVIA PHARMA INC"
    assert parsed.suffixes == ("inc",)


def test_names_without_spaces_match_through_the_compact_key() -> None:
    assert normalize_organization_name("OrvexaBio").compact == (
        normalize_organization_name("Orvexa Bio").compact
    )


def test_a_name_that_is_only_a_legal_form_is_kept() -> None:
    assert normalize_organization_name("Inc").normalized == "inc"


def test_match_keys_are_unique_and_ordered() -> None:
    keys = normalize_organization_name("Orvexa Bio, Inc.").match_keys()
    assert [method for method, _ in keys][0] == "exact"
    assert len({key for _, key in keys}) == len(keys)


@pytest.mark.parametrize("value", [None, 42, ["a"]])
def test_non_string_names_are_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="must be a string"):
        normalize_organization_name(value)  # type: ignore[arg-type]


def test_absurdly_long_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="longer than"):
        normalize_organization_name("x" * (MAX_ORG_NAME_LENGTH + 1))


def test_lookalike_companies_are_not_interchangeable() -> None:
    bio = normalize_organization_name("Orvexa Bio")
    labs = normalize_organization_name("Orvexa Labs")
    assert bio.expanded != labs.expanded
    assert shares_distinguishing_words(bio, labs) is False


def test_a_legal_suffix_does_not_make_a_different_company() -> None:
    assert shares_distinguishing_words(
        normalize_organization_name("Zentavia Pharmaceuticals"),
        normalize_organization_name("Zentavia Pharmaceuticals Inc"),
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Fixture Oncology, Inc.", OrganizationType.COMPANY),
        ("Nexoria Genetics Corp.", OrganizationType.COMPANY),
        ("Halcyra Bio", OrganizationType.COMPANY),
        ("Nexoria Gen.", OrganizationType.COMPANY),
        ("Harrowgate University", OrganizationType.UNIVERSITY),
        ("Univ. of Harrowgate", OrganizationType.UNIVERSITY),
        ("St. Aldric Medical Center", OrganizationType.HOSPITAL),
        ("Fixture Clinic Osaka", OrganizationType.HOSPITAL),
        ("National Institute for Synthetic Health", OrganizationType.GOVERNMENT),
        ("Synthetic Research Council", OrganizationType.GOVERNMENT),
        ("Riverbend Research Institute", OrganizationType.NONPROFIT),
        ("Some Charitable Foundation", OrganizationType.NONPROFIT),
        ("Mystery Entity", OrganizationType.UNKNOWN),
    ],
)
def test_classification(name: str, expected: OrganizationType) -> None:
    assert classify_organization_type(name).organization_type is expected


def test_classification_explains_itself() -> None:
    result = classify_organization_type("Harrowgate University")
    assert "university" in result.reason and result.confidence >= 0.8


@pytest.mark.parametrize(
    ("affiliation", "expected"),
    [
        ("Fixture Oncology, Inc., Boston, MA, USA.", ["Fixture Oncology, Inc"]),
        ("Fixture University, Department of Medicine.", ["Fixture University"]),
        (
            "Department of Oncology, Fixture Immunotherapies GmbH, Munich, Germany.",
            ["Fixture Immunotherapies GmbH"],
        ),
        ("Fixture Vaccines GmbH, Berlin, Germany", ["Fixture Vaccines GmbH"]),
        ("Fixture Labs; Fixture University", ["Fixture Labs", "Fixture University"]),
        ("someone@example.org", []),
        ("", []),
        ("   ", []),
    ],
)
def test_affiliation_parsing(affiliation: str, expected: list[str]) -> None:
    assert split_affiliation(affiliation) == expected


def test_affiliation_parsing_caps_the_number_of_names() -> None:
    crowded = "; ".join(f"Fixture Labs {i}" for i in range(10))
    assert len(split_affiliation(crowded)) <= 3
