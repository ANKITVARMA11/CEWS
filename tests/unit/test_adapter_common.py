"""Unit tests for the helpers shared by the source adapters."""

from __future__ import annotations

from typing import Any

import pytest

from cews.ingestion.adapters.common import (
    as_list,
    clean_text,
    clip,
    dig,
    option_bool,
    option_query,
    or_query,
    search_terms,
    unique_texts,
)
from cews.ingestion.errors import AdapterConfigError
from cews.settings import load_settings
from support_sources import registry_config

pytestmark = pytest.mark.unit


def config(**options: Any) -> Any:
    return registry_config("pubmed", options=options)


def test_search_terms_default_to_the_monitored_areas() -> None:
    settings = load_settings(env_file=None)
    assert search_terms(settings, config()) == settings.therapeutic_area_list


def test_query_terms_option_replaces_them() -> None:
    settings = load_settings(env_file=None)
    assert search_terms(settings, config(query_terms=["CRISPR", " base  editing "])) == [
        "CRISPR",
        "base editing",
    ]


@pytest.mark.parametrize(
    ("overrides", "options", "message"),
    [
        ({"therapeutic_areas": ""}, {}, "no search terms"),
        ({}, {"query_terms": "CRISPR"}, "list of strings"),
        ({}, {"query_terms": [1, 2]}, "list of strings"),
        ({}, {"query_terms": ['say "hi"']}, "quotes"),
        ({}, {"query_terms": [f"term{i}" for i in range(60)]}, "at most"),
    ],
)
def test_invalid_search_terms(
    overrides: dict[str, Any], options: dict[str, Any], message: str
) -> None:
    settings = load_settings(env_file=None, overrides=overrides)
    with pytest.raises(AdapterConfigError, match=message):
        search_terms(settings, config(**options))


@pytest.mark.parametrize(
    ("terms", "quote", "expected"),
    [
        (["Oncology"], True, "Oncology"),
        (["Oncology", "Neurology"], True, "(Oncology OR Neurology)"),
        (["Rare Diseases"], True, '"Rare Diseases"'),
        (["Rare Diseases"], False, "Rare Diseases"),  # PubMed keeps automatic term mapping
        (["CAR-T"], False, '"CAR-T"'),  # punctuation always quoted
        (["CAR-T", "mRNA"], True, '("CAR-T" OR mRNA)'),
    ],
)
def test_or_query(terms: list[str], quote: bool, expected: str) -> None:
    assert or_query(terms, quote_phrases=quote) == expected


def test_option_query_and_option_bool() -> None:
    assert option_query(config()) is None
    assert option_query(config(query="  x  ")) == "x"
    with pytest.raises(AdapterConfigError):
        option_query(config(query=""))
    assert option_bool(config(), "store_abstracts", True) is True
    assert option_bool(config(store_abstracts=False), "store_abstracts", True) is False
    with pytest.raises(AdapterConfigError, match="store_abstracts"):
        option_bool(config(store_abstracts="no"), "store_abstracts", True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<p>Hello <b>world</b></p>", "Hello world"),
        ("a &amp; b", "a & b"),
        ("  spaced   out \n text ", "spaced out text"),
        ("&lt;notatag&gt;", "<notatag>"),
        ("", None),
        (None, None),
        ("   ", None),
        (42, "42"),
    ],
)
def test_clean_text(value: Any, expected: str | None) -> None:
    assert clean_text(value) == expected


def test_clip() -> None:
    assert clip("abcdef", 10) == "abcdef"
    assert clip("abcdef", 4) == "abc…"
    assert len(clip("x" * 500, 255) or "") == 255
    assert clip(None, 5) is None


def test_dig_and_as_list() -> None:
    data = {"a": {"b": {"c": 1}}}
    assert dig(data, "a", "b", "c") == 1
    assert dig(data, "a", "missing", "c") is None
    assert dig(None, "a") is None
    assert as_list(None) == [] and as_list([1]) == [1] and as_list("x") == ["x"]


def test_unique_texts_keeps_order_and_caps() -> None:
    assert unique_texts(["b", "a", "b", None, "  a  "]) == ["b", "a"]
    assert unique_texts(["<i>x</i>", "x"]) == ["x"]
    assert len(unique_texts([f"item {i}" for i in range(100)], limit=10)) == 10
