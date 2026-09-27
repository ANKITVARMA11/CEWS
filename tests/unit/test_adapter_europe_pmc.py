"""Unit tests for the Europe PMC adapter: query building, cursor paging, preprint handling."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from typing import Any

import pytest

from cews.ingestion.adapters.europe_pmc import EuropePmcAdapter
from cews.ingestion.errors import SourceParseError
from cews.ingestion.results import CollectionWindow, RawPage, RequestSpec
from cews.settings import Settings, load_settings
from support_sources import EuropePmcServer, build_adapter, fixture_json, registry_config

pytestmark = pytest.mark.unit

WINDOW = CollectionWindow(
    datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 21, tzinfo=UTC), "incremental"
)


def adapter(settings: Settings | None = None, **config: Any) -> EuropePmcAdapter:
    built = build_adapter(
        EuropePmcAdapter, registry_config("europe_pmc", **config), EuropePmcServer(), settings
    )
    assert isinstance(built, EuropePmcAdapter)
    return built


def results(page: int = 1) -> list[dict[str, Any]]:
    return fixture_json(f"europe_pmc/search_page_{page}.json")["resultList"]["result"]


def _page(body: Any, cursor: str = "*") -> RawPage:
    request = RequestSpec(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"cursorMark": cursor}
    )
    return RawPage(request, 200, json.dumps(body).encode(), {}, datetime.now(UTC))


# --------------------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------------------
def test_query_filters_by_first_publication_date() -> None:
    params = adapter().build_query(WINDOW, None).params
    assert "FIRST_PDATE:[2026-08-01 TO 2026-08-20]" in params["query"]
    assert params["resultType"] == "core" and params["format"] == "json"
    assert params["cursorMark"] == "*" and params["synonym"] == "false"


def test_preprints_only_while_pubmed_is_enabled() -> None:
    assert adapter().source_codes() == ["PPR"]
    assert "AND (SRC:PPR)" in adapter().build_query(WINDOW, None).params["query"]


def test_everything_is_collected_when_pubmed_is_disabled() -> None:
    settings = load_settings(env_file=None, overrides={"enable_pubmed": False})
    built = adapter(settings)
    assert built.source_codes() is None
    assert "SRC:" not in built.build_query(WINDOW, None).params["query"]


def test_sources_option_overrides_the_default() -> None:
    built = adapter(options={"sources": ["ppr", "MED"]})
    assert built.source_codes() == ["PPR", "MED"]
    assert "(SRC:PPR OR SRC:MED)" in built.build_query(WINDOW, None).params["query"]


@pytest.mark.parametrize("value", [["NOPE"], [], "PPR", [7]])
def test_invalid_sources_option_is_reported(value: Any) -> None:
    assert any("sources" in p for p in adapter(options={"sources": value}).validate_configuration())


def test_page_size_is_capped_and_constant_while_paging() -> None:
    built = adapter(page_size=4000)
    first = built.build_query(WINDOW, None).params
    later = built.build_query(WINDOW, "FIXTURE_CURSOR_2").params
    assert first["pageSize"] == 1000 == later["pageSize"]
    assert later["cursorMark"] == "FIXTURE_CURSOR_2"


def test_search_terms_are_quoted_for_lucene() -> None:
    query = adapter().build_query(WINDOW, None).params["query"]
    assert '"Rare Diseases"' in query and "Oncology" in query


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def test_first_page_reports_the_hit_count_and_next_cursor() -> None:
    parsed = adapter().parse_response(_page(fixture_json("europe_pmc/search_page_1.json")))
    assert len(parsed.items) == 2
    assert parsed.next_cursor == "FIXTURE_CURSOR_2"
    assert parsed.total_available == 3


def test_paging_stops_when_the_cursor_stops_moving() -> None:
    body = fixture_json("europe_pmc/search_page_2.json")
    parsed = adapter().parse_response(_page(body, cursor="FIXTURE_CURSOR_2"))
    assert len(parsed.items) == 1
    assert parsed.next_cursor is None  # nextCursorMark equals the cursor we sent
    assert parsed.total_available is None  # only reported for the first page


def test_paging_stops_on_an_empty_page() -> None:
    body = {"hitCount": 3, "nextCursorMark": "NEXT", "resultList": {"result": []}}
    assert adapter().parse_response(_page(body, cursor="OTHER")).next_cursor is None


def test_zero_hits_is_a_valid_answer() -> None:
    parsed = adapter().parse_response(_page({"hitCount": 0, "resultList": {"result": []}}))
    assert parsed.items == () and parsed.next_cursor is None


@pytest.mark.parametrize("body", [[], "text", {"unexpected": True}])
def test_unexpected_responses_are_rejected(body: Any) -> None:
    with pytest.raises(SourceParseError):
        adapter().parse_response(_page(body))


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def test_preprint_record_mapping() -> None:
    record = adapter().normalize_record(results()[0])
    data, detail = record.data, record.detail
    assert data.source_record_id == "PPR:PPR990001"
    assert data.source_url == "https://europepmc.org/article/PPR/PPR990001"
    assert data.title == "A fixture preprint on mRNA cancer vaccines"  # markup removed
    assert data.abstract is not None and "<h4>" not in data.abstract
    assert data.published_at == datetime(2026, 8, 5, tzinfo=UTC)
    assert data.updated_at_source == datetime(2026, 8, 7, tzinfo=UTC)  # first index date
    assert detail["journal"] == "bioRxiv"  # preprint server
    assert detail["publication_date"] == date(2026, 8, 5)
    assert detail["publication_type"] == "preprint"
    assert detail["authors"] == ["Doe J", "Roe R"]
    # affiliations come from both the nested details and the flat field
    assert detail["affiliations"] == [
        "Fixture Vaccines GmbH, Berlin, Germany",
        "Fixture University",
    ]


def test_journal_article_record_mapping() -> None:
    record = adapter().normalize_record(results()[1])
    assert record.data.source_record_id == "MED:99000010"
    assert record.detail["journal"] == "Journal of Fixture Immunology"
    assert record.detail["citation_count"] == 3
    payload = record.data.raw_payload
    assert payload is not None
    assert payload["pmid"] == "99000010" and payload["doi"] == "10.0000/fixture.0010"
    assert payload["mesh_terms"] == ["Melanoma"] and payload["keywords"] == ["checkpoint inhibitor"]


def test_records_without_a_journal_or_abstract_are_accepted() -> None:
    record = adapter().normalize_record(results(2)[0])
    assert record.data.source_record_id == "PPR:PPR990002"
    assert record.detail["journal"] == "medRxiv"
    assert record.data.abstract is None


def test_abstract_storage_can_be_turned_off() -> None:
    record = adapter(options={"store_abstracts": False}).normalize_record(results()[0])
    assert record.data.abstract is None
    assert record.data.raw_payload is not None and "abstract" not in record.data.raw_payload


@pytest.mark.parametrize(
    "item",
    [
        {},
        {"id": "PPR1"},
        {"source": "PPR"},
        {"id": "bad id", "source": "PPR"},
        {"id": "PPR1", "source": "NOPE"},
    ],
)
def test_records_without_a_valid_id_are_rejected(item: dict[str, Any]) -> None:
    with pytest.raises(SourceParseError, match="id/source"):
        adapter().normalize_record(item)


def test_health_check_uses_a_single_result_search() -> None:
    built = adapter()
    url = built.health_check_url()
    assert url is not None and "resultType=idlist" in url and "pageSize=1" in url
    assert built.health_check().status.value == "ok"
