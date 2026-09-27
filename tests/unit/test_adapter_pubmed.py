"""Unit tests for the PubMed adapter: two-step fetch, XML parsing, the 10,000-record cap."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest

from cews.ingestion.adapters.pubmed import (
    SEARCH_CAP,
    PubMedAdapter,
    PubMedPage,
    choose_publication_date,
    parse_pubmed_xml,
)
from cews.ingestion.errors import SourceParseError
from cews.ingestion.results import CollectionWindow, RequestSpec
from cews.settings import Settings, load_settings
from support_sources import PubMedServer, build_adapter, fixture_bytes, registry_config

pytestmark = pytest.mark.unit

WINDOW = CollectionWindow(
    datetime(2026, 8, 10, tzinfo=UTC), datetime(2026, 8, 15, tzinfo=UTC), "incremental"
)
PAGE_1 = "pubmed/efetch_99000001_99000002.xml"


def adapter(
    settings: Settings | None = None, server: PubMedServer | None = None, **config: Any
) -> PubMedAdapter:
    built = build_adapter(
        PubMedAdapter,
        registry_config("pubmed", **config),
        server or PubMedServer(),
        settings,
    )
    assert isinstance(built, PubMedAdapter)
    return built


def articles(name: str = PAGE_1) -> list[dict[str, Any]]:
    return parse_pubmed_xml(fixture_bytes(name))


# --------------------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------------------
def test_search_uses_the_entrez_date_range_and_paging() -> None:
    params = adapter().build_query(WINDOW, None).params
    assert params["db"] == "pubmed" and params["datetype"] == "edat"
    assert params["mindate"] == "2026/08/10" and params["maxdate"] == "2026/08/14"
    assert params["retstart"] == 0 and params["retmax"] == 200
    assert params["tool"] == "cews"
    assert "api_key" not in params and "email" not in params


def test_identifying_parameters_are_sent_when_configured() -> None:
    settings = load_settings(
        env_file=None, overrides={"ncbi_email": "team@example.org", "ncbi_api_key": "secret-key"}
    )
    params = adapter(settings).build_query(WINDOW, None).params
    assert params["email"] == "team@example.org" and params["api_key"] == "secret-key"


def test_search_terms_are_not_quoted_so_pubmed_can_map_them() -> None:
    term = adapter().build_query(WINDOW, None).params["term"]
    assert "Rare Diseases" in term and '"Rare Diseases"' not in term
    assert '"CAR-T"' in term  # punctuation still quoted


def test_cursor_becomes_retstart() -> None:
    assert adapter().build_query(WINDOW, "400").params["retstart"] == 400


def test_page_size_above_the_cap_is_a_configuration_problem() -> None:
    assert any("page_size" in p for p in adapter(page_size=20000).validate_configuration())


# --------------------------------------------------------------------------------------
# Two-step fetch
# --------------------------------------------------------------------------------------
def test_fetch_page_runs_esearch_then_efetch() -> None:
    server = PubMedServer()
    page = adapter(server=server).fetch_page(adapter(server=server).build_query(WINDOW, None))
    assert isinstance(page, PubMedPage)
    assert page.pmids == ("99000001", "99000002") and page.total == 3 and page.retstart == 0
    assert b"PubmedArticleSet" in page.content
    assert [r.url.path.rsplit("/", 1)[-1] for r in server.requests] == [
        "esearch.fcgi",
        "efetch.fcgi",
    ]


def test_no_results_means_no_efetch_call() -> None:
    server = PubMedServer()
    server.esearch_override = {"esearchresult": {"count": "0", "retstart": "0", "idlist": []}}
    built = adapter(server=server)
    page = built.fetch_page(built.build_query(WINDOW, None))
    assert page.content == b"" and built.parse_response(page).items == ()
    assert len(server.requests) == 1


def test_esearch_error_is_reported() -> None:
    server = PubMedServer()
    server.esearch_override = {"esearchresult": {"ERROR": "Invalid db name"}}
    built = adapter(server=server)
    with pytest.raises(SourceParseError, match="Invalid db name"):
        built.fetch_page(built.build_query(WINDOW, None))


def test_paging_follows_retstart_until_the_total_is_reached() -> None:
    built = adapter(page_size=2)
    first = built.parse_response(built.fetch_page(built.build_query(WINDOW, None)))
    assert first.next_cursor == "2" and first.total_available == 3
    second = built.parse_response(built.fetch_page(built.build_query(WINDOW, "2")))
    assert second.next_cursor is None and len(second.items) == 1


def test_the_ten_thousand_cap_is_reported_and_respected() -> None:
    built = adapter()
    page = PubMedPage(
        RequestSpec(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"mindate": "2026/08/01", "maxdate": "2026/08/07"},
        ),
        200,
        fixture_bytes(PAGE_1),
        {},
        datetime.now(UTC),
        pmids=tuple(str(i) for i in range(200)),
        total=25_000,
        retstart=0,
    )
    parsed = built.parse_response(page)
    assert parsed.total_available == SEARCH_CAP
    assert len(parsed.warnings) == 1
    assert "25000" in parsed.warnings[0] and "window_slice_days" in parsed.warnings[0]

    at_cap = PubMedPage(
        page.request,
        200,
        b"",
        {},
        datetime.now(UTC),
        pmids=("1",),
        total=25_000,
        retstart=SEARCH_CAP - 1,
    )
    assert built.parse_response(at_cap).next_cursor is None  # never pages past the cap
    assert built.parse_response(at_cap).warnings == ()  # warned once per slice, not per page


def test_pages_from_elsewhere_are_rejected() -> None:
    from cews.ingestion.results import RawPage

    plain = RawPage(
        RequestSpec("https://eutils.ncbi.nlm.nih.gov/x"), 200, b"", {}, datetime.now(UTC)
    )
    with pytest.raises(SourceParseError, match="PubMedAdapter.fetch_page"):
        adapter().parse_response(plain)


# --------------------------------------------------------------------------------------
# XML parsing
# --------------------------------------------------------------------------------------
def test_structured_abstract_authors_and_identifiers() -> None:
    first = articles()[0]
    assert first["pmid"] == "99000001"
    assert (
        first["title"] == "Targeting KRAS with FX-101: a fixture study."
    )  # inline markup flattened
    assert (
        first["abstract"]
        == "BACKGROUND: Fixture text for testing only. RESULTS: More fixture text."
    )
    assert first["authors"] == ["Jane Doe", "Richard Roe"]
    assert first["affiliations"] == [
        "Fixture Oncology, Inc., Boston, MA, USA.",
        "Fixture University, Department of Medicine.",
    ]  # repeated affiliation kept once
    assert first["doi"] == "10.0000/fixture.0001"
    assert first["mesh_terms"] == ["Carcinoma, Non-Small-Cell Lung"]
    assert first["keywords"] == ["KRAS"]
    assert first["journal"] == "Journal of Fixture Oncology"
    assert first["entrez_date"] == "2026-08-12" and first["electronic_date"] == "2026-08-11"


def test_medline_date_collective_author_and_missing_abstract() -> None:
    second = articles()[1]
    assert second["publication_date"] == "2026-07-01"  # "2026 Jul-Aug"
    assert second["publication_date_precision"] == "month"
    assert second["authors"] == ["Fixture Gene Therapy Consortium"]
    assert second["abstract"] is None and second["affiliations"] == []
    assert second["pmcid"] == "PMC99000002"


def test_year_only_dates_fall_back_to_the_index_date() -> None:
    third = articles("pubmed/efetch_99000003.xml")[0]
    assert third["publication_date"] == "2026-01-01"
    assert third["publication_date_precision"] == "year"
    # counting it under January would distort monthly activity
    assert choose_publication_date(third) == "2026-08-14"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (
            {
                "electronic_date": "2026-05-02",
                "publication_date": "2026-06-01",
                "publication_date_precision": "month",
                "entrez_date": "2026-07-01",
            },
            "2026-05-02",
        ),
        (
            {
                "publication_date": "2026-06-01",
                "publication_date_precision": "month",
                "entrez_date": "2026-07-01",
            },
            "2026-06-01",
        ),
        (
            {
                "publication_date": "2026-01-01",
                "publication_date_precision": "year",
                "entrez_date": "2026-07-01",
            },
            "2026-07-01",
        ),
        ({"publication_date": "2026-01-01", "publication_date_precision": "year"}, "2026-01-01"),
        ({}, None),
    ],
)
def test_publication_date_choice(item: dict[str, Any], expected: str | None) -> None:
    assert choose_publication_date(item) == expected


def test_empty_and_invalid_documents() -> None:
    assert parse_pubmed_xml(b"") == []
    assert parse_pubmed_xml(b"   ") == []
    with pytest.raises(SourceParseError, match="invalid XML"):
        parse_pubmed_xml(b"<PubmedArticleSet><PubmedArticle>")
    with pytest.raises(SourceParseError, match="PubmedArticleSet"):
        parse_pubmed_xml(b"<html><body>error</body></html>")


def test_book_records_are_skipped() -> None:
    document = (
        b"<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>1</PMID></BookDocument>"
        b"</PubmedBookArticle></PubmedArticleSet>"
    )
    assert parse_pubmed_xml(document) == []


def test_entity_expansion_attacks_are_refused() -> None:
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol">'
        b'<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">]>'
        b"<PubmedArticleSet><PubmedArticle>&lol2;</PubmedArticle></PubmedArticleSet>"
    )
    with pytest.raises(SourceParseError):
        parse_pubmed_xml(bomb)


def test_external_entity_references_are_refused() -> None:
    external = (
        b'<?xml version="1.0"?><!DOCTYPE t [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
        b"<PubmedArticleSet><PubmedArticle>&x;</PubmedArticle></PubmedArticleSet>"
    )
    with pytest.raises(SourceParseError):
        parse_pubmed_xml(external)


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------
def test_record_mapping() -> None:
    record = adapter().normalize_record(articles()[0])
    data, detail = record.data, record.detail
    assert data.source_record_id == "99000001"
    assert data.source_url == "https://pubmed.ncbi.nlm.nih.gov/99000001/"
    assert data.record_type == "publication"
    assert data.published_at == datetime(2026, 8, 11, tzinfo=UTC)
    assert data.updated_at_source == datetime(2026, 8, 14, tzinfo=UTC)  # DateRevised
    assert detail["publication_identifier"] == "99000001"
    assert detail["journal"] == "Journal of Fixture Oncology"
    assert detail["publication_date"] == date(2026, 8, 11)
    assert detail["publication_type"] == "Journal Article"
    assert detail["citation_count"] is None  # PubMed does not provide one
    assert data.raw_payload is not None and data.raw_payload["doi"] == "10.0000/fixture.0001"


def test_abstract_storage_can_be_turned_off() -> None:
    built = adapter(options={"store_abstracts": False})
    record = built.normalize_record(articles()[0])
    assert record.data.abstract is None
    assert record.data.raw_payload is not None and "abstract" not in record.data.raw_payload


def test_invalid_store_abstracts_option_is_reported() -> None:
    assert any(
        "store_abstracts" in p
        for p in adapter(options={"store_abstracts": "yes"}).validate_configuration()
    )


@pytest.mark.parametrize("pmid", ["", "abc", "12345678901"])
def test_records_without_a_valid_pmid_are_rejected(pmid: str) -> None:
    with pytest.raises(SourceParseError, match="PMID"):
        adapter().normalize_record({"pmid": pmid})


def test_health_check_uses_einfo() -> None:
    built = adapter()
    assert built.health_check_url() is not None and "einfo.fcgi" in built.health_check_url()
    assert built.health_check().status.value == "ok"


def test_api_key_never_appears_in_error_output() -> None:
    settings = load_settings(env_file=None, overrides={"ncbi_api_key": "super-secret-key"})
    server = PubMedServer()
    server.fail_always = True
    built = adapter(settings, server=server)
    health = built.health_check()
    assert "super-secret-key" not in health.message
    with pytest.raises(Exception) as excinfo:
        built.fetch_page(built.build_query(WINDOW, None))
    assert "super-secret-key" not in str(excinfo.value)
