"""Unit tests for the ClinicalTrials.gov adapter: query building, privacy, field mapping."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest

from cews.ingestion.adapters.clinical_trials_gov import (
    FIELDS,
    ClinicalTrialsGovAdapter,
    strip_personal_data,
)
from cews.ingestion.errors import SourceParseError
from cews.ingestion.results import CollectionWindow, RawPage, RequestSpec
from cews.settings import Settings, load_settings
from support_sources import ClinicalTrialsServer, build_adapter, fixture_json, registry_config

pytestmark = pytest.mark.unit

WINDOW = CollectionWindow(
    datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 8, 21, tzinfo=UTC), "incremental"
)
PERSONAL_VALUES = (
    "Placeholder Contact",
    "placeholder@example.invalid",
    "000-000-0000",
    "Placeholder Official",
    "Placeholder Site Contact",
    "site@example.invalid",
)


def adapter(settings: Settings | None = None, **config: Any) -> ClinicalTrialsGovAdapter:
    built = build_adapter(
        ClinicalTrialsGovAdapter,
        registry_config("clinical_trials_gov", **config),
        ClinicalTrialsServer(),
        settings,
    )
    assert isinstance(built, ClinicalTrialsGovAdapter)
    return built


def studies(page: int = 1) -> list[Mapping[str, Any]]:
    return fixture_json(f"clinical_trials_gov/studies_page_{page}.json")["studies"]


# --------------------------------------------------------------------------------------
# Query
# --------------------------------------------------------------------------------------
def test_query_filters_on_last_update_date_and_study_type() -> None:
    query = adapter().build_query(WINDOW, None)
    assert query.url.endswith("/studies")
    advanced = query.params["filter.advanced"]
    assert "AREA[LastUpdatePostDate]RANGE[2026-08-01,2026-08-20]" in advanced  # end is inclusive
    assert "AREA[StudyType]INTERVENTIONAL" in advanced
    assert query.params["format"] == "json"
    assert "pageToken" not in query.params


def test_query_searches_the_monitored_therapeutic_areas() -> None:
    term = adapter().build_query(WINDOW, None).params["query.term"]
    assert term.startswith("(") and " OR " in term
    assert "Oncology" in term and '"Rare Diseases"' in term and '"CAR-T"' in term


def test_query_can_be_overridden_by_registry_options() -> None:
    terms = adapter(options={"query_terms": ["CRISPR", "base editing"]})
    assert terms.build_query(WINDOW, None).params["query.term"] == '(CRISPR OR "base editing")'
    whole = adapter(options={"query": "AREA[LeadSponsorClass]INDUSTRY"})
    assert whole.build_query(WINDOW, None).params["query.term"] == "AREA[LeadSponsorClass]INDUSTRY"


def test_study_type_option() -> None:
    every = adapter(options={"study_type": "all"}).build_query(WINDOW, None)
    assert "StudyType" not in every.params["filter.advanced"]
    observational = adapter(options={"study_type": "OBSERVATIONAL"}).build_query(WINDOW, None)
    assert "AREA[StudyType]OBSERVATIONAL" in observational.params["filter.advanced"]
    assert adapter(options={"study_type": "nonsense"}).validate_configuration()


def test_page_size_is_capped_at_the_api_maximum() -> None:
    query = adapter(page_size=5000).build_query(WINDOW, None)
    assert query.params["pageSize"] == 1000 and query.page_size == 1000


def test_cursor_becomes_the_page_token() -> None:
    query = adapter().build_query(WINDOW, "TOKEN")
    assert query.params["pageToken"] == "TOKEN"


def test_requested_fields_exclude_personal_data() -> None:
    requested = adapter().build_query(WINDOW, None).params["fields"].split(",")
    assert "LeadSponsorName" in requested and "LocationCountry" in requested
    banned = ("Contact", "OverallOfficial", "Investigator", "Email", "Phone", "LocationFacility")
    # "OfficialTitle" is the trial's formal title, not a person.
    assert not any(word in field for field in requested for word in banned)
    assert set(requested) == set(FIELDS)


def test_configuration_problems_are_reported() -> None:
    no_terms = adapter(settings=load_settings(env_file=None, overrides={"therapeutic_areas": ""}))
    assert any("no search terms" in p for p in no_terms.validate_configuration())
    quoted = adapter(options={"query_terms": ['bad " term']})
    assert any("quotes" in p for p in quoted.validate_configuration())


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
def _page(body: Any) -> RawPage:
    return RawPage(
        RequestSpec("https://clinicaltrials.gov/api/v2/studies"),
        200,
        json.dumps(body).encode(),
        {},
        datetime.now(UTC),
    )


def test_parse_reads_studies_and_the_next_token() -> None:
    parsed = adapter().parse_response(
        _page(fixture_json("clinical_trials_gov/studies_page_1.json"))
    )
    assert len(parsed.items) == 2 and parsed.next_cursor == "FIXTURE_PAGE_TOKEN_2"


def test_parse_stops_on_the_last_page() -> None:
    parsed = adapter().parse_response(
        _page(fixture_json("clinical_trials_gov/studies_page_2.json"))
    )
    assert len(parsed.items) == 1 and parsed.next_cursor is None


@pytest.mark.parametrize("body", [{"studies": "nope"}, [], {"studies": [], "nextPageToken": 7}])
def test_parse_rejects_unexpected_shapes(body: Any) -> None:
    with pytest.raises(SourceParseError):
        adapter().parse_response(_page(body))


# --------------------------------------------------------------------------------------
# Normalization and privacy
# --------------------------------------------------------------------------------------
def test_personal_fields_are_removed_at_any_depth() -> None:
    cleaned = strip_personal_data(studies()[0])
    text = json.dumps(cleaned)
    assert all(value not in text for value in PERSONAL_VALUES)
    locations = cleaned["protocolSection"]["contactsLocationsModule"]["locations"]
    assert [loc["country"] for loc in locations] == ["United States", "Japan", "United States"]


def test_stored_record_contains_no_personal_data() -> None:
    record = adapter().normalize_record(studies()[0])
    stored = json.dumps(record.data.raw_payload) + str(record.detail) + str(record.data.abstract)
    assert all(value not in stored for value in PERSONAL_VALUES)


def test_full_record_is_mapped() -> None:
    record = adapter().normalize_record(studies()[0])
    data, detail = record.data, record.detail
    assert data.source_record_id == "NCT09000001"
    assert data.source_url == "https://clinicaltrials.gov/study/NCT09000001"
    assert data.record_type == "clinical_trial"
    assert data.title is not None and data.title.startswith("A Study of FX-101")
    assert data.abstract is not None and "<b>" not in data.abstract  # HTML removed
    assert data.published_at == datetime(2026, 2, 10, tzinfo=UTC)  # first posted
    assert data.updated_at_source == datetime(2026, 8, 12, tzinfo=UTC)  # last update
    assert detail["trial_identifier"] == "NCT09000001"
    assert detail["sponsor_name"] == "Fixture Oncology, Inc."
    assert detail["phase"] == "PHASE1/PHASE2"
    assert detail["status"] == "RECRUITING"
    assert detail["enrollment"] == 120
    assert detail["start_date"] == date(2026, 3, 1)  # month-only date
    assert detail["completion_date"] == date(2028, 6, 30)
    assert detail["intervention"] == "FX-101; Placebo comparator"
    assert detail["condition"] == "Non-small Cell Lung Cancer; Solid Tumor"
    assert detail["countries"] == ["Japan", "United States"]  # de-duplicated and sorted


def test_sparse_records_are_handled() -> None:
    record = adapter().normalize_record(studies(2)[0])
    assert (
        record.data.title == "Registry of Fixture CAR-T Recipients"
    )  # falls back to officialTitle
    assert record.data.published_at == datetime(2026, 8, 18, tzinfo=UTC)  # no first-posted date
    assert record.detail["phase"] is None and record.detail["condition"] is None
    assert record.detail["countries"] is None


def test_early_phase_and_missing_optional_sections() -> None:
    record = adapter().normalize_record(studies()[1])
    assert record.detail["phase"] == "EARLY_PHASE1"
    assert record.data.abstract is None
    assert record.detail["intervention"] == "FX-AAV9"


@pytest.mark.parametrize(
    "study",
    [
        {},
        {"protocolSection": {}},
        {"protocolSection": {"identificationModule": {"nctId": "12345678"}}},
        {"protocolSection": {"identificationModule": {"nctId": "NCT1"}}},
    ],
)
def test_records_without_a_valid_nct_id_are_rejected(study: Any) -> None:
    with pytest.raises(SourceParseError, match="NCT id"):
        adapter().normalize_record(study)


def test_long_values_are_trimmed_to_column_limits() -> None:
    study = json.loads(json.dumps(studies()[0]))
    study["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"] = "x" * 400
    detail = adapter().normalize_record(study).detail
    assert len(detail["sponsor_name"]) == 255


def test_window_slices_do_not_overlap_by_a_whole_day() -> None:
    first, second = WINDOW.slices(10)
    one = adapter().build_query(first, None).params["filter.advanced"]
    two = adapter().build_query(second, None).params["filter.advanced"]
    assert "RANGE[2026-08-01,2026-08-10]" in one
    assert "RANGE[2026-08-11,2026-08-20]" in two


def test_health_check_uses_the_statistics_endpoint() -> None:
    built = adapter()
    assert built.health_check_url() == "https://clinicaltrials.gov/api/v2/stats/size"
    health = built.health_check()
    assert health.status.value == "ok"


def test_adapter_without_base_url_reports_configuration_problems() -> None:
    config = replace(registry_config("clinical_trials_gov"), base_url=None)
    built = build_adapter(ClinicalTrialsGovAdapter, config, ClinicalTrialsServer())
    assert any("base_url" in problem for problem in built.validate_configuration())
    assert built.health_check_url() is None


def test_slices_cover_the_whole_window() -> None:
    covered = WINDOW.slices(7)
    assert covered[0].start == WINDOW.start and covered[-1].end == WINDOW.end
    assert all(a.end == b.start for a, b in zip(covered, covered[1:], strict=False))
    assert sum((s.end - s.start for s in covered), timedelta()) == WINDOW.end - WINDOW.start
