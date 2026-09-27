"""ClinicalTrials.gov adapter (public API v2, no key).

Request: ``GET {base_url}/studies`` with

* ``query.term`` - the monitored therapeutic areas joined with OR (or ``options.query``);
* ``filter.advanced`` - ``AREA[LastUpdatePostDate]RANGE[start,end]`` for the collection window,
  plus ``AREA[StudyType]INTERVENTIONAL`` unless ``options.study_type`` says otherwise;
* ``fields`` - an explicit list of institutional fields only;
* ``pageSize`` (capped at 1000 by the API) and ``pageToken`` for cursor pagination. The last
  page is the one without ``nextPageToken``.

Privacy: study records can contain names, phone numbers and e-mail addresses of investigators
and site contacts. CEWS never requests those fields and removes ``centralContacts``,
``overallOfficials`` and per-location ``contacts`` from anything it stores, in case the API
returns them anyway. Only the country of each location is kept.

API behaviour verified on 2026-09-22 against the official search syntax documentation and a
live-verified community reference (see ``docs/data_sources.md``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

from cews.constants import SourceType
from cews.ingestion.adapters.common import (
    as_list,
    clean_text,
    clip,
    dig,
    option_query,
    or_query,
    search_terms,
    unique_texts,
)
from cews.ingestion.base import SourceAdapter, register_adapter
from cews.ingestion.errors import AdapterConfigError, SourceParseError
from cews.ingestion.results import (
    CollectionWindow,
    NormalizedRecord,
    ParsedPage,
    RawPage,
    RequestSpec,
)
from cews.normalization.dates import normalize_date, to_utc_datetime

MAX_PAGE_SIZE = 1000
NCT_ID = re.compile(r"^NCT\d{8}$")
STUDY_URL = "https://clinicaltrials.gov/study/{nct_id}"

# Institutional fields only. No contact, official or investigator fields.
FIELDS: tuple[str, ...] = (
    "NCTId",
    "BriefTitle",
    "OfficialTitle",
    "OverallStatus",
    "StudyType",
    "Phase",
    "EnrollmentCount",
    "EnrollmentType",
    "StartDate",
    "PrimaryCompletionDate",
    "CompletionDate",
    "StudyFirstPostDate",
    "LastUpdatePostDate",
    "LeadSponsorName",
    "LeadSponsorClass",
    "CollaboratorName",
    "Condition",
    "Keyword",
    "InterventionType",
    "InterventionName",
    "LocationCountry",
    "BriefSummary",
)
PERSONAL_KEYS = frozenset({"centralContacts", "overallOfficials", "contacts"})
_STUDY_TYPES = frozenset({"INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS", "ALL"})


def strip_personal_data(value: Any) -> Any:
    """Return a copy of ``value`` without contact and investigator details, at any depth."""
    if isinstance(value, Mapping):
        return {k: strip_personal_data(v) for k, v in value.items() if k not in PERSONAL_KEYS}
    if isinstance(value, list):
        return [strip_personal_data(item) for item in value]
    return value


@register_adapter
class ClinicalTrialsGovAdapter(SourceAdapter):
    """Clinical trials from ClinicalTrials.gov, filtered by last-update date."""

    source_name = "clinical_trials_gov"
    source_type = SourceType.CLINICAL_TRIAL

    # ---- configuration -------------------------------------------------------------------
    def validate_configuration(self) -> list[str]:
        """Check the base URL, search terms and study-type option."""
        problems = super().validate_configuration()
        try:
            self._query_term()
            self._study_type()
        except AdapterConfigError as exc:
            problems.append(str(exc))
        return problems

    def health_check_url(self) -> str | None:
        """A small registry-statistics document, not a search."""
        return f"{self.config.base_url}/stats/size" if self.config.base_url else None

    def _query_term(self) -> str:
        return option_query(self.config) or or_query(
            search_terms(self.settings, self.config), quote_phrases=True
        )

    def _study_type(self) -> str:
        value = str(self.config.options.get("study_type", "INTERVENTIONAL")).upper()
        if value not in _STUDY_TYPES:
            raise AdapterConfigError(
                f"{self.source_name}: options.study_type must be one of {sorted(_STUDY_TYPES)}"
            )
        return value

    # ---- request / response --------------------------------------------------------------
    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        """One page of studies last updated inside ``window`` (dates are inclusive)."""
        first = window.start.date()
        last = (window.end - timedelta(microseconds=1)).date()
        advanced = f"AREA[LastUpdatePostDate]RANGE[{first.isoformat()},{last.isoformat()}]"
        study_type = self._study_type()
        if study_type != "ALL":
            advanced += f" AND AREA[StudyType]{study_type}"
        page_size = min(self.config.page_size, MAX_PAGE_SIZE)
        params: dict[str, Any] = {
            "format": "json",
            "query.term": self._query_term(),
            "filter.advanced": advanced,
            "fields": ",".join(FIELDS),
            "pageSize": page_size,
        }
        if cursor:
            params["pageToken"] = cursor
        return RequestSpec(f"{self.config.base_url}/studies", params=params, page_size=page_size)

    def parse_response(self, page: RawPage) -> ParsedPage:
        """Studies on the page; the cursor is ``nextPageToken`` (absent on the last page)."""
        body = page.json()
        if not isinstance(body, dict) or not isinstance(body.get("studies", []), list):
            raise SourceParseError("expected an object with a 'studies' list")
        token = body.get("nextPageToken")
        if token is not None and not isinstance(token, str):
            raise SourceParseError("nextPageToken must be a string")
        studies = tuple(s for s in body.get("studies", []) if isinstance(s, dict))
        total = body.get("totalCount")
        return ParsedPage(studies, token or None, total if isinstance(total, int) else None)

    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        """One study as a clinical-trial record (personal data removed)."""
        study = strip_personal_data(item)
        section = study.get("protocolSection") or {}
        nct_id = str(dig(section, "identificationModule", "nctId") or "").strip()
        if not NCT_ID.match(nct_id):
            raise SourceParseError(f"missing or malformed NCT id: {nct_id!r}")

        status = section.get("statusModule") or {}
        sponsor = dig(section, "sponsorCollaboratorsModule", "leadSponsor") or {}
        design = section.get("designModule") or {}
        interventions = as_list(dig(section, "armsInterventionsModule", "interventions"))
        locations = as_list(dig(section, "contactsLocationsModule", "locations"))
        conditions = unique_texts(as_list(dig(section, "conditionsModule", "conditions")))
        phases = [str(p) for p in as_list(design.get("phases")) if p]
        enrollment = dig(design, "enrollmentInfo", "count")

        first_posted = to_utc_datetime(dig(status, "studyFirstPostDateStruct", "date"))
        last_update = to_utc_datetime(dig(status, "lastUpdatePostDateStruct", "date"))
        title = clean_text(dig(section, "identificationModule", "briefTitle")) or clean_text(
            dig(section, "identificationModule", "officialTitle")
        )
        return self.build_record(
            source_record_id=nct_id,
            title=title,
            abstract=clean_text(dig(section, "descriptionModule", "briefSummary")),
            source_url=STUDY_URL.format(nct_id=nct_id),
            published_at=first_posted or last_update,
            updated_at_source=last_update,
            payload=study,
            detail={
                "trial_identifier": nct_id,
                "sponsor_name": clip(clean_text(sponsor.get("name")), 255),
                "phase": clip("/".join(phases), 32) or None,
                "status": clip(clean_text(status.get("overallStatus")), 64),
                "enrollment": (
                    enrollment if isinstance(enrollment, int) and enrollment >= 0 else None
                ),
                "start_date": normalize_date(dig(status, "startDateStruct", "date")),
                "completion_date": normalize_date(dig(status, "completionDateStruct", "date")),
                "intervention": "; ".join(
                    unique_texts(i.get("name") for i in interventions if isinstance(i, dict))
                )
                or None,
                "condition": "; ".join(conditions) or None,
                "countries": sorted(
                    {
                        c
                        for c in (
                            clean_text(loc.get("country"))
                            for loc in locations
                            if isinstance(loc, dict)
                        )
                        if c
                    }
                )
                or None,
            },
        )
