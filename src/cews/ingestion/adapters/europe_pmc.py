"""Europe PMC adapter (REST search API, no key).

Request: ``GET {base_url}/search`` with ``format=json``, ``resultType=core`` (includes abstracts
and affiliations), a constant ``pageSize`` (at most 1000) and ``cursorMark`` paging (``*`` for
the first page, then ``nextCursorMark``). The query combines the search terms with
``FIRST_PDATE:[start TO end]`` for the collection window.

Overlap with PubMed: Europe PMC indexes all of PubMed. To avoid counting the same article twice,
the adapter collects only preprints (``SRC:PPR``: bioRxiv, medRxiv and others) while the PubMed
source is enabled, and everything when it is not. ``options.sources`` (for example
``[PPR, MED]``) overrides this.

Behaviour verified on 2026-09-22 (see ``docs/data_sources.md``): every response is HTTP 200,
even for a malformed query, so responses are validated here; an unknown ``SRC`` silently
returns nothing, so source codes are checked against a known list; the page size must stay
constant for the whole walk.
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
    option_bool,
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
FIRST_CURSOR = "*"
ARTICLE_URL = "https://europepmc.org/article/{source}/{id}"
# Europe PMC source codes (controlled list).
KNOWN_SOURCES = frozenset({"MED", "PMC", "PPR", "AGR", "CBA", "CTX", "ETH", "HIR", "NBK", "PAT"})
_RECORD_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


@register_adapter
class EuropePmcAdapter(SourceAdapter):
    """Publications and preprints from Europe PMC, by first publication date."""

    source_name = "europe_pmc"
    source_type = SourceType.PUBLICATION

    def validate_configuration(self) -> list[str]:
        """Check the base URL, the search terms, the source codes and the options."""
        problems = super().validate_configuration()
        try:
            self._terms()
            self.source_codes()
            option_bool(self.config, "store_abstracts", True)
        except AdapterConfigError as exc:
            problems.append(str(exc))
        return problems

    def health_check_url(self) -> str | None:
        """A one-result search, the lightest documented call."""
        if not self.config.base_url:
            return None
        return (
            f"{self.config.base_url}/search?query=cancer&format=json&resultType=idlist&pageSize=1"
        )

    def _terms(self) -> str:
        return option_query(self.config) or or_query(
            search_terms(self.settings, self.config), quote_phrases=True
        )

    def source_codes(self) -> list[str] | None:
        """Europe PMC sources to include; None means all of them.

        Raises:
            AdapterConfigError: for an unknown or malformed ``options.sources`` value.
        """
        raw = self.config.options.get("sources")
        if raw is None:
            return ["PPR"] if self.settings.enable_pubmed else None
        if not isinstance(raw, list) or not raw or not all(isinstance(s, str) for s in raw):
            raise AdapterConfigError(f"{self.source_name}: options.sources must be a list of codes")
        codes = [s.strip().upper() for s in raw]
        unknown = sorted(set(codes) - KNOWN_SOURCES)
        if unknown:
            raise AdapterConfigError(
                f"{self.source_name}: options.sources has unknown codes {unknown}; "
                f"known codes: {sorted(KNOWN_SOURCES)}"
            )
        return codes

    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        """One page of records first published inside ``window`` (dates are inclusive)."""
        first = window.start.date().isoformat()
        last = (window.end - timedelta(microseconds=1)).date().isoformat()
        query = f"{self._terms()} AND FIRST_PDATE:[{first} TO {last}]"
        codes = self.source_codes()
        if codes:
            query += " AND (" + " OR ".join(f"SRC:{code}" for code in codes) + ")"
        page_size = min(self.config.page_size, MAX_PAGE_SIZE)
        params = {
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": page_size,
            "cursorMark": cursor or FIRST_CURSOR,
            "synonym": "false",
        }
        return RequestSpec(f"{self.config.base_url}/search", params=params, page_size=page_size)

    def parse_response(self, page: RawPage) -> ParsedPage:
        """Results on the page. Paging stops when a page is empty or the cursor stops moving."""
        body = page.json()
        if not isinstance(body, dict):
            raise SourceParseError("expected a JSON object")
        results = dig(body, "resultList", "result")
        if results is None and "hitCount" not in body:
            raise SourceParseError("response has neither 'resultList' nor 'hitCount'")
        items = tuple(r for r in as_list(results) if isinstance(r, dict))
        sent = str(page.request.params.get("cursorMark", FIRST_CURSOR))
        nxt = body.get("nextCursorMark")
        more = bool(items) and isinstance(nxt, str) and bool(nxt) and nxt != sent
        hits = body.get("hitCount")
        total = hits if isinstance(hits, int) and sent == FIRST_CURSOR else None
        return ParsedPage(items, nxt if more else None, total)

    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        """One search result as a publication record."""
        source = str(item.get("source") or "").strip().upper()
        record_id = str(item.get("id") or "").strip()
        if source not in KNOWN_SOURCES or not _RECORD_ID.match(record_id):
            raise SourceParseError(f"missing or malformed id/source: {source!r}:{record_id!r}")
        store_abstracts = option_bool(self.config, "store_abstracts", True)

        authors_raw = as_list(dig(item, "authorList", "author"))
        authors = unique_texts(a.get("fullName") for a in authors_raw if isinstance(a, dict))
        affiliation_values: list[Any] = []
        for author in authors_raw:
            if not isinstance(author, dict):
                continue
            affiliation_values.append(author.get("affiliation"))
            details = as_list(dig(author, "authorAffiliationDetailsList", "authorAffiliation"))
            affiliation_values.extend(d.get("affiliation") for d in details if isinstance(d, dict))
        affiliations = unique_texts(affiliation_values)
        pub_types = unique_texts(as_list(dig(item, "pubTypeList", "pubType")))
        journal = clean_text(dig(item, "journalInfo", "journal", "title")) or clean_text(
            dig(item, "bookOrReportDetails", "publisher")
        )
        published = item.get("firstPublicationDate") or item.get("pubYear")
        cited = item.get("citedByCount")

        payload = {
            key: item.get(key)
            for key in (
                "id",
                "source",
                "pmid",
                "pmcid",
                "doi",
                "title",
                "pubYear",
                "firstPublicationDate",
                "firstIndexDate",
                "isOpenAccess",
                "citedByCount",
                "language",
            )
            if item.get(key) is not None
        }
        payload.update(
            {
                "authors": authors,
                "affiliations": affiliations,
                "publication_types": pub_types,
                "journal": journal,
                "keywords": unique_texts(as_list(dig(item, "keywordList", "keyword"))),
                "mesh_terms": unique_texts(
                    h.get("descriptorName")
                    for h in as_list(dig(item, "meshHeadingList", "meshHeading"))
                    if isinstance(h, dict)
                ),
            }
        )
        abstract = clean_text(item.get("abstractText")) if store_abstracts else None
        if abstract:
            payload["abstract"] = abstract
        return self.build_record(
            source_record_id=f"{source}:{record_id}",
            title=clean_text(item.get("title")),
            abstract=abstract,
            source_url=ARTICLE_URL.format(source=source, id=record_id),
            published_at=to_utc_datetime(published),
            updated_at_source=to_utc_datetime(item.get("firstIndexDate")),
            payload=payload,
            detail={
                "publication_identifier": clip(f"{source}:{record_id}", 128),
                "journal": clip(journal, 255),
                "publication_date": normalize_date(published),
                "authors": authors or None,
                "affiliations": affiliations or None,
                "citation_count": cited if isinstance(cited, int) and cited >= 0 else None,
                "publication_type": clip(pub_types[0], 64) if pub_types else None,
            },
        )
