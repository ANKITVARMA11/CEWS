"""PubMed adapter (NCBI E-utilities, no key required; ``NCBI_API_KEY`` raises the rate limit).

Each page takes two requests:

1. ``esearch.fcgi`` (JSON) - PMIDs of records added to PubMed inside the window
   (``datetype=edat``, ``mindate``/``maxdate`` inclusive, ``retstart``/``retmax`` paging);
2. ``efetch.fcgi`` (XML) - the full citations for those PMIDs, parsed with ``defusedxml``.

Limits (NCBI documentation, verified 2026-09-22):

* 3 requests/second without an API key, 10 with one (``requests_per_second`` in the registry
  stays below 3, and each page makes two requests);
* one search can only reach its first 10,000 records (``retstart + retmax <= 10,000``). When a
  date slice matches more, the run keeps the first 10,000 and records a warning; shorten
  ``window_slice_days`` or narrow the query to get everything;
* NCBI asks clients to identify themselves: ``tool=cews`` and ``email`` (``NCBI_EMAIL``).

Abstracts are publisher-copyrighted. They are stored for internal analysis by default; set
``options.store_abstracts: false`` in the registry if your organization's policy says otherwise.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as SafeET
from defusedxml import DefusedXmlException

from cews.constants import SourceType
from cews.ingestion.adapters.common import (
    clean_text,
    clip,
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

SEARCH_CAP = 10_000
PMID = re.compile(r"^\d{1,9}$")
ARTICLE_URL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
TOOL_NAME = "cews"
MAX_MESH_TERMS = 40


@dataclass(frozen=True)
class PubMedPage(RawPage):
    """An efetch XML page plus the esearch facts needed for paging."""

    pmids: tuple[str, ...] = ()
    total: int = 0
    retstart: int = 0


# --------------------------------------------------------------------------------------
# XML parsing
# --------------------------------------------------------------------------------------
def _text(element: Element | None) -> str | None:
    return clean_text("".join(element.itertext())) if element is not None else None


def _dated(element: Element | None) -> tuple[date | None, str | None]:
    """Read a PubMed date element and how precise it is ("day", "month" or "year").

    Handles Year/Month/Day children (month as number or name) and ``MedlineDate`` strings
    such as ``2026 Jul-Aug`` or ``2023 Winter``.
    """
    if element is None:
        return None, None
    medline = element.findtext("MedlineDate")
    if medline:
        value = normalize_date(medline)
        has_month = bool(re.search(r"\d{4}\s+[A-Za-z]", medline.strip()))
        return value, ("month" if has_month else "year") if value else None
    year = element.findtext("Year")
    if not year:
        return None, None
    month = element.findtext("Month") or ""
    day = element.findtext("Day") or ""
    if month and not month.isdigit():
        value = normalize_date(f"{year} {month} {day}".strip())
    else:
        value = normalize_date("-".join(p for p in (year, month, day) if p))
    precision = "day" if month and day else "month" if month else "year"
    return value, precision if value else None


def _date(element: Element | None) -> date | None:
    """Read a PubMed date element (see :func:`_dated`)."""
    return _dated(element)[0]


def _author(element: Element) -> str | None:
    collective = element.findtext("CollectiveName")
    if collective:
        return clean_text(collective)
    names = [element.findtext("ForeName"), element.findtext("LastName")]
    return clean_text(" ".join(n for n in names if n)) or clean_text(element.findtext("LastName"))


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def parse_article(article: Element) -> dict[str, Any]:
    """One ``PubmedArticle`` element as a plain dictionary."""
    citation = article.find("MedlineCitation")
    if citation is None:
        raise SourceParseError("PubmedArticle without MedlineCitation")
    body = citation.find("Article")
    pubmed_data = article.find("PubmedData")
    ids = pubmed_data.find("ArticleIdList") if pubmed_data is not None else None

    def article_id(kind: str) -> str | None:
        if ids is None:
            return None
        for node in ids.findall("ArticleId"):
            if node.get("IdType") == kind and node.text:
                return node.text.strip()
        return None

    history = pubmed_data.find("History") if pubmed_data is not None else None
    entrez = None
    if history is not None:
        for node in history.findall("PubMedPubDate"):
            if node.get("PubStatus") in ("entrez", "pubmed"):
                entrez = _date(node)
                if node.get("PubStatus") == "entrez":
                    break

    abstract_parts = []
    authors: list[str | None] = []
    affiliations: list[str | None] = []
    if body is not None:
        for part in body.findall("Abstract/AbstractText"):
            text = _text(part)
            if text:
                label = part.get("Label")
                abstract_parts.append(f"{label}: {text}" if label else text)
        for node in body.findall("AuthorList/Author"):
            authors.append(_author(node))
            affiliations.extend(_text(a) for a in node.findall("AffiliationInfo/Affiliation"))

    electronic = None
    if body is not None:
        for node in body.findall("ArticleDate"):
            if node.get("DateType", "Electronic") == "Electronic":
                electronic = _date(node)
                break

    publication_date, precision = (
        _dated(body.find("Journal/JournalIssue/PubDate")) if body is not None else (None, None)
    )
    return {
        "pmid": (citation.findtext("PMID") or "").strip(),
        "title": _text(body.find("ArticleTitle")) if body is not None else None,
        "abstract": " ".join(abstract_parts) or None,
        "journal": _text(body.find("Journal/Title")) if body is not None else None,
        "journal_iso": _text(body.find("Journal/ISOAbbreviation")) if body is not None else None,
        "publication_date": _iso(publication_date),
        "publication_date_precision": precision,
        "electronic_date": _iso(electronic),
        "entrez_date": _iso(entrez),
        "revised_date": _iso(_date(citation.find("DateRevised"))),
        "authors": unique_texts(authors),
        "affiliations": unique_texts(affiliations),
        "publication_types": (
            unique_texts(n.text for n in body.findall("PublicationTypeList/PublicationType"))
            if body is not None
            else []
        ),
        "language": citation.findtext("Article/Language"),
        "mesh_terms": unique_texts(
            (n.text for n in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")),
            limit=MAX_MESH_TERMS,
        ),
        "keywords": unique_texts(
            (n.text for n in citation.findall("KeywordList/Keyword")), limit=MAX_MESH_TERMS
        ),
        "doi": article_id("doi"),
        "pmcid": article_id("pmc"),
    }


def parse_pubmed_xml(content: bytes) -> list[dict[str, Any]]:
    """Parse an efetch ``PubmedArticleSet`` document.

    Book records (``PubmedBookArticle``) are skipped. The parser refuses entity expansion and
    external references (``defusedxml``); the standard DOCTYPE line NCBI sends is accepted.

    Raises:
        SourceParseError: if the document is not well-formed or not a PubmedArticleSet.
    """
    if not content.strip():
        return []
    try:
        root = SafeET.fromstring(content)
    except (SafeET.ParseError, DefusedXmlException) as exc:
        raise SourceParseError(f"efetch returned invalid XML: {exc}") from exc
    if root.tag != "PubmedArticleSet":
        raise SourceParseError(f"expected PubmedArticleSet, got {root.tag}")
    return [parse_article(article) for article in root.findall("PubmedArticle")]


# --------------------------------------------------------------------------------------
# Adapter
# --------------------------------------------------------------------------------------
def choose_publication_date(item: Mapping[str, Any]) -> str | None:
    """The date an article counts under in monthly activity.

    Order: the electronic publication date; the journal issue date when it has at least month
    precision; the date the record entered PubMed; a year-only issue date as a last resort. A
    year-only date would otherwise place every such article in January.
    """
    if item.get("electronic_date"):
        return str(item["electronic_date"])
    issue = item.get("publication_date")
    if issue and item.get("publication_date_precision") in ("day", "month"):
        return str(issue)
    return item.get("entrez_date") or issue


@register_adapter
class PubMedAdapter(SourceAdapter):
    """Publications added to PubMed (by Entrez date)."""

    source_name = "pubmed"
    source_type = SourceType.PUBLICATION

    def validate_configuration(self) -> list[str]:
        """Check the base URL, the search terms and the options."""
        problems = super().validate_configuration()
        try:
            self._term()
            option_bool(self.config, "store_abstracts", True)
        except AdapterConfigError as exc:
            problems.append(str(exc))
        if self.config.page_size > SEARCH_CAP:
            problems.append(f"{self.source_name}: page_size must not exceed {SEARCH_CAP}")
        return problems

    def health_check_url(self) -> str | None:
        """The E-utilities database description for PubMed."""
        if not self.config.base_url:
            return None
        return f"{self.config.base_url}/einfo.fcgi?db=pubmed&retmode=json"

    def _term(self) -> str:
        return option_query(self.config) or or_query(
            search_terms(self.settings, self.config), quote_phrases=False
        )

    def _identity(self) -> dict[str, str]:
        params = {"tool": TOOL_NAME}
        if self.settings.ncbi_email:
            params["email"] = self.settings.ncbi_email
        if self.settings.ncbi_api_key is not None:
            params["api_key"] = self.settings.ncbi_api_key.get_secret_value()
        return params

    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        """The esearch request for one page of PMIDs added inside ``window``."""
        first = window.start.date()
        last = (window.end - timedelta(microseconds=1)).date()
        page_size = min(self.config.page_size, SEARCH_CAP)
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": self._term(),
            "datetype": "edat",
            "mindate": first.strftime("%Y/%m/%d"),
            "maxdate": last.strftime("%Y/%m/%d"),
            "retmode": "json",
            "retmax": page_size,
            "retstart": int(cursor or 0),
            **self._identity(),
        }
        return RequestSpec(
            f"{self.config.base_url}/esearch.fcgi", params=params, page_size=page_size
        )

    def fetch_page(self, query: RequestSpec) -> RawPage:
        """Run esearch, then efetch for the PMIDs it returned.

        Raises:
            SourceParseError: if esearch reports an error or returns an unexpected shape.
        """
        response = self.http.get(query.url, params=query.params)
        try:
            body = response.json()
        except ValueError as exc:
            raise SourceParseError(f"esearch returned invalid JSON: {exc}") from exc
        result = body.get("esearchresult") if isinstance(body, dict) else None
        if not isinstance(result, dict):
            raise SourceParseError("esearch response has no 'esearchresult'")
        if result.get("ERROR"):
            raise SourceParseError(f"esearch error: {str(result['ERROR'])[:200]}")
        try:
            total = int(result.get("count", 0))
            retstart = int(result.get("retstart", query.params.get("retstart", 0)))
            pmids = tuple(str(p) for p in result.get("idlist", []))
        except (TypeError, ValueError) as exc:
            raise SourceParseError(f"esearch returned malformed counts: {exc}") from exc

        content = b""
        if pmids:
            fetched = self.http.get(
                f"{self.config.base_url}/efetch.fcgi",
                params={
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "retmode": "xml",
                    **self._identity(),
                },
            )
            content = fetched.content
        return PubMedPage(
            request=query,
            status_code=response.status_code,
            content=content,
            headers=dict(response.headers),
            fetched_at=datetime.now(UTC),
            pmids=pmids,
            total=total,
            retstart=retstart,
        )

    def parse_response(self, page: RawPage) -> ParsedPage:
        """Citations on the page, the next ``retstart`` and a warning at the 10,000 cap."""
        if not isinstance(page, PubMedPage):
            raise SourceParseError("PubMed pages must come from PubMedAdapter.fetch_page")
        items = tuple(parse_pubmed_xml(page.content))
        next_start = page.retstart + len(page.pmids)
        warnings: tuple[str, ...] = ()
        if page.retstart == 0 and page.total > SEARCH_CAP:
            params = page.request.params
            warnings = (
                f"PubMed matched {page.total} records for {params.get('mindate')} to "
                f"{params.get('maxdate')}; only the first {SEARCH_CAP} can be retrieved. "
                "Shorten window_slice_days or narrow the query.",
            )
        more = bool(page.pmids) and next_start < min(page.total, SEARCH_CAP)
        return ParsedPage(
            items,
            str(next_start) if more else None,
            min(page.total, SEARCH_CAP) if page.retstart == 0 else None,
            warnings=warnings,
        )

    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        """One citation as a publication record."""
        pmid = str(item.get("pmid") or "").strip()
        if not PMID.match(pmid):
            raise SourceParseError(f"missing or malformed PMID: {pmid!r}")
        store_abstracts = option_bool(self.config, "store_abstracts", True)
        payload = dict(item)
        if not store_abstracts:
            payload.pop("abstract", None)
        published = choose_publication_date(item)
        types = item.get("publication_types") or []
        return self.build_record(
            source_record_id=pmid,
            title=item.get("title"),
            abstract=item.get("abstract") if store_abstracts else None,
            source_url=ARTICLE_URL.format(pmid=pmid),
            published_at=to_utc_datetime(published),
            updated_at_source=to_utc_datetime(item.get("revised_date") or item.get("entrez_date")),
            payload=payload,
            detail={
                "publication_identifier": pmid,
                "journal": clip(item.get("journal"), 255),
                "publication_date": normalize_date(published),
                "authors": item.get("authors") or None,
                "affiliations": item.get("affiliations") or None,
                "citation_count": None,
                "publication_type": clip(types[0], 64) if types else None,
            },
        )
