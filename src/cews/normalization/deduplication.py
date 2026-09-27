"""Record hashing and cross-source de-duplication.

:func:`create_record_hash` drives change detection during upserts: the same record fetched
again produces the same digest, so nothing is rewritten.

The rest of the module finds the *same work* collected from different sources - a paper indexed
by both PubMed and Europe PMC, or a preprint that later appears as a journal article. Matching
is deterministic, in order of reliability: DOI, then PubMed ID, then an exact normalized title
within the same year. Duplicates keep their rows (they are evidence of where the work was
found) but are pointed at a canonical record, so activity counts never count one paper twice.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"cannot hash value of type {type(value).__name__}")


def _clean(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


def create_record_hash(
    *,
    source: str,
    source_record_id: str,
    title: str | None = None,
    abstract: str | None = None,
    published_at: datetime | None = None,
    payload: Mapping[str, Any] | None = None,
) -> str:
    """Return a stable SHA-256 hex digest of a record's content.

    The digest changes when the title, abstract, publication time or payload changes, and
    is unaffected by whitespace differences and dictionary key order. Fetch time is not
    part of the hash, so re-fetching an unchanged record yields the same digest.

    Raises:
        ValueError: if ``source`` or ``source_record_id`` is empty.
        TypeError: if the payload contains values that cannot be serialized.
    """
    if not source or not source_record_id:
        raise ValueError("source and source_record_id are required to hash a record")
    canonical = json.dumps(
        {
            "source": source,
            "source_record_id": source_record_id,
            "title": _clean(title),
            "abstract": _clean(abstract),
            "published_at": published_at.isoformat() if published_at else None,
            "payload": dict(payload) if payload else {},
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Cross-source duplicates
# --------------------------------------------------------------------------------------
DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:", "doi.org/")
_TITLE_NOISE = re.compile(r"[^a-z0-9 ]+")
_PMID = re.compile(r"^\d{1,9}$")


def normalize_doi(value: Any) -> str | None:
    """Return a bare lower-case DOI (``10.xxxx/yyyy``), or None if it is not one."""
    if not isinstance(value, str):
        return None
    text = value.strip().casefold()
    for prefix in DOI_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
    text = text.strip().strip(".")
    return text if text.startswith("10.") and "/" in text else None


def normalize_title(value: Any) -> str | None:
    """Return a title reduced to lower-case words, for exact comparison."""
    if not isinstance(value, str):
        return None
    text = _TITLE_NOISE.sub(" ", value.casefold())
    words = text.split()
    return " ".join(words) if len(words) >= 4 else None  # too short to be distinctive


def duplicate_keys(
    *,
    title: str | None,
    published_at: datetime | None,
    payload: Mapping[str, Any] | None,
    source_record_id: str | None = None,
    source: str | None = None,
) -> list[tuple[str, str]]:
    """Return ``(kind, key)`` pairs identifying the work, strongest first.

    ``doi`` and ``pmid`` come from the stored payload (and, for PubMed, from the record id).
    ``title`` is a fallback that also includes the publication year, so unrelated papers that
    share a generic title in different years are not merged. Callers apply it only across
    different sources (see :func:`find_duplicates`).
    """
    keys: list[tuple[str, str]] = []
    payload = payload or {}
    doi = normalize_doi(payload.get("doi"))
    if doi:
        keys.append(("doi", doi))
    pmid = payload.get("pmid")
    if source == "pubmed" and source_record_id and _PMID.match(str(source_record_id)):
        pmid = source_record_id
    if isinstance(pmid, str | int) and _PMID.match(str(pmid)):
        keys.append(("pmid", str(pmid)))
    normalized_title = normalize_title(title)
    if normalized_title:
        year = published_at.year if published_at else 0
        keys.append(("title", f"{year}:{normalized_title}"))
    return keys


@dataclass(frozen=True)
class DuplicateLink:
    """One record identified as a copy of another."""

    record_id: int
    canonical_id: int
    reason: str


@dataclass(frozen=True)
class DeduplicationSummary:
    """What a de-duplication pass found."""

    examined: int
    duplicates: int
    by_reason: dict[str, int]
    cleared: int = 0


def find_duplicates(records: Sequence[Any]) -> list[DuplicateLink]:
    """Return the duplicates among ``records`` (objects with the SourceRecord columns).

    The record seen first (lowest id) is the canonical one; every later record sharing a key
    points at it. Records are compared only within the same record type.
    """
    canonical: dict[tuple[str, str, str], int] = {}
    sources: dict[int, str] = {}
    links: list[DuplicateLink] = []
    for record in sorted(records, key=lambda r: r.id):
        sources[record.id] = record.source
        keys = duplicate_keys(
            title=record.title,
            published_at=record.published_at,
            payload=record.raw_payload_json,
            source_record_id=record.source_record_id,
            source=record.source,
        )
        match: tuple[str, int] | None = None
        for kind, key in keys:
            existing = canonical.get((record.record_type, kind, key))
            if existing is None or existing == record.id:
                continue
            # A shared identifier is proof; a shared title is only a hint, and two records from
            # the *same* source with the same title are usually different works.
            if kind == "title" and sources.get(existing) == record.source:
                continue
            match = (kind, existing)
            break
        if match is not None:
            links.append(DuplicateLink(record.id, match[1], match[0]))
            continue
        for kind, key in keys:
            canonical.setdefault((record.record_type, kind, key), record.id)
    return links
