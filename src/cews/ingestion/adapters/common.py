"""Helpers shared by the source adapters: search terms, text cleanup and safe field access.

Search scope: by default every adapter searches for the monitored therapeutic areas
(``THERAPEUTIC_AREAS`` in ``.env``). A registry entry can override this with
``options.query_terms`` (a list of terms) or a complete source-specific ``options.query``.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from cews.ingestion.errors import AdapterConfigError
from cews.ingestion.registry import SourceConfig
from cews.settings import Settings

_TAG = re.compile(r"<[^>]+>")
_SIMPLE_TERM = re.compile(r"^[\w ]+$")
MAX_TERMS = 50
MAX_LIST_ITEMS = 50


def search_terms(settings: Settings, config: SourceConfig) -> list[str]:
    """Terms to search for: ``options.query_terms`` if set, else the monitored areas.

    Raises:
        AdapterConfigError: if ``options.query_terms`` is not a list of strings, or there are
            no terms at all.
    """
    raw = config.options.get("query_terms")
    if raw is None:
        terms = settings.therapeutic_area_list
    elif isinstance(raw, list) and all(isinstance(t, str) for t in raw):
        terms = [" ".join(t.split()) for t in raw if t.strip()]
    else:
        raise AdapterConfigError(f"{config.id}: options.query_terms must be a list of strings")
    if not terms:
        raise AdapterConfigError(
            f"{config.id}: no search terms; set THERAPEUTIC_AREAS or options.query_terms"
        )
    if len(terms) > MAX_TERMS:
        raise AdapterConfigError(f"{config.id}: at most {MAX_TERMS} search terms are supported")
    for term in terms:
        if '"' in term or "\\" in term:
            raise AdapterConfigError(f"{config.id}: search terms must not contain quotes: {term!r}")
    return terms


def or_query(terms: Sequence[str], *, quote_phrases: bool) -> str:
    """Join terms with OR. Terms with punctuation are always quoted; multi-word terms too when
    ``quote_phrases`` is true (Lucene-style sources). PubMed leaves plain phrases unquoted so its
    automatic term mapping (MeSH synonyms) still applies.
    """
    parts = []
    for term in terms:
        simple = bool(_SIMPLE_TERM.match(term))
        needs_quotes = not simple or (quote_phrases and " " in term)
        parts.append(f'"{term}"' if needs_quotes else term)
    return "(" + " OR ".join(parts) + ")" if len(parts) > 1 else parts[0]


def option_query(config: SourceConfig) -> str | None:
    """A complete query override from ``options.query``, if configured.

    Raises:
        AdapterConfigError: if the option is present but not a non-empty string.
    """
    value = config.options.get("query")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdapterConfigError(f"{config.id}: options.query must be a non-empty string")
    return value.strip()


def option_bool(config: SourceConfig, name: str, default: bool) -> bool:
    """Read a boolean option.

    Raises:
        AdapterConfigError: if the option is present but not a boolean.
    """
    value = config.options.get(name, default)
    if not isinstance(value, bool):
        raise AdapterConfigError(f"{config.id}: options.{name} must be true or false")
    return value


def clean_text(value: Any) -> str | None:
    """Plain text from ``value``: HTML tags removed, entities decoded, whitespace collapsed."""
    if value is None:
        return None
    text = html.unescape(_TAG.sub(" ", str(value)))
    text = " ".join(text.split())
    return text or None


def clip(value: str | None, limit: int) -> str | None:
    """Truncate ``value`` to ``limit`` characters (PostgreSQL enforces column lengths)."""
    if value is None:
        return None
    return value if len(value) <= limit else value[: limit - 1] + "…"


def dig(data: Any, *path: str) -> Any:
    """Follow ``path`` through nested mappings; None as soon as a key is missing."""
    current = data
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def as_list(value: Any) -> list[Any]:
    """``value`` as a list: lists unchanged, None as empty, anything else wrapped."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def unique_texts(values: Iterable[Any], limit: int = MAX_LIST_ITEMS) -> list[str]:
    """Distinct cleaned strings in first-seen order, capped at ``limit``."""
    seen: dict[str, None] = {}
    for value in values:
        text = clean_text(value)
        if text and text.casefold() not in {k.casefold() for k in seen}:
            seen[text] = None
        if len(seen) >= limit:
            break
    return list(seen)
