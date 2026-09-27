"""Identifier and URL validation helpers.

Phase 2 provides URL validation used when storing source links. Source-specific identifier
validation (NCT ids, PMIDs, DOIs, patent numbers) is added with the adapters.
"""

from __future__ import annotations

from urllib.parse import urlparse

from cews.constants import MAX_URL_LENGTH


def is_valid_url(value: str | None) -> bool:
    """Return True for an http(s) URL with a host and a length within the database limit."""
    if not value or len(value) > MAX_URL_LENGTH or any(ch.isspace() for ch in value):
        return False
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
