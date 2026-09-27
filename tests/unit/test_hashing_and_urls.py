"""Unit tests for record hashing and URL validation."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from cews.constants import MAX_URL_LENGTH
from cews.normalization.deduplication import create_record_hash
from cews.normalization.identifiers import is_valid_url

pytestmark = pytest.mark.unit

BASE = {"source": "s", "source_record_id": "1", "title": "A title", "abstract": "Body"}


def test_hash_is_a_stable_sha256_digest() -> None:
    digest = create_record_hash(**BASE)
    assert len(digest) == 64
    assert digest == create_record_hash(**BASE)


def test_hash_ignores_whitespace_and_key_order() -> None:
    first = create_record_hash(**{**BASE, "title": "A   title"}, payload={"x": 1, "y": [1, 2]})
    second = create_record_hash(**BASE, payload={"y": [1, 2], "x": 1})
    assert first == second


@pytest.mark.parametrize(
    "change",
    [
        {"title": "Another title"},
        {"abstract": "Different"},
        {"source": "t"},
        {"source_record_id": "2"},
        {"payload": {"x": 2}},
        {"published_at": datetime(2026, 1, 1, tzinfo=UTC)},
    ],
)
def test_hash_changes_when_content_changes(change: dict[str, object]) -> None:
    assert create_record_hash(**BASE, payload={"x": 1}) != create_record_hash(
        **{**BASE, **change}, **({} if "payload" in change else {"payload": {"x": 1}})
    )


def test_hash_handles_dates_in_payload() -> None:
    digest = create_record_hash(**BASE, payload={"when": date(2026, 1, 2)})
    assert digest == create_record_hash(**BASE, payload={"when": date(2026, 1, 2)})


def test_hash_rejects_unserializable_payload() -> None:
    with pytest.raises(TypeError):
        create_record_hash(**BASE, payload={"bad": object()})


@pytest.mark.parametrize("field", ["source", "source_record_id"])
def test_hash_requires_source_identity(field: str) -> None:
    with pytest.raises(ValueError, match="required"):
        create_record_hash(**{**BASE, field: ""})


@pytest.mark.parametrize(
    "url",
    [
        "https://example.org/path?q=1",
        "http://localhost:8000/x",
        "https://example.invalid/synthetic/1",
    ],
)
def test_valid_urls(url: str) -> None:
    assert is_valid_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "example.org",
        "ftp://example.org/file",
        "javascript:alert(1)",
        "http://",
        "https://exa mple.org",
        "https://example.org/" + "a" * MAX_URL_LENGTH,
    ],
)
def test_invalid_urls(url: str | None) -> None:
    assert is_valid_url(url) is False
