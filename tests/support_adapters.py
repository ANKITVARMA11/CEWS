"""A reference adapter and a fake paginated JSON API, used by contract and ingestion tests.

``FakeApi`` behaves like a typical research API: ``GET /v1/records`` with ``from``/``to``
date filters, ``limit`` and an opaque ``cursor``, returning ``{"total", "items", "next"}``.
It can be told to fail pages, send HTTP 429, or return malformed items. Every request goes
through ``httpx.MockTransport``; unknown hosts raise, so tests can never reach the network.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs

import httpx

from cews.constants import SourceType
from cews.ingestion.base import SourceAdapter
from cews.ingestion.errors import SourceParseError
from cews.ingestion.registry import SourceConfig
from cews.ingestion.results import (
    CollectionWindow,
    NormalizedRecord,
    ParsedPage,
    RawPage,
    RequestSpec,
)

API_HOST = "api.example.test"
API_BASE = f"https://{API_HOST}"
START = datetime(2026, 1, 1, tzinfo=UTC)


def make_items(count: int, *, start: datetime = START, step_days: int = 3) -> list[dict[str, Any]]:
    """Items spread ``step_days`` apart, published from ``start`` onwards."""
    return [
        {
            "id": f"EX-{i:04d}",
            "title": f"Example record {i}",
            "abstract": f"Abstract for record {i}",
            "published": (start + timedelta(days=i * step_days)).isoformat(),
            "journal": "Example Journal",
        }
        for i in range(count)
    ]


@dataclass
class FakeApi:
    """In-memory fake of a paginated, date-filtered JSON API."""

    items: list[dict[str, Any]] = field(default_factory=list)
    fail_pages: set[int] = field(default_factory=set)  # request numbers answered with 503
    fail_always: bool = False
    rate_limit_first: int = 0  # first N requests answered with 429
    not_found: bool = False
    malformed_ids: set[str] = field(default_factory=set)  # items returned without a title field
    broken_json_pages: set[int] = field(default_factory=set)
    requests: list[httpx.Request] = field(default_factory=list)

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host != API_HOST:
            raise AssertionError(f"unexpected host in test: {request.url.host}")
        self.requests.append(request)
        number = len(self.requests)
        if self.not_found:
            return httpx.Response(404, request=request)
        if number <= self.rate_limit_first:
            return httpx.Response(429, headers={"Retry-After": "2"}, request=request)
        if self.fail_always or number in self.fail_pages:
            return httpx.Response(503, request=request)
        if request.url.path == "/v1/health":
            return httpx.Response(200, json={"ok": True}, request=request)
        if number in self.broken_json_pages:
            return httpx.Response(200, content=b"{not json", request=request)
        query = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        low = datetime.fromisoformat(query["from"])
        high = datetime.fromisoformat(query["to"])
        limit = int(query.get("limit", "10"))
        offset = int(query.get("cursor", "0"))
        matching = [i for i in self.items if low <= datetime.fromisoformat(i["published"]) < high]
        page = matching[offset : offset + limit]
        served = [
            {
                k: v
                for k, v in item.items()
                if not (item["id"] in self.malformed_ids and k == "title")
            }
            for item in page
        ]
        nxt = offset + limit if offset + limit < len(matching) else None
        body = {
            "total": len(matching),
            "items": served,
            "next": str(nxt) if nxt is not None else None,
        }
        return httpx.Response(
            200, json=body, headers={"X-RateLimit-Remaining": "99"}, request=request
        )

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


class ExampleAdapter(SourceAdapter):
    """Reference adapter for the fake API. Real adapters follow the same pattern."""

    source_name = "example_source"
    source_type = SourceType.PUBLICATION

    def health_check_url(self) -> str | None:
        return f"{API_BASE}/v1/health"

    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        params: dict[str, Any] = {
            "from": window.start.isoformat(),
            "to": window.end.isoformat(),
            "limit": self.config.page_size,
        }
        if cursor:
            params["cursor"] = cursor
        return RequestSpec(f"{API_BASE}/v1/records", params=params, page_size=self.config.page_size)

    def parse_response(self, page: RawPage) -> ParsedPage:
        body = page.json()
        if not isinstance(body, dict) or not isinstance(body.get("items"), list):
            raise SourceParseError("expected an object with an 'items' list")
        return ParsedPage(tuple(body["items"]), body.get("next"), body.get("total"))

    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        published = datetime.fromisoformat(item["published"])
        return self.build_record(
            source_record_id=item["id"],
            title=item["title"],
            abstract=item.get("abstract"),
            source_url=f"{API_BASE}/records/{item['id']}",
            published_at=published,
            payload=dict(item),
            detail={
                "publication_identifier": item["id"],
                "journal": item.get("journal"),
                "publication_date": published.date(),
            },
        )


def example_config(**overrides: Any) -> SourceConfig:
    """Registry entry for the example source (small pages so pagination is exercised)."""
    values: dict[str, Any] = {
        "id": "example_source",
        "source_type": SourceType.PUBLICATION,
        "env_flag": "ENABLE_PUBMED",
        "base_url": API_BASE,
        "page_size": 5,
        "max_pages_per_run": 100,
        "window_slice_days": 30,
        "requests_per_second": 1000.0,
        "burst": 1000,
    }
    values.update(overrides)
    return SourceConfig(**values)


def no_sleep(_seconds: float) -> None:
    """Replacement for time.sleep in tests."""


def items_between(
    items: list[dict[str, Any]], low: datetime, high: datetime
) -> list[Mapping[str, Any]]:
    """Items published in ``[low, high)``."""
    return [i for i in items if low <= datetime.fromisoformat(i["published"]) < high]
