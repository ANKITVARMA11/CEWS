"""Fake servers for the real source adapters, backed by ``data/fixtures/sources``.

Each server answers only its source's real host names, serves the fixture pages the adapter's
paging parameters ask for, records every request, and can be told to fail (``fail_always``) or
to answer a path with a given status (``status_for``). Traffic goes through
``httpx.MockTransport``; nothing can reach the network.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx

from cews.ingestion.base import SourceAdapter, build_http_client
from cews.ingestion.registry import SourceConfig, load_source_registry
from cews.settings import Settings, load_settings
from support import REGISTRY_FILE, REPO_ROOT
from support_adapters import no_sleep

FIXTURES = REPO_ROOT / "data" / "fixtures" / "sources"
FEED_A = "https://news.fixture-a.test/rss.xml"
FEED_B = "https://ir.fixture-b.test/atom.xml"


def fixture_bytes(relative: str) -> bytes:
    """Raw bytes of a fixture file."""
    return (FIXTURES / relative).read_bytes()


def fixture_json(relative: str) -> Any:
    """A parsed JSON fixture."""
    return json.loads(fixture_bytes(relative))


def registry_config(source_id: str, **overrides: Any) -> SourceConfig:
    """The shipped registry entry for ``source_id``, without rate-limit waits in tests."""
    config = load_source_registry(REGISTRY_FILE).get(source_id)
    return replace(config, **{"requests_per_second": 1000.0, "burst": 1000, **overrides})


class FixtureServer:
    """Base class: host check, request log, forced failures."""

    hosts: frozenset[str] = frozenset()

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_always = False
        self.status_for: dict[tuple[str, str], int] = {}  # (host, path) -> status code

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.host not in self.hosts:
            raise AssertionError(f"unexpected host in test: {request.url.host}")
        self.requests.append(request)
        if self.fail_always:
            return httpx.Response(503, request=request)
        forced = self.status_for.get((request.url.host, request.url.path))
        if forced is not None:
            return httpx.Response(forced, request=request)
        return self.respond(request)

    def respond(self, request: httpx.Request) -> httpx.Response:
        raise NotImplementedError

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def paths(self) -> list[str]:
        return [r.url.path for r in self.requests]


def _json(request: httpx.Request, data: Any) -> httpx.Response:
    return httpx.Response(200, json=data, request=request)


class ClinicalTrialsServer(FixtureServer):
    """``/api/v2/studies`` with ``pageToken`` paging, and ``/api/v2/stats/size``."""

    hosts = frozenset({"clinicaltrials.gov"})

    def respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v2/stats/size":
            return _json(request, fixture_json("clinical_trials_gov/stats_size.json"))
        if path == "/api/v2/studies":
            token = request.url.params.get("pageToken")
            if token is None:
                return _json(request, fixture_json("clinical_trials_gov/studies_page_1.json"))
            if token == "FIXTURE_PAGE_TOKEN_2":
                return _json(request, fixture_json("clinical_trials_gov/studies_page_2.json"))
            return httpx.Response(400, request=request)
        return httpx.Response(404, request=request)


class PubMedServer(FixtureServer):
    """``esearch`` (by ``retstart``), ``efetch`` (by ``id`` list) and ``einfo``."""

    hosts = frozenset({"eutils.ncbi.nlm.nih.gov"})

    def __init__(self) -> None:
        super().__init__()
        self.esearch_override: dict[str, Any] | None = None

    def respond(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.rsplit("/", 1)[-1]
        params = request.url.params
        if path == "einfo.fcgi":
            return _json(request, fixture_json("pubmed/einfo.json"))
        if path == "esearch.fcgi":
            if self.esearch_override is not None:
                return _json(request, self.esearch_override)
            start = params.get("retstart", "0")
            if start in ("0", "2"):
                return _json(request, fixture_json(f"pubmed/esearch_retstart_{start}.json"))
            empty = fixture_json("pubmed/esearch_retstart_2.json")
            empty["esearchresult"].update({"retstart": start, "idlist": []})
            return _json(request, empty)
        if path == "efetch.fcgi":
            ids = params.get("id", "")
            name = f"pubmed/efetch_{ids.replace(',', '_')}.xml"
            if (FIXTURES / name).is_file():
                return httpx.Response(200, content=fixture_bytes(name), request=request)
            return httpx.Response(
                200, content=b"<PubmedArticleSet></PubmedArticleSet>", request=request
            )
        return httpx.Response(404, request=request)


class EuropePmcServer(FixtureServer):
    """``/search`` with ``cursorMark`` paging; ``resultType=idlist`` answers health checks."""

    hosts = frozenset({"www.ebi.ac.uk"})

    def respond(self, request: httpx.Request) -> httpx.Response:
        if request.url.path != "/europepmc/webservices/rest/search":
            return httpx.Response(404, request=request)
        params = request.url.params
        if params.get("resultType") == "idlist":
            return _json(request, fixture_json("europe_pmc/health.json"))
        cursor = params.get("cursorMark", "*")
        if cursor == "*":
            return _json(request, fixture_json("europe_pmc/search_page_1.json"))
        if cursor == "FIXTURE_CURSOR_2":
            return _json(request, fixture_json("europe_pmc/search_page_2.json"))
        return _json(request, {"hitCount": 3, "resultList": {"result": []}})


class RssServer(FixtureServer):
    """Two feeds on two hosts; ``robots`` sets each host's robots.txt (None means HTTP 404)."""

    hosts = frozenset({"news.fixture-a.test", "ir.fixture-b.test"})

    def __init__(self) -> None:
        super().__init__()
        self.robots: dict[str, bytes | None] = {
            "news.fixture-a.test": fixture_bytes("generic_rss/robots.txt"),
            "ir.fixture-b.test": None,
        }
        self.feeds: dict[str, bytes] = {
            "news.fixture-a.test": fixture_bytes("generic_rss/fixture_a_rss.xml"),
            "ir.fixture-b.test": fixture_bytes("generic_rss/fixture_b_atom.xml"),
        }

    def respond(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/robots.txt":
            body = self.robots.get(host)
            if body is None:
                return httpx.Response(404, request=request)
            return httpx.Response(200, content=body, request=request)
        if request.url.path in ("/rss.xml", "/atom.xml"):
            return httpx.Response(200, content=self.feeds[host], request=request)
        return httpx.Response(404, request=request)


def build_adapter(
    adapter_cls: type[SourceAdapter],
    config: SourceConfig,
    server: FixtureServer,
    settings: Settings | None = None,
) -> SourceAdapter:
    """An adapter wired to a fixture server, with no rate-limit or retry waiting."""
    settings = settings or load_settings(env_file=None)
    http = build_http_client(
        settings, config, transport=server.transport(), sleep=no_sleep, jitter=False
    )
    return adapter_cls(settings, config, http)


def rss_config(**overrides: Any) -> SourceConfig:
    """The generic_rss registry entry with the two fixture feeds (override with ``feeds=``)."""
    overrides.setdefault("feeds", (FEED_A, FEED_B))
    return registry_config("generic_rss", **overrides)


__all__ = [
    "FEED_A",
    "FEED_B",
    "FIXTURES",
    "ClinicalTrialsServer",
    "EuropePmcServer",
    "FixtureServer",
    "PubMedServer",
    "RssServer",
    "build_adapter",
    "fixture_bytes",
    "fixture_json",
    "registry_config",
    "rss_config",
]
