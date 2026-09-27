"""Unit tests for the shared HTTP client (URL safety, rate limiting, retries, statistics)."""

from __future__ import annotations

import httpx
import pytest

from cews.ingestion.errors import PermanentSourceError, TransientSourceError
from cews.ingestion.http_client import USER_AGENT, HttpClient, validate_request_url
from cews.ingestion.rate_limiter import RateLimiter

pytestmark = pytest.mark.unit

HOST = "api.example.test"


def _client(
    handler, *, retries: int = 2, allowed=(HOST,), limiter: RateLimiter | None = None
) -> HttpClient:
    return HttpClient(
        limiter=limiter or RateLimiter(1000, 1000),
        timeout_seconds=5,
        max_retries=retries,
        allowed_hosts=allowed,
        jitter=False,
        transport=httpx.MockTransport(handler),
        sleep=lambda s: None,
    )


@pytest.mark.parametrize(
    "url",
    [
        "ftp://api.example.test/x",
        "file:///etc/passwd",
        "https://user:pw@api.example.test/x",
        "https:///nohost",
        "https://evil.example.test/x",
    ],
)
def test_unsafe_urls_are_blocked(url: str) -> None:
    with pytest.raises(PermanentSourceError):
        validate_request_url(url, {HOST})


def test_empty_allow_list_permits_any_http_host() -> None:
    assert validate_request_url("https://anything.example.org/feed.xml") == "anything.example.org"


def test_blocked_url_makes_no_request() -> None:
    calls: list[httpx.Request] = []
    client = _client(lambda r: calls.append(r) or httpx.Response(200))
    with pytest.raises(PermanentSourceError):
        client.get("https://other.example.test/x")
    assert calls == []


def test_successful_get_sends_user_agent_and_params_and_records_stats() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ok": True}, headers={"X-RateLimit-Remaining": "41"})

    client = _client(handler)
    response = client.get(f"https://{HOST}/v1", params={"q": "crispr", "page": 2})
    assert response.json() == {"ok": True}
    assert seen[0].headers["User-Agent"] == USER_AGENT
    assert seen[0].url.params["q"] == "crispr" and seen[0].url.params["page"] == "2"
    stats = client.stats.as_dict()
    assert stats["requests"] == 1 and stats["retries"] == 0 and stats["last_status"] == 200
    assert stats["rate_limit_headers"] == {"x-ratelimit-remaining": "41"}


def test_retries_and_429_are_counted_and_slow_the_limiter() -> None:
    answers = iter([429, 503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(answers)
        return httpx.Response(code, headers={"Retry-After": "4"} if code == 429 else {})

    limiter = RateLimiter(1000, 1)
    penalties: list[float] = []
    original = limiter.penalize
    limiter.penalize = lambda s: (penalties.append(s), original(s))  # type: ignore[method-assign]
    client = _client(handler, limiter=limiter)
    assert client.get(f"https://{HOST}/v1").status_code == 200
    stats = client.stats
    assert (stats.requests, stats.retries, stats.rate_limited) == (3, 2, 1)
    assert penalties == [4.0]


def test_exhausted_retries_raise_transient_error() -> None:
    client = _client(lambda r: httpx.Response(503), retries=1)
    with pytest.raises(TransientSourceError):
        client.get(f"https://{HOST}/v1")
    assert client.stats.requests == 2


def test_network_error_is_counted_and_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200)

    client = _client(handler)
    assert client.get(f"https://{HOST}/v1").status_code == 200
    assert client.stats.requests == 2


def test_redirect_to_disallowed_host_is_refused() -> None:
    contacted: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        contacted.append(request.url.host or "")
        if request.url.host == HOST:
            return httpx.Response(302, headers={"Location": "https://elsewhere.example.test/x"})
        return httpx.Response(200)

    client = _client(handler, allowed=(HOST,))
    # the target is checked before it is contacted, so it is never requested
    with pytest.raises(PermanentSourceError, match="allow-list"):
        client.get(f"https://{HOST}/v1")
    assert contacted == [HOST]


def test_reset_stats_and_context_manager() -> None:
    with _client(lambda r: httpx.Response(200)) as client:
        client.get(f"https://{HOST}/v1")
        client.reset_stats()
        assert client.stats.requests == 0


@pytest.mark.parametrize(("timeout", "retries"), [(0, 1), (-1, 1), (5, -1)])
def test_invalid_client_parameters(timeout: float, retries: int) -> None:
    with pytest.raises(ValueError):
        HttpClient(limiter=RateLimiter(1), timeout_seconds=timeout, max_retries=retries)
