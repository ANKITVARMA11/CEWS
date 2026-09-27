"""HTTP client shared by all source adapters.

Every request goes through the same steps: URL validation (http/https only, no embedded
credentials, host must be on the source's allow-list), rate limiting, the request itself with
a timeout, and retry with backoff. Statistics (requests, retries, HTTP 429 responses, seconds
spent waiting, and the server's rate-limit headers) are collected for the ingestion audit.

Tests pass an ``httpx.MockTransport`` so no real network traffic ever happens.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import httpx

from cews import __version__
from cews.ingestion.errors import PermanentSourceError, TransientSourceError
from cews.ingestion.rate_limiter import RateLimiter, apply_rate_limit
from cews.ingestion.retry import DEFAULT_BASE_DELAY, DEFAULT_MAX_DELAY, retry_request

LOGGER = logging.getLogger(__name__)

USER_AGENT = f"CEWS/{__version__} (local competitive-intelligence proof of concept)"
MAX_REDIRECTS = 5
RATE_LIMIT_HEADERS = (
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
    "ratelimit-limit",
    "ratelimit-remaining",
    "ratelimit-reset",
)


@dataclass
class RequestStats:
    """Counters for the requests made by one client."""

    requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    wait_seconds: float = 0.0
    last_status: int | None = None
    rate_limit_headers: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy of the counters."""
        return {
            "requests": self.requests,
            "retries": self.retries,
            "rate_limited": self.rate_limited,
            "wait_seconds": round(self.wait_seconds, 3),
            "last_status": self.last_status,
            "rate_limit_headers": dict(self.rate_limit_headers),
        }


def validate_request_url(url: str, allowed_hosts: Iterable[str] = ()) -> str:
    """Return the host of ``url`` after checking it is safe to request.

    Raises:
        PermanentSourceError: for non-http(s) URLs, URLs with credentials, a missing host, or a
            host outside ``allowed_hosts`` (when the allow-list is not empty).
    """
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise PermanentSourceError(f"invalid URL: {exc}") from exc
    if parts.scheme not in {"http", "https"}:
        raise PermanentSourceError(f"only http(s) URLs are allowed, got scheme {parts.scheme!r}")
    if parts.username or parts.password:
        raise PermanentSourceError("URLs must not contain credentials")
    host = (parts.hostname or "").lower()
    if not host:
        raise PermanentSourceError("URL has no host")
    allowed = {h.lower() for h in allowed_hosts}
    if allowed and host not in allowed:
        raise PermanentSourceError(f"host {host!r} is not on this source's allow-list")
    return host


class HttpClient:
    """Rate-limited, retrying HTTP client for one source."""

    def __init__(
        self,
        *,
        limiter: RateLimiter,
        timeout_seconds: float,
        max_retries: int,
        allowed_hosts: Iterable[str] = (),
        base_delay: float = DEFAULT_BASE_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        jitter: bool = True,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        """Create a client.

        Raises:
            ValueError: for a non-positive timeout or negative retry count.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self.limiter = limiter
        self.max_retries = max_retries
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._jitter = jitter
        self._sleep = sleep
        self._stats = RequestStats()
        self._stats_lock = threading.Lock()
        headers = {"User-Agent": USER_AGENT, **dict(default_headers or {})}
        # Redirects are followed by hand (see get) so a redirect target is checked against the
        # allow-list *before* it is contacted, not after.
        self._client = httpx.Client(
            timeout=timeout_seconds, transport=transport, headers=headers, follow_redirects=False
        )

    # -- statistics -----------------------------------------------------------------------
    @property
    def stats(self) -> RequestStats:
        """A copy of the current counters."""
        with self._stats_lock:
            return RequestStats(
                **{
                    **self._stats.__dict__,
                    "rate_limit_headers": dict(self._stats.rate_limit_headers),
                }
            )

    def reset_stats(self) -> None:
        """Start counting from zero (called at the start of each collection)."""
        with self._stats_lock:
            self._stats = RequestStats()

    def _record(self, response: httpx.Response | None, waited: float) -> None:
        with self._stats_lock:
            self._stats.requests += 1
            self._stats.wait_seconds += waited
            if response is not None:
                self._stats.last_status = response.status_code
                if response.status_code == 429:
                    self._stats.rate_limited += 1
                for name in RATE_LIMIT_HEADERS:
                    if name in response.headers:
                        self._stats.rate_limit_headers[name] = response.headers[name]

    def _on_retry(self, error: TransientSourceError, delay: float) -> None:
        with self._stats_lock:
            self._stats.retries += 1
            self._stats.wait_seconds += delay
        if error.status_code == 429:
            self.limiter.penalize(delay)

    # -- requests -------------------------------------------------------------------------
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """GET ``url`` with rate limiting and retries. Returns a successful response.

        Raises:
            PermanentSourceError: for a blocked URL or a non-retryable HTTP error.
            TransientSourceError: when retries are exhausted.
        """
        host = validate_request_url(url, self.allowed_hosts)

        def send() -> httpx.Response:
            target = httpx.URL(url)
            # Passing an empty params mapping would REPLACE a query string already in the URL,
            # so only pass params when there are some.
            query = dict(params) if params else None
            for _ in range(MAX_REDIRECTS + 1):
                waited = apply_rate_limit(self.limiter)
                try:
                    response = self._client.get(target, params=query, headers=dict(headers or {}))
                except httpx.HTTPError:
                    self._record(None, waited)
                    raise
                self._record(response, waited)
                if not response.is_redirect:
                    return response
                location = response.headers.get("Location", "")
                if not location:
                    raise PermanentSourceError(
                        f"HTTP {response.status_code} from {response.url.host} without a Location"
                    )
                target = response.url.join(location)
                validate_request_url(str(target), self.allowed_hosts)
                query = None  # the redirect target carries its own query string
            raise PermanentSourceError(f"too many redirects (more than {MAX_REDIRECTS})")

        LOGGER.debug("GET %s", host)
        return retry_request(
            send,
            max_retries=self.max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            jitter=self._jitter,
            sleep=self._sleep,
            on_retry=self._on_retry,
        )

    def close(self) -> None:
        """Close the underlying connection pool."""
        self._client.close()

    def __enter__(self) -> HttpClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
