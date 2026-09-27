"""Tests for the standard-library transport and the URL query-string handling.

The transport is exercised against a throwaway HTTP server on localhost, so it really goes
through ``urllib`` without touching the internet.
"""

from __future__ import annotations

import gzip
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import httpx
import pytest

from cews.ingestion.base import build_http_client
from cews.ingestion.errors import PermanentSourceError
from cews.ingestion.http_client import HttpClient
from cews.ingestion.rate_limiter import RateLimiter
from cews.ingestion.stdlib_transport import StdlibTransport
from cews.settings import load_settings
from support_sources import registry_config

pytestmark = pytest.mark.unit

RECEIVED: list[dict[str, Any]] = []


class _Handler(BaseHTTPRequestHandler):
    """Answers a few fixed paths and records what it was sent."""

    protocol_version = "HTTP/1.0"

    def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        RECEIVED.append({"path": self.path, "headers": dict(self.headers)})
        if self.path.startswith("/json"):
            self._send(200, json.dumps({"path": self.path}).encode())
        elif self.path == "/gzip":
            self._send(200, gzip.compress(b'{"compressed": true}'), encoding="gzip")
        elif self.path == "/forbidden":
            self._send(403, b"<html>403</html>")
        elif self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/elsewhere")
            self.end_headers()
        else:
            self._send(404, b"not found")

    def _send(self, status: int, body: bytes, encoding: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Keep the test output quiet."""


@pytest.fixture
def server() -> Iterator[str]:
    RECEIVED.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _client(**kwargs: Any) -> HttpClient:
    return HttpClient(
        limiter=RateLimiter(1000.0),
        timeout_seconds=10,
        max_retries=0,
        allowed_hosts={"127.0.0.1"},
        transport=StdlibTransport(),
        **kwargs,
    )


def test_successful_request(server: str) -> None:
    with _client() as client:
        response = client.get(f"{server}/json")
    assert response.status_code == 200 and response.json() == {"path": "/json"}


def test_query_string_and_params_reach_the_server(server: str) -> None:
    with _client() as client:
        client.get(f"{server}/json?query=cancer&format=json")
        client.get(f"{server}/json", params={"query": "cancer", "pageSize": 1})
    assert RECEIVED[0]["path"] == "/json?query=cancer&format=json"
    assert RECEIVED[1]["path"] == "/json?query=cancer&pageSize=1"


def test_user_agent_is_sent(server: str) -> None:
    with _client() as client:
        client.get(f"{server}/json")
    assert RECEIVED[0]["headers"]["User-Agent"].startswith("CEWS/")


def test_gzip_responses_are_decompressed(server: str) -> None:
    with _client() as client:
        response = client.get(f"{server}/gzip")
    assert response.json() == {"compressed": True}
    assert RECEIVED[0]["headers"]["Accept-Encoding"] == "gzip, deflate"


def test_error_responses_become_source_errors(server: str) -> None:
    with _client() as client, pytest.raises(PermanentSourceError) as excinfo:
        client.get(f"{server}/forbidden")
    assert "403" in str(excinfo.value)
    assert "127.0.0.1" in str(excinfo.value)


def test_a_redirect_to_another_host_is_refused_before_it_is_followed(server: str) -> None:
    with _client() as client, pytest.raises(PermanentSourceError, match="allow-list"):
        client.get(f"{server}/redirect")
    # only the original request was made; the redirect target was never contacted
    assert [entry["path"] for entry in RECEIVED] == ["/redirect"]


def test_connection_failures_are_reported(server: str) -> None:
    unreachable = "http://127.0.0.1:9"  # discard port
    with _client() as client, pytest.raises(Exception) as excinfo:
        client.get(f"{unreachable}/json")
    assert "127.0.0.1" in str(excinfo.value) or "connect" in str(excinfo.value).lower()


def test_registry_option_selects_the_transport() -> None:
    settings = load_settings(env_file=None)
    stdlib = build_http_client(settings, registry_config("clinical_trials_gov"))
    default = build_http_client(settings, registry_config("pubmed"))
    try:
        assert isinstance(stdlib._client._transport, StdlibTransport)
        assert isinstance(default._client._transport, httpx.HTTPTransport)
    finally:
        stdlib.close()
        default.close()


def test_an_explicit_transport_always_wins() -> None:
    mock = httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    client = build_http_client(
        load_settings(env_file=None), registry_config("clinical_trials_gov"), transport=mock
    )
    try:
        assert client._client._transport is mock  # tests are never sent over the network
    finally:
        client.close()
