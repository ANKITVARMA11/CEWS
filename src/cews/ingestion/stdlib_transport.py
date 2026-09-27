"""An httpx transport backed by the standard library, for hosts that reject httpx.

ClinicalTrials.gov sits behind a firewall that fingerprints the TLS handshake and answers
``403 Forbidden`` to ``httpx`` (and to ``requests``' async cousins) while accepting the same
request from Python's standard library. Rather than change how CEWS talks to every source, a
source can opt into this transport in ``config/source_registry.yaml``::

    options:
      http_transport: stdlib

Everything else - rate limiting, retries, host allow-lists, redirect checks, statistics - is
unchanged, because this only replaces the layer that puts bytes on the wire.

Notes:

* redirects are **not** followed here; httpx handles them, so its host checks still apply;
* gzip and deflate responses are decompressed (``br``/``zstd`` are never requested);
* proxy environment variables (``HTTPS_PROXY`` and friends) are honoured by urllib;
* network failures are raised as the httpx errors the retry logic already understands.
"""

from __future__ import annotations

import gzip
import logging
import socket
import urllib.error
import urllib.request
import zlib
from typing import Any

import httpx

LOGGER = logging.getLogger(__name__)

ACCEPT_ENCODING = "gzip, deflate"
DEFAULT_TIMEOUT = 30.0
STDLIB = "stdlib"


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    """Return redirects to the caller instead of following them."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def _decompress(body: bytes, encoding: str) -> bytes:
    encoding = (encoding or "").lower().strip()
    try:
        if encoding == "gzip":
            return gzip.decompress(body)
        if encoding == "deflate":
            return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, zlib.error) as exc:
        raise httpx.DecodingError(f"cannot decode {encoding} response: {exc}") from exc
    return body


class StdlibTransport(httpx.BaseTransport):
    """Sends requests with ``urllib.request`` and returns httpx responses."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Build an opener that does not follow redirects."""
        self._timeout = timeout
        self._opener = urllib.request.build_opener(_NoRedirects)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Perform one request. Network problems become httpx errors so retries still work."""
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in ("host", "accept-encoding", "connection")
        }
        headers["Accept-Encoding"] = ACCEPT_ENCODING
        body = request.read() or None
        raw = urllib.request.Request(
            str(request.url), data=body, headers=headers, method=request.method
        )
        timeout = self._timeout
        extension = request.extensions.get("timeout")
        if isinstance(extension, dict) and extension.get("read"):
            timeout = float(extension["read"])

        try:
            with self._opener.open(raw, timeout=timeout) as response:
                content = _decompress(response.read(), response.headers.get("Content-Encoding", ""))
                return httpx.Response(
                    response.status,
                    headers=_clean_headers(response.headers.items()),
                    content=content,
                    request=request,
                )
        except urllib.error.HTTPError as error:  # 4xx and 5xx still carry a body
            content = _decompress(error.read(), error.headers.get("Content-Encoding", ""))
            return httpx.Response(
                error.code,
                headers=_clean_headers(error.headers.items()),
                content=content,
                request=request,
            )
        except TimeoutError as error:
            raise httpx.ReadTimeout(str(error) or "read timed out", request=request) from error
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, TimeoutError | socket.timeout):
                raise httpx.ReadTimeout(str(reason), request=request) from error
            raise httpx.ConnectError(str(reason), request=request) from error
        except OSError as error:  # pragma: no cover - rare socket-level failures
            raise httpx.ConnectError(str(error), request=request) from error


def _clean_headers(items: Any) -> list[tuple[str, str]]:
    """Drop headers that describe the transfer we have already undone."""
    skip = {"content-encoding", "content-length", "transfer-encoding"}
    return [(name, value) for name, value in items if name.lower() not in skip]
