"""Unit tests for retry with backoff, Retry-After parsing, and response classification."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import httpx
import pytest

from cews.ingestion.errors import PermanentSourceError, TransientSourceError
from cews.ingestion.retry import (
    backoff_delay,
    classify_response,
    parse_retry_after,
    retry_request,
)

pytestmark = pytest.mark.unit

URL = "https://api.example.test/x"


def _sender(responses: list[int | Exception], headers: dict[int, dict[str, str]] | None = None):
    """Return (send, calls) where send replays ``responses`` in order."""
    items: Iterator[int | Exception] = iter(responses)
    calls: list[int] = []

    def send() -> httpx.Response:
        item = next(items)
        calls.append(1)
        if isinstance(item, Exception):
            raise item
        return httpx.Response(
            item, headers=(headers or {}).get(item, {}), request=httpx.Request("GET", URL)
        )

    return send, calls


def test_success_on_first_attempt_does_not_sleep() -> None:
    send, calls = _sender([200])
    sleeps: list[float] = []
    assert retry_request(send, max_retries=3, sleep=sleeps.append).status_code == 200
    assert len(calls) == 1 and sleeps == []


def test_transient_failures_are_retried_with_exponential_backoff() -> None:
    send, calls = _sender([500, 502, 503, 200])
    sleeps: list[float] = []
    response = retry_request(send, max_retries=3, jitter=False, base_delay=1.0, sleep=sleeps.append)
    assert response.status_code == 200
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_retries_are_exhausted_then_the_error_is_raised() -> None:
    send, calls = _sender([503, 503, 503])
    with pytest.raises(TransientSourceError) as excinfo:
        retry_request(send, max_retries=2, jitter=False, sleep=lambda s: None)
    assert excinfo.value.status_code == 503
    assert len(calls) == 3


def test_zero_retries_means_one_attempt() -> None:
    send, calls = _sender([503])
    with pytest.raises(TransientSourceError):
        retry_request(send, max_retries=0, sleep=lambda s: None)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 422])
def test_permanent_errors_are_not_retried(status: int) -> None:
    send, calls = _sender([status, 200])
    with pytest.raises(PermanentSourceError) as excinfo:
        retry_request(send, max_retries=5, sleep=lambda s: None)
    assert excinfo.value.status_code == status
    assert len(calls) == 1


@pytest.mark.parametrize(
    "error",
    [httpx.ConnectTimeout("slow"), httpx.ReadTimeout("slow"), httpx.ConnectError("refused")],
)
def test_network_errors_are_retried(error: Exception) -> None:
    send, calls = _sender([error, 200])
    assert retry_request(send, max_retries=1, sleep=lambda s: None).status_code == 200
    assert len(calls) == 2


def test_retry_after_header_is_honoured() -> None:
    send, _ = _sender([429, 200], headers={429: {"Retry-After": "7"}})
    sleeps: list[float] = []
    retry_request(send, max_retries=2, jitter=False, sleep=sleeps.append)
    assert sleeps == [7.0]


def test_retry_after_is_capped_by_max_delay() -> None:
    send, _ = _sender([429, 200], headers={429: {"Retry-After": "3600"}})
    sleeps: list[float] = []
    retry_request(send, max_retries=2, jitter=False, max_delay=30, sleep=sleeps.append)
    assert sleeps == [30.0]


def test_on_retry_callback_sees_each_retry() -> None:
    send, _ = _sender([503, 429, 200], headers={429: {"Retry-After": "2"}})
    seen: list[tuple[int | None, float]] = []
    retry_request(
        send,
        max_retries=3,
        jitter=False,
        sleep=lambda s: None,
        on_retry=lambda error, delay: seen.append((error.status_code, delay)),
    )
    assert seen == [(503, 1.0), (429, 2.0)]


def test_negative_retries_rejected() -> None:
    with pytest.raises(ValueError):
        retry_request(lambda: httpx.Response(200), max_retries=-1)


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [(1, 1.0), (2, 2.0), (3, 4.0), (4, 8.0), (10, 60.0)],
)
def test_backoff_delay_curve(attempt: int, expected: float) -> None:
    assert backoff_delay(attempt, base_delay=1.0, max_delay=60.0) == expected


def test_backoff_jitter_stays_within_25_percent() -> None:
    import random

    rng = random.Random(1)
    values = [backoff_delay(3, jitter=True, rng=rng) for _ in range(200)]
    assert all(4.0 <= v <= 5.0 for v in values)
    assert len(set(values)) > 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("5", 5.0),
        ("0", 0.0),
        ("-3", 0.0),
        ("1.5", 1.5),
        ("soon", None),
    ],
)
def test_parse_retry_after_seconds(value: str | None, expected: float | None) -> None:
    assert parse_retry_after(value) == expected


def test_parse_retry_after_http_date() -> None:
    now = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)
    assert parse_retry_after("Tue, 01 Sep 2026 12:00:30 GMT", now=now) == 30.0
    assert parse_retry_after("Tue, 01 Sep 2026 11:00:00 GMT", now=now) == 0.0


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (200, None),
        (302, None),
        (408, TransientSourceError),
        (429, TransientSourceError),
        (599, TransientSourceError),
        (404, PermanentSourceError),
    ],
)
def test_classify_response(status: int, kind: type[Exception] | None) -> None:
    response = httpx.Response(status, request=httpx.Request("GET", URL))
    if kind is None:
        assert classify_response(response) is response
    else:
        with pytest.raises(kind):
            classify_response(response)
