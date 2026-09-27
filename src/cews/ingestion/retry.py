"""Retry with exponential backoff for source requests, built on tenacity.

``retry_request`` calls a request function, classifies the outcome, and retries only
:class:`~cews.ingestion.errors.TransientSourceError` (timeouts, connection errors, HTTP 429
and 5xx). A server's ``Retry-After`` value is honoured, capped at ``max_delay``. Permanent
errors (other 4xx responses) are raised immediately.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import tenacity

from cews.ingestion.errors import PermanentSourceError, TransientSourceError

LOGGER = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
DEFAULT_BASE_DELAY = 1.0
DEFAULT_MAX_DELAY = 60.0


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP date) into seconds, or None."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - (now or datetime.now(UTC))).total_seconds()
    return max(0.0, seconds)


def classify_response(response: httpx.Response) -> httpx.Response:
    """Return ``response`` if it succeeded; raise the right error type otherwise.

    Raises:
        TransientSourceError: for 408, 425, 429 and 5xx responses.
        PermanentSourceError: for other 4xx responses.
    """
    status = response.status_code
    if status < 400:
        return response
    host = response.request.url.host if response.request is not None else "source"
    if status in RETRYABLE_STATUS_CODES or status >= 500:
        raise TransientSourceError(
            f"HTTP {status} from {host}",
            status_code=status,
            retry_after=parse_retry_after(response.headers.get("Retry-After")),
        )
    raise PermanentSourceError(f"HTTP {status} from {host}", status_code=status)


def backoff_delay(
    attempt: int,
    *,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    retry_after: float | None = None,
    jitter: bool = False,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based).

    Exponential: ``base_delay * 2 ** (attempt - 1)``, capped at ``max_delay``. A server's
    ``retry_after`` wins over the computed value (still capped). Optional jitter adds up to 25%.
    """
    delay = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
    if retry_after is not None:
        delay = min(max_delay, max(retry_after, 0.0))
    if jitter and delay > 0:
        delay += (rng or random).uniform(0, 0.25 * delay)
    return delay


def retry_request(
    send: Callable[[], httpx.Response],
    *,
    max_retries: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: bool = True,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[TransientSourceError, float], None] | None = None,
) -> httpx.Response:
    """Call ``send`` until it returns a successful response or retries run out.

    Args:
        send: Performs one HTTP request and returns the response.
        max_retries: Retries after the first attempt (0 means a single attempt).
        base_delay, max_delay: Backoff shape in seconds.
        jitter: Add up to 25% random jitter to each delay (disable in tests).
        sleep: Wait function (injectable for tests).
        on_retry: Called with the error and the delay before each retry.

    Raises:
        TransientSourceError: after the last retry fails.
        PermanentSourceError: immediately, for non-retryable responses.
        ValueError: if ``max_retries`` is negative.
    """
    if max_retries < 0:
        raise ValueError("max_retries must not be negative")

    def attempt() -> httpx.Response:
        try:
            response = send()
        except httpx.TimeoutException as exc:
            raise TransientSourceError(f"timeout: {type(exc).__name__}") from exc
        except httpx.TransportError as exc:
            raise TransientSourceError(f"connection error: {type(exc).__name__}") from exc
        return classify_response(response)

    def wait(state: tenacity.RetryCallState) -> float:
        error = state.outcome.exception() if state.outcome else None
        retry_after = error.retry_after if isinstance(error, TransientSourceError) else None
        delay = backoff_delay(
            state.attempt_number,
            base_delay=base_delay,
            max_delay=max_delay,
            retry_after=retry_after,
            jitter=jitter,
        )
        if isinstance(error, TransientSourceError):
            LOGGER.info(
                "retrying after %s in %.1fs (attempt %d)", error, delay, state.attempt_number
            )
            if on_retry is not None:
                on_retry(error, delay)
        return delay

    retrying = tenacity.Retrying(
        stop=tenacity.stop_after_attempt(max_retries + 1),
        wait=wait,
        retry=tenacity.retry_if_exception_type(TransientSourceError),
        sleep=sleep,
        reraise=True,
    )
    return retrying(attempt)
