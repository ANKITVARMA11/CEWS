"""Token-bucket rate limiter for source requests.

One limiter is created per source, from ``requests_per_second`` and ``burst`` in
``config/source_registry.yaml``. It is thread-safe: concurrent callers reserve slots in turn
and sleep outside the lock. ``clock`` and ``sleep`` are injectable so tests run instantly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Allows ``burst`` immediate requests, then one request every ``1 / requests_per_second``."""

    def __init__(
        self,
        requests_per_second: float,
        burst: int = 1,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create a limiter.

        Raises:
            ValueError: if the rate is not positive or ``burst`` is below 1.
        """
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")
        self.requests_per_second = float(requests_per_second)
        self.burst = int(burst)
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._tokens = float(burst)
        self._updated = clock()
        self.total_wait_seconds = 0.0
        self.acquisitions = 0

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._updated)
        self._tokens = min(float(self.burst), self._tokens + elapsed * self.requests_per_second)
        self._updated = now

    def acquire(self) -> float:
        """Wait (if needed) until a request is allowed. Returns the seconds waited."""
        with self._lock:
            self._refill(self._clock())
            self._tokens -= 1.0
            wait = -self._tokens / self.requests_per_second if self._tokens < 0 else 0.0
            self.total_wait_seconds += wait
            self.acquisitions += 1
        if wait > 0:
            self._sleep(wait)
        return wait

    def penalize(self, seconds: float) -> None:
        """Delay future requests by ``seconds`` (used when a server answers HTTP 429)."""
        if seconds <= 0:
            return
        with self._lock:
            self._refill(self._clock())
            self._tokens -= seconds * self.requests_per_second


def apply_rate_limit(limiter: RateLimiter) -> float:
    """Apply ``limiter`` before a request and return the seconds waited."""
    return limiter.acquire()
