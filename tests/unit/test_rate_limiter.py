"""Unit tests for the token-bucket rate limiter."""

from __future__ import annotations

import threading

import pytest

from cews.ingestion.rate_limiter import RateLimiter, apply_rate_limit

pytestmark = pytest.mark.unit


class FakeClock:
    """Clock whose time only moves when the code under test sleeps (or a test advances it)."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 6))
        self.now += seconds


def _limiter(rate: float, burst: int = 1) -> tuple[RateLimiter, FakeClock]:
    clock = FakeClock()
    return RateLimiter(rate, burst, clock=clock.time, sleep=clock.sleep), clock


def test_first_request_is_immediate_then_requests_are_spaced() -> None:
    limiter, clock = _limiter(2.0)
    waits = [limiter.acquire() for _ in range(4)]
    assert waits == [0.0, 0.5, 0.5, 0.5]
    assert clock.now == pytest.approx(1.5)


def test_rate_is_respected_over_many_requests() -> None:
    limiter, clock = _limiter(4.0)
    for _ in range(41):
        apply_rate_limit(limiter)
    assert clock.now == pytest.approx(10.0)  # 40 intervals of 0.25 s
    assert limiter.acquisitions == 41
    assert limiter.total_wait_seconds == pytest.approx(10.0)


def test_burst_allows_immediate_requests_then_throttles() -> None:
    limiter, _ = _limiter(1.0, burst=3)
    assert [limiter.acquire() for _ in range(5)] == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_idle_time_refills_tokens_up_to_burst() -> None:
    limiter, clock = _limiter(1.0, burst=2)
    limiter.acquire()
    limiter.acquire()
    clock.now += 100
    assert [limiter.acquire() for _ in range(3)] == [0.0, 0.0, 1.0]


def test_penalize_delays_the_next_request() -> None:
    limiter, _ = _limiter(1.0)
    limiter.acquire()
    limiter.penalize(10)
    assert limiter.acquire() == pytest.approx(11.0)


def test_penalize_ignores_non_positive_values() -> None:
    limiter, _ = _limiter(1.0)
    limiter.penalize(0)
    limiter.penalize(-5)
    assert limiter.acquire() == 0.0


@pytest.mark.parametrize(("rate", "burst"), [(0, 1), (-1, 1), (1, 0)])
def test_invalid_parameters(rate: float, burst: int) -> None:
    with pytest.raises(ValueError):
        RateLimiter(rate, burst)


def test_concurrent_callers_share_one_budget() -> None:
    lock = threading.Lock()
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        with lock:
            clock.sleeps.append(seconds)

    limiter = RateLimiter(10.0, clock=clock.time, sleep=sleep)
    threads = [threading.Thread(target=limiter.acquire) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # With a frozen clock, reservations stack up: 19 callers wait 0.1, 0.2, ... 1.9 s.
    assert sorted(round(s, 6) for s in clock.sleeps) == [round(0.1 * i, 6) for i in range(1, 20)]
