"""Unit tests for the calendar period helpers."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from cews.constants import PeriodType
from cews.features.time_windows import (
    add_months,
    align_series,
    complete_months,
    iter_months,
    month_range,
    month_start,
    period_start,
    quarter_start,
    split_series,
    year_start,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2026, 8, 17), date(2026, 8, 1)),
        (date(2026, 1, 1), date(2026, 1, 1)),
        (date(2026, 12, 31), date(2026, 12, 1)),
        (datetime(2026, 8, 17, 23, 30, tzinfo=UTC), date(2026, 8, 1)),
    ],
)
def test_month_start(value: date | datetime, expected: date) -> None:
    assert month_start(value) == expected


def test_month_start_uses_utc() -> None:
    """A timestamp late on the last day of a month elsewhere is still that month in UTC."""
    ist = timezone(timedelta(hours=5, minutes=30))
    assert month_start(datetime(2026, 9, 1, 2, 0, tzinfo=ist)) == date(2026, 8, 1)


@pytest.mark.parametrize(
    ("month", "expected"),
    [(1, 1), (2, 1), (3, 1), (4, 4), (6, 4), (7, 7), (9, 7), (10, 10), (12, 10)],
)
def test_quarter_start(month: int, expected: int) -> None:
    assert quarter_start(date(2026, month, 15)) == date(2026, expected, 1)


def test_year_start() -> None:
    assert year_start(date(2026, 8, 17)) == date(2026, 1, 1)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (PeriodType.MONTH, date(2026, 8, 1)),
        (PeriodType.QUARTER, date(2026, 7, 1)),
        (PeriodType.YEAR, date(2026, 1, 1)),
    ],
)
def test_period_start(kind: PeriodType, expected: date) -> None:
    assert period_start(date(2026, 8, 17), kind) == expected


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (date(2026, 8, 1), 1, date(2026, 9, 1)),
        (date(2026, 12, 1), 1, date(2027, 1, 1)),
        (date(2026, 1, 1), -1, date(2025, 12, 1)),
        (date(2026, 6, 1), 0, date(2026, 6, 1)),
        (date(2026, 6, 1), -18, date(2024, 12, 1)),
    ],
)
def test_add_months(start: date, months: int, expected: date) -> None:
    assert add_months(start, months) == expected


def test_month_range_is_oldest_first_and_ends_with_the_given_month() -> None:
    months = month_range(date(2026, 8, 17), 4)
    assert months == [date(2026, 5, 1), date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)]


def test_month_range_crosses_a_year() -> None:
    assert month_range(date(2026, 2, 1), 4)[0] == date(2025, 11, 1)


def test_month_range_needs_a_positive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        month_range(date(2026, 8, 1), 0)


def test_iter_months_is_inclusive() -> None:
    months = list(iter_months(date(2026, 6, 10), date(2026, 8, 20)))
    assert months == [date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)]
    assert list(iter_months(date(2026, 8, 1), date(2026, 8, 31))) == [date(2026, 8, 1)]
    assert list(iter_months(date(2026, 9, 1), date(2026, 8, 1))) == []


def test_align_series_fills_quiet_months_with_zero() -> None:
    """A month with no records is a real zero, not missing data."""
    months = month_range(date(2026, 8, 1), 3)
    assert align_series({date(2026, 7, 1): 5.0}, months) == [0.0, 5.0, 0.0]
    assert align_series({}, months) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("values", "recent", "previous", "expected"),
    [
        ([1, 2, 3, 4, 5, 6], 3, 3, ([4, 5, 6], [1, 2, 3])),
        ([1, 2, 3, 4, 5, 6, 7], 3, 3, ([5, 6, 7], [2, 3, 4])),
        ([1, 2], 3, 3, ([1, 2], [])),
        ([], 3, 3, ([], [])),
        ([1, 2, 3, 4], 1, 2, ([4], [2, 3])),
    ],
)
def test_split_series(
    values: list[float], recent: int, previous: int, expected: tuple[list[float], list[float]]
) -> None:
    assert split_series(values, recent, previous) == expected


@pytest.mark.parametrize(("recent", "previous"), [(0, 3), (3, 0), (-1, 2)])
def test_split_series_needs_positive_windows(recent: int, previous: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        split_series([1, 2, 3], recent, previous)


def test_complete_months_drops_the_month_in_progress() -> None:
    months = month_range(date(2026, 8, 20), 3)
    assert complete_months(months, date(2026, 8, 20)) == [date(2026, 6, 1), date(2026, 7, 1)]
