"""Calendar periods for activity counting.

Trend features are measured over whole months, quarters and years, never over the two-hour
collection cycle. Everything here works on ``date`` objects anchored to the first day of a
period, which is how ``activity_aggregates.period`` stores them.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, date, datetime

from cews.constants import PeriodType

MONTHS_PER_QUARTER = 3


def month_start(value: date | datetime) -> date:
    """First day of the month containing ``value``."""
    moment = value.astimezone(UTC).date() if isinstance(value, datetime) else value
    return date(moment.year, moment.month, 1)


def quarter_start(value: date | datetime) -> date:
    """First day of the quarter containing ``value``."""
    first = month_start(value)
    return date(first.year, first.month - (first.month - 1) % MONTHS_PER_QUARTER, 1)


def year_start(value: date | datetime) -> date:
    """First day of the year containing ``value``."""
    return date(month_start(value).year, 1, 1)


def period_start(value: date | datetime, period_type: PeriodType) -> date:
    """First day of the period of the given type containing ``value``."""
    if period_type is PeriodType.MONTH:
        return month_start(value)
    if period_type is PeriodType.QUARTER:
        return quarter_start(value)
    return year_start(value)


def add_months(start: date, months: int) -> date:
    """Move a period start forward (or back) by whole months."""
    index = start.year * 12 + (start.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def month_range(end: date | datetime, months: int) -> list[date]:
    """The ``months`` month-starts ending with the month containing ``end``, oldest first.

    Raises:
        ValueError: if ``months`` is not positive.
    """
    if months < 1:
        raise ValueError("months must be positive")
    last = month_start(end)
    return [add_months(last, -offset) for offset in range(months - 1, -1, -1)]


def iter_months(start: date | datetime, end: date | datetime) -> Iterator[date]:
    """Month-starts from ``start`` to ``end`` inclusive, oldest first."""
    current, last = month_start(start), month_start(end)
    while current <= last:
        yield current
        current = add_months(current, 1)


def align_series(counts: dict[date, float], months: Sequence[date]) -> list[float]:
    """Line up sparse counts with a month list, filling missing months with zero.

    A month with no records is a real zero for trend purposes, not missing data: the source was
    collected and nothing was published.
    """
    return [float(counts.get(month, 0.0)) for month in months]


def split_series(
    values: Sequence[float], recent: int, previous: int
) -> tuple[list[float], list[float]]:
    """Split a series into its most recent window and the window before it.

    Returns ``(recent, previous)`` as lists, shorter than asked for if the series is short.

    Raises:
        ValueError: if either window length is not positive.
    """
    if recent < 1 or previous < 1:
        raise ValueError("window lengths must be positive")
    recent_values = list(values[-recent:]) if values else []
    earlier = list(values[:-recent]) if len(values) > recent else []
    return recent_values, earlier[-previous:]


def complete_months(values: Sequence[date], as_of: date | datetime) -> list[date]:
    """Drop the month containing ``as_of``: it is still being filled and would look like a dip."""
    current = month_start(as_of)
    return [month for month in values if month < current]
