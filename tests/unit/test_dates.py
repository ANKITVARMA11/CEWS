"""Unit tests for cews.normalization.dates."""

from __future__ import annotations

import time
from datetime import UTC, date, datetime, timedelta, timezone

import pytest

from cews.normalization.dates import normalize_date, to_utc_datetime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # ISO and near-ISO (ClinicalTrials.gov, Europe PMC)
        ("2026-03-15", date(2026, 3, 15)),
        ("2026-03", date(2026, 3, 1)),
        ("2026", date(2026, 1, 1)),
        ("2026/03/15", date(2026, 3, 15)),
        ("2026-3-5", date(2026, 3, 5)),
        ("2026-08-20T10:30:00Z", date(2026, 8, 20)),
        ("2026-08-20T10:30:00+00:00", date(2026, 8, 20)),
        # PubMed styles
        ("2024 Mar 5", date(2024, 3, 5)),
        ("2024 Mar", date(2024, 3, 1)),
        ("2024 Mar-Apr", date(2024, 3, 1)),
        ("2023 Winter", date(2023, 12, 1)),
        ("2023 Spring", date(2023, 3, 1)),
        ("Sept 2025", date(2025, 9, 1)),
        # Long forms
        ("March 2026", date(2026, 3, 1)),
        ("15 March 2026", date(2026, 3, 15)),
        ("March 15, 2026", date(2026, 3, 15)),
        # Objects
        (date(2026, 5, 1), date(2026, 5, 1)),
        (datetime(2026, 5, 1, 12, 0, tzinfo=UTC), date(2026, 5, 1)),
        (datetime(2026, 5, 1, 23, 0, tzinfo=timezone(timedelta(hours=-5))), date(2026, 5, 2)),
        # Padding and whitespace
        ("  2026-03-15  ", date(2026, 3, 15)),
    ],
)
def test_supported_formats(value: object, expected: date) -> None:
    assert normalize_date(value) == expected


@pytest.mark.parametrize(
    "value", [None, "", "   ", "garbage", "not a date", "1850", "2500", "2026-13-01", 42, [], {}]
)
def test_unreadable_values_return_none(value: object) -> None:
    assert normalize_date(value) is None


def test_impossible_days_are_clamped_to_the_month() -> None:
    assert normalize_date("2026-02-30") == date(2026, 2, 28)
    assert normalize_date("2024-02-31") == date(2024, 2, 29)  # leap year


def test_struct_time_from_feedparser() -> None:
    assert normalize_date(time.struct_time((2026, 8, 11, 13, 0, 0, 0, 0, 0))) == date(2026, 8, 11)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-08-20T10:30:00+05:30", datetime(2026, 8, 20, 5, 0, tzinfo=UTC)),
        ("2026-08", datetime(2026, 8, 1, tzinfo=UTC)),
        (date(2026, 8, 20), datetime(2026, 8, 20, tzinfo=UTC)),
        (datetime(2026, 8, 20, 6, 0), datetime(2026, 8, 20, 6, 0, tzinfo=UTC)),  # naive means UTC
        ("nonsense", None),
        (None, None),
    ],
)
def test_to_utc_datetime(value: object, expected: datetime | None) -> None:
    assert to_utc_datetime(value) == expected


def test_to_utc_datetime_is_always_timezone_aware() -> None:
    for value in ("2026-08-20", "2026", datetime(2026, 1, 1), date(2026, 1, 1)):
        moment = to_utc_datetime(value)
        assert moment is not None and moment.tzinfo is not None
