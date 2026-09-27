"""Date normalization for source records.

Sources write dates in many shapes: ``2026-03-15``, ``2026-03`` (ClinicalTrials.gov),
``2024 Mar-Apr`` or ``2023 Winter`` (PubMed ``MedlineDate``), ``15 March 2026``, ISO timestamps,
and RSS ``struct_time`` values. :func:`normalize_date` turns all of them into a
:class:`datetime.date`. A partial date resolves to the first day of its month or year, which is
the right granularity for monthly and quarterly activity counts.
"""

from __future__ import annotations

import calendar
import re
import time
from datetime import UTC, date, datetime

_MONTHS: dict[str, int] = {}
for _number in range(1, 13):
    _MONTHS[calendar.month_name[_number].lower()] = _number
    _MONTHS[calendar.month_abbr[_number].lower()] = _number
_MONTHS["sept"] = 9
_SEASONS = {"spring": 3, "summer": 6, "fall": 9, "autumn": 9, "winter": 12}

_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[T ].*)?$")
_YEAR = re.compile(r"^(\d{4})$")
_YEAR_WORD = re.compile(r"^(\d{4})\s+([A-Za-z]+)\.?(?:\s*-\s*[A-Za-z]+\.?)?(?:\s+(\d{1,2}))?")
_WORD_YEAR = re.compile(r"^(?:(\d{1,2})\s+)?([A-Za-z]+)\.?,?\s+(?:(\d{1,2}),?\s+)?(\d{4})$")
MIN_YEAR, MAX_YEAR = 1900, 2100


def _build(year: int, month: int = 1, day: int = 1) -> date | None:
    if not MIN_YEAR <= year <= MAX_YEAR or not 1 <= month <= 12:
        return None
    day = min(max(day, 1), calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _month(word: str) -> int | None:
    key = word.lower().rstrip(".")
    return _MONTHS.get(key) or _SEASONS.get(key)


def normalize_date(value: object) -> date | None:
    """Return ``value`` as a date, or None when it cannot be read.

    Accepts ``date``/``datetime`` objects, ``time.struct_time``, and strings such as
    ``2026-03-15``, ``2026/03``, ``2026``, ``2024 Mar 5``, ``2024 Mar-Apr``, ``2023 Winter``,
    ``March 2026``, ``15 March 2026``, ``March 15, 2026`` and ISO timestamps. Timezone-aware
    datetimes are converted to UTC first. Years outside 1900-2100 are rejected.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value.astimezone(UTC) if value.tzinfo else value
        return moment.date()
    if isinstance(value, date):
        return value
    if isinstance(value, time.struct_time):
        return _build(value.tm_year, value.tm_mon, value.tm_mday)
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    if "T" in text and re.match(r"^\d{4}-\d{2}-\d{2}T", text):
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is not None:
            return normalize_date(parsed)
    if match := _ISO.match(text):
        year, month, day = match.groups()
        return _build(int(year), int(month), int(day or 1))
    if match := _YEAR.match(text):
        return _build(int(match.group(1)))
    if match := _YEAR_WORD.match(text):
        month = _month(match.group(2))
        if month is None:
            return _build(int(match.group(1)))
        return _build(int(match.group(1)), month, int(match.group(3) or 1))
    if match := _WORD_YEAR.match(text):
        month = _month(match.group(2))
        if month is None:
            return None
        day = match.group(1) or match.group(3) or "1"
        return _build(int(match.group(4)), month, int(day))
    return None


def to_utc_datetime(value: object) -> datetime | None:
    """Return ``value`` as a timezone-aware UTC datetime (midnight for date-only values).

    Naive datetimes are assumed to be UTC. Returns None when the value cannot be read.
    """
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, time.struct_time):
        return datetime.fromtimestamp(calendar.timegm(value), tz=UTC)
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value.strip()):
        try:
            return to_utc_datetime(datetime.fromisoformat(value.strip().replace("Z", "+00:00")))
        except ValueError:
            pass
    day = normalize_date(value)
    return datetime(day.year, day.month, day.day, tzinfo=UTC) if day else None
