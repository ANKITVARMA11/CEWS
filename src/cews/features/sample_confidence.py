"""Confidence inputs: sample size, freshness and completeness.

These do not measure activity; they measure how much the activity numbers can be trusted. Each
returns 0 to 1 and feeds the confidence score that is shown next to every analytical score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

DEFAULT_SATURATION_K = 25.0
DEFAULT_FRESHNESS_HALF_LIFE_DAYS = 30.0


def calculate_sample_confidence(records: float, *, k: float = DEFAULT_SATURATION_K) -> float:
    """How much a count can be trusted, from 0 (nothing) to 1 (plenty).

    ``1 - exp(-N / k)``: confidence climbs quickly through the first records and flattens, so
    ten records are worth far more than two, while three hundred are barely better than two
    hundred. ``k`` is the saturation constant (25 by default: about 63% confidence at 25
    records).

    Raises:
        ValueError: for a negative or non-finite count, or a non-positive k.
    """
    if records < 0 or not math.isfinite(records):
        raise ValueError("record count must be finite and not negative")
    if k <= 0 or not math.isfinite(k):
        raise ValueError("k must be a positive, finite number")
    return 1.0 - math.exp(-records / k)


def calculate_data_freshness(
    last_updated: datetime | None,
    *,
    as_of: datetime | None = None,
    half_life_days: float = DEFAULT_FRESHNESS_HALF_LIFE_DAYS,
) -> float:
    """How current the underlying data is, from 0 (stale or never collected) to 1 (just now).

    Confidence halves every ``half_life_days``, so a source last collected two half-lives ago
    contributes a quarter as much.

    Raises:
        ValueError: for a naive datetime or a non-positive half-life.
    """
    if half_life_days <= 0 or not math.isfinite(half_life_days):
        raise ValueError("half_life_days must be a positive, finite number")
    if last_updated is None:
        return 0.0
    if last_updated.tzinfo is None:
        raise ValueError("last_updated must be timezone-aware")
    now = as_of or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    age_days = (now - last_updated).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return float(0.5 ** (age_days / half_life_days))


def calculate_data_completeness(present: Sequence[bool] | Mapping[str, bool]) -> float:
    """The share of expected inputs that are actually present, from 0 to 1.

    Accepts a sequence of flags or a mapping of name to flag (for example one entry per expected
    month, or per required field).

    Raises:
        ValueError: if nothing is expected.
    """
    flags = list(present.values()) if isinstance(present, Mapping) else list(present)
    if not flags:
        raise ValueError("completeness needs at least one expected input")
    return sum(1 for flag in flags if flag) / len(flags)
