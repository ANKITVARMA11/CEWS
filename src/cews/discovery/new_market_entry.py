"""Detecting a competitor's first move into a therapeutic area.

A company that has never worked in an area and suddenly files patents and starts trials there is
one of the strongest early signals in competitive intelligence, and it is invisible in any
ranking of totals: the numbers are small precisely because the work is new.

An entry is recorded when a competitor has at least ``min_records`` records in a research area
inside the recent window and had **nothing** in it over a much longer stretch before that. The
long quiet stretch is what separates a genuine entry from normal fluctuation, and it is why an
entry cannot be declared until there is enough history to be quiet in.

Entries are looked for at topic level by default rather than at the level of whole therapeutic
areas. A broad area accumulates incidental mentions from unrelated work, so a company almost
never looks absent from one; the meaningful signal is the first move into a specific modality or
technology. Pass ``areas_only=True`` to restrict it to whole areas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import PeriodType
from cews.database.models import ActivityAggregate, Organization, Topic
from cews.features.time_windows import add_months, month_start

LOGGER = logging.getLogger(__name__)

AREA_TOPIC_TYPE = "therapeutic_area"
DEFAULT_WINDOW_MONTHS = 6
DEFAULT_QUIET_MONTHS = 18
DEFAULT_MIN_RECORDS = 3


@dataclass(frozen=True)
class MarketEntry:
    """A competitor's first significant activity in an area."""

    organization_id: int
    organization_name: str
    topic_id: int
    topic_name: str
    records_in_window: int
    quiet_months: int
    first_period: date | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored with the score that used it."""
        return {
            "organization": self.organization_name,
            "area": self.topic_name,
            "records_in_window": self.records_in_window,
            "quiet_months_before": self.quiet_months,
            "first_period": self.first_period.isoformat() if self.first_period else None,
        }

    @property
    def description(self) -> str:
        """One sentence stating the observed fact, with no interpretation."""
        return (
            f"{self.organization_name} recorded {self.records_in_window} record(s) in "
            f"{self.topic_name} after {self.quiet_months} month(s) with none"
        )


def _totals(
    session: Session, start: date, end: date, topic_type: str | None
) -> dict[tuple[int, int], tuple[int, date]]:
    """Records per (organization, topic) between two months, with the earliest month."""
    query = (
        select(
            ActivityAggregate.organization_id,
            ActivityAggregate.topic_id,
            func.sum(ActivityAggregate.activity_count),
            func.min(ActivityAggregate.period),
        )
        .where(
            ActivityAggregate.period_type == PeriodType.MONTH.value,
            ActivityAggregate.organization_id.is_not(None),
            ActivityAggregate.topic_id.is_not(None),
            ActivityAggregate.period >= start,
            ActivityAggregate.period < end,
        )
        .group_by(ActivityAggregate.organization_id, ActivityAggregate.topic_id)
    )
    if topic_type is not None:
        query = query.join(Topic, Topic.id == ActivityAggregate.topic_id).where(
            Topic.topic_type == topic_type
        )
    return {
        (int(organization_id), int(topic_id)): (int(total or 0), first)
        for organization_id, topic_id, total, first in session.execute(query).all()
    }


def detect_new_therapeutic_area_entry(
    session: Session,
    *,
    as_of: datetime | date | None = None,
    window_months: int = DEFAULT_WINDOW_MONTHS,
    quiet_months: int = DEFAULT_QUIET_MONTHS,
    min_records: int = DEFAULT_MIN_RECORDS,
    organization_ids: set[int] | None = None,
    areas_only: bool = False,
) -> list[MarketEntry]:
    """Find competitors whose activity in an area began inside the recent window.

    Args:
        as_of: the end of the recent window (default: now). The month in progress is excluded.
        window_months: how far back "recently" reaches.
        quiet_months: how long before that the competitor must have been absent.
        min_records: how much activity counts as a real entry rather than a passing mention.
        organization_ids: limit to these competitors.
        areas_only: restrict to whole therapeutic areas instead of individual topics.

    Raises:
        ValueError: if any window length is not positive.
    """
    if window_months < 1 or quiet_months < 1 or min_records < 1:
        raise ValueError("window_months, quiet_months and min_records must be positive")

    end = month_start(as_of or datetime.now(UTC))
    window_start = add_months(end, -window_months)
    quiet_start = add_months(window_start, -quiet_months)

    recent = _totals(session, window_start, end, AREA_TOPIC_TYPE if areas_only else None)
    before = _totals(session, quiet_start, window_start, AREA_TOPIC_TYPE if areas_only else None)
    if not recent:
        return []

    names = {
        organization.id: organization.canonical_name
        for organization in session.scalars(select(Organization))
    }
    topics = {topic.id: topic.canonical_name for topic in session.scalars(select(Topic))}

    entries: list[MarketEntry] = []
    for (organization_id, topic_id), (count, first) in sorted(recent.items()):
        if organization_ids is not None and organization_id not in organization_ids:
            continue
        if count < min_records:
            continue
        if before.get((organization_id, topic_id), (0, None))[0] > 0:
            continue  # they were already there
        entries.append(
            MarketEntry(
                organization_id=organization_id,
                organization_name=names.get(organization_id, f"organization {organization_id}"),
                topic_id=topic_id,
                topic_name=topics.get(topic_id, f"topic {topic_id}"),
                records_in_window=count,
                quiet_months=quiet_months,
                first_period=first,
            )
        )
    LOGGER.info("detected %d new therapeutic area entr(ies)", len(entries))
    return sorted(entries, key=lambda entry: (-entry.records_in_window, entry.organization_name))
