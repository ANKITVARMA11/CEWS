"""Counting activity into calendar periods.

Records are grouped into monthly buckets per topic, per competitor, per competitor-within-topic
and overall, for each source type, and stored in ``activity_aggregates``. Quarters and years are
summed from the months, so the periods always agree with each other.

Two numbers are kept per bucket. ``activity_count`` is how many records there were.
``weighted_activity`` sums the link confidence instead, so weaker evidence counts for less (an
author's affiliation is stored at half the weight of a sponsored trial). Scoring uses the
weighted figure; the plain count is what people recognise on a dashboard.

Counting is done by the database, and re-running it rewrites the same rows rather than adding
more.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from cews.constants import PeriodType, SourceType
from cews.database.models import ActivityAggregate, RecordOrganization, RecordTopic, SourceRecord
from cews.features.time_windows import month_start, quarter_start, year_start

LOGGER = logging.getLogger(__name__)

# (period, period_type, organization_id, topic_id, source_type); None means "all".
Bucket = tuple[date, str, int | None, int | None, str]


@dataclass
class AggregationSummary:
    """What one aggregation pass wrote."""

    months: int = 0
    rows_written: int = 0
    rows_updated: int = 0
    rows_removed: int = 0
    records_counted: int = 0
    first_period: date | None = None
    last_period: date | None = None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "months": self.months,
            "rows_written": self.rows_written,
            "rows_updated": self.rows_updated,
            "rows_removed": self.rows_removed,
            "records_counted": self.records_counted,
            "first_period": self.first_period.isoformat() if self.first_period else None,
            "last_period": self.last_period.isoformat() if self.last_period else None,
            "warnings": list(self.warnings),
        }


def month_expression(session: Session) -> ColumnElement[str]:
    """A database expression giving the first day of a record's month, as ``YYYY-MM-DD``.

    SQLite and PostgreSQL spell this differently, so the grouping stays in the database on both
    instead of pulling every record into Python.
    """
    if session.get_bind().dialect.name == "postgresql":
        return func.to_char(func.date_trunc("month", SourceRecord.published_at), "YYYY-MM-DD")
    return func.strftime("%Y-%m-01", SourceRecord.published_at)


def _base(session: Session, start: date | None, end: date) -> Select[Any]:
    query = select(month_expression(session).label("period"), SourceRecord.record_type).where(
        SourceRecord.published_at.is_not(None),
        SourceRecord.published_at < datetime(end.year, end.month, end.day, tzinfo=UTC),
        SourceRecord.duplicate_of_id.is_(None),
    )
    if start is not None:
        query = query.where(
            SourceRecord.published_at >= datetime(start.year, start.month, start.day, tzinfo=UTC)
        )
    return query


def _counts(
    session: Session, start: date | None, end: date
) -> tuple[dict[Bucket, tuple[int, float, int]], int]:
    """Count records per bucket with one grouped query per breakdown."""
    buckets: dict[Bucket, tuple[int, float, int]] = {}
    month = month_expression(session)
    monthly = PeriodType.MONTH.value

    def collect(query: Select[Any], has_organization: bool, has_topic: bool) -> None:
        for row in session.execute(query).all():
            key: Bucket = (
                date.fromisoformat(str(row.period)),
                monthly,
                int(row.organization_id) if has_organization else None,
                int(row.topic_id) if has_topic else None,
                str(row.record_type),
            )
            buckets[key] = (int(row.records), float(row.weighted or 0.0), int(row.records))

    overall = (
        _base(session, start, end)
        .add_columns(
            func.count(func.distinct(SourceRecord.id)).label("records"),
            func.count(func.distinct(SourceRecord.id)).label("weighted"),
        )
        .group_by(month, SourceRecord.record_type)
    )
    collect(overall, False, False)

    by_organization = (
        _base(session, start, end)
        .add_columns(
            RecordOrganization.organization_id.label("organization_id"),
            func.count(func.distinct(SourceRecord.id)).label("records"),
            func.sum(RecordOrganization.confidence).label("weighted"),
        )
        .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
        .group_by(month, SourceRecord.record_type, RecordOrganization.organization_id)
    )
    collect(by_organization, True, False)

    by_topic = (
        _base(session, start, end)
        .add_columns(
            RecordTopic.topic_id.label("topic_id"),
            func.count(func.distinct(SourceRecord.id)).label("records"),
            func.sum(RecordTopic.confidence).label("weighted"),
        )
        .join(RecordTopic, RecordTopic.source_record_id == SourceRecord.id)
        .group_by(month, SourceRecord.record_type, RecordTopic.topic_id)
    )
    collect(by_topic, False, True)

    by_both = (
        _base(session, start, end)
        .add_columns(
            RecordOrganization.organization_id.label("organization_id"),
            RecordTopic.topic_id.label("topic_id"),
            func.count(func.distinct(SourceRecord.id)).label("records"),
            func.sum(RecordOrganization.confidence * RecordTopic.confidence).label("weighted"),
        )
        .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
        .join(RecordTopic, RecordTopic.source_record_id == SourceRecord.id)
        .group_by(
            month,
            SourceRecord.record_type,
            RecordOrganization.organization_id,
            RecordTopic.topic_id,
        )
    )
    collect(by_both, True, True)

    counted = session.scalar(
        _base(session, start, end).with_only_columns(func.count(func.distinct(SourceRecord.id)))
    )
    return buckets, int(counted or 0)


def _roll_up(monthly: dict[Bucket, tuple[int, float, int]]) -> dict[Bucket, tuple[int, float, int]]:
    """Sum monthly buckets into quarters and years."""
    wider: dict[Bucket, tuple[int, float, int]] = {}
    for (period, _, organization_id, topic_id, source_type), values in monthly.items():
        for period_type, start in (
            (PeriodType.QUARTER, quarter_start(period)),
            (PeriodType.YEAR, year_start(period)),
        ):
            key: Bucket = (start, period_type.value, organization_id, topic_id, source_type)
            count, weighted, unique = wider.get(key, (0, 0.0, 0))
            wider[key] = (count + values[0], weighted + values[1], unique + values[2])
    return wider


def aggregate_monthly_activity(
    session: Session,
    *,
    as_of: datetime | date | None = None,
    since: date | None = None,
    rebuild: bool = False,
    is_synthetic: bool = False,
) -> AggregationSummary:
    """Recount activity into ``activity_aggregates`` and report what changed.

    Args:
        as_of: only records published before this point count (default: now). The month
            containing it is excluded, because a part-finished month looks like a decline.
        since: only recount from this month onward (default: everything).
        rebuild: delete stored rows in range that no longer have records, instead of leaving a
            stale count behind.
        is_synthetic: mark the rows as demo data.

    Raises:
        ValueError: if ``since`` is later than the window end.
    """
    end = month_start(as_of or datetime.now(UTC))  # the month in progress is excluded
    start = month_start(since) if since else None
    if start is not None and start > end:
        raise ValueError("since must not be later than as_of")

    summary = AggregationSummary()
    monthly, summary.records_counted = _counts(session, start, end)
    if not monthly:
        summary.warnings.append("no records with a publication date in this window")
        return summary

    wanted = {**monthly, **_roll_up(monthly)}
    months = sorted({period for period, kind, *_ in wanted if kind == PeriodType.MONTH.value})
    summary.months = len(months)
    summary.first_period, summary.last_period = months[0], months[-1]

    stored = select(ActivityAggregate)
    if start is not None:
        stored = stored.where(ActivityAggregate.period >= year_start(start))
    existing: dict[Bucket, ActivityAggregate] = {
        (row.period, row.period_type, row.organization_id, row.topic_id, row.source_type): row
        for row in session.scalars(stored)
    }

    for key, (count, raw_weighted, unique) in wanted.items():
        weighted = round(raw_weighted, 4)  # compare against what is actually stored
        row = existing.pop(key, None)
        if row is None:
            period, period_type, organization_id, topic_id, source_type = key
            session.add(
                ActivityAggregate(
                    period=period,
                    period_type=period_type,
                    organization_id=organization_id,
                    topic_id=topic_id,
                    source_type=source_type,
                    activity_count=count,
                    weighted_activity=weighted,
                    unique_record_count=unique,
                    is_synthetic=is_synthetic,
                )
            )
            summary.rows_written += 1
        elif (row.activity_count, row.unique_record_count) != (count, unique) or abs(
            row.weighted_activity - weighted
        ) > 1e-9:
            row.activity_count = count
            row.weighted_activity = weighted
            row.unique_record_count = unique
            summary.rows_updated += 1

    if rebuild:
        for row in existing.values():
            session.delete(row)
            summary.rows_removed += 1
    elif existing:
        summary.warnings.append(
            f"{len(existing)} stored period(s) no longer have records; use --rebuild to drop them"
        )
    session.flush()
    LOGGER.info("aggregated activity: %s", summary.as_dict())
    return summary


def monthly_series(
    session: Session,
    months: Sequence[date],
    *,
    organization_id: int | None = None,
    topic_id: int | None = None,
    source_types: Iterable[str] | None = None,
) -> dict[str, list[float]]:
    """Weighted monthly activity per source type for one entity, aligned to ``months``.

    A month with no records reads as zero: the sources were collected and nothing was published.
    """
    wanted = list(source_types) if source_types is not None else [t.value for t in SourceType]
    query = select(
        ActivityAggregate.period,
        ActivityAggregate.source_type,
        ActivityAggregate.weighted_activity,
    ).where(
        ActivityAggregate.period_type == PeriodType.MONTH.value,
        ActivityAggregate.period.in_(list(months)),
        ActivityAggregate.source_type.in_(wanted),
    )
    query = query.where(
        ActivityAggregate.organization_id.is_(None)
        if organization_id is None
        else ActivityAggregate.organization_id == organization_id
    )
    query = query.where(
        ActivityAggregate.topic_id.is_(None)
        if topic_id is None
        else ActivityAggregate.topic_id == topic_id
    )

    found: dict[str, dict[date, float]] = {source: {} for source in wanted}
    for period, source_type, weighted in session.execute(query).all():
        found.setdefault(str(source_type), {})[period] = float(weighted or 0.0)
    return {
        source: [found.get(source, {}).get(month, 0.0) for month in months] for source in wanted
    }
