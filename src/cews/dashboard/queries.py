"""Fetching what the dashboard shows.

Plain functions over a session, returning plain data. They are deliberately free of any
presentation library so the same queries can serve the Streamlit dashboard, the API and the
exports, and so they can be tested without starting a web server.

Two rules run through all of them:

* **Nothing is computed here.** Scores, forecasts and anomalies are read as stored, with the
  components and reasons the pipeline recorded.
* **Where the data came from is never hidden.** Every view can say whether it is looking at
  synthetic demo data or live records, because a dashboard that cannot tell you that is a
  dashboard you cannot trust in front of leadership.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import EntityType, PeriodType, ScoreType, SourceType
from cews.database.models import (
    ActivityAggregate,
    Anomaly,
    Forecast,
    IngestionRun,
    Insight,
    Organization,
    RecordOrganization,
    RecordTopic,
    ReviewQueueItem,
    Score,
    SourceCheckpoint,
    SourceRecord,
    TherapeuticArea,
    Topic,
)
from cews.features.time_windows import month_range

LOGGER = logging.getLogger(__name__)

RECENT_MONTHS = 24


@dataclass(frozen=True)
class DataOrigin:
    """Whether the numbers on screen come from demo data or real collection."""

    synthetic_records: int
    live_records: int

    @property
    def is_demo(self) -> bool:
        """True when any synthetic record is present."""
        return self.synthetic_records > 0

    @property
    def is_mixed(self) -> bool:
        """True when both kinds are present, which should not normally happen."""
        return self.synthetic_records > 0 and self.live_records > 0

    @property
    def label(self) -> str:
        """A short description for the banner."""
        if self.is_mixed:
            return "MIXED DATA: both synthetic demo records and live records are present"
        if self.is_demo:
            return "SYNTHETIC DEMO DATA: invented organizations and artificial activity"
        if self.live_records:
            return "Live data collected from public sources"
        return "No records collected yet"


@dataclass
class Overview:
    """The handful of numbers that belong at the top of the first screen."""

    origin: DataOrigin
    competitors: int = 0
    topics: int = 0
    records: int = 0
    records_by_type: dict[str, int] = field(default_factory=dict)
    last_collected: datetime | None = None
    score_date: date | None = None
    emerging_trends: int = 0
    high_priority_threats: int = 0
    opportunities: int = 0
    low_confidence: int = 0
    anomalies_recent: int = 0
    needs_review: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "origin": self.origin.label,
            "competitors": self.competitors,
            "topics": self.topics,
            "records": self.records,
            "records_by_type": dict(self.records_by_type),
            "last_collected": self.last_collected.isoformat() if self.last_collected else None,
            "score_date": self.score_date.isoformat() if self.score_date else None,
            "emerging_trends": self.emerging_trends,
            "high_priority_threats": self.high_priority_threats,
            "opportunities": self.opportunities,
            "low_confidence": self.low_confidence,
            "anomalies_recent": self.anomalies_recent,
            "needs_review": self.needs_review,
        }


def data_origin(session: Session) -> DataOrigin:
    """Whether the stored records are synthetic, live, or an unintended mixture."""
    rows = session.execute(
        select(SourceRecord.is_synthetic, func.count()).group_by(SourceRecord.is_synthetic)
    ).all()
    counts = {bool(flag): int(total) for flag, total in rows}
    return DataOrigin(synthetic_records=counts.get(True, 0), live_records=counts.get(False, 0))


def latest_score_date(session: Session) -> date | None:
    """The most recent date scores were stored for."""
    return session.scalar(select(func.max(Score.score_date)))


def scores_for(
    session: Session,
    score_type: str,
    *,
    score_date: date | None = None,
    entity_type: str | None = None,
    with_context: bool | None = None,
) -> list[dict[str, Any]]:
    """Stored scores of one type, highest first, with their breakdown.

    Args:
        score_type: which score to read.
        score_date: the date to read (default: the most recent).
        entity_type: limit to topics or competitors.
        with_context: True for scores limited to a therapeutic area, False for overall ones.
    """
    when = score_date or latest_score_date(session)
    if when is None:
        return []
    query = select(Score).where(Score.score_date == when, Score.score_type == score_type)
    if entity_type is not None:
        query = query.where(Score.entity_type == entity_type)
    if with_context is True:
        query = query.where(Score.context_key != "")
    elif with_context is False:
        query = query.where(Score.context_key == "")

    names = entity_names(session)
    rows: list[dict[str, Any]] = []
    for row in session.scalars(query):
        breakdown = row.component_json or {}
        rows.append(
            {
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "entity": names.get((row.entity_type, row.entity_id), f"#{row.entity_id}"),
                "context": row.context_key,
                "score": float(row.score_value),
                "confidence": float(row.confidence_score),
                "category": breakdown.get("category", ""),
                "qualified": bool(breakdown.get("qualified", False)),
                "failed_rules": [
                    gate["name"] for gate in breakdown.get("gates", []) if not gate.get("passed")
                ],
                "sample_size": breakdown.get("sample_size", 0),
                "unavailable": breakdown.get("unavailable", []),
                "explanation": breakdown.get("explanation", ""),
                "components": breakdown.get("components", {}),
                "notes": breakdown.get("notes", []),
                "inputs": breakdown.get("inputs", {}),
                "scoring_version": row.scoring_version,
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def entity_names(session: Session) -> dict[tuple[str, int], str]:
    """Display names for every topic and organization."""
    names: dict[tuple[str, int], str] = {}
    for topic_id, name in session.execute(select(Topic.id, Topic.canonical_name)).all():
        names[(EntityType.TOPIC.value, int(topic_id))] = str(name)
    for organization_id, name in session.execute(
        select(Organization.id, Organization.canonical_name)
    ).all():
        names[(EntityType.COMPETITOR.value, int(organization_id))] = str(name)
    return names


def area_names(session: Session) -> dict[str, str]:
    """Therapeutic area keys mapped to their display names."""
    return {
        str(key): str(name)
        for key, name in session.execute(
            select(TherapeuticArea.key, TherapeuticArea.canonical_name)
        ).all()
    }


def overview(session: Session, *, score_date: date | None = None) -> Overview:
    """The summary numbers for the first screen."""
    origin = data_origin(session)
    when = score_date or latest_score_date(session)
    result = Overview(origin=origin, score_date=when)
    result.records = origin.synthetic_records + origin.live_records
    result.records_by_type = {
        str(record_type): int(total)
        for record_type, total in session.execute(
            select(SourceRecord.record_type, func.count()).group_by(SourceRecord.record_type)
        ).all()
    }
    result.competitors = int(
        session.scalar(
            select(func.count())
            .select_from(Organization)
            .where(
                (Organization.manually_included.is_(True))
                | (Organization.discovered_automatically.is_(True))
            )
        )
        or 0
    )
    result.topics = int(
        session.scalar(select(func.count()).select_from(Topic).where(Topic.active.is_(True))) or 0
    )
    result.last_collected = session.scalar(select(func.max(SourceRecord.fetched_at)))
    result.needs_review = int(
        session.scalar(
            select(func.count())
            .select_from(ReviewQueueItem)
            .where(ReviewQueueItem.status == "pending")
        )
        or 0
    )
    if when is not None:
        trends = scores_for(session, ScoreType.TREND.value, score_date=when)
        result.emerging_trends = sum(1 for row in trends if row["qualified"])
        opportunities = scores_for(session, ScoreType.OPPORTUNITY.value, score_date=when)
        result.opportunities = sum(1 for row in opportunities if row["qualified"])
        threats = scores_for(session, ScoreType.THREAT.value, score_date=when, with_context=False)
        result.high_priority_threats = sum(1 for row in threats if row["score"] >= 80)
        result.low_confidence = sum(
            1 for row in trends + opportunities + threats if row["confidence"] < 60
        )
    result.anomalies_recent = int(session.scalar(select(func.count()).select_from(Anomaly)) or 0)
    return result


def monthly_activity(
    session: Session,
    entity_type: str,
    entity_id: int,
    *,
    months: int = RECENT_MONTHS,
    as_of: date | None = None,
) -> tuple[list[date], list[float]]:
    """Recorded activity per month for one entity, oldest first."""
    end = as_of or session.scalar(select(func.max(ActivityAggregate.period))) or date.today()
    wanted = month_range(end, months)
    column = (
        ActivityAggregate.topic_id
        if entity_type == EntityType.TOPIC.value
        else ActivityAggregate.organization_id
    )
    other = (
        ActivityAggregate.organization_id
        if entity_type == EntityType.TOPIC.value
        else ActivityAggregate.topic_id
    )
    rows = session.execute(
        select(ActivityAggregate.period, func.sum(ActivityAggregate.activity_count))
        .where(
            ActivityAggregate.period_type == PeriodType.MONTH.value,
            column == entity_id,
            other.is_(None),
            ActivityAggregate.period.in_(wanted),
        )
        .group_by(ActivityAggregate.period)
    ).all()
    found = {period: float(total or 0) for period, total in rows}
    return wanted, [found.get(month, 0.0) for month in wanted]


def forecast_for(session: Session, entity_type: str, entity_id: int) -> dict[str, Any] | None:
    """The most recent forecast for one entity, with its interval and model."""
    when = session.scalar(
        select(func.max(Forecast.forecast_date)).where(
            Forecast.entity_type == entity_type, Forecast.entity_id == entity_id
        )
    )
    if when is None:
        return None
    rows = list(
        session.scalars(
            select(Forecast)
            .where(
                Forecast.entity_type == entity_type,
                Forecast.entity_id == entity_id,
                Forecast.forecast_date == when,
            )
            .order_by(Forecast.target_period)
        )
    )
    if not rows:
        return None
    return {
        "forecast_date": when,
        "model": rows[0].model_name,
        "metric_name": rows[0].backtest_metric_name,
        "metric": rows[0].backtest_metric,
        "training_months": rows[0].training_months,
        "months": [row.target_period for row in rows],
        "predicted": [float(row.predicted_value) for row in rows],
        "lower": [float(row.lower_bound) for row in rows],
        "upper": [float(row.upper_bound) for row in rows],
    }


def recent_anomalies(session: Session, *, limit: int = 20) -> list[dict[str, Any]]:
    """Unusual months, newest first, with what kind of unusual each one is."""
    names = entity_names(session)
    rows = session.scalars(
        select(Anomaly)
        .order_by(Anomaly.anomaly_date.desc(), Anomaly.confidence.desc())
        .limit(limit)
    )
    return [
        {
            "date": row.anomaly_date,
            "entity_type": row.entity_type,
            "entity_id": row.entity_id,
            "entity": names.get((row.entity_type, row.entity_id), f"#{row.entity_id}"),
            "observed": float(row.observed_value),
            "expected_lower": float(row.expected_lower),
            "expected_upper": float(row.expected_upper),
            "kind": row.anomaly_class,
            "confidence": float(row.confidence),
            "explanation": (row.evidence_json or {}).get("explanation", ""),
        }
        for row in rows
    ]


def evidence_records(
    session: Session, entity_type: str, entity_id: int, *, limit: int = 15
) -> list[dict[str, Any]]:
    """The source records behind an entity, newest first, with links back to the source."""
    if entity_type == EntityType.TOPIC.value:
        joined = (
            select(SourceRecord)
            .join(RecordTopic, RecordTopic.source_record_id == SourceRecord.id)
            .where(RecordTopic.topic_id == entity_id)
        )
    else:
        joined = (
            select(SourceRecord)
            .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
            .where(RecordOrganization.organization_id == entity_id)
        )
    rows = session.scalars(joined.order_by(SourceRecord.published_at.desc()).limit(limit))
    return [
        {
            "source": row.source,
            "type": row.record_type,
            "identifier": row.source_record_id,
            "title": row.title,
            "published": row.published_at.date() if row.published_at else None,
            "url": row.source_url,
        }
        for row in rows
    ]


def source_health(session: Session) -> list[dict[str, Any]]:
    """Each source's last run, with its checkpoint and circuit-breaker state."""
    checkpoints = {row.source: row for row in session.scalars(select(SourceCheckpoint))}
    latest: dict[str, IngestionRun] = {
        run.source: run
        for run in session.scalars(select(IngestionRun).order_by(IngestionRun.start_time))
    }
    now = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for source in sorted(set(checkpoints) | set(latest)):
        state = checkpoints.get(source)
        run = latest.get(source)
        paused = bool(state and state.circuit_open_until and state.circuit_open_until > now)
        rows.append(
            {
                "source": source,
                "last_status": (state.last_status if state else None)
                or (run.status if run else None),
                "last_success": state.last_success_at if state else None,
                "consecutive_failures": state.consecutive_failures if state else 0,
                "paused": paused,
                "records_last_run": run.records_inserted + run.records_updated if run else 0,
                "last_run_at": run.start_time if run else None,
            }
        )
    return rows


def review_items(session: Session, *, limit: int = 25) -> list[dict[str, Any]]:
    """Entity decisions waiting for a person to confirm."""
    rows = session.scalars(
        select(ReviewQueueItem)
        .where(ReviewQueueItem.status == "pending")
        .order_by(ReviewQueueItem.created_at.desc())
        .limit(limit)
    )
    return [
        {
            "kind": row.queue_type,
            "subject": row.subject_ref,
            "detail": row.payload_json,
            "raised": row.created_at,
        }
        for row in rows
    ]


def recent_insights(session: Session, *, limit: int = 20) -> list[dict[str, Any]]:
    """Generated insights, newest first (empty until the rule engine is built)."""
    rows = session.scalars(
        select(Insight)
        .order_by(Insight.insight_date.desc(), Insight.confidence_score.desc())
        .limit(limit)
    )
    return [
        {
            "date": row.insight_date,
            "title": row.title,
            "severity": row.severity,
            "observed_fact": row.observed_fact,
            "interpretation": row.interpretation,
            "recommended_review": row.recommended_review,
            "confidence": float(row.confidence_score),
        }
        for row in rows
    ]


def record_counts_by_source(session: Session) -> dict[str, int]:
    """How many records each source has contributed."""
    return {
        str(source): int(total)
        for source, total in session.execute(
            select(SourceRecord.source, func.count()).group_by(SourceRecord.source)
        ).all()
    }


def source_types_present(session: Session) -> Sequence[str]:
    """Which kinds of record exist at all, so a chart does not imply a missing source is zero."""
    return [
        str(record_type)
        for record_type, in session.execute(select(SourceRecord.record_type).distinct()).all()
        if record_type in {item.value for item in SourceType}
    ]
