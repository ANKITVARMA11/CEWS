"""Threat modifiers: the specific moves that matter beyond raw growth.

Growth rates alone miss the things an analyst would actually flag: a competitor appearing in a
therapeutic area it has never worked in, trials moving into later phases, enrolment jumping,
patents landing in a topic we have already identified as trending.

Each modifier adds a small, capped number of points to a threat score, is computed from records
(never inferred), carries the evidence that produced it, and is displayed separately from the
base score so nobody has to take the adjustment on trust. The total is capped as well, so
modifiers can sharpen a ranking but never manufacture a threat on their own.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import PeriodType, SourceType
from cews.database.models import ActivityAggregate, ClinicalTrial, RecordOrganization, SourceRecord
from cews.discovery.new_market_entry import detect_new_therapeutic_area_entry
from cews.features.time_windows import add_months, month_range
from cews.scoring.config import ThreatModifiers

LOGGER = logging.getLogger(__name__)

RECENT_MONTHS = 3
HISTORY_MONTHS = 12
# An entry needs a longer look-back than a growth window: the point is a long silence.
ENTRY_RECENT_MONTHS = 6
ENTRY_HISTORY_MONTHS = 18
PHASE_ORDER: dict[str, int] = {
    "EARLY_PHASE1": 1,
    "PHASE1": 2,
    "PHASE1/PHASE2": 3,
    "PHASE2": 4,
    "PHASE2/PHASE3": 5,
    "PHASE3": 6,
    "PHASE4": 7,
}
ENROLMENT_JUMP = 1.5  # recent enrolment at least half as much again as the previous window


@dataclass(frozen=True)
class Modifier:
    """One capped adjustment, with the evidence behind it."""

    name: str
    points: float
    detail: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored beside the score."""
        return {
            "name": self.name,
            "points": round(self.points, 2),
            "detail": self.detail,
            "evidence": dict(self.evidence),
        }


@dataclass
class ModifierSet:
    """Every modifier that applied to one competitor, and the capped total."""

    applied: list[Modifier] = field(default_factory=list)
    total: float = 0.0
    capped_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "total_points": round(self.total, 2),
            "capped_at": self.capped_at,
            "applied": [modifier.as_dict() for modifier in self.applied],
        }

    def summary(self) -> str:
        """A readable list of what was added, for the explanation."""
        if not self.applied:
            return ""
        parts = "; ".join(f"{m.detail} (+{m.points:.0f})" for m in self.applied)
        capped = f", capped at {self.capped_at:.0f}" if self.capped_at is not None else ""
        return f"Adjusted by {self.total:.0f} point(s): {parts}{capped}."


def _phase_rank(phase: str | None) -> int:
    if not phase:
        return 0
    return PHASE_ORDER.get(phase.strip().upper(), 0)


def new_area_entries(
    session: Session,
    organization_id: int,
    *,
    as_of: date,
    recent_months: int = ENTRY_RECENT_MONTHS,
    history_months: int = ENTRY_HISTORY_MONTHS,
) -> list[int]:
    """Topic ids this competitor has just moved into.

    Delegates to :func:`cews.discovery.new_market_entry.detect_new_therapeutic_area_entry` so
    there is a single definition of what an entry is. That rule requires a few records inside the
    recent window and none at all over a long stretch before it, which keeps a one-record blip in
    a quiet topic from being reported as a company entering a new field.
    """
    entries = detect_new_therapeutic_area_entry(
        session,
        as_of=as_of,
        window_months=recent_months,
        quiet_months=history_months,
        organization_ids={organization_id},
    )
    return sorted(entry.topic_id for entry in entries)


def phase_progression(
    session: Session, organization_id: int, *, as_of: date, recent_months: int = RECENT_MONTHS
) -> tuple[str | None, str | None]:
    """The competitor's furthest trial phase recently, and before that.

    Returns ``(recent, earlier)`` phase labels; either may be None when there were no trials.
    """
    recent_start = month_range(as_of, recent_months)[0]

    def furthest(start: date | None, end: date | None) -> str | None:
        query = (
            select(ClinicalTrial.phase)
            .join(SourceRecord, SourceRecord.id == ClinicalTrial.source_record_id)
            .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
            .where(RecordOrganization.organization_id == organization_id)
        )
        if start is not None:
            query = query.where(SourceRecord.published_at >= _as_datetime(start))
        if end is not None:
            query = query.where(SourceRecord.published_at < _as_datetime(end))
        phases = [row.phase for row in session.execute(query).all() if row.phase]
        if not phases:
            return None
        return max(phases, key=_phase_rank)

    return furthest(recent_start, None), furthest(None, recent_start)


def enrolment_change(
    session: Session, organization_id: int, *, as_of: date, recent_months: int = RECENT_MONTHS
) -> tuple[float, float]:
    """Total planned enrolment in the recent window and the window before it."""
    recent = month_range(as_of, recent_months)
    previous_start = add_months(recent[0], -recent_months)

    def total(start: date, end: date) -> float:
        value = session.scalar(
            select(func.sum(ClinicalTrial.enrollment))
            .join(SourceRecord, SourceRecord.id == ClinicalTrial.source_record_id)
            .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
            .where(
                RecordOrganization.organization_id == organization_id,
                SourceRecord.published_at >= _as_datetime(start),
                SourceRecord.published_at < _as_datetime(end),
            )
        )
        return float(value or 0.0)

    return total(recent[0], add_months(recent[-1], 1)), total(previous_start, recent[0])


def patents_in_topics(
    session: Session,
    organization_id: int,
    topic_ids: Sequence[int],
    *,
    as_of: date,
    recent_months: int = RECENT_MONTHS,
) -> int:
    """How many patent records this competitor filed recently in the given topics."""
    if not topic_ids:
        return 0
    recent = month_range(as_of, recent_months)
    value = session.scalar(
        select(func.sum(ActivityAggregate.activity_count)).where(
            ActivityAggregate.period_type == PeriodType.MONTH.value,
            ActivityAggregate.organization_id == organization_id,
            ActivityAggregate.topic_id.in_(list(topic_ids)),
            ActivityAggregate.source_type == SourceType.PATENT.value,
            ActivityAggregate.period >= recent[0],
            ActivityAggregate.period < add_months(recent[-1], 1),
        )
    )
    return int(value or 0)


def _as_datetime(value: date) -> Any:
    from datetime import UTC, datetime

    return datetime(value.year, value.month, value.day, tzinfo=UTC)


def collect_modifiers(
    session: Session,
    organization_id: int,
    *,
    as_of: date,
    config: ThreatModifiers,
    supporting_sources: int,
    trending_topic_ids: Sequence[int] = (),
    topic_names: Mapping[int, str] | None = None,
) -> ModifierSet:
    """Work out which modifiers apply to one competitor, each capped by configuration."""
    result = ModifierSet()
    if not config.enabled:
        return result
    names = topic_names or {}

    entries = new_area_entries(session, organization_id, as_of=as_of)
    if entries:
        cap = config.cap_for("new_therapeutic_area_entry")
        if cap > 0:
            listed = ", ".join(names.get(topic_id, str(topic_id)) for topic_id in entries[:3])
            result.applied.append(
                Modifier(
                    "new_therapeutic_area_entry",
                    min(cap, cap * min(len(entries), 2) / 2),
                    f"first activity in {listed}",
                    {"topic_ids": entries[:10], "count": len(entries)},
                )
            )

    recent_phase, earlier_phase = phase_progression(session, organization_id, as_of=as_of)
    if _phase_rank(recent_phase) > _phase_rank(earlier_phase):
        cap = config.cap_for("phase_progression")
        if cap > 0:
            result.applied.append(
                Modifier(
                    "phase_progression",
                    cap,
                    f"trials reached {recent_phase} (previously {earlier_phase or 'none'})",
                    {"recent_phase": recent_phase, "previous_phase": earlier_phase},
                )
            )

    recent_enrolment, previous_enrolment = enrolment_change(session, organization_id, as_of=as_of)
    if previous_enrolment > 0 and recent_enrolment >= previous_enrolment * ENROLMENT_JUMP:
        cap = config.cap_for("large_enrollment_increase")
        if cap > 0:
            result.applied.append(
                Modifier(
                    "large_enrollment_increase",
                    cap,
                    (
                        f"planned enrolment rose from {previous_enrolment:.0f} to "
                        f"{recent_enrolment:.0f}"
                    ),
                    {"recent": recent_enrolment, "previous": previous_enrolment},
                )
            )

    patents = patents_in_topics(session, organization_id, trending_topic_ids, as_of=as_of)
    if patents > 0:
        cap = config.cap_for("monitored_topic_patent_activity")
        if cap > 0:
            result.applied.append(
                Modifier(
                    "monitored_topic_patent_activity",
                    cap,
                    f"{patents} patent record(s) in topics currently trending",
                    {"patent_records": patents, "topic_ids": list(trending_topic_ids)[:10]},
                )
            )

    if supporting_sources >= 3:
        cap = config.cap_for("multiple_supporting_sources")
        if cap > 0:
            result.applied.append(
                Modifier(
                    "multiple_supporting_sources",
                    cap,
                    f"{supporting_sources} independent sources show growth",
                    {"supporting_sources": supporting_sources},
                )
            )

    raw_total = sum(modifier.points for modifier in result.applied)
    result.total = min(raw_total, config.total_cap_points)
    if raw_total > config.total_cap_points:
        result.capped_at = config.total_cap_points
    return result
