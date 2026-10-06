"""Sweeping the stored data for problems the pipeline should never produce.

Every check here reads what is already in the database; none of them recompute a score or fix
anything. They exist to catch drift between what the schema promises (a foreign key, a 0-100
range, a required field) and what actually ended up on disk - the kind of problem that a
one-off script, a partial migration, or a bug elsewhere could introduce without any single
component noticing.

Each check returns a :class:`QualityCheckResult` with a pass/fail verdict, how many rows were
affected, and up to a handful of examples so a failure is immediately actionable rather than
just a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import PeriodType
from cews.database.models import (
    ActivityAggregate,
    Forecast,
    IngestionRun,
    Insight,
    InsightEvidence,
    Organization,
    RecordOrganization,
    RecordTopic,
    Score,
    SourceRecord,
    Topic,
)

MAX_EXAMPLES = 5
DEFAULT_FRESHNESS_DAYS = 7
DEFAULT_MIN_SUCCESS_RATE = 0.5


@dataclass(frozen=True)
class QualityCheckResult:
    """The outcome of one check."""

    name: str
    passed: bool
    affected_count: int
    total_count: int
    examples: tuple[str, ...] = ()
    detail: str = ""

    @property
    def affected_rate(self) -> float:
        """Share of rows affected, 0 when there was nothing to check."""
        return self.affected_count / self.total_count if self.total_count else 0.0

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "name": self.name,
            "passed": self.passed,
            "affected_count": self.affected_count,
            "total_count": self.total_count,
            "affected_rate": round(self.affected_rate, 4),
            "examples": list(self.examples),
            "detail": self.detail,
        }


@dataclass
class DataQualityReport:
    """Every check run together, in one pass."""

    checks: list[QualityCheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when every check passed."""
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> tuple[QualityCheckResult, ...]:
        """The checks that did not pass, for a short summary."""
        return tuple(check for check in self.checks if not check.passed)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "passed": self.passed,
            "checks": [check.as_dict() for check in self.checks],
            "failed": [check.name for check in self.failed_checks],
        }


def _result(
    name: str, affected: list[Any], total: int, *, describe: Any = str
) -> QualityCheckResult:
    return QualityCheckResult(
        name=name,
        passed=not affected,
        affected_count=len(affected),
        total_count=total,
        examples=tuple(describe(item) for item in affected[:MAX_EXAMPLES]),
    )


# --------------------------------------------------------------------------------------
# 1. Required-field completeness
# --------------------------------------------------------------------------------------
def check_required_fields(session: Session) -> QualityCheckResult:
    """Every source record should have a title and a publication date to be usable at all."""
    total = int(session.scalar(select(func.count()).select_from(SourceRecord)) or 0)
    missing = list(
        session.scalars(
            select(SourceRecord.source_record_id).where(
                (SourceRecord.title.is_(None))
                | (SourceRecord.title == "")
                | (SourceRecord.published_at.is_(None))
            )
        )
    )
    return _result("required_fields", missing, total)


# --------------------------------------------------------------------------------------
# 2. Duplicate rate
# --------------------------------------------------------------------------------------
def check_duplicate_rate(session: Session, *, max_rate: float = 0.5) -> QualityCheckResult:
    """How much of the collected data turned out to be a copy of something already stored.

    This is a rate to watch, not a hard defect: some duplication across sources is normal (the
    same trial reported by two registries). It fails only when the rate is implausibly high,
    which usually means an ingestion bug is re-collecting the same records under new ids.
    """
    total = int(session.scalar(select(func.count()).select_from(SourceRecord)) or 0)
    duplicates = int(
        session.scalar(
            select(func.count())
            .select_from(SourceRecord)
            .where(SourceRecord.duplicate_of_id.is_not(None))
        )
        or 0
    )
    rate = duplicates / total if total else 0.0
    return QualityCheckResult(
        name="duplicate_rate",
        passed=rate <= max_rate,
        affected_count=duplicates,
        total_count=total,
        detail=f"{rate:.1%} of records are marked as duplicates (threshold {max_rate:.0%})",
    )


# --------------------------------------------------------------------------------------
# 3. Date validity
# --------------------------------------------------------------------------------------
def check_date_validity(session: Session, *, as_of: datetime | None = None) -> QualityCheckResult:
    """Dates that could not possibly be right: published in the future, or a forecast that
    targets a period on or before the date it was made from."""
    now = as_of or datetime.now(UTC)
    total = int(session.scalar(select(func.count()).select_from(SourceRecord)) or 0)
    future_records = list(
        session.scalars(
            select(SourceRecord.source_record_id).where(SourceRecord.published_at > now)
        )
    )
    backwards_forecasts = list(
        session.scalars(select(Forecast.id).where(Forecast.target_period <= Forecast.forecast_date))
    )
    misaligned_periods = list(
        session.scalars(
            select(ActivityAggregate.id).where(
                ActivityAggregate.period_type == PeriodType.MONTH.value,
                func.strftime("%d", ActivityAggregate.period) != "01",
            )
        )
    )
    affected = [
        *(f"future publication: {value}" for value in future_records),
        *(f"forecast {value} targets on/before its own date" for value in backwards_forecasts),
        *(
            f"activity_aggregate {value} not aligned to a month boundary"
            for value in misaligned_periods
        ),
    ]
    return _result("date_validity", affected, total or 1)


# --------------------------------------------------------------------------------------
# 4. Referential integrity
# --------------------------------------------------------------------------------------
def check_referential_integrity(session: Session) -> QualityCheckResult:
    """Every link table row should point at something that still exists.

    Foreign keys with ``ondelete=CASCADE`` should make this impossible in normal operation; this
    check exists for the same reason a smoke detector exists even with a working stove - to
    catch the case where something upstream (a manual edit, a bulk load with constraints off,
    an older SQLite file) let a reference go stale.
    """
    orphans: list[str] = []
    orphans += [
        f"record_topics.{rt_id} -> missing topic"
        for rt_id in session.scalars(
            select(RecordTopic.id).where(
                ~select(Topic.id).where(Topic.id == RecordTopic.topic_id).exists()
            )
        )
    ]
    orphans += [
        f"record_organizations.{ro_id} -> missing organization"
        for ro_id in session.scalars(
            select(RecordOrganization.id).where(
                ~select(Organization.id)
                .where(Organization.id == RecordOrganization.organization_id)
                .exists()
            )
        )
    ]
    orphans += [
        f"insight_evidence.{ie_id} -> missing source record"
        for ie_id in session.scalars(
            select(InsightEvidence.id).where(
                ~select(SourceRecord.id)
                .where(SourceRecord.id == InsightEvidence.source_record_id)
                .exists()
            )
        )
    ]
    orphans += [
        f"insight_evidence.{ie_id} -> missing insight"
        for ie_id in session.scalars(
            select(InsightEvidence.id).where(
                ~select(Insight.id).where(Insight.id == InsightEvidence.insight_id).exists()
            )
        )
    ]
    total = int(
        (session.scalar(select(func.count()).select_from(RecordTopic)) or 0)
        + (session.scalar(select(func.count()).select_from(RecordOrganization)) or 0)
        + (session.scalar(select(func.count()).select_from(InsightEvidence)) or 0)
    )
    return _result("referential_integrity", orphans, total or 1)


# --------------------------------------------------------------------------------------
# 5. Score, confidence and link-confidence ranges
# --------------------------------------------------------------------------------------
def check_score_ranges(session: Session) -> QualityCheckResult:
    """Every stored score, confidence and link weight must be on the scale it promises."""
    problems: list[str] = []
    for row_id, value in session.execute(
        select(Score.id, Score.score_value).where(
            (Score.score_value < 0) | (Score.score_value > 100)
        )
    ).all():
        problems.append(f"score {row_id} has score_value={value}, outside 0-100")
    for row_id, value in session.execute(
        select(Score.id, Score.confidence_score).where(
            (Score.confidence_score < 0) | (Score.confidence_score > 100)
        )
    ).all():
        problems.append(f"score {row_id} has confidence_score={value}, outside 0-100")
    for row_id, confidence in session.execute(
        select(RecordTopic.id, RecordTopic.confidence).where(
            (RecordTopic.confidence < 0) | (RecordTopic.confidence > 1)
        )
    ).all():
        problems.append(f"record_topics {row_id} has confidence={confidence}, outside 0-1")
    for row_id, confidence in session.execute(
        select(RecordOrganization.id, RecordOrganization.confidence).where(
            (RecordOrganization.confidence < 0) | (RecordOrganization.confidence > 1)
        )
    ).all():
        problems.append(f"record_organizations {row_id} has confidence={confidence}, outside 0-1")
    total = int(
        (session.scalar(select(func.count()).select_from(Score)) or 0)
        + (session.scalar(select(func.count()).select_from(RecordTopic)) or 0)
        + (session.scalar(select(func.count()).select_from(RecordOrganization)) or 0)
    )
    return _result("score_ranges", problems, total or 1)


# --------------------------------------------------------------------------------------
# 6. Ingestion success rate
# --------------------------------------------------------------------------------------
def check_ingestion_success_rate(
    session: Session, *, min_rate: float = DEFAULT_MIN_SUCCESS_RATE
) -> QualityCheckResult:
    """How often collection runs actually succeeded, per source over all recorded runs."""
    total = int(session.scalar(select(func.count()).select_from(IngestionRun)) or 0)
    if total == 0:
        return QualityCheckResult(
            "ingestion_success_rate", True, 0, 0, detail="no runs recorded yet"
        )
    failed = int(
        session.scalar(
            select(func.count()).select_from(IngestionRun).where(IngestionRun.status == "failed")
        )
        or 0
    )
    rate = 1.0 - failed / total
    return QualityCheckResult(
        name="ingestion_success_rate",
        passed=rate >= min_rate,
        affected_count=failed,
        total_count=total,
        detail=f"{rate:.0%} of {total} run(s) succeeded (threshold {min_rate:.0%})",
    )


# --------------------------------------------------------------------------------------
# 7. Source freshness
# --------------------------------------------------------------------------------------
def check_source_freshness(
    session: Session, *, as_of: datetime | None = None, max_age_days: float = DEFAULT_FRESHNESS_DAYS
) -> QualityCheckResult:
    """How long it has been since each source last successfully collected anything."""
    now = as_of or datetime.now(UTC)
    rows = session.execute(
        select(IngestionRun.source, func.max(IngestionRun.end_time))
        .where(IngestionRun.status == "succeeded")
        .group_by(IngestionRun.source)
    ).all()
    if not rows:
        return QualityCheckResult(
            "source_freshness", True, 0, 0, detail="no successful runs recorded yet"
        )
    stale = [
        f"{source}: last succeeded {(now - last).days} day(s) ago"
        for source, last in rows
        if last is not None and (now - last).total_seconds() / 86400.0 > max_age_days
    ]
    return _result("source_freshness", stale, len(rows), describe=str)


# --------------------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------------------
def run_data_quality_checks(
    session: Session, *, as_of: datetime | None = None
) -> DataQualityReport:
    """Run every data-quality check and return the combined report."""
    return DataQualityReport(
        checks=[
            check_required_fields(session),
            check_duplicate_rate(session),
            check_date_validity(session, as_of=as_of),
            check_referential_integrity(session),
            check_score_ranges(session),
            check_ingestion_success_rate(session),
            check_source_freshness(session, as_of=as_of),
        ]
    )
