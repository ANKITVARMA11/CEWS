"""SQLAlchemy 2.x ORM models for the CEWS analytical repository.

Design notes:

* Only metadata, abstracts, normalized entities, features, scores and links are stored. No
  PDFs or full patent documents.
* ``UTCDateTime`` guarantees timezone-aware UTC datetimes on every backend (SQLite drops the
  timezone; this type restores it) and rejects naive datetimes.
* JSON columns use ``JSONB`` on PostgreSQL and plain JSON elsewhere.
* Every table that can hold or derive from demo data carries ``is_synthetic`` so synthetic
  and live data are never mixed silently.
* Constraint and index names follow a naming convention so Alembic migrations are stable.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from cews.constants import (
    AlertRating,
    PeriodType,
    ProcessingStatus,
    RunStatus,
    SourceType,
    TopicStatus,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JSON_TYPE = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC ``DateTime`` that behaves identically on SQLite and PostgreSQL."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Any) -> datetime | None:
        """Convert to UTC before storing; naive datetimes are rejected."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime not allowed; pass a timezone-aware datetime")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Any) -> datetime | None:
        """Return timezone-aware UTC datetimes regardless of backend."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def _in(column: str, enum_cls: type[StrEnum]) -> str:
    values = ", ".join(f"'{member.value}'" for member in enum_cls)
    return f"{column} IN ({values})"


class Base(DeclarativeBase):
    """Declarative base with a stable constraint naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


# ======================================================================================
# Core records
# ======================================================================================
class SourceRecord(Base):
    """One record collected from a source (trial, publication, patent, award, announcement)."""

    __tablename__ = "source_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    source_record_id: Mapped[str] = mapped_column(String(256), nullable=False)
    record_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(1024))
    title: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    raw_payload_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime, index=True)
    updated_at_source: Mapped[datetime | None] = mapped_column(UTCDateTime)
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    processing_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=ProcessingStatus.NEW.value, index=True
    )
    # Set when the same work was collected from more than one source (same DOI, PMID or title).
    # The duplicate keeps its row for evidence; activity counts use the canonical record only.
    duplicate_of_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_records.id", ondelete="SET NULL"), index=True
    )
    duplicate_reason: Mapped[str | None] = mapped_column(String(32))
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    trial: Mapped[ClinicalTrial | None] = relationship(
        back_populates="record", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    publication: Mapped[Publication | None] = relationship(
        back_populates="record", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    patent: Mapped[Patent | None] = relationship(
        back_populates="record", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    funding: Mapped[FundingAward | None] = relationship(
        back_populates="record", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )
    announcement: Mapped[Announcement | None] = relationship(
        back_populates="record", uselist=False, cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        UniqueConstraint("source", "source_record_id", name="uq_source_records_source_record"),
        Index("ix_source_records_source_url", "source_url"),
        Index("ix_source_records_type_published", "record_type", "published_at"),
        CheckConstraint(_in("record_type", SourceType), name="record_type"),
        CheckConstraint(_in("processing_status", ProcessingStatus), name="processing_status"),
    )


class Organization(Base):
    """A normalized organization (company, university, agency, ...)."""

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    organization_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    country: Mapped[str | None] = mapped_column(String(64))
    website: Mapped[str | None] = mapped_column(String(512))
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )
    discovered_automatically: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    manually_included: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    manually_excluded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    aliases: Mapped[list[OrganizationAlias]] = relationship(
        back_populates="organization", cascade="all, delete-orphan", passive_deletes=True
    )


class OrganizationAlias(Base):
    """An alternative name of an organization, with how and how confidently it was matched."""

    __tablename__ = "organization_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_alias: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    alias_type: Mapped[str] = mapped_column(String(32), nullable=False, default="variant")
    match_method: Mapped[str] = mapped_column(String(32), nullable=False, default="deterministic")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    # How often this spelling has been seen. The most frequent one becomes the display name.
    mention_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    organization: Mapped[Organization] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("organization_id", "normalized_alias", name="uq_org_aliases_org_alias"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )


class TherapeuticArea(Base):
    """A therapeutic area from the taxonomy (may have a parent area)."""

    __tablename__ = "therapeutic_areas"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("therapeutic_areas.id", ondelete="SET NULL"), index=True
    )


class Topic(Base):
    """A research topic: taxonomy topics and AI-discovered candidates (pending review)."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    topic_type: Mapped[str] = mapped_column(String(32), nullable=False, default="topic")
    parent_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), index=True
    )
    therapeutic_area_id: Mapped[int | None] = mapped_column(
        ForeignKey("therapeutic_areas.id", ondelete="SET NULL"), index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TopicStatus.ACTIVE.value
    )
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    aliases: Mapped[list[TopicAlias]] = relationship(
        back_populates="topic", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (CheckConstraint(_in("status", TopicStatus), name="status"),)


class TopicAlias(Base):
    """A synonym or abbreviation that maps to a topic."""

    __tablename__ = "topic_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    alias: Mapped[str] = mapped_column(String(255), nullable=False)
    matching_method: Mapped[str] = mapped_column(String(32), nullable=False, default="keyword")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    topic: Mapped[Topic] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("topic_id", "alias", name="uq_topic_aliases_topic_alias"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )


class RecordTopic(Base):
    """Assignment of a source record to a topic, with method and confidence."""

    __tablename__ = "record_topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    topic_id: Mapped[int] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    matching_method: Mapped[str] = mapped_column(String(32), nullable=False, default="keyword")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    __table_args__ = (
        UniqueConstraint(
            "source_record_id", "topic_id", "matching_method", name="uq_record_topics_assignment"
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )


class RecordOrganization(Base):
    """Link between a source record and an organization, with relationship type and confidence."""

    __tablename__ = "record_organizations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    relationship_type: Mapped[str] = mapped_column(String(32), nullable=False)
    match_method: Mapped[str] = mapped_column(String(32), nullable=False, default="deterministic")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    __table_args__ = (
        UniqueConstraint(
            "source_record_id",
            "organization_id",
            "relationship_type",
            name="uq_record_orgs_link",
        ),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
    )


# ======================================================================================
# Record details (one row per source record, keyed by source_record_id)
# ======================================================================================
class ClinicalTrial(Base):
    """Clinical trial details. ``sponsor_name`` preserves the original source spelling."""

    __tablename__ = "clinical_trials"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    trial_identifier: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    sponsor_name: Mapped[str | None] = mapped_column(String(255))
    sponsor_organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )
    phase: Mapped[str | None] = mapped_column(String(32))
    status: Mapped[str | None] = mapped_column(String(64))
    enrollment: Mapped[int | None] = mapped_column(Integer)
    start_date: Mapped[date | None] = mapped_column(Date)
    completion_date: Mapped[date | None] = mapped_column(Date)
    intervention: Mapped[str | None] = mapped_column(Text)
    condition: Mapped[str | None] = mapped_column(Text)
    countries: Mapped[list[str] | None] = mapped_column(JSON_TYPE)

    record: Mapped[SourceRecord] = relationship(back_populates="trial")

    __table_args__ = (CheckConstraint("enrollment IS NULL OR enrollment >= 0", name="enrollment"),)


class Publication(Base):
    """Publication details."""

    __tablename__ = "publications"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    publication_identifier: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    journal: Mapped[str | None] = mapped_column(String(255))
    publication_date: Mapped[date | None] = mapped_column(Date)
    authors: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    affiliations: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    citation_count: Mapped[int | None] = mapped_column(Integer)
    publication_type: Mapped[str | None] = mapped_column(String(64))

    record: Mapped[SourceRecord] = relationship(back_populates="publication")


class Patent(Base):
    """Patent details. ``assignee_name`` preserves the original source spelling."""

    __tablename__ = "patents"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    patent_identifier: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    application_date: Mapped[date | None] = mapped_column(Date)
    publication_date: Mapped[date | None] = mapped_column(Date)
    grant_date: Mapped[date | None] = mapped_column(Date)
    assignee_name: Mapped[str | None] = mapped_column(String(255))
    assignee_organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )
    inventors: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    patent_classifications: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    patent_family_id: Mapped[str | None] = mapped_column(String(64), index=True)
    legal_status: Mapped[str | None] = mapped_column(String(64))

    record: Mapped[SourceRecord] = relationship(back_populates="patent")


class FundingAward(Base):
    """Research grant or funding award. ``recipient_name`` preserves the source spelling."""

    __tablename__ = "funding_awards"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    award_identifier: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    recipient_name: Mapped[str | None] = mapped_column(String(255))
    recipient_organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )
    agency: Mapped[str | None] = mapped_column(String(255))
    amount: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(8))
    start_date: Mapped[date | None] = mapped_column(Date)
    end_date: Mapped[date | None] = mapped_column(Date)

    record: Mapped[SourceRecord] = relationship(back_populates="funding")

    __table_args__ = (CheckConstraint("amount IS NULL OR amount >= 0", name="amount"),)


class Announcement(Base):
    """Company announcement. ``organization_name`` preserves the source spelling."""

    __tablename__ = "announcements"

    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), primary_key=True
    )
    organization_name: Mapped[str | None] = mapped_column(String(255))
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="SET NULL"), index=True
    )
    announcement_type: Mapped[str | None] = mapped_column(String(64))
    announcement_date: Mapped[date | None] = mapped_column(Date)
    partner_organizations: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    detected_topics: Mapped[list[str] | None] = mapped_column(JSON_TYPE)

    record: Mapped[SourceRecord] = relationship(back_populates="announcement")


# ======================================================================================
# Analytics
# ======================================================================================
class ActivityAggregate(Base):
    """Activity counts per period, source type and (optionally) organization and topic.

    A NULL ``organization_id`` or ``topic_id`` means "all". Uniqueness treats NULL as 0.
    """

    __tablename__ = "activity_aggregates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    period: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    period_type: Mapped[str] = mapped_column(String(16), nullable=False)
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id", ondelete="CASCADE"), index=True
    )
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="CASCADE"), index=True
    )
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    activity_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    weighted_activity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    unique_record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        CheckConstraint(_in("period_type", PeriodType), name="period_type"),
        CheckConstraint(_in("source_type", SourceType), name="source_type"),
        CheckConstraint("activity_count >= 0", name="activity_count"),
    )


Index(
    "uq_activity_aggregates_key",
    ActivityAggregate.period,
    ActivityAggregate.period_type,
    ActivityAggregate.source_type,
    func.coalesce(ActivityAggregate.organization_id, 0),
    func.coalesce(ActivityAggregate.topic_id, 0),
    unique=True,
)


class FeatureValue(Base):
    """A raw and normalized feature value for an entity, with its cohort and sample size."""

    __tablename__ = "feature_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feature_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    context_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    feature_name: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_value: Mapped[float | None] = mapped_column(Float)
    normalized_value: Mapped[float | None] = mapped_column(Float)
    normalization_method: Mapped[str | None] = mapped_column(String(32))
    comparison_cohort: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    cohort_size: Mapped[int | None] = mapped_column(Integer)
    sample_size: Mapped[int | None] = mapped_column(Integer)
    computed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "feature_date",
            "entity_type",
            "entity_id",
            "context_key",
            "feature_name",
            "comparison_cohort",
            name="uq_feature_values_key",
        ),
        Index("ix_feature_values_entity", "entity_type", "entity_id"),
        CheckConstraint(
            "normalized_value IS NULL OR (normalized_value >= 0 AND normalized_value <= 100)",
            name="normalized_range",
        ),
    )


class Score(Base):
    """A 0-100 score with its confidence and full component breakdown."""

    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    score_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    context_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    score_type: Mapped[str] = mapped_column(String(32), nullable=False)
    score_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    component_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    scoring_version: Mapped[str] = mapped_column(String(32), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "score_date",
            "entity_type",
            "entity_id",
            "context_key",
            "score_type",
            "scoring_version",
            name="uq_scores_key",
        ),
        Index("ix_scores_entity", "entity_type", "entity_id"),
        CheckConstraint("score_value >= 0 AND score_value <= 100", name="score_range"),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 100", name="confidence_range"
        ),
    )


class Forecast(Base):
    """A monthly activity forecast with bounds, model name and backtest error."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="all")
    forecast_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    target_period: Mapped[date] = mapped_column(Date, nullable=False)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    lower_bound: Mapped[float] = mapped_column(Float, nullable=False)
    upper_bound: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    backtest_metric_name: Mapped[str | None] = mapped_column(String(32))
    backtest_metric: Mapped[float | None] = mapped_column(Float)
    training_months: Mapped[int | None] = mapped_column(Integer)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint(
            "entity_type",
            "entity_id",
            "source_type",
            "forecast_date",
            "target_period",
            name="uq_forecasts_key",
        ),
        CheckConstraint("lower_bound <= upper_bound", name="bounds_ordered"),
    )


class Anomaly(Base):
    """A detected anomaly in an activity time series, with its expected range and evidence."""

    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    anomaly_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    metric: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_value: Mapped[float] = mapped_column(Float, nullable=False)
    expected_lower: Mapped[float] = mapped_column(Float, nullable=False)
    expected_upper: Mapped[float] = mapped_column(Float, nullable=False)
    deviation: Mapped[float] = mapped_column(Float, nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    anomaly_class: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        Index("ix_anomalies_entity", "entity_type", "entity_id"),
        CheckConstraint("confidence >= 0 AND confidence <= 100", name="confidence_range"),
    )


class Insight(Base):
    """A rule-generated insight, with fact and interpretation kept in separate columns."""

    __tablename__ = "insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    insight_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    insight_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    observed_fact: Mapped[str] = mapped_column(Text, nullable=False)
    interpretation: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_review: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    dedupe_key: Mapped[str] = mapped_column(String(128), nullable=False, default="", index=True)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    evidence: Mapped[list[InsightEvidence]] = relationship(
        back_populates="insight", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        Index("ix_insights_entity", "entity_type", "entity_id"),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 100", name="confidence_range"
        ),
    )


class InsightEvidence(Base):
    """Link from an insight to a supporting source record."""

    __tablename__ = "insight_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    insight_id: Mapped[int] = mapped_column(
        ForeignKey("insights.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    evidence_role: Mapped[str] = mapped_column(String(32), nullable=False, default="supporting")
    evidence_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    insight: Mapped[Insight] = relationship(back_populates="evidence")

    __table_args__ = (
        UniqueConstraint(
            "insight_id", "source_record_id", "evidence_role", name="uq_insight_evidence_link"
        ),
    )


class AlertReview(Base):
    """Expert review of an insight/alert, used to compute alert precision and lead time."""

    __tablename__ = "alert_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    insight_id: Mapped[int] = mapped_column(
        ForeignKey("insights.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[str] = mapped_column(String(32), nullable=False)
    reviewer: Mapped[str | None] = mapped_column(String(128))
    comment: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (CheckConstraint(_in("rating", AlertRating), name="rating"),)


# ======================================================================================
# Operations: ingestion history, checkpoints, jobs, evaluation, AI outputs
# ======================================================================================
class IngestionRun(Base):
    """Audit row for one source collection run."""

    __tablename__ = "ingestion_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    start_time: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    end_time: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    records_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_received: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_inserted: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_updated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    records_skipped: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    warnings_json: Mapped[list[str] | None] = mapped_column(JSON_TYPE)
    error_summary: Mapped[str | None] = mapped_column(Text)
    collection_mode: Mapped[str | None] = mapped_column(String(16))
    rate_limit_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (CheckConstraint(_in("status", RunStatus), name="status"),)


class SourceCheckpoint(Base):
    """Per-source checkpoint and circuit-breaker state.

    ``checkpoint_json`` is the last *successful* checkpoint; ``attempted_checkpoint_json`` is
    the checkpoint the most recent run was trying to reach (it differs after a failure).
    """

    __tablename__ = "source_checkpoints"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    attempted_checkpoint_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    last_full_refresh_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_attempted_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    last_status: Mapped[str | None] = mapped_column(String(16))
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    circuit_open_until: Mapped[datetime | None] = mapped_column(UTCDateTime)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)


class JobRun(Base):
    """Audit row for a scheduler job (a full cycle across sources and analysis)."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    job_name: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False, default="schedule")
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    dry_run: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    summary_json: Mapped[dict[str, Any] | None] = mapped_column(JSON_TYPE)
    error_summary: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (CheckConstraint(_in("status", RunStatus), name="status"),)


class JobLock(Base):
    """Named lock that prevents overlapping job executions."""

    __tablename__ = "job_locks"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    owner: Mapped[str] = mapped_column(String(64), nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False)


class EvaluationRun(Base):
    """One evaluation (backtest, data-quality run, ablation) with its configuration snapshot."""

    __tablename__ = "evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    evaluation_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    evaluation_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    configuration_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class AIExtraction(Base):
    """Cached output of a local AI layer for one record, with its grounding result."""

    __tablename__ = "ai_extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_record_id: Mapped[int] = mapped_column(
        ForeignKey("source_records.id", ondelete="CASCADE"), nullable=False, index=True
    )
    extraction_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1")
    output_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "source_record_id",
            "extraction_type",
            "model",
            "prompt_version",
            name="uq_ai_extractions_key",
        ),
    )


class ReviewQueueItem(Base):
    """Item awaiting human review (uncertain org match, AI-discovered topic, extraction)."""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    queue_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    subject_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, default=utcnow)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime)

    __table_args__ = (
        CheckConstraint("status IN ('pending', 'accepted', 'rejected')", name="status"),
        UniqueConstraint("queue_type", "subject_ref", name="uq_review_queue_subject"),
    )


DETAIL_MODELS: dict[str, type[Base]] = {
    SourceType.CLINICAL_TRIAL.value: ClinicalTrial,
    SourceType.PUBLICATION.value: Publication,
    SourceType.PATENT.value: Patent,
    SourceType.FUNDING.value: FundingAward,
    SourceType.ANNOUNCEMENT.value: Announcement,
}

# Tables whose rows are removed by ``reset_synthetic_data`` when ``is_synthetic`` is true.
SYNTHETIC_TABLES: tuple[str, ...] = (
    "source_records",
    "organizations",
    "activity_aggregates",
    "feature_values",
    "scores",
    "forecasts",
    "anomalies",
    "insights",
    "ingestion_runs",
    "evaluation_runs",
)
