"""Integration tests: schema, constraints, migrations, engines and sessions."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, inspect, select, text
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from cews.database.connection import (
    DatabaseError,
    build_database_url,
    create_db_engine,
    create_memory_engine,
    create_session_factory,
    database_is_initialized,
    describe_url,
    session_scope,
)
from cews.database.migrations import (
    current_revision,
    downgrade_database,
    head_revision,
    upgrade_database,
)
from cews.database.models import (
    ActivityAggregate,
    Base,
    ClinicalTrial,
    Forecast,
    Insight,
    InsightEvidence,
    Organization,
    OrganizationAlias,
    RecordTopic,
    Score,
    SourceRecord,
    Topic,
)
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
SPEC_TABLES = {
    "source_records",
    "organizations",
    "organization_aliases",
    "therapeutic_areas",
    "topics",
    "topic_aliases",
    "clinical_trials",
    "publications",
    "patents",
    "funding_awards",
    "announcements",
    "record_topics",
    "record_organizations",
    "activity_aggregates",
    "feature_values",
    "scores",
    "forecasts",
    "insights",
    "insight_evidence",
    "ingestion_runs",
    "evaluation_runs",
}


def _record(key: str = "1", **overrides: object) -> SourceRecord:
    values: dict[str, object] = {
        "source": "test_source",
        "source_record_id": key,
        "record_type": "publication",
        "fetched_at": NOW,
        "content_hash": "0" * 64,
    }
    values.update(overrides)
    return SourceRecord(**values)


# --------------------------------------------------------------------------------------
# Migrations
# --------------------------------------------------------------------------------------
def test_migration_creates_every_table(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names())
    assert tables >= SPEC_TABLES
    assert tables - {"alembic_version"} == set(Base.metadata.tables)
    assert head_revision() is not None
    assert current_revision(engine) == head_revision()
    assert database_is_initialized(engine)


def test_upgrade_is_idempotent(engine: Engine) -> None:
    head = head_revision()
    assert upgrade_database(engine) == head
    assert upgrade_database(engine) == head


@pytest.mark.filterwarnings("ignore:Skipped unsupported reflection")
@pytest.mark.filterwarnings("ignore:autogenerate skipping")
def test_migrations_match_the_models(engine: Engine) -> None:
    with engine.connect() as connection:
        context = MigrationContext.configure(connection, opts={"compare_type": True})
        assert compare_metadata(context, Base.metadata) == []


def test_downgrade_removes_everything(tmp_path: Path) -> None:
    file_engine = create_db_engine(f"sqlite:///{tmp_path / 'd.db'}")
    upgrade_database(file_engine)
    assert downgrade_database(file_engine) is None
    assert set(inspect(file_engine).get_table_names()) <= {"alembic_version"}
    file_engine.dispose()


def test_migration_failure_is_reported(tmp_path: Path) -> None:
    with pytest.raises(DatabaseError, match="Alembic environment not found"):
        upgrade_database(create_memory_engine(), migrations_dir=tmp_path)


def test_required_indexes_exist(engine: Engine) -> None:
    inspector = inspect(engine)

    def indexed(table: str) -> set[str]:
        return {
            column for index in inspector.get_indexes(table) for column in index["column_names"]
        }

    assert {"published_at", "content_hash", "source_url", "fetched_at"} <= indexed("source_records")
    assert "score_date" in indexed("scores")
    assert {"start_time", "source"} <= indexed("ingestion_runs")
    assert "organization_id" in indexed("record_organizations")
    assert "topic_id" in indexed("record_topics")
    assert "organization_id" in indexed("activity_aggregates")
    unique = {tuple(u["column_names"]) for u in inspector.get_unique_constraints("source_records")}
    assert ("source", "source_record_id") in unique


# --------------------------------------------------------------------------------------
# Constraints
# --------------------------------------------------------------------------------------
def test_source_and_record_id_are_unique(session: Session) -> None:
    session.add(_record("dup"))
    session.flush()
    session.add(_record("dup"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_same_record_id_in_different_sources_is_allowed(session: Session) -> None:
    session.add_all([_record("x", source="a"), _record("x", source="b")])
    session.flush()


def test_foreign_keys_are_enforced(session: Session) -> None:
    session.add(RecordTopic(source_record_id=999, topic_id=999))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize(
    "bad",
    [
        {"record_type": "podcast"},
        {"processing_status": "weird"},
    ],
)
def test_source_record_check_constraints(session: Session, bad: dict[str, object]) -> None:
    session.add(_record("chk", **bad))
    with pytest.raises(IntegrityError):
        session.flush()


@pytest.mark.parametrize("value", [-0.1, 100.5])
def test_score_range_is_enforced(session: Session, value: float) -> None:
    session.add(
        Score(
            score_date=date(2026, 9, 1),
            entity_type="topic",
            entity_id=1,
            score_type="trend",
            score_value=value,
            confidence_score=50,
            component_json={},
            scoring_version="1.0.0",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_valid_score_round_trips_json(session: Session) -> None:
    session.add(
        Score(
            score_date=date(2026, 9, 1),
            entity_type="topic",
            entity_id=1,
            score_type="trend",
            score_value=72.5,
            confidence_score=64,
            component_json={"velocity": {"raw": 0.4, "weight": 0.3}},
            scoring_version="1.0.0",
        )
    )
    session.flush()
    session.expire_all()
    stored = session.scalars(select(Score)).one()
    assert stored.component_json["velocity"]["weight"] == 0.3
    assert stored.context_key == ""


def test_score_key_is_unique(session: Session) -> None:
    def make() -> Score:
        return Score(
            score_date=date(2026, 9, 1),
            entity_type="topic",
            entity_id=1,
            score_type="trend",
            score_value=50,
            confidence_score=50,
            component_json={},
            scoring_version="1.0.0",
        )

    session.add(make())
    session.flush()
    session.add(make())
    with pytest.raises(IntegrityError):
        session.flush()


def test_forecast_bounds_must_be_ordered(session: Session) -> None:
    session.add(
        Forecast(
            entity_type="topic",
            entity_id=1,
            forecast_date=date(2026, 9, 1),
            target_period=date(2026, 10, 1),
            predicted_value=5,
            lower_bound=9,
            upper_bound=3,
            model_name="naive",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_negative_enrollment_is_rejected(session: Session) -> None:
    record = _record("trial", record_type="clinical_trial")
    session.add(record)
    session.flush()
    session.add(ClinicalTrial(source_record_id=record.id, trial_identifier="T1", enrollment=-5))
    with pytest.raises(IntegrityError):
        session.flush()


def test_alias_confidence_must_be_a_probability(session: Session) -> None:
    org = Organization(canonical_name="Acme", normalized_name="acme")
    session.add(org)
    session.flush()
    session.add(
        OrganizationAlias(
            organization_id=org.id, alias="ACME", normalized_alias="acme", confidence=1.5
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_activity_aggregate_uniqueness_treats_null_as_all(session: Session) -> None:
    def make(**extra: object) -> ActivityAggregate:
        return ActivityAggregate(
            period=date(2026, 8, 1),
            period_type="month",
            source_type="patent",
            activity_count=3,
            **extra,
        )

    session.add(make())
    session.flush()
    session.add(make())  # second "all organizations / all topics" row for the same key
    with pytest.raises(IntegrityError):
        session.flush()


def test_activity_aggregate_rejects_bad_period_type(session: Session) -> None:
    session.add(
        ActivityAggregate(period=date(2026, 8, 1), period_type="decade", source_type="patent")
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_record_cascades(session: Session) -> None:
    record = _record("cascade", record_type="clinical_trial")
    topic = Topic(key="t", canonical_name="T")
    session.add_all([record, topic])
    session.flush()
    session.add(ClinicalTrial(source_record_id=record.id, trial_identifier="T1"))
    session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id))
    insight = Insight(
        insight_date=date(2026, 9, 1),
        severity="info",
        insight_type="trend",
        entity_type="topic",
        entity_id=topic.id,
        title="t",
        observed_fact="f",
        interpretation="i",
        recommended_review="r",
        confidence_score=50,
    )
    session.add(insight)
    session.flush()
    session.add(InsightEvidence(insight_id=insight.id, source_record_id=record.id))
    session.flush()
    session.expire_all()

    session.delete(session.get(SourceRecord, record.id))
    session.flush()
    for table in ("clinical_trials", "record_topics", "insight_evidence"):
        assert session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one() == 0
    assert session.get(Topic, topic.id) is not None and session.get(Insight, insight.id) is not None


# --------------------------------------------------------------------------------------
# Time zones
# --------------------------------------------------------------------------------------
def test_datetimes_round_trip_as_utc(session: Session) -> None:
    ist = timezone(timedelta(hours=5, minutes=30))
    session.add(_record("tz", published_at=datetime(2026, 1, 1, 12, 0, tzinfo=ist)))
    session.flush()
    session.expire_all()
    stored = session.scalars(select(SourceRecord)).one()
    assert stored.published_at == datetime(2026, 1, 1, 6, 30, tzinfo=UTC)
    assert stored.published_at is not None and stored.published_at.tzinfo is not None


def test_naive_datetimes_are_rejected(session: Session) -> None:
    session.add(_record("naive", published_at=datetime(2026, 1, 1, 12, 0)))
    with pytest.raises((ValueError, StatementError)):
        session.flush()


# --------------------------------------------------------------------------------------
# Engines and sessions
# --------------------------------------------------------------------------------------
def test_sqlite_file_engine_creates_directory_and_enables_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "cews.db"
    file_engine = create_db_engine(f"sqlite:///{path}")
    with file_engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert str(connection.execute(text("PRAGMA journal_mode")).scalar_one()).lower() == "wal"
    assert path.parent.is_dir()
    file_engine.dispose()


def test_memory_engine_shares_one_database() -> None:
    memory = create_memory_engine()
    with memory.begin() as connection:
        connection.execute(text("CREATE TABLE t (x INTEGER)"))
    with memory.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM t")).scalar_one() == 0


def test_build_database_url_for_each_backend(settings: Settings, tmp_path: Path) -> None:
    sqlite_url = build_database_url(settings)
    assert sqlite_url.get_backend_name() == "sqlite" and sqlite_url.database == str(
        tmp_path / "cews.db"
    )
    postgres = load_settings(
        env_file=None,
        overrides={
            "database_backend": "postgresql",
            "database_url": "postgresql+psycopg://user:hunter2@db.example:5432/cews",
        },
    )
    url = build_database_url(postgres)
    assert url.get_backend_name() == "postgresql" and url.host == "db.example"
    assert "hunter2" not in describe_url(url)


def test_invalid_url_is_a_database_error() -> None:
    with pytest.raises(DatabaseError, match="invalid database URL"):
        create_db_engine("not a url")


def test_session_scope_commits_and_rolls_back(engine: Engine) -> None:
    factory = create_session_factory(engine)
    with session_scope(factory) as work:
        work.add(_record("kept"))
    with pytest.raises(RuntimeError), session_scope(factory) as work:
        work.add(_record("discarded"))
        work.flush()
        raise RuntimeError("boom")
    with session_scope(factory) as work:
        keys = {row.source_record_id for row in work.scalars(select(SourceRecord))}
    assert keys == {"kept"}


def test_database_is_initialized_is_false_for_an_empty_database() -> None:
    assert database_is_initialized(create_memory_engine()) is False
