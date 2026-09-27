"""Tests for the read-only dashboard queries."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType, ScoreType
from cews.dashboard import queries
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.settings import Settings

pytestmark = pytest.mark.integration


def test_the_origin_of_the_numbers_is_always_stated(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    """A dashboard that cannot say where its figures came from cannot be trusted."""
    with session_scope(dashboard_session_factory) as session:
        origin = queries.data_origin(session)
    assert origin.is_demo is True and origin.is_mixed is False
    assert "SYNTHETIC" in origin.label


def test_an_empty_database_says_so() -> None:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        origin = queries.data_origin(session)
        summary = queries.overview(session)
    assert origin.is_demo is False and origin.live_records == 0
    assert "No records" in origin.label
    assert summary.records == 0 and summary.score_date is None
    engine.dispose()


def test_the_overview_counts_what_matters(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        summary = queries.overview(session)
    assert summary.records > 0 and summary.competitors > 0 and summary.topics > 0
    assert summary.records_by_type and sum(summary.records_by_type.values()) == summary.records
    assert summary.score_date is not None
    assert summary.emerging_trends <= summary.topics
    assert summary.last_collected is not None


def test_scores_arrive_with_their_reasoning(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
    assert rows
    assert rows == sorted(rows, key=lambda row: row["score"], reverse=True)
    first = rows[0]
    assert first["entity"] and first["explanation"]
    assert set(first["components"])
    assert isinstance(first["qualified"], bool)
    assert all(isinstance(rule, str) for rule in first["failed_rules"])


def test_a_high_score_without_the_evidence_is_visible_as_such(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    """The demo's six-record topic must reach the dashboard with its failed rules attached."""
    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
    thin = next(row for row in rows if row["entity"].startswith("siRNA"))
    assert thin["score"] > 60 and thin["qualified"] is False
    assert thin["failed_rules"]


def test_overall_and_area_threat_scores_can_be_told_apart(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        overall = queries.scores_for(session, ScoreType.THREAT.value, with_context=False)
        by_area = queries.scores_for(session, ScoreType.THREAT.value, with_context=True)
    assert overall and all(row["context"] == "" for row in overall)
    assert all(row["context"] for row in by_area)


def test_activity_reads_back_month_by_month(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
        months, series = queries.monthly_activity(
            session, EntityType.TOPIC.value, rows[0]["entity_id"], months=12
        )
    assert len(months) == len(series) == 12
    assert months == sorted(months) and isinstance(months[0], date)
    assert sum(series) > 0


def test_a_forecast_comes_with_its_interval_and_model(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
        forecast = None
        for row in rows:
            forecast = queries.forecast_for(session, EntityType.TOPIC.value, row["entity_id"])
            if forecast:
                break
    assert forecast is not None
    assert forecast["model"] and forecast["months"]
    assert len(forecast["predicted"]) == len(forecast["lower"]) == len(forecast["months"])
    for value, low, high in zip(
        forecast["predicted"], forecast["lower"], forecast["upper"], strict=True
    ):
        assert low <= value <= high


def test_a_missing_forecast_is_none_not_an_invented_one(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        assert queries.forecast_for(session, EntityType.TOPIC.value, 999999) is None


def test_anomalies_carry_what_kind_they_are(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        rows = queries.recent_anomalies(session, limit=10)
    assert rows
    assert all(row["kind"] and row["entity"] for row in rows)
    assert all(row["expected_lower"] <= row["expected_upper"] for row in rows)
    assert rows == sorted(rows, key=lambda row: row["date"], reverse=True)


def test_evidence_links_back_to_real_records(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
        records = queries.evidence_records(
            session, EntityType.TOPIC.value, rows[0]["entity_id"], limit=5
        )
    assert records and len(records) <= 5
    assert all(record["identifier"] and record["type"] for record in records)


def test_source_health_and_review_are_readable(
    dashboard_session_factory: sessionmaker[Session],
) -> None:
    with session_scope(dashboard_session_factory) as session:
        health = queries.source_health(session)
        review = queries.review_items(session)
        counts = queries.record_counts_by_source(session)
    assert isinstance(health, list)  # empty until a fetch has run
    assert isinstance(review, list)
    assert counts and sum(counts.values()) > 0


def test_nothing_is_computed_by_the_dashboard_layer(
    dashboard_session_factory: sessionmaker[Session], dashboard_database: tuple[object, Settings]
) -> None:
    """Scores shown must be exactly the stored ones, never recomputed on the way out."""
    from sqlalchemy import select

    from cews.database.models import Score

    with session_scope(dashboard_session_factory) as session:
        rows = queries.scores_for(session, ScoreType.TREND.value)
        stored = {
            (row.entity_id, row.context_key): float(row.score_value)
            for row in session.scalars(
                select(Score).where(Score.score_type == ScoreType.TREND.value)
            )
        }
    for row in rows:
        assert row["score"] == stored[(row["entity_id"], row["context"])]
