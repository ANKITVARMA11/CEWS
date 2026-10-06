"""Unit tests for the five insight rules and their orchestration.

Each rule is tested against hand-built ``Score`` rows carrying exactly the ``component_json``
shape the real scoring pipeline stores, so these tests catch a rule reading the wrong key without
needing a full scoring run. ``find_new_market_entries`` is the one exception: it reads
``activity_aggregates`` directly (mirroring how the threat score itself gets this fact), so it is
tested against those instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType, InsightType, PeriodType, ScoreType, SourceType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ActivityAggregate, Insight, Organization, Score, Topic
from cews.insights.rule_engine import (
    find_competitor_movements,
    find_emerging_trends,
    find_new_market_entries,
    find_opportunities,
    find_patent_surges,
    generate_insights,
)
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit

SCORE_DATE = date(2026, 8, 1)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


def _topic(session: Session, name: str = "CRISPR gene editing") -> Topic:
    topic = Topic(key=name.lower().replace(" ", "_"), canonical_name=name, topic_type="technology")
    session.add(topic)
    session.flush()
    return topic


def _organization(session: Session, name: str = "Example Pharma") -> Organization:
    organization = Organization(canonical_name=name, normalized_name=name.casefold())
    session.add(organization)
    session.flush()
    return organization


def _score(
    session: Session,
    *,
    entity_type: str,
    entity_id: int,
    score_type: str,
    value: float,
    confidence: float,
    breakdown: dict[str, object],
    context_key: str = "",
) -> Score:
    row = Score(
        score_date=SCORE_DATE,
        entity_type=entity_type,
        entity_id=entity_id,
        context_key=context_key,
        score_type=score_type,
        score_value=value,
        confidence_score=confidence,
        component_json=breakdown,
        scoring_version="1.0.0",
    )
    session.add(row)
    session.flush()
    return row


def _link_evidence(
    session: Session, topic: Topic | None = None, organization: Organization | None = None
) -> None:
    from cews.database.models import RecordOrganization, RecordTopic, SourceRecord

    record = SourceRecord(
        source="test",
        source_record_id="ev1",
        record_type="publication",
        fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
        published_at=datetime(2026, 7, 1, tzinfo=UTC),
        content_hash="e" * 64,
    )
    session.add(record)
    session.flush()
    if topic is not None:
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
    if organization is not None:
        session.add(
            RecordOrganization(
                source_record_id=record.id,
                organization_id=organization.id,
                relationship_type="sponsor",
                confidence=1.0,
            )
        )
    session.flush()


# --------------------------------------------------------------------------------------
# Emerging trend
# --------------------------------------------------------------------------------------
def test_a_qualified_trend_produces_an_insight(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=78.0,
            confidence=90.0,
            breakdown={
                "qualified": True,
                "sample_size": 442,
                "components": {"velocity": {"available": True}, "momentum": {"available": True}},
            },
        )
        candidates = find_emerging_trends(
            session, SCORE_DATE, {(EntityType.TOPIC.value, topic.id): "CRISPR"}
        )
    assert len(candidates) == 1
    assert candidates[0].insight_type is InsightType.EMERGING_TREND
    assert candidates[0].evidence_record_ids


def test_an_unqualified_trend_produces_nothing(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=95.0,  # a high score alone is not enough
            confidence=30.0,
            breakdown={"qualified": False, "sample_size": 6, "components": {}},
        )
        candidates = find_emerging_trends(session, SCORE_DATE, {})
    assert candidates == []


def test_a_qualified_trend_with_no_evidence_is_still_returned_by_the_rule(
    factory: sessionmaker[Session],
) -> None:
    """Evidence rejection happens in the orchestrator, not inside each rule."""
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=78.0,
            confidence=90.0,
            breakdown={"qualified": True, "sample_size": 442, "components": {}},
        )
        candidates = find_emerging_trends(session, SCORE_DATE, {})
    assert len(candidates) == 1
    assert candidates[0].evidence_record_ids == []


# --------------------------------------------------------------------------------------
# Competitor movement
# --------------------------------------------------------------------------------------
def test_a_significant_rank_move_produces_an_insight(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization = _organization(session)
        _link_evidence(session, organization=organization)
        _score(
            session,
            entity_type=EntityType.COMPETITOR.value,
            entity_id=organization.id,
            score_type=ScoreType.INNOVATION.value,
            value=71.0,
            confidence=90.0,
            breakdown={"components": {"rank": {"detail": {"rank": 2, "of": 11, "rank_change": 4}}}},
        )
        candidates = find_competitor_movements(session, SCORE_DATE, {}, min_confidence=60.0)
    assert len(candidates) == 1
    assert "moved up" in candidates[0].text.title


def test_a_small_rank_move_is_not_worth_an_insight(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization = _organization(session)
        _score(
            session,
            entity_type=EntityType.COMPETITOR.value,
            entity_id=organization.id,
            score_type=ScoreType.INNOVATION.value,
            value=71.0,
            confidence=90.0,
            breakdown={"components": {"rank": {"detail": {"rank": 5, "of": 11, "rank_change": 1}}}},
        )
        candidates = find_competitor_movements(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


def test_a_first_ever_run_has_no_rank_history_to_move_from(factory: sessionmaker[Session]) -> None:
    """No history is a fact, not a zero: it must not be read as 'moved 0 places'."""
    with session_scope(factory) as session:
        organization = _organization(session)
        _score(
            session,
            entity_type=EntityType.COMPETITOR.value,
            entity_id=organization.id,
            score_type=ScoreType.INNOVATION.value,
            value=71.0,
            confidence=90.0,
            breakdown={
                "components": {"rank": {"detail": {"rank": 1, "of": 11}}}
            },  # no rank_change key
        )
        candidates = find_competitor_movements(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


def test_low_confidence_movement_is_not_surfaced(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization = _organization(session)
        _score(
            session,
            entity_type=EntityType.COMPETITOR.value,
            entity_id=organization.id,
            score_type=ScoreType.INNOVATION.value,
            value=71.0,
            confidence=20.0,
            breakdown={"components": {"rank": {"detail": {"rank": 1, "of": 11, "rank_change": 5}}}},
        )
        candidates = find_competitor_movements(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


# --------------------------------------------------------------------------------------
# New market entry
# --------------------------------------------------------------------------------------
def _add_activity(
    session: Session, organization_id: int, topic_id: int, period: date, count: int
) -> None:
    session.add(
        ActivityAggregate(
            period=period,
            period_type=PeriodType.MONTH.value,
            organization_id=organization_id,
            topic_id=topic_id,
            source_type=SourceType.CLINICAL_TRIAL.value,
            activity_count=count,
            weighted_activity=float(count),
            unique_record_count=count,
        )
    )
    session.flush()


def test_a_genuine_first_move_into_a_topic_is_found(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session, "Neurodegeneration")
        organization = _organization(session, "Zentavia Pharma")
        _link_evidence(session, organization=organization)
        _add_activity(session, organization.id, topic.id, date(2026, 3, 1), 13)
        candidates = find_new_market_entries(
            session, SCORE_DATE, {(EntityType.COMPETITOR.value, organization.id): "Zentavia Pharma"}
        )
    assert len(candidates) == 1
    assert "Neurodegeneration" in candidates[0].text.title
    assert candidates[0].confidence == 100.0


def test_ongoing_activity_is_not_a_new_entry(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        organization = _organization(session)
        _add_activity(session, organization.id, topic.id, date(2025, 1, 1), 5)  # long-standing
        _add_activity(session, organization.id, topic.id, date(2026, 3, 1), 13)
        candidates = find_new_market_entries(session, SCORE_DATE, {})
    assert candidates == []


# --------------------------------------------------------------------------------------
# Patent surge
# --------------------------------------------------------------------------------------
def test_high_patent_growth_produces_a_surge_insight(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=70.0,
            confidence=90.0,
            breakdown={
                "components": {
                    "patent_growth": {"available": True, "normalized": 90.0, "raw_value": 0.5}
                }
            },
        )
        candidates = find_patent_surges(session, SCORE_DATE, {}, min_confidence=60.0)
    assert len(candidates) == 1
    assert candidates[0].insight_type is InsightType.PATENT_SURGE


def test_patent_growth_below_the_percentile_bar_is_not_a_surge(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=70.0,
            confidence=90.0,
            breakdown={
                "components": {
                    "patent_growth": {"available": True, "normalized": 50.0, "raw_value": 0.5}
                }
            },
        )
        candidates = find_patent_surges(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


def test_no_patent_data_at_all_is_not_a_surge(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=70.0,
            confidence=90.0,
            breakdown={"components": {"patent_growth": {"available": False}}},
        )
        candidates = find_patent_surges(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


def test_negative_patent_growth_is_not_a_surge_even_at_a_high_percentile(
    factory: sessionmaker[Session],
) -> None:
    """A component can rank highly within its cohort while still declining in absolute terms."""
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=70.0,
            confidence=90.0,
            breakdown={
                "components": {
                    "patent_growth": {"available": True, "normalized": 95.0, "raw_value": -0.2}
                }
            },
        )
        candidates = find_patent_surges(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates == []


def test_patent_surge_evidence_is_patent_only(factory: sessionmaker[Session]) -> None:
    from cews.database.models import RecordTopic, SourceRecord

    with session_scope(factory) as session:
        topic = _topic(session)
        patent = SourceRecord(
            source="t",
            source_record_id="p1",
            record_type="patent",
            fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
            published_at=datetime(2026, 7, 1, tzinfo=UTC),
            content_hash="p" * 64,
        )
        paper = SourceRecord(
            source="t",
            source_record_id="pub1",
            record_type="publication",
            fetched_at=datetime(2026, 7, 1, tzinfo=UTC),
            published_at=datetime(2026, 7, 1, tzinfo=UTC),
            content_hash="q" * 64,
        )
        session.add_all([patent, paper])
        session.flush()
        session.add(RecordTopic(source_record_id=patent.id, topic_id=topic.id, confidence=1.0))
        session.add(RecordTopic(source_record_id=paper.id, topic_id=topic.id, confidence=1.0))
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=70.0,
            confidence=90.0,
            breakdown={
                "components": {
                    "patent_growth": {"available": True, "normalized": 90.0, "raw_value": 0.5}
                }
            },
        )
        candidates = find_patent_surges(session, SCORE_DATE, {}, min_confidence=60.0)
    assert candidates[0].evidence_record_ids == [patent.id]


# --------------------------------------------------------------------------------------
# Opportunity
# --------------------------------------------------------------------------------------
def test_a_qualified_opportunity_produces_an_insight(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.OPPORTUNITY.value,
            value=73.0,
            confidence=94.0,
            breakdown={
                "qualified": True,
                "components": {"low_competition": {"normalized": 85.0}},
            },
        )
        candidates = find_opportunities(session, SCORE_DATE, {})
    assert len(candidates) == 1
    assert "very few competitors" in candidates[0].text.observed_fact


def test_an_unqualified_opportunity_produces_nothing(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.OPPORTUNITY.value,
            value=90.0,
            confidence=30.0,
            breakdown={"qualified": False, "components": {}},
        )
        candidates = find_opportunities(session, SCORE_DATE, {})
    assert candidates == []


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def test_a_candidate_with_no_evidence_is_dropped_not_stored(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=78.0,
            confidence=90.0,
            breakdown={"qualified": True, "sample_size": 442, "components": {}},
        )
        run = generate_insights(session, settings, score_date=SCORE_DATE)
    assert run.candidates_found == 1
    assert run.rejected_no_evidence == 1
    assert run.stored == 0


def test_rerunning_suppresses_everything_as_a_duplicate(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=78.0,
            confidence=90.0,
            breakdown={"qualified": True, "sample_size": 442, "components": {}},
        )
        first = generate_insights(session, settings, score_date=SCORE_DATE)
    with session_scope(factory) as session:
        second = generate_insights(session, settings, score_date=SCORE_DATE)
    assert first.stored == 1
    assert second.stored == 0 and second.duplicates_suppressed == 1


def test_store_false_finds_without_writing(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=78.0,
            confidence=90.0,
            breakdown={"qualified": True, "sample_size": 442, "components": {}},
        )
        run = generate_insights(session, settings, score_date=SCORE_DATE, store=False)
    assert run.candidates_found == 1 and run.stored == 0  # found, but nothing was written
    with session_scope(factory) as session:
        assert session.query(Insight).count() == 0


def test_no_scores_at_all_is_reported_not_guessed(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        run = generate_insights(session, settings)
    assert run.candidates_found == 0
    assert any("cews score" in warning for warning in run.warnings)


def test_the_most_recent_score_date_is_used_by_default(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _link_evidence(session, topic=topic)
        older = Score(
            score_date=date(2026, 7, 1),
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            score_value=78.0,
            confidence_score=90.0,
            component_json={"qualified": True, "sample_size": 442, "components": {}},
            scoring_version="1.0.0",
        )
        session.add(older)
        _score(
            session,
            entity_type=EntityType.TOPIC.value,
            entity_id=topic.id,
            score_type=ScoreType.TREND.value,
            value=80.0,
            confidence=90.0,
            breakdown={"qualified": True, "sample_size": 500, "components": {}},
        )
        run = generate_insights(session, settings)
    assert run.insight_date == SCORE_DATE  # the later of the two, not the older one
