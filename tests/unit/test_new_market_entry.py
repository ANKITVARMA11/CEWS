"""Unit tests for detecting a competitor's first move into a research area."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import PeriodType, SourceType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ActivityAggregate, Organization, Topic
from cews.discovery.new_market_entry import detect_new_therapeutic_area_entry
from cews.features.time_windows import add_months

pytestmark = pytest.mark.unit

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
WINDOW_START = date(2026, 3, 1)  # six months back


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def setup(
    session: Session, *, topic_type: str = "modality", name: str = "New topic"
) -> tuple[int, int]:
    organization = Organization(canonical_name="Example Pharma", normalized_name="example pharma")
    topic = Topic(key="t1", canonical_name=name, topic_type=topic_type)
    session.add_all([organization, topic])
    session.flush()
    return organization.id, topic.id


def add_activity(
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


def test_a_first_move_into_a_topic_is_detected(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, WINDOW_START, 8)
        entries = detect_new_therapeutic_area_entry(session, as_of=AS_OF)
    assert len(entries) == 1
    assert entries[0].organization_name == "Example Pharma"
    assert entries[0].records_in_window == 8
    assert "after 18 month(s) with none" in entries[0].description


def test_a_company_already_working_there_is_not_an_entry(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, date(2025, 6, 1), 4)  # before the window
        add_activity(session, organization_id, topic_id, WINDOW_START, 9)
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF) == []


def test_a_single_record_is_not_an_entry(factory: sessionmaker[Session]) -> None:
    """One passing mention in a quiet topic is noise, not a company entering a field."""
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, WINDOW_START, 1)
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF) == []


def test_the_threshold_is_configurable(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, WINDOW_START, 2)
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF, min_records=2)
        assert not detect_new_therapeutic_area_entry(session, as_of=AS_OF, min_records=5)


def test_activity_older_than_the_window_does_not_count_as_entry(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, add_months(WINDOW_START, -2), 9)
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF) == []


def test_areas_only_ignores_finer_topics(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session, topic_type="modality")
        add_activity(session, organization_id, topic_id, WINDOW_START, 9)
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF, areas_only=True) == []
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF, areas_only=False)


def test_it_can_be_limited_to_certain_competitors(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, WINDOW_START, 9)
        assert detect_new_therapeutic_area_entry(
            session, as_of=AS_OF, organization_ids={organization_id}
        )
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF, organization_ids={999}) == []


def test_an_empty_database_reports_nothing(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        assert detect_new_therapeutic_area_entry(session, as_of=AS_OF) == []


@pytest.mark.parametrize(
    ("window", "quiet", "minimum"), [(0, 18, 3), (6, 0, 3), (6, 18, 0), (-1, 18, 3)]
)
def test_invalid_windows_are_refused(
    factory: sessionmaker[Session], window: int, quiet: int, minimum: int
) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        detect_new_therapeutic_area_entry(
            session, as_of=AS_OF, window_months=window, quiet_months=quiet, min_records=minimum
        )


def test_the_result_is_json_friendly(factory: sessionmaker[Session]) -> None:
    import json

    with session_scope(factory) as session:
        organization_id, topic_id = setup(session)
        add_activity(session, organization_id, topic_id, WINDOW_START, 6)
        entry = detect_new_therapeutic_area_entry(session, as_of=AS_OF)[0]
    json.dumps(entry.as_dict())
    assert entry.as_dict()["records_in_window"] == 6
