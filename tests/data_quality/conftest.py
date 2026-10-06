"""Shared fixtures for the data-quality checks."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.database.models import SourceRecord


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def make_record(
    session: Session,
    identifier: str,
    *,
    title: str | None = "A title",
    published: datetime | None = datetime(2026, 1, 1, tzinfo=UTC),
    record_type: str = "publication",
) -> SourceRecord:
    """A minimal, otherwise-valid source record for one test to perturb."""
    record = SourceRecord(
        source="test",
        source_record_id=identifier,
        record_type=record_type,
        title=title,
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        published_at=published,
        content_hash=f"{identifier:0>64}"[:64],
    )
    session.add(record)
    session.flush()
    return record
