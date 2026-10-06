"""How much collected data turned out to be a copy of something already stored."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tests.data_quality.conftest import make_record

from cews.database.connection import session_scope
from cews.validation.data_quality import check_duplicate_rate

pytestmark = pytest.mark.unit


def test_no_duplicates_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1")
        make_record(session, "r2")
        result = check_duplicate_rate(session)
    assert result.passed and result.affected_count == 0


def test_some_duplication_is_normal_and_still_passes(factory: sessionmaker[Session]) -> None:
    """The same trial reported by two registries is expected, not a defect."""
    with session_scope(factory) as session:
        original = make_record(session, "r1")
        duplicate = make_record(session, "r2")
        duplicate.duplicate_of_id = original.id
        result = check_duplicate_rate(session)
    assert result.passed
    assert result.affected_count == 1 and result.total_count == 2


def test_an_implausibly_high_rate_fails(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        original = make_record(session, "r0")
        for index in range(1, 10):
            duplicate = make_record(session, f"r{index}")
            duplicate.duplicate_of_id = original.id
        result = check_duplicate_rate(session, max_rate=0.5)
    assert not result.passed
    assert result.affected_count == 9 and result.total_count == 10


def test_the_threshold_is_configurable(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        original = make_record(session, "r1")
        duplicate = make_record(session, "r2")
        duplicate.duplicate_of_id = original.id
        strict = check_duplicate_rate(session, max_rate=0.1)
        lenient = check_duplicate_rate(session, max_rate=0.9)
    assert not strict.passed and lenient.passed


def test_no_records_at_all_is_not_a_failure(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        result = check_duplicate_rate(session)
    assert result.passed and result.total_count == 0
