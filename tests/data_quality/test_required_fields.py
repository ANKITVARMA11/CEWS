"""Every source record needs a title and a publication date to be usable at all."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tests.data_quality.conftest import make_record

from cews.database.connection import session_scope
from cews.validation.data_quality import check_required_fields

pytestmark = pytest.mark.unit


def test_a_complete_record_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1")
        result = check_required_fields(session)
    assert result.passed and result.affected_count == 0 and result.total_count == 1


def test_a_missing_title_is_caught(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", title=None)
        result = check_required_fields(session)
    assert not result.passed
    assert result.affected_count == 1
    assert "r1" in result.examples[0]


def test_an_empty_string_title_is_also_caught(factory: sessionmaker[Session]) -> None:
    """A blank title is exactly as useless as a missing one."""
    with session_scope(factory) as session:
        make_record(session, "r1", title="")
        result = check_required_fields(session)
    assert not result.passed and result.affected_count == 1


def test_a_missing_publication_date_is_caught(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", published=None)
        result = check_required_fields(session)
    assert not result.passed and result.affected_count == 1


def test_multiple_problems_are_all_counted(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", title=None)
        make_record(session, "r2", published=None)
        make_record(session, "r3")  # fine
        result = check_required_fields(session)
    assert result.affected_count == 2 and result.total_count == 3


def test_no_records_at_all_is_not_a_failure(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        result = check_required_fields(session)
    assert result.passed and result.total_count == 0
