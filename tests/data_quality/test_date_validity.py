"""Dates that could not possibly be right: future publications, backwards forecasts, and
activity periods that do not sit on a real calendar boundary."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy.orm import Session, sessionmaker
from tests.data_quality.conftest import make_record

from cews.database.connection import session_scope
from cews.database.models import ActivityAggregate, Forecast
from cews.validation.data_quality import check_date_validity

pytestmark = pytest.mark.unit

NOW = datetime(2026, 9, 1, tzinfo=UTC)


def test_ordinary_dates_pass(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", published=datetime(2026, 1, 1, tzinfo=UTC))
        result = check_date_validity(session, as_of=NOW)
    assert result.passed


def test_a_future_publication_date_is_caught(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", published=NOW + timedelta(days=30))
        result = check_date_validity(session, as_of=NOW)
    assert not result.passed
    assert "future publication" in result.examples[0]


def test_a_publication_dated_exactly_now_is_not_flagged(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", published=NOW)
        result = check_date_validity(session, as_of=NOW)
    assert result.passed


def test_a_forecast_that_targets_its_own_date_is_caught(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        session.add(
            Forecast(
                entity_type="topic",
                entity_id=1,
                forecast_date=date(2026, 8, 1),
                target_period=date(2026, 8, 1),
                predicted_value=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
                model_name="naive",
            )
        )
        result = check_date_validity(session, as_of=NOW)
    assert not result.passed
    assert "targets on/before" in result.examples[0]


def test_a_forecast_that_targets_the_future_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        session.add(
            Forecast(
                entity_type="topic",
                entity_id=1,
                forecast_date=date(2026, 8, 1),
                target_period=date(2026, 9, 1),
                predicted_value=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
                model_name="naive",
            )
        )
        result = check_date_validity(session, as_of=NOW)
    assert result.passed


def test_a_monthly_period_not_on_the_first_of_the_month_is_caught(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        session.add(
            ActivityAggregate(period=date(2026, 8, 15), period_type="month", source_type="patent")
        )
        result = check_date_validity(session, as_of=NOW)
    assert not result.passed
    assert "not aligned to a month boundary" in result.examples[0]


def test_a_properly_aligned_monthly_period_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        session.add(
            ActivityAggregate(period=date(2026, 8, 1), period_type="month", source_type="patent")
        )
        result = check_date_validity(session, as_of=NOW)
    assert result.passed


def test_quarter_and_year_periods_are_not_checked_against_the_month_rule(
    factory: sessionmaker[Session],
) -> None:
    """A quarter legitimately starts on the 1st of its first month; the rule is month-specific."""
    with session_scope(factory) as session:
        session.add(
            ActivityAggregate(period=date(2026, 7, 1), period_type="quarter", source_type="patent")
        )
        result = check_date_validity(session, as_of=NOW)
    assert result.passed


def test_multiple_kinds_of_problem_are_all_counted(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        make_record(session, "r1", published=NOW + timedelta(days=1))
        session.add(
            Forecast(
                entity_type="topic",
                entity_id=1,
                forecast_date=date(2026, 8, 1),
                target_period=date(2026, 7, 1),
                predicted_value=1.0,
                lower_bound=0.0,
                upper_bound=2.0,
                model_name="naive",
            )
        )
        result = check_date_validity(session, as_of=NOW)
    assert result.affected_count == 2


def test_no_data_at_all_is_not_a_failure(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        result = check_date_validity(session, as_of=NOW)
    assert result.passed
