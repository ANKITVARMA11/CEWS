"""Unit tests for the evaluation orchestrator.

These use an empty in-memory database, so sections that need data report that plainly. The
expensive sections (backtest, robustness) are exercised for real in the integration tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import EvaluationRun, Forecast
from cews.settings import Settings, load_settings
from cews.validation import evaluation_report
from cews.validation.evaluation_report import (
    DEFAULT_SECTIONS,
    GROUPS,
    SECTIONS,
    latest_stored_evaluations,
    run_evaluation,
    summarize_forecast_metrics,
)

pytestmark = pytest.mark.unit

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


def _forecast(session: Session, entity_id: int, model: str, mase: float | None, day: date) -> None:
    session.add(
        Forecast(
            entity_type="topic",
            entity_id=entity_id,
            forecast_date=day,
            target_period=date(day.year, day.month + 1, 1),
            predicted_value=1.0,
            lower_bound=0.0,
            upper_bound=2.0,
            model_name=model,
            backtest_metric_name="mase",
            backtest_metric=mase,
        )
    )


# --------------------------------------------------------------------------------------
# Structure: three kinds of evidence, kept apart
# --------------------------------------------------------------------------------------
def test_every_section_belongs_to_exactly_one_group() -> None:
    flat = [name for names in GROUPS.values() for name in names]
    assert sorted(flat) == sorted(SECTIONS) and len(flat) == len(set(flat))


def test_the_three_headings_are_algorithm_backtest_and_expert_validation() -> None:
    assert set(GROUPS) == {"algorithm", "backtest", "expert_validation"}


def test_algorithm_output_is_never_filed_under_expert_validation() -> None:
    assert "data_quality" not in GROUPS["expert_validation"]
    assert "alerts" not in GROUPS["algorithm"] and "alerts" not in GROUPS["backtest"]
    assert "backtest" not in GROUPS["expert_validation"]


def test_the_default_run_needs_no_file() -> None:
    assert "expert_sheet" not in DEFAULT_SECTIONS


def test_a_report_groups_only_what_ran(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(session, settings, sections=("data_quality", "alerts"), as_of=AS_OF)
    grouped = report.grouped()
    assert list(grouped["algorithm"]) == ["data_quality"]
    assert list(grouped["expert_validation"]) == ["alerts"]
    assert grouped["backtest"] == {}


# --------------------------------------------------------------------------------------
# Running sections
# --------------------------------------------------------------------------------------
def test_data_quality_and_alerts_run_on_an_empty_database(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(session, settings, sections=("data_quality", "alerts"), as_of=AS_OF)
    assert report.errors == {}
    assert report.sections["data_quality"]["passed"] is True
    assert report.sections["alerts"]["reviewed"] == 0


def test_a_section_that_cannot_run_is_recorded_and_the_rest_still_run(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(
            session, settings, sections=("forecast", "data_quality"), as_of=AS_OF
        )
    assert "no forecasts stored yet" in report.errors["forecast"]
    assert "data_quality" in report.sections and "forecast" not in report.sections


def test_a_missing_benchmark_file_is_an_error_entry_not_a_crash(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = load_settings(
        env_file=None, overrides={"benchmark_topics_file": tmp_path / "absent.yaml"}
    )
    with session_scope(factory) as session:
        report = run_evaluation(session, settings, sections=("benchmarks",), as_of=AS_OF)
    assert "benchmark file not found" in report.errors["benchmarks"]


def test_a_genuine_bug_is_not_swallowed(
    factory: sessionmaker[Session], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only expected failures (bad config, missing file) become error entries."""

    def broken(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("a real bug")

    monkeypatch.setattr(evaluation_report, "run_data_quality_checks", broken)
    with session_scope(factory) as session, pytest.raises(RuntimeError, match="real bug"):
        run_evaluation(session, settings, sections=("data_quality",), as_of=AS_OF)


def test_an_unknown_section_is_refused(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="unknown section"):
        run_evaluation(session, settings, sections=("nonsense",))


@pytest.mark.parametrize(("horizon", "k"), [(0, 5), (3, 0)])
def test_non_positive_parameters_are_refused(
    factory: sessionmaker[Session], settings: Settings, horizon: int, k: int
) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        run_evaluation(session, settings, horizon_months=horizon, k=k)


def test_supplying_an_expert_file_adds_the_expert_sheet_section(
    factory: sessionmaker[Session], settings: Settings, tmp_path: Path
) -> None:
    sheet = tmp_path / "s.csv"
    sheet.write_text("topic,score,expert_rating\nA,90,relevant\n", encoding="utf-8")
    with session_scope(factory) as session:
        report = run_evaluation(
            session, settings, sections=("alerts",), as_of=AS_OF, expert_file=sheet
        )
    assert report.sections["expert_sheet"]["rated"] == 1


def test_the_ai_ablation_runs_against_the_shipped_fixture(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(session, settings, sections=("ai_ablation",), as_of=AS_OF)
    ablation = report.sections["ai_ablation"]
    assert ablation["announcement_extraction"]["labelled_count"] >= 50
    assert ablation["org_matching"].startswith("skipped")


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------
def test_each_successful_section_is_stored_with_its_configuration(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(
            session,
            settings,
            sections=("data_quality", "alerts"),
            as_of=AS_OF,
            horizon_months=4,
            k=7,
        )
    with session_scope(factory) as session:
        rows = {row.evaluation_type: row for row in session.scalars(select(EvaluationRun))}
    assert set(rows) == {"data_quality", "alerts"}
    assert rows["alerts"].evaluation_id == f"{report.evaluation_id}-alerts"
    assert rows["alerts"].configuration_json["horizon_months"] == 4
    assert rows["alerts"].configuration_json["k"] == 7
    assert rows["alerts"].metrics_json == report.sections["alerts"]


def test_a_section_that_failed_is_not_stored(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        run_evaluation(session, settings, sections=("forecast",), as_of=AS_OF)
    with session_scope(factory) as session:
        assert session.scalars(select(EvaluationRun)).all() == []


def test_no_store_writes_nothing(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        run_evaluation(session, settings, sections=("data_quality",), as_of=AS_OF, store=False)
    with session_scope(factory) as session:
        assert session.scalars(select(EvaluationRun)).all() == []


def test_two_runs_get_distinct_ids(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        first = run_evaluation(session, settings, sections=("alerts",), as_of=AS_OF)
        second = run_evaluation(session, settings, sections=("alerts",), as_of=AS_OF)
    assert first.evaluation_id != second.evaluation_id


def test_the_latest_stored_result_of_each_kind_is_returned(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        run_evaluation(session, settings, sections=("alerts",), as_of=AS_OF)
        second = run_evaluation(session, settings, sections=("alerts", "data_quality"), as_of=AS_OF)
    with session_scope(factory) as session:
        latest = latest_stored_evaluations(session)
    assert set(latest) == {"alerts", "data_quality"}
    assert latest["alerts"]["evaluation_id"] == f"{second.evaluation_id}-alerts"


def test_the_report_is_json_friendly(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        report = run_evaluation(
            session, settings, sections=("data_quality", "alerts", "ai_ablation"), as_of=AS_OF
        )
    payload = json.loads(json.dumps(report.as_dict()))
    assert {"algorithm", "backtest", "expert_validation", "errors"} <= set(payload)


# --------------------------------------------------------------------------------------
# Forecast summary
# --------------------------------------------------------------------------------------
def test_the_forecast_summary_uses_each_entitys_latest_forecast(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        _forecast(session, 1, "holt_linear", 9.0, date(2026, 6, 1))  # superseded
        _forecast(session, 1, "arima", 0.5, date(2026, 7, 1))
        _forecast(session, 2, "naive", 1.5, date(2026, 7, 1))
        session.flush()
        summary = summarize_forecast_metrics(session)
    assert summary["entities"] == 2
    assert summary["models_chosen"] == {"arima": 1, "naive": 1}
    assert summary["median_mase"] == pytest.approx(1.0)
    assert summary["best_mase"] == 0.5 and summary["worst_mase"] == 1.5


def test_an_entity_with_no_backtest_metric_is_counted_but_not_averaged(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        _forecast(session, 1, "naive", None, date(2026, 7, 1))
        _forecast(session, 2, "arima", 0.8, date(2026, 7, 1))
        session.flush()
        summary = summarize_forecast_metrics(session)
    assert summary["entities"] == 2 and summary["with_backtest_metric"] == 1
    assert summary["mean_mase"] == pytest.approx(0.8)


def test_multiple_target_months_do_not_inflate_the_entity_count(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        for month in (8, 9, 10):
            session.add(
                Forecast(
                    entity_type="topic",
                    entity_id=1,
                    forecast_date=date(2026, 7, 1),
                    target_period=date(2026, month, 1),
                    predicted_value=1.0,
                    lower_bound=0.0,
                    upper_bound=2.0,
                    model_name="arima",
                    backtest_metric_name="mase",
                    backtest_metric=0.7,
                )
            )
        session.flush()
        assert summarize_forecast_metrics(session)["entities"] == 1


def test_no_forecasts_at_all_is_an_error_naming_the_fix(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="cews forecast"):
        summarize_forecast_metrics(session)
