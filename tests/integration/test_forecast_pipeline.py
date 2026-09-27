"""Integration tests for the forecasting and anomaly pass."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Anomaly as AnomalyRow
from cews.database.models import Forecast
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity
from cews.forecasting.pipeline import ForecastRun, run_forecasting
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.settings import Settings, load_settings
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
LAST_COMPLETE = date(2026, 8, 1)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


@pytest.fixture(scope="module")
def prepared(taxonomy: Taxonomy) -> Iterator[tuple[sessionmaker[Session], Settings]]:
    """The full demo dataset with activity counted, built once."""
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    settings = load_settings(
        env_file=None,
        overrides={
            "topic_taxonomy_file": TAXONOMY_FILE,
            "scoring_config_file": SCORING_FILE,
            "competitor_mode": "AUTO",
            "min_competitor_evidence_count": 5,
        },
    )
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig()))
        normalize_records(session, settings, taxonomy)
        discover_competitors(session, settings, as_of=AS_OF)
        aggregate_monthly_activity(session, as_of=AS_OF, is_synthetic=True)
    yield factory, settings
    engine.dispose()


def run(prepared: tuple[sessionmaker[Session], Settings], **kwargs: object) -> ForecastRun:
    factory, settings = prepared
    with session_scope(factory) as session:
        return run_forecasting(session, settings, as_of=AS_OF, is_synthetic=True, **kwargs)


@pytest.fixture(scope="module")
def computed(prepared: tuple[sessionmaker[Session], Settings]) -> ForecastRun:
    """One pass, shared by the tests that only read the result.

    Backtesting five models across every entity is the slow part of the suite, so it runs once
    here; the tests that check storage do their own runs.
    """
    return run(prepared, store=False)


def named(result: ForecastRun, prefix: str) -> object:
    return next(entity for entity in result.entities if entity.name.startswith(prefix))


# --------------------------------------------------------------------------------------
# Forecasts
# --------------------------------------------------------------------------------------
def test_every_active_entity_is_forecast(computed: ForecastRun) -> None:
    result = computed
    assert result.entities
    kinds = {entity.entity_type for entity in result.entities}
    assert kinds == {EntityType.TOPIC.value, EntityType.COMPETITOR.value}
    assert all(entity.selection.forecast is not None for entity in result.entities)


def test_quiet_entities_are_skipped_not_forecast_as_zero(computed: ForecastRun) -> None:
    result = computed
    assert result.skipped
    assert not set(result.skipped) & {entity.name for entity in result.entities}


def test_the_month_in_progress_is_not_forecast_from(computed: ForecastRun) -> None:
    """A part-finished month would look like a collapse and drag every forecast down."""
    result = computed
    assert result.as_of == LAST_COMPLETE
    assert result.entities[0].months[-1] == LAST_COMPLETE


def test_forecasts_look_forward_from_the_last_complete_month(computed: ForecastRun) -> None:
    entity = computed.entities[0]
    assert entity.forecast_months[0] == date(2026, 9, 1)
    assert len(entity.forecast_months) == 6


def test_a_model_is_chosen_per_entity(computed: ForecastRun) -> None:
    """Different shapes of history deserve different models."""
    chosen = {entity.selection.model for entity in computed.entities}
    assert len(chosen) > 1
    assert all(model for model in chosen)


def test_each_forecast_keeps_the_comparison_that_chose_it(computed: ForecastRun) -> None:
    entity = computed.entities[0]
    payload = entity.selection.as_dict()
    assert payload["reason"] and payload["candidates"]
    assert set(payload["candidates"]) >= {"naive", "holt_linear"}


def test_intervals_surround_the_prediction(computed: ForecastRun) -> None:
    for entity in computed.entities:
        forecast = entity.selection.forecast
        assert forecast is not None
        for value, low, high in zip(
            forecast.predictions, forecast.lower, forecast.upper, strict=True
        ):
            assert low <= value <= high
            assert low >= 0.0  # counts cannot be negative


# --------------------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------------------
def test_the_designed_spike_is_found_and_called_a_spike(computed: ForecastRun) -> None:
    """The demo puts one month of patents in AAV gene delivery and nothing else unusual there."""
    aav = named(computed, "AAV")
    spikes = {
        found.period: found for found in aav.anomalies if found.kind.value == "one_time_spike"
    }
    designed = spikes.get(date(2026, 5, 1))
    assert designed is not None, f"expected a spike in May 2026, found {sorted(spikes)}"
    assert designed.confidence > 80
    assert designed.observed > designed.expected_upper
    # it is the most convincing one in that series, whatever else the noise produced
    assert designed.confidence == max(found.confidence for found in aav.anomalies)


def test_the_thin_topic_is_flagged_only_faintly(computed: ForecastRun) -> None:
    thin = named(computed, "siRNA")
    assert thin.anomalies
    assert all(found.confidence < 50 for found in thin.anomalies)


def test_anomalies_can_be_turned_off(prepared: tuple[sessionmaker[Session], Settings]) -> None:
    result = run(prepared, store=False, detect_anomalies=False, history_months=18)
    assert result.entities and not result.anomalies()


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------
def test_forecasts_and_anomalies_are_stored(
    prepared: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, _ = prepared
    result = run(prepared, history_months=18, horizon=2)
    with session_scope(factory) as session:
        forecasts = list(session.scalars(select(Forecast)))
        anomalies = list(session.scalars(select(AnomalyRow)))
    assert len(forecasts) == sum(len(entity.forecast_months) for entity in result.entities)
    assert anomalies
    row = forecasts[0]
    assert row.forecast_date == LAST_COMPLETE and row.target_period > LAST_COMPLETE
    assert row.lower_bound <= row.predicted_value <= row.upper_bound
    assert row.model_name and row.backtest_metric_name == "mase"
    assert row.training_months and row.is_synthetic is True
    anomaly = anomalies[0]
    assert anomaly.evidence_json and "explanation" in anomaly.evidence_json
    assert anomaly.expected_lower <= anomaly.expected_upper


def test_rerunning_replaces_rather_than_duplicating(
    prepared: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, _ = prepared
    run(prepared, history_months=18, horizon=2)
    with session_scope(factory) as session:
        before = session.scalar(select(func.count()).select_from(Forecast))
        anomalies_before = session.scalar(select(func.count()).select_from(AnomalyRow))

    second = run(prepared, history_months=18, horizon=2)
    assert second.forecasts_written == 0  # everything was already there
    assert second.forecasts_updated > 0
    assert second.anomalies_written == 0
    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(Forecast)) == before
        assert session.scalar(select(func.count()).select_from(AnomalyRow)) == anomalies_before


def test_nothing_is_written_when_storing_is_off(computed: ForecastRun) -> None:
    result = computed
    assert result.entities
    assert result.forecasts_written == 0 and result.anomalies_written == 0


def test_an_empty_database_says_what_to_run() -> None:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    settings = load_settings(
        env_file=None,
        overrides={"topic_taxonomy_file": TAXONOMY_FILE, "scoring_config_file": SCORING_FILE},
    )
    with session_scope(factory) as session:
        result = run_forecasting(session, settings, as_of=AS_OF)
    assert result.entities == []
    assert any("cews normalize" in warning for warning in result.warnings)
    engine.dispose()


@pytest.mark.parametrize(("horizon", "history"), [(0, 30), (6, 0), (-1, 30)])
def test_invalid_settings_are_refused(
    prepared: tuple[sessionmaker[Session], Settings], horizon: int, history: int
) -> None:
    factory, settings = prepared
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        run_forecasting(
            session, settings, as_of=AS_OF, horizon=horizon, history_months=history, store=False
        )
