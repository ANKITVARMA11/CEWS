"""Integration tests for the feature pass: aggregation, computation and storage.

The demo dataset is built with known shapes (a sustained trend, a decline, a one-month spike, a
six-record topic), so these tests check the features describe what the data actually does.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType, PeriodType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ActivityAggregate, FeatureValue, Topic
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity, monthly_series
from cews.features.pipeline import EntityFeatures, FeatureRun, compute_features
from cews.features.time_windows import month_range
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.settings import Settings, load_settings
from support import TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def settings() -> Settings:
    return load_settings(
        env_file=None,
        overrides={
            "topic_taxonomy_file": TAXONOMY_FILE,
            "min_competitor_evidence_count": 5,
            "competitor_mode": "AUTO",
        },
    )


@pytest.fixture
def prepared(
    factory: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> sessionmaker[Session]:
    """Demo data, normalized, with competitors chosen and activity counted."""
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.3)))
        normalize_records(session, settings, taxonomy)
        discover_competitors(session, settings, as_of=AS_OF)
        aggregate_monthly_activity(session, as_of=AS_OF, is_synthetic=True)
    return factory


def run_features(factory: sessionmaker[Session], settings: Settings, **kwargs: Any) -> FeatureRun:
    with session_scope(factory) as session:
        return compute_features(session, settings, as_of=AS_OF, is_synthetic=True, **kwargs)


def topic_named(run: FeatureRun, prefix: str) -> EntityFeatures:
    return next(entity for entity in run.entities if entity.name.startswith(prefix))


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def test_activity_is_counted_into_months_quarters_and_years(
    prepared: sessionmaker[Session],
) -> None:
    with session_scope(prepared) as session:
        kinds = set(session.scalars(select(ActivityAggregate.period_type).distinct()))
        assert kinds == {p.value for p in PeriodType}


def test_quarters_equal_the_sum_of_their_months(prepared: sessionmaker[Session]) -> None:
    with session_scope(prepared) as session:
        months = session.scalar(
            select(func.sum(ActivityAggregate.activity_count)).where(
                ActivityAggregate.period_type == "month",
                ActivityAggregate.organization_id.is_(None),
                ActivityAggregate.topic_id.is_(None),
                ActivityAggregate.period >= date(2026, 1, 1),
                ActivityAggregate.period < date(2026, 4, 1),
            )
        )
        quarter = session.scalar(
            select(func.sum(ActivityAggregate.activity_count)).where(
                ActivityAggregate.period_type == "quarter",
                ActivityAggregate.organization_id.is_(None),
                ActivityAggregate.topic_id.is_(None),
                ActivityAggregate.period == date(2026, 1, 1),
            )
        )
    assert months == quarter and months > 0


def test_recounting_changes_nothing(prepared: sessionmaker[Session]) -> None:
    with session_scope(prepared) as session:
        again = aggregate_monthly_activity(session, as_of=AS_OF, is_synthetic=True)
    assert (again.rows_written, again.rows_updated) == (0, 0)


def test_the_month_in_progress_is_excluded(prepared: sessionmaker[Session]) -> None:
    """A part-finished month would look like a collapse in activity."""
    with session_scope(prepared) as session:
        mid_month = aggregate_monthly_activity(
            session, as_of=datetime(2026, 8, 20, tzinfo=UTC), is_synthetic=True
        )
    assert mid_month.last_period == date(2026, 7, 1)


def test_the_series_reads_back_aligned_to_months(prepared: sessionmaker[Session]) -> None:
    months = month_range(date(2026, 8, 1), 12)
    with session_scope(prepared) as session:
        topic = session.scalar(select(Topic).where(Topic.key == "crispr_gene_editing"))
        assert topic is not None
        series = monthly_series(session, months, topic_id=topic.id)
    assert set(series) >= {"publication", "clinical_trial", "patent"}
    assert all(len(values) == 12 for values in series.values())
    assert sum(series["publication"]) > 0


# --------------------------------------------------------------------------------------
# The features describe the data
# --------------------------------------------------------------------------------------
def test_a_declining_topic_has_negative_momentum_and_velocity(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    run = run_features(prepared, settings)
    declining = topic_named(run, "Immune checkpoint")
    assert declining.momentum.momentum < 0
    assert declining.velocity is not None and declining.velocity.slope < 0
    # Most sources do not show a rise. A couple still can: on a smaller sample the monthly
    # counts are noisy, which is exactly why agreement is a fraction and not a yes/no.
    assert declining.agreement.agreement < 0.5


def test_a_one_month_spike_is_not_called_steady_growth(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    """AAV gene delivery is flat with a single month of patents in the demo data."""
    run = run_features(prepared, settings)
    spiky = topic_named(run, "AAV")
    assert spiky.consistency is not None
    assert spiky.consistency.consistency < 0.6
    assert spiky.velocity is not None and spiky.velocity.r_squared < 0.5


def test_a_topic_with_six_records_is_kept_but_flagged(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    """The demo's deliberate false trend: large growth, almost no evidence."""
    run = run_features(prepared, settings)
    thin = topic_named(run, "siRNA")
    assert thin.momentum.momentum > 1.0  # looks spectacular
    assert thin.total_records < 10
    assert thin.sample_confidence < 0.3  # and confidence says not to believe it
    assert len(thin.agreement.supporting) < 2  # one source cannot corroborate itself
    assert thin.name in run.skipped_low_sample


def test_a_busy_topic_is_trusted(prepared: sessionmaker[Session], settings: Settings) -> None:
    run = run_features(prepared, settings)
    busy = topic_named(run, "CRISPR")
    assert busy.total_records > 100
    assert busy.sample_confidence > 0.95
    assert busy.agreement.multi_source is True


def test_competition_density_is_measured_for_topics_only(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    run = run_features(prepared, settings)
    topics = [e for e in run.entities if e.entity_type == EntityType.TOPIC.value]
    competitors = [e for e in run.entities if e.entity_type == EntityType.COMPETITOR.value]
    assert topics and competitors
    assert all(entity.competition is not None for entity in topics)
    assert all(entity.competition is None for entity in competitors)
    crowded = topic_named(run, "CRISPR")
    assert crowded.competition is not None and crowded.competition.active_competitors >= 2


def test_entities_with_no_activity_are_left_out_of_the_ranking(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    """An empty topic would otherwise shift everyone else's percentile."""
    run = run_features(prepared, settings)
    assert run.skipped_no_data
    ranked = {entity.name for entity in run.entities}
    assert not (ranked & set(run.skipped_no_data))
    assert all(entity.total_records > 0 for entity in run.entities)


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------
def test_features_are_stored_with_their_cohort(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    run = run_features(prepared, settings)
    assert run.values_written > 0
    with session_scope(prepared) as session:
        rows = list(session.scalars(select(FeatureValue)))
    assert len(rows) == run.values_written
    for row in rows:
        assert row.feature_date == run.feature_date
        assert row.normalized_value is not None and 0.0 <= row.normalized_value <= 100.0
        assert row.comparison_cohort.startswith(
            (EntityType.TOPIC.value, EntityType.COMPETITOR.value)
        )
        assert row.cohort_size and row.cohort_size > 0
        assert row.normalization_method == "percentile"
        assert row.is_synthetic is True


def test_topics_and_competitors_are_ranked_separately(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    """A topic's velocity is only meaningful against other topics."""
    run_features(prepared, settings)
    with session_scope(prepared) as session:
        cohorts = set(
            session.scalars(
                select(FeatureValue.comparison_cohort).where(
                    FeatureValue.feature_name == "velocity"
                )
            )
        )
    assert len(cohorts) == 2
    assert any(cohort.startswith("topic/") for cohort in cohorts)
    assert any(cohort.startswith("competitor/") for cohort in cohorts)


def test_rerunning_the_pass_does_not_duplicate_rows(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    first = run_features(prepared, settings)
    second = run_features(prepared, settings)
    assert second.values_written == 0
    assert second.values_updated == first.values_written
    with session_scope(prepared) as session:
        assert (
            session.scalar(select(func.count()).select_from(FeatureValue)) == first.values_written
        )


def test_nothing_is_written_when_storing_is_off(
    prepared: sessionmaker[Session], settings: Settings
) -> None:
    run = run_features(prepared, settings, store=False)
    assert run.entities and run.values_written == 0
    with session_scope(prepared) as session:
        assert session.scalar(select(func.count()).select_from(FeatureValue)) == 0


def test_the_run_reports_what_it_used(prepared: sessionmaker[Session], settings: Settings) -> None:
    summary = run_features(prepared, settings).as_dict()
    assert summary["settings"]["normalization_method"] == "percentile"
    assert summary["settings"]["velocity_window_months"] == 12
    assert summary["topics"] > 0 and summary["competitors"] > 0


def test_an_empty_database_says_so(factory: sessionmaker[Session], settings: Settings) -> None:
    run = run_features(factory, settings)
    assert run.entities == []
    assert any("no active topics" in warning for warning in run.warnings)
