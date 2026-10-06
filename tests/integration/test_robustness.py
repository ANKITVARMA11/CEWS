"""Integration tests for the sensitivity/robustness checks.

Weight, normalization-method and source-removal sensitivity recompute the trend ranking
in-memory from one shared ``compute_features(store=False)`` call, so these tests build the demo
dataset once (normalized, competitors discovered) and reuse it, the same way the backtest tests
do. The low-sample-confidence check is the one exception: it reads stored scores, so it is tested
against a database that has actually been scored.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import FeatureValue, Score
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import EntityFeatures, compute_features
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.scoring.config import ScoringConfig, load_scoring_config
from cews.scoring.pipeline import run_scoring
from cews.settings import Settings, load_settings
from cews.validation.robustness import (
    low_sample_confidence_check,
    normalization_method_sensitivity,
    run_robustness_checks,
    source_removal_sensitivity,
    weight_sensitivity,
)
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


@pytest.fixture(scope="module")
def normalized(taxonomy: Taxonomy) -> Iterator[tuple[sessionmaker[Session], Settings]]:
    """The full demo dataset, normalized with competitors chosen, but not scored - the
    weight/normalization/source-removal checks compute everything they need in memory."""
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
        # compute_features reads activity_aggregates directly and never populates it itself;
        # run_robustness_checks does this call for its own callers, but this fixture also feeds
        # compute_features directly (via _topics) and the `scored` fixture below, so it is done
        # once here for everything built on top of this fixture.
        aggregate_monthly_activity(session, is_synthetic=True)
    yield factory, settings
    engine.dispose()


@pytest.fixture(scope="module")
def scored(
    normalized: tuple[sessionmaker[Session], Settings],
) -> tuple[sessionmaker[Session], Settings]:
    """The same dataset, with one real features + score pass on top, for the checks that read
    stored scores (low_sample_confidence_check)."""
    factory, settings = normalized
    with session_scope(factory) as session:
        compute_features(session, settings, as_of=AS_OF, store=True, is_synthetic=True)
        run_scoring(session, settings, as_of=AS_OF, store=True, is_synthetic=True)
    return factory, settings


def _topics(
    normalized: tuple[sessionmaker[Session], Settings],
) -> tuple[list[EntityFeatures], ScoringConfig]:
    factory, settings = normalized
    with session_scope(factory) as session:
        feature_run = compute_features(session, settings, as_of=AS_OF, store=False)
        topics = [e for e in feature_run.entities if e.entity_type == EntityType.TOPIC.value]
        config = load_scoring_config(settings)
    return topics, config


# --------------------------------------------------------------------------------------
# Overall orchestration
# --------------------------------------------------------------------------------------
def test_a_full_run_produces_every_dimension(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_robustness_checks(session, settings, as_of=AS_OF)
    dimensions = {result.dimension.split(":")[0] for result in report.results}
    assert {"weight", "normalization", "source_removed", "window_months"} <= dimensions


def test_nothing_is_written_to_the_database(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        before_scores = session.scalar(select(func.count()).select_from(Score))
        before_features = session.scalar(select(func.count()).select_from(FeatureValue))
    with session_scope(factory) as session:
        run_robustness_checks(session, settings, as_of=AS_OF)
    with session_scope(factory) as session:
        after_scores = session.scalar(select(func.count()).select_from(Score))
        after_features = session.scalar(select(func.count()).select_from(FeatureValue))
    assert before_scores == after_scores
    assert before_features == after_features


def test_every_stability_metric_is_on_its_stated_scale(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_robustness_checks(session, settings, as_of=AS_OF)
    for result in report.results:
        assert 0.0 <= result.stability.top_k_overlap <= 1.0
        if result.stability.spearman.coefficient is not None:
            assert -1.0 <= result.stability.spearman.coefficient <= 1.0


def test_the_report_is_json_friendly(normalized: tuple[sessionmaker[Session], Settings]) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_robustness_checks(session, settings, as_of=AS_OF)
    json.dumps(report.as_dict())


def test_an_empty_database_reports_it_cannot_run() -> None:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    settings = load_settings(
        env_file=None,
        overrides={"topic_taxonomy_file": TAXONOMY_FILE, "scoring_config_file": SCORING_FILE},
    )
    with session_scope(factory) as session:
        report = run_robustness_checks(session, settings, as_of=AS_OF)
    assert report.results == []
    assert any("fewer than 2" in warning for warning in report.warnings)
    engine.dispose()


# --------------------------------------------------------------------------------------
# Weight sensitivity
# --------------------------------------------------------------------------------------
def test_weight_sensitivity_covers_every_trend_component(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    topics, config = _topics(normalized)
    results = weight_sensitivity(topics, config)
    dimensions = {result.dimension for result in results}
    assert dimensions == {
        "weight:velocity",
        "weight:momentum",
        "weight:patent_growth",
        "weight:funding_growth",
        "weight:consistency",
    }


def test_a_zero_perturbation_leaves_the_ranking_unchanged(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    """Removing 0% of a weight is a no-op; the ranking must come back perfectly stable."""
    topics, config = _topics(normalized)
    results = weight_sensitivity(topics, config, perturbation=0.0)
    for result in results:
        assert result.stability.spearman.coefficient == pytest.approx(1.0)
        assert result.stability.top_k_overlap == 1.0


# --------------------------------------------------------------------------------------
# Normalization method sensitivity
# --------------------------------------------------------------------------------------
def test_normalization_sensitivity_covers_every_alternative_method(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    topics, config = _topics(normalized)
    results = normalization_method_sensitivity(topics, config)
    dimensions = {result.dimension for result in results}
    assert dimensions == {
        "normalization:minmax",
        "normalization:robust_zscore",
        "normalization:winsorized_minmax",
    }


# --------------------------------------------------------------------------------------
# Source removal sensitivity
# --------------------------------------------------------------------------------------
def test_source_removal_covers_every_source_present_in_the_data(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    topics, config = _topics(normalized)
    results = source_removal_sensitivity(topics, config)
    present_sources = {source for topic in topics for source in topic.available_sources}
    dimensions = {result.dimension.removeprefix("source_removed:") for result in results}
    assert dimensions == present_sources


def test_removing_a_source_never_crashes_even_when_it_is_dominant(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    """A topic with only one source type contributing must still produce a comparable ranking
    (or be cleanly excluded from the comparison), never raise."""
    topics, config = _topics(normalized)
    results = source_removal_sensitivity(topics, config)
    assert results  # the demo data has more than one source type present


# --------------------------------------------------------------------------------------
# Low-sample confidence check
# --------------------------------------------------------------------------------------
def test_low_sample_confidence_is_none_with_nothing_scored(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, _ = normalized
    with session_scope(factory) as session:
        assert low_sample_confidence_check(session) is None


def test_low_sample_confidence_reads_real_stored_scores(
    scored: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, _ = scored
    with session_scope(factory) as session:
        result = low_sample_confidence_check(session)
    assert result is not None
    assert result.dimension == "low_sample_confidence"
    assert result.stability.spearman.coefficient is not None
    assert -1.0 <= result.stability.spearman.coefficient <= 1.0
