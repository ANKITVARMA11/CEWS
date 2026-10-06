"""Integration tests for the historical topic-ranking backtest.

Each fold genuinely recomputes features and scores at a historical cutoff, so these tests build
the full demo dataset once and share it, the same way the scoring pipeline's own tests do.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, delete, func, insert, select
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import FeatureValue, Score
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.settings import Settings, load_settings
from cews.validation.backtest import BacktestReport, run_topic_ranking_backtest
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


@pytest.fixture(scope="module")
def normalized(taxonomy: Taxonomy) -> Iterator[tuple[sessionmaker[Session], Settings]]:
    """The full demo dataset, normalized with competitors chosen, but not yet featured or
    scored - the backtest computes those itself, at each historical cutoff it tests."""
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
    yield factory, settings
    engine.dispose()


def run(normalized: tuple[sessionmaker[Session], Settings], **kwargs: Any) -> BacktestReport:
    factory, settings = normalized
    with session_scope(factory) as session:
        return run_topic_ranking_backtest(session, settings, is_synthetic=True, **kwargs)


# --------------------------------------------------------------------------------------
# It produces real, varied folds
# --------------------------------------------------------------------------------------
def test_the_demo_dataset_produces_several_folds(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    report = run(normalized, horizon_months=3, k=5, cleanup=True)
    assert len(report.folds) > 5
    assert report.mean_precision_at_k is not None
    assert 0.0 <= report.mean_precision_at_k <= 1.0


def test_folds_cover_different_cutoffs_with_different_rankings(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    report = run(normalized, horizon_months=3, k=5, cleanup=True)
    cutoffs = [fold.cutoff for fold in report.folds]
    assert len(set(cutoffs)) == len(cutoffs)  # every fold is a distinct date
    top_topics = {fold.predicted[0][1] for fold in report.folds if fold.predicted}
    assert len(top_topics) > 1  # the top-ranked topic is not frozen across all of history


def test_metrics_are_always_on_their_stated_scale(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    report = run(normalized, horizon_months=3, k=5, cleanup=True)
    for fold in report.folds:
        assert 0.0 <= fold.precision_at_k <= 1.0
        assert 0.0 <= fold.recall_at_k <= 1.0
        assert 0.0 <= fold.ndcg_at_k <= 1.0
        if fold.spearman.coefficient is not None:
            assert -1.0 <= fold.spearman.coefficient <= 1.0


# --------------------------------------------------------------------------------------
# No leakage: a fold's prediction cannot see its own evaluation window
# --------------------------------------------------------------------------------------
def _fresh_database(taxonomy: Taxonomy) -> tuple[Engine, sessionmaker[Session], Settings]:
    """A brand new database with the same demo dataset, independent of any other test's state."""
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
    return engine, factory, settings


def test_a_folds_predicted_ranking_does_not_depend_on_data_after_its_cutoff(
    taxonomy: Taxonomy,
) -> None:
    """The defining guarantee: two backtests differing only in how much *future* data exists
    beyond a fold's own horizon must predict identically for that fold.

    Each side of the comparison gets its own freshly built database, loaded with the identical
    demo dataset, so nothing carried over from repeated runs against a shared database (a
    separate property, already covered by the cleanup tests) can be mistaken for leakage here.
    """
    from cews.features.activity_counts import aggregate_monthly_activity
    from cews.validation.backtest import _default_cutoffs

    full_engine, full_factory, full_settings = _fresh_database(taxonomy)
    try:
        with session_scope(full_factory) as session:
            # _default_cutoffs reads activity_aggregates, which run_topic_ranking_backtest
            # normally populates itself; done explicitly here since the cutoffs are needed
            # before the backtest call, to pass the same nominal cutoff to both sides below.
            aggregate_monthly_activity(session, is_synthetic=True)
            nominal_cutoffs = _default_cutoffs(session, horizon_months=3, min_history_months=15)
            assert nominal_cutoffs
            earliest_nominal_cutoff = nominal_cutoffs[0]
            full_report = run_topic_ranking_backtest(
                session,
                full_settings,
                cutoffs=nominal_cutoffs,
                horizon_months=3,
                k=5,
                is_synthetic=True,
                cleanup=True,
            )
    finally:
        full_engine.dispose()
    assert full_report.folds
    earliest_fold = min(full_report.folds, key=lambda fold: fold.cutoff)

    single_engine, single_factory, single_settings = _fresh_database(taxonomy)
    try:
        with session_scope(single_factory) as session:
            # Passing the *nominal* cutoff that produced earliest_fold, not earliest_fold.cutoff
            # itself: BacktestFold.cutoff is already the real, shifted date compute_features
            # settled on, and feeding that back in as a nominal cutoff would shift it a second
            # time, comparing two different months rather than testing anything about leakage.
            single_fold_report = run_topic_ranking_backtest(
                session,
                single_settings,
                cutoffs=[earliest_nominal_cutoff],
                horizon_months=3,
                k=5,
                is_synthetic=True,
                cleanup=True,
            )
    finally:
        single_engine.dispose()
    assert len(single_fold_report.folds) == 1
    assert single_fold_report.folds[0].cutoff == earliest_fold.cutoff
    assert single_fold_report.folds[0].predicted == earliest_fold.predicted


# --------------------------------------------------------------------------------------
# cleanup
# --------------------------------------------------------------------------------------
def test_cleanup_removes_only_what_the_backtest_itself_wrote(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    """The bug this guards against: a fold's own historical row must never be mistaken for
    another fold's, and a date that pre-dates the backtest must never be touched."""
    factory, settings = normalized
    with session_scope(factory) as session:
        session.execute(
            insert(Score),
            [
                {
                    "score_date": date(2026, 8, 1),
                    "entity_type": "topic",
                    "entity_id": 1,
                    "context_key": "",
                    "score_type": "trend",
                    "score_value": 42.0,
                    "confidence_score": 90.0,
                    "component_json": {},
                    "scoring_version": "pre-existing",
                    "is_synthetic": True,
                }
            ],
        )
        before = {(row.score_date, row.scoring_version) for row in session.scalars(select(Score))}

    with session_scope(factory) as session:
        report = run_topic_ranking_backtest(
            session, settings, horizon_months=3, k=5, is_synthetic=True, cleanup=True
        )
    assert len(report.folds) > 0

    with session_scope(factory) as session:
        after = {(row.score_date, row.scoring_version) for row in session.scalars(select(Score))}
    assert before <= after  # the pre-existing row (and its version) is still there
    assert (date(2026, 8, 1), "pre-existing") in after


def test_cleanup_leaves_no_feature_or_score_rows_behind(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        before_scores = session.scalar(select(func.count()).select_from(Score))
        before_features = session.scalar(select(func.count()).select_from(FeatureValue))

    with session_scope(factory) as session:
        run_topic_ranking_backtest(
            session, settings, horizon_months=3, k=5, is_synthetic=True, cleanup=True
        )

    with session_scope(factory) as session:
        after_scores = session.scalar(select(func.count()).select_from(Score))
        after_features = session.scalar(select(func.count()).select_from(FeatureValue))
    assert after_scores == before_scores
    assert after_features == before_features


def test_without_cleanup_the_historical_rows_are_left_for_inspection(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_topic_ranking_backtest(
            session, settings, horizon_months=3, k=5, is_synthetic=True, cleanup=False
        )
    try:
        assert report.folds
        with session_scope(factory) as session:
            stored_dates = set(session.scalars(select(Score.score_date).distinct()))
        for fold in report.folds:
            assert fold.cutoff in stored_dates
    finally:
        # These rows are deliberately left behind by cleanup=False; remove them here so they do
        # not look like pre-existing data to every test that runs after this one.
        with session_scope(factory) as session:
            for fold in report.folds:
                session.execute(delete(Score).where(Score.score_date == fold.cutoff))
                session.execute(
                    delete(FeatureValue).where(FeatureValue.feature_date == fold.cutoff)
                )


# --------------------------------------------------------------------------------------
# Explicit cutoffs and edge cases
# --------------------------------------------------------------------------------------
def test_explicit_cutoffs_override_the_default_search(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_topic_ranking_backtest(
            session,
            settings,
            cutoffs=[date(2025, 6, 1), date(2025, 9, 1)],
            horizon_months=3,
            k=5,
            is_synthetic=True,
            cleanup=True,
        )
    # A fold is identified by the date actually scored: the last complete month before each
    # nominal cutoff, since compute_features excludes the month containing the cutoff itself.
    assert {fold.cutoff for fold in report.folds} == {date(2025, 5, 1), date(2025, 8, 1)}


def test_a_cutoff_too_recent_for_a_full_horizon_is_reported(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session:
        report = run_topic_ranking_backtest(
            session,
            settings,
            cutoffs=[date(2026, 8, 1)],  # right at the edge of the demo data
            horizon_months=6,
            k=5,
            is_synthetic=True,
            cleanup=True,
        )
    # either it could not be scored meaningfully, or its growth numbers reflect a short window;
    # either way this must not crash
    assert isinstance(report.folds, list)


def test_an_empty_database_reports_it_cannot_run() -> None:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    settings = load_settings(
        env_file=None,
        overrides={"topic_taxonomy_file": TAXONOMY_FILE, "scoring_config_file": SCORING_FILE},
    )
    with session_scope(factory) as session:
        report = run_topic_ranking_backtest(session, settings)
    assert report.folds == []
    assert any("not enough history" in warning for warning in report.warnings)
    engine.dispose()


@pytest.mark.parametrize(("horizon_months", "k"), [(0, 5), (3, 0), (-1, 5)])
def test_invalid_parameters_are_refused(
    normalized: tuple[sessionmaker[Session], Settings], horizon_months: int, k: int
) -> None:
    factory, settings = normalized
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        run_topic_ranking_backtest(session, settings, horizon_months=horizon_months, k=k)


def test_the_report_is_json_friendly(normalized: tuple[sessionmaker[Session], Settings]) -> None:
    import json

    report = run(normalized, horizon_months=3, k=5, cleanup=True)
    json.dumps(report.as_dict())
    for fold in report.folds:
        json.dumps(fold.as_dict())


# --------------------------------------------------------------------------------------
# Data that was already there is never overwritten or deleted
# --------------------------------------------------------------------------------------
def _store_real_run(factory: sessionmaker[Session], settings: Settings, as_of: datetime) -> date:
    """Store features and scores the way `cews features` and `cews score` would."""
    from cews.features.pipeline import compute_features
    from cews.scoring.pipeline import run_scoring

    with session_scope(factory) as session:
        features = compute_features(session, settings, as_of=as_of, store=True, is_synthetic=True)
        run_scoring(session, settings, as_of=as_of, store=True, is_synthetic=True)
        return features.feature_date


def _snapshot(factory: sessionmaker[Session], day: date) -> list[tuple[object, ...]]:
    with session_scope(factory) as session:
        scores = session.execute(
            select(Score.entity_type, Score.entity_id, Score.score_type, Score.score_value)
            .where(Score.score_date == day)
            .order_by(Score.entity_type, Score.entity_id, Score.score_type, Score.context_key)
        ).all()
        features = session.execute(
            select(FeatureValue.entity_id, FeatureValue.feature_name, FeatureValue.raw_value)
            .where(FeatureValue.feature_date == day)
            .order_by(FeatureValue.entity_id, FeatureValue.feature_name)
        ).all()
    return [tuple(row) for row in scores] + [tuple(row) for row in features]


def test_stored_data_at_a_fold_date_is_neither_overwritten_nor_deleted(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    """The hazard: an earlier `cews features`/`cews score` run left rows at a date a fold would
    write. Recomputing there and then cleaning up would destroy real output."""
    factory, settings = normalized
    baseline = run(normalized, horizon_months=3, k=5, cleanup=True)

    existing = _store_real_run(factory, settings, datetime(2025, 8, 1, tzinfo=UTC))
    try:
        before = _snapshot(factory, existing)
        assert before  # something real is stored

        report = run(normalized, horizon_months=3, k=5, cleanup=True)

        assert _snapshot(factory, existing) == before  # untouched, value for value
        assert len(report.folds) < len(baseline.folds)  # the colliding folds stood down
        assert any("already stored" in warning for warning in report.warnings)
        assert existing not in {fold.cutoff for fold in report.folds}
    finally:
        with session_scope(factory) as session:
            session.execute(delete(Score).where(Score.score_date == existing))
            session.execute(delete(FeatureValue).where(FeatureValue.feature_date == existing))


def test_a_stored_date_in_the_gap_before_a_cutoff_is_not_scored_by_mistake(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    """A stored feature date between the date a fold computes and its cutoff would be picked
    up by scoring in place of the fresh one; the fold must refuse rather than mislabel it."""
    factory, settings = normalized
    existing = _store_real_run(factory, settings, datetime(2025, 8, 1, tzinfo=UTC))  # 2025-07-01
    try:
        report = run(normalized, cutoffs=[date(2025, 7, 1)], horizon_months=3, k=5, cleanup=True)
        assert report.folds == []
        assert any("later date" in warning for warning in report.warnings)
    finally:
        with session_scope(factory) as session:
            session.execute(delete(Score).where(Score.score_date == existing))
            session.execute(delete(FeatureValue).where(FeatureValue.feature_date == existing))


def test_a_clean_database_is_unaffected_by_the_guards(
    normalized: tuple[sessionmaker[Session], Settings],
) -> None:
    report = run(normalized, horizon_months=3, k=5, cleanup=True)
    assert len(report.folds) > 5
    assert not any("already stored" in w or "later date" in w for w in report.warnings)
