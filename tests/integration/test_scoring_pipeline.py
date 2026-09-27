"""Integration tests for the scoring pass: inputs, confidence, storage and reruns."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType, ScoreType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Score, TherapeuticArea
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import compute_features
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.scoring.config import load_scoring_config
from cews.scoring.inputs import load_entity_inputs
from cews.scoring.pipeline import ScoringRun, run_scoring
from cews.scoring.store import latest_score_date, load_scores
from cews.settings import Settings, load_settings
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


@pytest.fixture(scope="module")
def rich_scored(taxonomy: Taxonomy) -> Iterator[tuple[sessionmaker[Session], Settings]]:
    """The full demo dataset, built once.

    An area is only scored once it has enough records behind it (``MIN_AREA_RECORDS``), which a
    reduced dataset never reaches - correctly, since an area should not be judged on a handful of
    records.
    """
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
        compute_features(session, settings, as_of=AS_OF, is_synthetic=True)
    yield factory, settings
    engine.dispose()


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
            "scoring_config_file": SCORING_FILE,
            "competitor_mode": "AUTO",
            "min_competitor_evidence_count": 5,
        },
    )


@pytest.fixture
def scored(
    factory: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> sessionmaker[Session]:
    """Demo data taken all the way through to stored features."""
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.3)))
        normalize_records(session, settings, taxonomy)
        discover_competitors(session, settings, as_of=AS_OF)
        aggregate_monthly_activity(session, as_of=AS_OF, is_synthetic=True)
        compute_features(session, settings, as_of=AS_OF, is_synthetic=True)
    return factory


def run(factory: sessionmaker[Session], settings: Settings, **kwargs: object) -> ScoringRun:
    with session_scope(factory) as session:
        return run_scoring(session, settings, as_of=AS_OF, is_synthetic=True, **kwargs)


def named(result_run: ScoringRun, prefix: str) -> object:
    return next(result for result in result_run.results if result.entity_name.startswith(prefix))


# --------------------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------------------
def test_every_entity_with_features_gets_a_confidence_score(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    assert result_run.results
    confidences = result_run.of_type(ScoreType.CONFIDENCE.value)
    entities = {(r.entity_type, r.entity_id) for r in result_run.results}
    assert len(confidences) == len(entities)  # one for every entity, whatever else it gets
    assert all(0.0 <= result.value <= 100.0 for result in result_run.results)
    assert result_run.score_date is not None


def test_thin_evidence_scores_low_and_is_gated(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    """The demo's six-record topic must not pass as a usable finding."""
    thin = named(run(scored, settings), "siRNA")
    assert thin.value < 60.0
    assert thin.qualified is False
    assert "usable_confidence" in thin.failed_gates
    assert "below" in thin.explain()


def test_a_well_evidenced_topic_scores_high(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    busy = named(run(scored, settings), "CRISPR")
    assert busy.value > 75.0
    assert busy.qualified is True
    assert busy.sample_size > 50


def test_a_first_run_does_not_punish_missing_history(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    busy = named(run(scored, settings), "CRISPR")
    assert "model_stability" in busy.unavailable
    assert busy.components["model_stability"].available is False
    used = [part.weight for part in busy.components.values() if part.available]
    assert sum(used) == pytest.approx(1.0)


def test_stability_is_measured_once_there_is_a_previous_run(
    scored: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    """A second feature date gives stability something to compare against."""
    earlier = AS_OF - timedelta(days=31)
    with session_scope(scored) as session:
        compute_features(session, settings, as_of=earlier, is_synthetic=True)
    result_run = run(scored, settings)
    busy = named(result_run, "CRISPR")
    assert "model_stability" not in busy.unavailable
    assert busy.components["model_stability"].available is True


# --------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------
def test_scores_are_stored_with_their_breakdown(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        rows = load_scores(session, result_run.score_date, score_type=ScoreType.CONFIDENCE.value)
    assert len(rows) == len(result_run.of_type(ScoreType.CONFIDENCE.value))
    row = rows[0]
    breakdown = row.component_json
    assert set(breakdown) >= {"components", "unavailable", "gates", "explanation", "inputs"}
    assert breakdown["scoring_version"] == result_run.version
    assert row.scoring_version == "1.0.0" and row.is_synthetic is True
    assert 0.0 <= row.score_value <= 100.0 and 0.0 <= row.confidence_score <= 100.0
    assert breakdown["inputs"]["available_sources"]
    # the stored explanation is the sentence a dashboard can show verbatim
    assert "confidence" in breakdown["explanation"]
    assert "record(s)" in breakdown["explanation"]


def test_rerunning_replaces_rather_than_duplicates(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    first = run(scored, settings)
    second = run(scored, settings)
    assert first.stored.written == len(first.results)
    assert second.stored.written == 0 and second.stored.updated == len(second.results)
    with session_scope(scored) as session:
        assert session.scalar(select(func.count()).select_from(Score)) == len(first.results)


def test_nothing_is_written_when_storing_is_off(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings, store=False)
    assert result_run.results and result_run.stored.written == 0
    with session_scope(scored) as session:
        assert session.scalar(select(func.count()).select_from(Score)) == 0


def test_the_latest_score_date_is_reported(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        assert latest_score_date(session) == result_run.score_date


def test_a_changed_weight_version_keeps_the_old_scores(
    scored: sessionmaker[Session], settings: Settings, tmp_path: object
) -> None:
    """Rescoring under different weights must not erase what was there before."""
    import yaml

    original = run(scored, settings)
    document = yaml.safe_load(SCORING_FILE.read_text(encoding="utf-8"))
    document["scoring_version"] = "2.0.0"
    path = tmp_path / "v2.yaml"  # type: ignore[operator]
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    updated = settings.model_copy(update={"scoring_config_file": path})

    second = run(scored, updated)
    assert second.version == "2.0.0"
    with session_scope(scored) as session:
        versions = set(session.scalars(select(Score.scoring_version)))
        total = session.scalar(select(func.count()).select_from(Score))
    assert versions == {"1.0.0", "2.0.0"}
    assert total == len(original.results) + len(second.results)


# --------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------
def test_inputs_carry_the_evidence_behind_a_score(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        entities = load_entity_inputs(session, result_run.score_date)
    inputs = next(entity for entity in entities.values() if entity.name.startswith("CRISPR"))
    assert inputs.sample_size > 0
    assert inputs.available_sources  # which sources actually had activity
    assert inputs.last_fetched_at is not None
    assert inputs.freshness(as_of=AS_OF) > 0.0
    assert "velocity" in inputs.normalized


def test_without_features_the_pass_says_what_to_run(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(factory, settings)
    assert result_run.results == []
    assert any("cews features" in warning for warning in result_run.warnings)


# --------------------------------------------------------------------------------------
# Trend and opportunity on the demo scenarios
# --------------------------------------------------------------------------------------
def test_growing_well_evidenced_topics_qualify_as_emerging_trends(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    """Which topics qualify depends on the sample, but what qualification means does not.

    Percentile ranking is relative, so on a smaller sample a different set of topics comes out
    on top. What must always hold is that anything called an emerging trend cleared every rule.
    """
    result_run = run(scored, settings)
    qualified = result_run.qualified(ScoreType.TREND.value)
    assert qualified, "no topic qualified; the rules cannot all be unsatisfiable"
    rules = load_scoring_config(settings).emerging_trend
    for trend in qualified:
        assert trend.value >= rules.min_trend_score
        assert trend.confidence >= rules.min_confidence
        assert trend.sample_size >= rules.min_sample_size
        assert trend.category in ("Emerging", "High priority")
        assert not trend.failed_gates


def test_the_false_trend_scores_high_but_never_qualifies(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    """The six-record topic has the highest momentum in the dataset. It is still not a finding."""
    result_run = run(scored, settings)
    trend = next(
        r for r in result_run.of_type(ScoreType.TREND.value) if r.entity_name.startswith("siRNA")
    )
    assert trend.value > 60.0  # the raw number looks impressive
    assert trend.qualified is False
    assert {"confidence_threshold", "minimum_evidence", "independent_sources"} <= set(
        trend.failed_gates
    )
    opportunity = next(
        r
        for r in result_run.of_type(ScoreType.OPPORTUNITY.value)
        if r.entity_name.startswith("siRNA")
    )
    assert opportunity.qualified is False


def test_a_declining_topic_scores_low(scored: sessionmaker[Session], settings: Settings) -> None:
    result_run = run(scored, settings)
    declining = next(
        r
        for r in result_run.of_type(ScoreType.TREND.value)
        if r.entity_name.startswith("Immune checkpoint")
    )
    assert declining.value < 50.0
    assert declining.qualified is False
    assert declining.category == "Low activity or declining"


def test_topics_and_competitors_each_get_their_own_scores(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    topics = {r.entity_name for r in result_run.results if r.entity_type == EntityType.TOPIC.value}
    for name in topics:
        kinds = {r.score_type for r in result_run.results if r.entity_name == name}
        assert kinds == {"confidence", "trend", "opportunity"}, name
    competitor_kinds = {
        r.score_type for r in result_run.results if r.entity_type == EntityType.COMPETITOR.value
    }
    assert competitor_kinds == {"confidence", "innovation", "threat"}


def test_every_stored_trend_score_can_be_explained(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        rows = load_scores(session, result_run.score_date, score_type=ScoreType.TREND.value)
    assert rows
    for row in rows:
        breakdown = row.component_json
        assert set(breakdown["components"]) == {
            "velocity",
            "momentum",
            "patent_growth",
            "funding_growth",
            "consistency",
        }
        assert breakdown["gates"] and "explanation" in breakdown
        contributions = sum(
            part["contribution"] for part in breakdown["components"].values() if part["available"]
        )
        assert contributions == pytest.approx(row.score_value, abs=0.05)


def test_qualified_counts_are_reported(scored: sessionmaker[Session], settings: Settings) -> None:
    summary = run(scored, settings).as_dict()
    assert summary["scores"]["trend"] > 0
    assert 0 <= summary["qualified"]["trend"] <= summary["scores"]["trend"]
    assert "confidence" not in summary["qualified"]


# --------------------------------------------------------------------------------------
# Competitor scores
# --------------------------------------------------------------------------------------
def test_every_competitor_gets_innovation_and_threat(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    result_run = run(scored, settings)
    competitors = {
        r.entity_id for r in result_run.results if r.entity_type == EntityType.COMPETITOR.value
    }
    innovation = {r.entity_id for r in result_run.of_type(ScoreType.INNOVATION.value)}
    overall_threat = {
        r.entity_id for r in result_run.of_type(ScoreType.THREAT.value) if not r.context_key
    }
    assert competitors and innovation == competitors == overall_threat


def test_competitors_are_ranked_by_innovation(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    ranked = run(scored, settings).of_type(ScoreType.INNOVATION.value)
    positions = [result.components["rank"].raw_value for result in ranked]
    assert positions == sorted(positions)  # highest score first means rank 1 first
    assert positions[0] == 1.0
    assert "Ranked 1 of" in " ".join(ranked[0].notes)


def test_threat_is_also_scored_within_each_area(
    rich_scored: tuple[sessionmaker[Session], Settings],
) -> None:
    factory, settings = rich_scored
    by_area = [r for r in run(factory, settings).of_type(ScoreType.THREAT.value) if r.context_key]
    assert by_area
    with session_scope(factory) as session:
        known = {area.key for area in session.scalars(select(TherapeuticArea))}
    for result in by_area:
        assert result.context_key in known  # the therapeutic area it is limited to
        assert " in " in result.entity_name
        assert "Limited to activity in" in " ".join(result.notes)


def test_the_demo_market_entry_is_credited_to_the_right_competitor(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    """Zentavia Pharma enters neurodegeneration in the demo data; nobody else does."""
    threats = [
        r
        for r in run(scored, settings).of_type(ScoreType.THREAT.value)
        if not r.context_key and "modifiers" in r.components
    ]
    entered = {
        r.entity_name
        for r in threats
        if any(
            item["name"] == "new_therapeutic_area_entry"
            for item in r.components["modifiers"].detail["applied"]
        )
    }
    assert entered == {"Zentavia Pharma"}


def test_modifiers_never_exceed_their_cap(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    config = load_scoring_config(settings)
    for result in run(scored, settings).of_type(ScoreType.THREAT.value):
        modifiers = result.components.get("modifiers")
        if modifiers is None:
            continue
        detail = modifiers.detail
        assert detail["total_points"] <= config.threat_modifiers.total_cap_points
        for item in detail["applied"]:
            assert item["points"] <= config.threat_modifiers.cap_for(item["name"])


def test_threat_never_claims_more_than_monitoring(
    scored: sessionmaker[Session], settings: Settings
) -> None:
    for result in run(scored, settings).of_type(ScoreType.THREAT.value):
        assert "not evidence of a legal, commercial or scientific threat" in " ".join(result.notes)


def test_stored_competitor_scores_carry_their_breakdown(
    rich_scored: tuple[sessionmaker[Session], Settings],
) -> None:
    scored, settings = rich_scored
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        innovation = load_scores(
            session, result_run.score_date, score_type=ScoreType.INNOVATION.value
        )
        threat = load_scores(session, result_run.score_date, score_type=ScoreType.THREAT.value)
    assert innovation and threat
    for row in innovation:
        assert set(row.component_json["components"]) >= {
            "patent",
            "clinical_trial",
            "publication",
            "funding",
        }
    assert any(row.context_key for row in threat)  # at least one is limited to an area
    assert all("explanation" in row.component_json for row in threat)


def test_area_scores_do_not_collide_with_overall_ones(
    rich_scored: tuple[sessionmaker[Session], Settings],
) -> None:
    """The same competitor has one overall threat score and one per area, kept apart."""
    scored, settings = rich_scored
    result_run = run(scored, settings)
    with session_scope(scored) as session:
        rows = load_scores(session, result_run.score_date, score_type=ScoreType.THREAT.value)
    keys = [(row.entity_id, row.context_key) for row in rows]
    assert len(keys) == len(set(keys))
    assert any(key[1] == "" for key in keys) and any(key[1] != "" for key in keys)
