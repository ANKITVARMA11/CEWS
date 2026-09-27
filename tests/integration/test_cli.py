"""Integration tests for the command-line interface (each test uses its own project root)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import func, select

from cews import __version__
from cews.cli import EXIT_CONFIG, EXIT_FAILURE, EXIT_OK, main
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import head_revision
from cews.database.models import SourceRecord, Topic
from cews.database.repositories import SourceRecordData, upsert_source_record
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """A project root of its own. Sources that would make network calls are off."""
    path = tmp_path / ".env"
    path.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./data/cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n"
        f"SOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "ENABLE_CLINICAL_TRIALS_GOV=false\nENABLE_PUBMED=false\nENABLE_EUROPE_PMC=false\n",
        encoding="utf-8",
    )
    return path


def enable(env_file: Path, *flags: str) -> None:
    """Turn settings on in an existing env file (the value replaces any earlier line)."""
    with env_file.open("a", encoding="utf-8") as handle:
        for flag in flags:
            handle.write(f"{flag}=true\n")


def _run(env_file: Path, *args: str) -> int:
    command, *rest = args
    return main([command, "--env-file", str(env_file), *rest])


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_check_env_before_and_after_init(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "check-env") == EXIT_OK
    out = capsys.readouterr().out
    assert "configuration is valid" in out and "not created yet" in out
    assert _run(env_file, "db-init") == EXIT_OK
    capsys.readouterr()
    assert _run(env_file, "check-env") == EXIT_OK
    assert "database reachable" in capsys.readouterr().out


def test_db_init_creates_schema_and_taxonomy(
    env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "db-init") == EXIT_OK
    out = capsys.readouterr().out
    # 8 taxonomy topics plus one topic per therapeutic area
    assert f"revision {head_revision()}" in out and "16 topic(s)" in out
    assert (tmp_path / "data" / "cews.db").is_file()
    assert _run(env_file, "db-init") == EXIT_OK  # idempotent
    assert "0 topic(s)" in capsys.readouterr().out


def test_db_init_can_skip_taxonomy(env_file: Path, tmp_path: Path) -> None:
    assert _run(env_file, "db-init", "--no-taxonomy") == EXIT_OK
    factory = create_session_factory(create_db_engine(f"sqlite:///{tmp_path / 'data' / 'cews.db'}"))
    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(Topic)) == 0


def test_db_status_reports_counts(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(env_file, "db-status") == EXIT_FAILURE  # not initialized yet
    assert "run db-init" in capsys.readouterr().out
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.1")
    capsys.readouterr()
    assert _run(env_file, "db-status") == EXIT_OK
    out = capsys.readouterr().out
    assert "up to date" in out and "synthetic" in out and "last run" in out


def test_seed_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "seed-demo") == EXIT_FAILURE
    assert "run db-init first" in capsys.readouterr().out


def test_seed_is_idempotent_and_labelled(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "seed-demo", "--scale", "0.1") == EXIT_OK
    first = capsys.readouterr().out
    assert "SYNTHETIC" in first and "inserted" in first
    assert _run(env_file, "seed-demo", "--scale", "0.1") == EXIT_OK
    assert "0 inserted, 0 updated" in capsys.readouterr().out


def test_seed_writes_fixture_files(env_file: Path, tmp_path: Path) -> None:
    _run(env_file, "db-init")
    assert _run(env_file, "seed-demo", "--scale", "0.1", "--write-fixtures") == EXIT_OK
    written = sorted(p.name for p in (tmp_path / "data" / "samples" / "demo").iterdir())
    assert "scenarios.json" in written and "synthetic_patents.jsonl" in written


def test_seed_refuses_to_mix_with_live_data(
    env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    factory = create_session_factory(create_db_engine(f"sqlite:///{tmp_path / 'data' / 'cews.db'}"))
    from datetime import UTC, datetime

    with session_scope(factory) as session:
        upsert_source_record(
            session,
            SourceRecordData(
                source="pubmed",
                source_record_id="1",
                record_type="publication",
                content_hash="a" * 64,
                fetched_at=datetime(2026, 9, 1, tzinfo=UTC),
            ),
        )
    capsys.readouterr()
    assert _run(env_file, "seed-demo", "--scale", "0.1") == EXIT_FAILURE
    assert "refusing" in capsys.readouterr().out
    assert _run(env_file, "seed-demo", "--scale", "0.1", "--allow-mixed") == EXIT_OK


def test_reset_requires_confirmation_and_keeps_live_data(
    env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.1")
    capsys.readouterr()
    assert _run(env_file, "reset-demo") == EXIT_CONFIG
    assert "--yes" in capsys.readouterr().out
    factory = create_session_factory(create_db_engine(f"sqlite:///{tmp_path / 'data' / 'cews.db'}"))
    with session_scope(factory) as session:
        assert (session.scalar(select(func.count()).select_from(SourceRecord)) or 0) > 100
    assert _run(env_file, "reset-demo", "--yes") == EXIT_OK
    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(SourceRecord)) == 0


def test_reset_on_missing_database_is_a_no_op(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "reset-demo", "--yes") == EXIT_OK
    assert "nothing to reset" in capsys.readouterr().out


def test_invalid_configuration_exits_with_config_code(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file.write_text(
        env_file.read_text(encoding="utf-8") + "FETCH_INTERVAL_MINUTES=0\n", encoding="utf-8"
    )
    assert _run(env_file, "check-env") == EXIT_CONFIG
    assert "FETCH_INTERVAL_MINUTES" in capsys.readouterr().out
    assert _run(env_file, "db-init") == EXIT_CONFIG


def test_missing_env_file_exits_with_config_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["db-init", "--env-file", str(tmp_path / "absent.env")]) == EXIT_CONFIG
    assert "not found" in capsys.readouterr().out


def test_invalid_demo_parameters_are_reported(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    assert _run(env_file, "seed-demo", "--months", "3") == EXIT_FAILURE
    assert "months must be" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# sources and fetch (Phase 3: the framework exists, no real adapter is registered yet)
# --------------------------------------------------------------------------------------
def test_sources_lists_the_registry(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "sources") == EXIT_OK
    out = capsys.readouterr().out
    for source in ("clinical_trials_gov", "pubmed", "europe_pmc", "openalex", "patents_uspto_bulk"):
        assert source in out
    assert "not yet" in out and "never run" in out


def test_sources_works_before_db_init(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(env_file, "sources") == EXIT_OK
    assert "never run" in capsys.readouterr().out


def test_sources_shows_which_adapters_exist(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "sources") == EXIT_OK
    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines() if line.strip()}
    assert "yes" in lines["clinical_trials_gov"]  # adapter implemented in Phase 4
    assert "not yet" in lines["openalex"]  # still to come
    assert "not yet" in lines["patents_uspto_bulk"]


def test_sources_health_reports_unconfigured_feeds(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only generic_rss is enabled here, and it has no feeds, so nothing is contacted.
    assert _run(env_file, "sources", "--health") == EXIT_OK
    assert "no URL configured" in capsys.readouterr().out


def test_fetch_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "fetch") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_fetch_skips_a_source_that_is_not_configured(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "fetch") == EXIT_OK  # only generic_rss is enabled, and it has no feeds
    out = capsys.readouterr().out
    assert "no feeds configured" in out
    assert "Result: skipped (0 of 1 source(s) ran)" in out


def test_fetch_refuses_to_add_live_data_to_the_demo_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.1")
    enable(env_file, "ENABLE_CLINICAL_TRIALS_GOV")
    capsys.readouterr()
    # The guard runs before any request, so no source is contacted.
    assert _run(env_file, "fetch") == EXIT_FAILURE
    out = capsys.readouterr().out
    assert "refusing to add live records" in out and "SQLITE_PATH" in out


def test_fetch_dry_run_and_explicit_disabled_source(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "fetch", "--dry-run", "--source", "openalex") == EXIT_OK
    out = capsys.readouterr().out
    assert "dry run" in out and "ENABLE_OPENALEX=false" in out


def test_fetch_unknown_source_is_a_usage_error(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "fetch", "--source", "nope") == EXIT_CONFIG
    assert "unknown source" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------------------
def test_features_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "features") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_features_reports_activity_and_ranks_entities(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.2")
    _run(env_file, "normalize")
    _run(env_file, "competitors")
    capsys.readouterr()
    assert _run(env_file, "features", "--as-of", "2026-09-01", "--top", "5") == EXIT_OK
    out = capsys.readouterr().out
    assert "record(s) over" in out and "month(s)" in out
    assert "velocity" in out and "momentum" in out
    assert "thin evidence" in out  # the demo's six-record topic is flagged, not hidden


def test_features_json_output(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.2")
    _run(env_file, "normalize")
    capsys.readouterr()
    assert _run(env_file, "features", "--as-of", "2026-09-01", "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["activity"]["records_counted"] > 0
    assert payload["features"]["settings"]["velocity_window_months"] == 12


def test_features_rejects_a_bad_date(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "features", "--as-of", "last tuesday") == EXIT_FAILURE
    assert "--as-of must be a date" in capsys.readouterr().out


def test_features_without_data_explains_what_to_run(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "features") == EXIT_OK
    assert "no records with a publication date" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------------------
def _prepare_scores(env_file: Path) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.2")
    _run(env_file, "normalize")
    _run(env_file, "competitors")
    _run(env_file, "features", "--as-of", "2026-09-01")


def test_score_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "score") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_score_without_features_says_what_to_run(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "score") == EXIT_OK
    assert "cews features" in capsys.readouterr().out


def test_score_reports_trends_and_opportunities(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--top", "40") == EXIT_OK
    out = capsys.readouterr().out
    assert "Emerging trends (passed every rule)" in out
    assert "Topics by trend score" in out and "opportunity score" in out
    assert "not a recommendation" in out
    assert "siRNA" in out  # scores high, shown with the rules it fails rather than hidden


def test_score_reports_competitor_scores(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--type", "threat") == EXIT_OK
    out = capsys.readouterr().out
    assert "monitoring priority" in out
    assert "not a legal or commercial claim" in out


def test_score_shows_innovation_ranking(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--type", "innovation") == EXIT_OK
    out = capsys.readouterr().out
    assert "innovation score" in out and "research output, not growth" in out


def test_score_can_show_one_kind_of_score(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--type", "trend") == EXIT_OK
    out = capsys.readouterr().out
    assert "Topics by trend score" in out and "opportunity score" not in out


def test_score_flags_results_that_should_not_be_acted_on(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--low-confidence") == EXIT_OK
    out = capsys.readouterr().out
    assert "should not be acted on" in out and "siRNA" in out


def test_score_json_includes_the_explanation(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["scoring_version"] == "1.0.0"
    assert payload["results"] and "explanation" in payload["results"][0]


def test_score_can_run_without_saving(env_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01", "--no-store") == EXIT_OK
    assert "0 written" in capsys.readouterr().out


def test_a_broken_scoring_configuration_is_a_usage_error(
    env_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    from support import SCORING_FILE

    _prepare_scores(env_file)
    document = yaml.safe_load(SCORING_FILE.read_text(encoding="utf-8"))
    document["weight_sets"]["trend_score"]["velocity"] = 0.9  # no longer sums to 1
    broken = tmp_path / "broken_scoring.yaml"
    broken.write_text(yaml.safe_dump(document), encoding="utf-8")
    env_file.write_text(
        env_file.read_text(encoding="utf-8").replace(str(SCORING_FILE), str(broken)),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert _run(env_file, "score", "--as-of", "2026-09-01") == EXIT_CONFIG
    assert "scoring configuration problem" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# forecast
# --------------------------------------------------------------------------------------
def test_forecast_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "forecast") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_forecast_reports_models_and_unusual_months(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert _run(env_file, "forecast", "--as-of", "2026-09-01", "--horizon", "3") == EXIT_OK
    out = capsys.readouterr().out
    assert "models chosen" in out and "written" in out
    assert "next 3 month(s)" in out
    assert "beat 'next month looks like this month'" in out


def test_forecast_can_skip_the_anomaly_pass(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert (
        _run(env_file, "forecast", "--as-of", "2026-09-01", "--horizon", "2", "--no-anomalies")
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "models chosen" in out and "kind" not in out


def test_forecast_json_carries_the_model_comparison(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert (
        _run(env_file, "forecast", "--as-of", "2026-09-01", "--horizon", "2", "--json") == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["run"]["models_chosen"]
    assert payload["forecasts"] and payload["forecasts"][0]["reason"]
    assert len(payload["forecasts"][0]["predictions"]) == 2


def test_forecast_can_run_without_saving(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_scores(env_file)
    capsys.readouterr()
    assert (
        _run(env_file, "forecast", "--as-of", "2026-09-01", "--horizon", "2", "--no-store")
        == EXIT_OK
    )
    assert "0 written" in capsys.readouterr().out


def test_forecast_without_data_says_what_to_run(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "forecast") == EXIT_OK
    out = capsys.readouterr().out
    assert "nothing to forecast" in out and "normalize" in out


# --------------------------------------------------------------------------------------
# extract-announcements
# --------------------------------------------------------------------------------------
def _prepare_normalized(env_file: Path) -> None:
    _run(env_file, "db-init")
    _run(env_file, "seed-demo", "--scale", "0.2")
    _run(env_file, "normalize")


def test_extract_announcements_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "extract-announcements") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_extract_announcements_classifies_with_the_keyword_fallback(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "extract-announcements") == EXIT_OK
    out = capsys.readouterr().out
    assert "ENABLE_AI_ANNOUNCEMENT_EXTRACTION=false" in out
    assert "classified" in out and "keyword=" in out
    assert "by type" in out


def test_extract_announcements_is_idempotent(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "extract-announcements") == EXIT_OK
    capsys.readouterr()
    assert _run(env_file, "extract-announcements") == EXIT_OK
    assert "no unclassified announcements found" in capsys.readouterr().out


def test_extract_announcements_json_output(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "extract-announcements", "--limit", "5", "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload
    assert len(payload) <= 5
    assert {"record_id", "announcement_type", "method"} <= set(payload[0])


def test_extract_announcements_json_is_pure_json_even_when_disabled(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--json must never be preceded by a plain-text note; a caller piping it into a JSON
    parser would otherwise break."""
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "extract-announcements", "--limit", "3", "--json") == EXIT_OK
    out = capsys.readouterr().out
    assert "note:" not in out
    json.loads(out)  # raises if anything but JSON was printed


def test_extract_announcements_json_with_nothing_pending_is_an_empty_array(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "extract-announcements", "--json") == EXIT_OK
    assert json.loads(capsys.readouterr().out) == []


def test_extract_announcements_can_run_without_saving(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "extract-announcements", "--limit", "3", "--no-store") == EXIT_OK
    capsys.readouterr()
    # nothing was saved, so a second run should still find the same records unclassified
    assert _run(env_file, "extract-announcements", "--limit", "3", "--json") == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 3


def test_extract_announcements_with_nothing_pending(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "extract-announcements") == EXIT_OK
    assert "no unclassified announcements found" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# discover-topics
# --------------------------------------------------------------------------------------
def test_discover_topics_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "discover-topics") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_discover_topics_with_too_few_unmatched_records(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fewer unmatched records than the minimum cluster size never needs the embedding model."""
    _run(env_file, "db-init")
    capsys.readouterr()
    assert _run(env_file, "discover-topics") == EXIT_OK
    out = capsys.readouterr().out
    assert "ENABLE_AI_TOPIC_DISCOVERY=false" in out
    assert "considered 0 unmatched record(s)" in out
    assert "new candidate(s)          : 0" in out


def test_discover_topics_wiring_with_a_stubbed_embedder(
    env_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercises the CLI's own plumbing (query, flags, printing) without a real model download."""
    import cews.ai.topic_discovery as topic_discovery_module
    from cews.ai.topic_discovery import DiscoveryRun, TopicCandidate

    _prepare_normalized(env_file)

    stub_run = DiscoveryRun(
        candidates=[
            TopicCandidate(
                label="Targeted Protein Degradation",
                terms=("protac", "degrader"),
                record_ids=(1, 2, 3, 4, 5),
                novelty=1.1,
                nearest_existing_topic=None,
                nearest_existing_similarity=0.0,
            )
        ],
        records_considered=190,
        records_clustered=5,
        clusters_matching_existing_topics=2,
        topics_created=1,
    )

    def fake_discover(
        session: object, settings: object, texts_by_record: object, **kwargs: object
    ) -> DiscoveryRun:
        assert texts_by_record  # the CLI did build a real text map from the database
        return stub_run

    monkeypatch.setattr(topic_discovery_module, "discover_topic_candidates", fake_discover)
    capsys.readouterr()
    assert _run(env_file, "discover-topics") == EXIT_OK
    out = capsys.readouterr().out
    assert "Targeted Protein Degradation" in out
    assert "new candidate(s)          : 1" in out
    assert "cews review" in out


def test_discover_topics_json_output(
    env_file: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import cews.ai.topic_discovery as topic_discovery_module
    from cews.ai.topic_discovery import DiscoveryRun, TopicCandidate

    _prepare_normalized(env_file)
    stub_run = DiscoveryRun(
        candidates=[
            TopicCandidate(
                label="Novel Cluster",
                terms=("novel",),
                record_ids=(1, 2, 3),
                novelty=0.9,
                nearest_existing_topic="CRISPR gene editing",
                nearest_existing_similarity=0.2,
            )
        ],
        records_considered=10,
        records_clustered=3,
        topics_created=1,
    )
    monkeypatch.setattr(
        topic_discovery_module,
        "discover_topic_candidates",
        lambda *args, **kwargs: stub_run,
    )
    capsys.readouterr()
    assert _run(env_file, "discover-topics", "--json") == EXIT_OK
    out = capsys.readouterr().out
    assert "note:" not in out
    payload = json.loads(out)
    assert payload["run"]["topics_created"] == 1
    assert payload["candidates"][0]["label"] == "Novel Cluster"


# --------------------------------------------------------------------------------------
# review approve / reject
# --------------------------------------------------------------------------------------
def _seed_topic_candidate(env_file: Path) -> int:
    """Insert one pending ai_topic_candidate item directly, bypassing the embedding model.

    Returns the review item's id.
    """
    from cews.constants import TopicStatus
    from cews.database.connection import create_db_engine, create_session_factory, session_scope
    from cews.database.models import ReviewQueueItem, Topic
    from cews.settings import load_settings

    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    with session_scope(factory) as session:
        topic = Topic(
            key="ai_candidate_protac_test",
            canonical_name="Protac, Disorders, Genetic",
            topic_type="ai_candidate",
            status=TopicStatus.PENDING_REVIEW.value,
            active=False,
        )
        session.add(topic)
        session.flush()
        item = ReviewQueueItem(
            queue_type="ai_topic_candidate",
            subject_ref=f"topic:{topic.id}",
            payload_json={"label": topic.canonical_name, "record_count": 10, "novelty": 1.2},
            status="pending",
        )
        session.add(item)
        session.flush()
        return item.id


def test_review_lists_a_discovered_topic_candidate_by_its_label(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    _seed_topic_candidate(env_file)
    capsys.readouterr()
    assert _run(env_file, "review") == EXIT_OK
    out = capsys.readouterr().out
    assert "Protac, Disorders, Genetic" in out  # not the raw "topic:N" subject_ref
    assert "record_count: 10" in out
    assert "cews review approve" in out


def test_review_approve_activates_the_candidate_topic(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cews.database.connection import create_db_engine, create_session_factory, session_scope
    from cews.database.models import Topic
    from cews.settings import load_settings

    _prepare_normalized(env_file)
    item_id = _seed_topic_candidate(env_file)
    capsys.readouterr()
    assert _run(env_file, "review", "approve", str(item_id)) == EXIT_OK
    out = capsys.readouterr().out
    assert "accepted" in out and "activated topic" in out

    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    with session_scope(factory) as session:
        topic = session.query(Topic).filter_by(key="ai_candidate_protac_test").one()
        assert topic.active is True


def test_review_reject_leaves_the_candidate_topic_inactive(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cews.database.connection import create_db_engine, create_session_factory, session_scope
    from cews.database.models import Topic
    from cews.settings import load_settings

    _prepare_normalized(env_file)
    item_id = _seed_topic_candidate(env_file)
    capsys.readouterr()
    assert _run(env_file, "review", "reject", str(item_id)) == EXIT_OK
    assert "rejected" in capsys.readouterr().out

    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    with session_scope(factory) as session:
        topic = session.query(Topic).filter_by(key="ai_candidate_protac_test").one()
        assert topic.active is False


def test_review_approve_on_an_organization_item_takes_no_data_action(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    _run(env_file, "review")
    listing = capsys.readouterr().out
    first_id = listing.split("[", 2)[1].split("]")[0]
    capsys.readouterr()
    assert _run(env_file, "review", "approve", first_id) == EXIT_OK
    assert "no automatic action" in capsys.readouterr().out


def test_review_approve_without_an_id_shows_usage(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "review", "approve") == EXIT_FAILURE
    assert "usage: cews review approve" in capsys.readouterr().out


def test_review_approve_a_missing_id_is_an_error(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    capsys.readouterr()
    assert _run(env_file, "review", "approve", "999999") == EXIT_FAILURE
    assert "no review item" in capsys.readouterr().out


def test_review_approve_twice_is_refused_the_second_time(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _prepare_normalized(env_file)
    item_id = _seed_topic_candidate(env_file)
    _run(env_file, "review", "approve", str(item_id))
    capsys.readouterr()
    assert _run(env_file, "review", "approve", str(item_id)) == EXIT_FAILURE
    assert "already accepted" in capsys.readouterr().out


def test_review_requires_an_initialized_database(
    env_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(env_file, "review") == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out
    assert _run(env_file, "review", "approve", "1") == EXIT_FAILURE
