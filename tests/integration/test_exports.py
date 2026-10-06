"""Integration tests for the Power BI export, on data built by the real pipeline.

The unit tests check each rule on a handful of hand-made rows; these check that the exported
model actually joins up at realistic scale, and that the command behaves as a person running it
would expect.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from cews.cli import EXIT_FAILURE, EXIT_OK, main
from cews.exports.powerbi_export import METADATA_FILE, TABLE_NAMES, TABLES
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An env file for a project whose database holds every kind of result."""
    root = tmp_path_factory.mktemp("export_project")
    env = root / ".env"
    env.write_text(
        f"PROJECT_ROOT={root}\nSQLITE_PATH=./data/cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n"
        f"SOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        f"BENCHMARK_TOPICS_FILE={SCORING_FILE.parent / 'benchmark_topics.yaml'}\n"
        f"ANNOUNCEMENT_LABELS_FILE={SCORING_FILE.parent / 'announcement_eval_labels.yaml'}\n"
        "ENABLE_CLINICAL_TRIALS_GOV=false\nENABLE_PUBMED=false\nENABLE_EUROPE_PMC=false\n",
        encoding="utf-8",
    )
    for command in (
        ["db-init"],
        ["seed-demo", "--scale", "0.2"],
        ["normalize"],
        ["competitors"],
        ["features", "--as-of", "2026-09-01"],
        ["score", "--as-of", "2026-09-01"],
        ["forecast", "--as-of", "2026-09-01", "--horizon", "3"],
        ["insights"],
        ["evaluate", "--only", "data_quality", "alerts"],
    ):
        assert main([command[0], "--env-file", str(env), *command[1:]]) == EXIT_OK, command
    return env


@pytest.fixture(scope="module")
def exported(project: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("powerbi")
    assert main(["export-powerbi", "--env-file", str(project), "--out", str(target)]) == EXIT_OK
    return target


def table(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / f"{name}.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def database_hashes(project: Path) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{project.parent / 'data' / 'cews.db'}?mode=ro", uri=True)
    try:
        names = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        result = {}
        for name in names:
            digest = hashlib.sha256()
            for row in connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid'):  # noqa: S608
                digest.update(repr(row).encode())
            result[name] = digest.hexdigest()
        return result
    finally:
        connection.close()


def test_every_table_is_written_and_populated_from_real_results(exported: Path) -> None:
    metadata = json.loads((exported / METADATA_FILE).read_text(encoding="utf-8"))
    assert set(metadata["tables"]) == set(TABLE_NAMES)
    for name in (
        "dim_date", "dim_organization", "dim_topic", "dim_source", "fact_activity", "fact_scores",
        "fact_score_components", "fact_forecasts", "fact_anomalies", "fact_insights",
        "fact_evidence", "fact_ingestion_runs", "fact_evaluations",
    ):  # fmt: skip
        if name == "fact_ingestion_runs":
            continue  # demo data is seeded, not fetched, so no collection run exists to report
        assert metadata["tables"][name]["rows"] > 0, name


def test_row_counts_match_the_database(project: Path, exported: Path) -> None:
    connection = sqlite3.connect(f"file:{project.parent / 'data' / 'cews.db'}?mode=ro", uri=True)
    try:
        for export_name, source_table in {
            "fact_scores": "scores", "fact_forecasts": "forecasts", "fact_anomalies": "anomalies",
            "fact_insights": "insights", "fact_evidence": "insight_evidence",
            "fact_activity": "activity_aggregates", "dim_organization": "organizations",
        }.items():  # fmt: skip
            expected = connection.execute(f"SELECT COUNT(*) FROM {source_table}").fetchone()[
                0
            ]  # noqa: S608
            assert len(table(exported, export_name)) == expected, export_name
    finally:
        connection.close()


def test_the_model_joins_up_at_realistic_scale(exported: Path) -> None:
    organizations = {r["organization_id"] for r in table(exported, "dim_organization")}
    topics = {r["topic_id"] for r in table(exported, "dim_topic")}
    dates = {r["date"] for r in table(exported, "dim_date")}
    scores = {r["score_id"] for r in table(exported, "fact_scores")}
    insights = {r["insight_id"] for r in table(exported, "fact_insights")}
    date_columns = {
        "fact_activity": ["period"], "fact_scores": ["score_date"],
        "fact_forecasts": ["forecast_date", "target_period"], "fact_anomalies": ["anomaly_date"],
        "fact_insights": ["insight_date"], "fact_evidence": ["published_date"],
        "fact_evaluations": ["evaluation_date", "period_start", "period_end"],
    }  # fmt: skip
    for name in (
        "fact_activity",
        "fact_scores",
        "fact_forecasts",
        "fact_anomalies",
        "fact_insights",
    ):
        for row in table(exported, name):
            assert row["organization_id"] in organizations | {""}, name
            assert row["topic_id"] in topics | {""}, name
    for name, columns in date_columns.items():
        for row in table(exported, name):
            for column in columns:
                assert row[column] in dates | {""}, (name, column)
    assert {r["score_id"] for r in table(exported, "fact_score_components")} <= scores
    assert {r["insight_id"] for r in table(exported, "fact_evidence")} <= insights


def test_every_insight_has_the_evidence_it_claims(exported: Path) -> None:
    counts: dict[str, int] = {}
    for row in table(exported, "fact_evidence"):
        counts[row["insight_id"]] = counts.get(row["insight_id"], 0) + 1
    for insight in table(exported, "fact_insights"):
        assert int(insight["evidence_count"]) == counts.get(insight["insight_id"], 0) > 0


def test_demo_data_is_marked_as_such_everywhere(exported: Path) -> None:
    metadata = json.loads((exported / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["is_synthetic"] is True and "SYNTHETIC" in metadata["data_origin"]
    for name in ("fact_scores", "fact_activity", "fact_insights", "fact_forecasts"):
        assert {r["is_synthetic"] for r in table(exported, name)} == {"1"}, name


def test_every_cell_matches_its_declared_type(exported: Path) -> None:
    for spec in TABLES:
        types = {column.name: column.type for column in spec.columns}
        for row in table(exported, spec.name):
            for name, value in row.items():
                if value == "":
                    continue
                kind = types[name]
                if kind == "int":
                    int(value)
                elif kind == "float":
                    float(value)
                elif kind == "flag":
                    assert value in {"0", "1"}
                elif kind == "date":
                    assert len(value) == 10 and value[4] == "-" and value[7] == "-"
                elif kind == "datetime":
                    assert value.endswith("Z") and "T" in value


def test_a_second_export_of_unchanged_data_is_byte_identical(
    project: Path, exported: Path, tmp_path: Path
) -> None:
    assert main(["export-powerbi", "--env-file", str(project), "--out", str(tmp_path)]) == EXIT_OK
    for spec in TABLES:
        assert (tmp_path / spec.filename).read_bytes() == (
            exported / spec.filename
        ).read_bytes(), spec.name


def test_exporting_never_changes_the_database(project: Path, tmp_path: Path) -> None:
    before = database_hashes(project)
    assert main(["export-powerbi", "--env-file", str(project), "--out", str(tmp_path)]) == EXIT_OK
    assert database_hashes(project) == before


def test_the_command_prints_a_summary_and_says_the_data_is_synthetic(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["export-powerbi", "--env-file", str(project), "--out", str(tmp_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "wrote 13 tables" in out and "fact_scores" in out and "SYNTHETIC DEMO DATA" in out
    assert "refresh_instructions.md" in out


def test_json_output_is_pure_json(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(["export-powerbi", "--env-file", str(project), "--out", str(tmp_path), "--json"])
        == EXIT_OK
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["row_counts"]["fact_scores"] > 0 and payload["is_synthetic"] is True


def test_the_default_folder_is_under_the_export_directory(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["export-powerbi", "--env-file", str(project)]) == EXIT_OK
    assert (project.parent / "data" / "exports" / "powerbi" / METADATA_FILE).is_file()
    capsys.readouterr()


def test_a_missing_database_is_a_clear_failure_not_a_new_empty_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n",
        encoding="utf-8",
    )
    assert main(["export-powerbi", "--env-file", str(env)]) == EXIT_FAILURE
    assert "cews db-init" in capsys.readouterr().out
    assert not (tmp_path / "absent.db").exists()


def test_an_unwritable_destination_is_reported(
    project: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "a_file"
    blocker.write_text("not a folder", encoding="utf-8")
    assert (
        main(["export-powerbi", "--env-file", str(project), "--out", str(blocker)]) == EXIT_FAILURE
    )
    assert "export failed" in capsys.readouterr().out
