"""Unit tests for the Power BI star-schema export."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import (
    ActivityAggregate,
    Anomaly,
    EvaluationRun,
    Forecast,
    IngestionRun,
    Insight,
    InsightEvidence,
    Organization,
    Score,
    SourceRecord,
    TherapeuticArea,
    Topic,
)
from cews.exports import powerbi_export
from cews.exports.csv_export import ExportError
from cews.exports.powerbi_export import (
    METADATA_FILE,
    TABLE_NAMES,
    TABLES,
    export_powerbi,
    flatten_metrics,
)

pytestmark = pytest.mark.unit

DAY = date(2026, 8, 1)
SPEC_TABLES = {
    "dim_date", "dim_organization", "dim_topic", "dim_source", "fact_activity", "fact_scores",
    "fact_forecasts", "fact_insights", "fact_evidence", "fact_ingestion_runs", "fact_evaluations",
}  # fmt: skip


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def read_table(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / f"{name}.csv").open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def seed(session: Session, *, title: str = "A paper", synthetic: bool = True) -> dict[str, int]:
    area = TherapeuticArea(key="onc", canonical_name="Oncology")
    session.add(area)
    session.flush()
    topic = Topic(
        key="car_t", canonical_name="CAR-T", topic_type="technology", therapeutic_area_id=area.id
    )
    org = Organization(
        canonical_name="Acme Bio",
        normalized_name="acme bio",
        is_synthetic=synthetic,
        discovered_automatically=True,
    )
    session.add_all([topic, org])
    session.flush()
    record = SourceRecord(
        source="synthetic_patents",
        source_record_id="P1",
        record_type="patent",
        title=title,
        fetched_at=datetime(2026, 8, 2, tzinfo=UTC),
        published_at=datetime(2026, 7, 15, tzinfo=UTC),
        content_hash="a" * 64,
        is_synthetic=synthetic,
        source_url="https://example.invalid/p1",
    )
    session.add(record)
    session.flush()
    session.add(
        ActivityAggregate(
            period=DAY,
            period_type="month",
            topic_id=topic.id,
            source_type="patent",
            activity_count=4,
            weighted_activity=3.5,
            unique_record_count=4,
            is_synthetic=synthetic,
        )
    )
    session.add(
        ActivityAggregate(
            period=DAY,
            period_type="month",
            organization_id=org.id,
            topic_id=None,
            source_type="patent",
            activity_count=2,
            weighted_activity=2.0,
            unique_record_count=2,
            is_synthetic=synthetic,
        )
    )
    score = Score(
        score_date=DAY,
        entity_type="topic",
        entity_id=topic.id,
        score_type="trend",
        score_value=71.5,
        confidence_score=88.0,
        scoring_version="1.0.0",
        is_synthetic=synthetic,
        component_json={
            "category": "Emerging",
            "qualified": True,
            "sample_size": 57.27,
            "explanation": "It scores well.",
            "components": {
                "velocity": {
                    "raw_value": 0.2,
                    "normalized": 80.0,
                    "weight": 0.5,
                    "contribution": 40.0,
                    "available": True,
                },
                "patent_growth": {
                    "raw_value": None,
                    "normalized": 0.0,
                    "weight": 0.0,
                    "contribution": 0.0,
                    "available": False,
                },
            },
        },
    )
    session.add(score)
    session.add(
        Forecast(
            entity_type="topic",
            entity_id=topic.id,
            forecast_date=DAY,
            target_period=date(2026, 9, 1),
            predicted_value=5.0,
            lower_bound=2.0,
            upper_bound=8.0,
            model_name="holt_winters",
            backtest_metric_name="mase",
            backtest_metric=0.7,
            training_months=24,
            is_synthetic=synthetic,
        )
    )
    session.add(
        Anomaly(
            anomaly_date=DAY,
            entity_type="competitor",
            entity_id=org.id,
            metric="activity",
            observed_value=20.0,
            expected_lower=2.0,
            expected_upper=9.0,
            deviation=6.0,
            method="robust_zscore",
            anomaly_class="one_time_spike",
            confidence=90.0,
            evidence_json={"explanation": "A spike."},
            is_synthetic=synthetic,
        )
    )
    insight = Insight(
        insight_date=DAY,
        severity="watch",
        insight_type="emerging_trend",
        entity_type="topic",
        entity_id=topic.id,
        title="CAR-T: emerging trend",
        observed_fact="fact",
        interpretation="means",
        recommended_review="check",
        confidence_score=88.0,
        is_synthetic=synthetic,
    )
    session.add(insight)
    session.flush()
    session.add(InsightEvidence(insight_id=insight.id, source_record_id=record.id))
    session.add(
        IngestionRun(
            job_id="j1",
            source="synthetic_patents",
            start_time=datetime(2026, 8, 2, 1, 0, tzinfo=UTC),
            end_time=datetime(2026, 8, 2, 1, 5, tzinfo=UTC),
            status="succeeded",
            records_inserted=4,
            warnings_json=["w"],
            is_synthetic=synthetic,
        )
    )
    session.add(
        EvaluationRun(
            evaluation_id="e1-backtest",
            evaluation_type="backtest",
            evaluation_date=DAY,
            configuration_json={},
            is_synthetic=synthetic,
            metrics_json={
                "folds": 12,
                "mean_precision_at_k": 0.48,
                "warnings": ["a", "b"],
                "nested": {"passed": True},
                "fold_detail": [{"big": 1}],
            },
        )
    )
    session.flush()
    return {"topic": topic.id, "org": org.id, "insight": insight.id, "score": score.id}


# --------------------------------------------------------------------------------------
# What is written
# --------------------------------------------------------------------------------------
def test_every_table_the_specification_names_is_exported() -> None:
    assert set(TABLE_NAMES) >= SPEC_TABLES


def test_the_two_extra_tables_are_present() -> None:
    assert {"fact_score_components", "fact_anomalies"} <= set(TABLE_NAMES)


def test_every_table_and_the_metadata_file_is_written(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {f"{n}.csv" for n in TABLE_NAMES} | {
        METADATA_FILE
    }


def test_headers_match_the_declared_columns_in_order(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    for spec in TABLES:
        header = (tmp_path / spec.filename).read_text(encoding="utf-8-sig").splitlines()[0]
        assert header.split(",") == [column.name for column in spec.columns], spec.name


def test_an_empty_database_still_produces_every_file_with_headers(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        result = export_powerbi(session, tmp_path)
    assert all(count == 0 for count in result.row_counts.values())
    for spec in TABLES:
        lines = (tmp_path / spec.filename).read_text(encoding="utf-8-sig").splitlines()
        assert len(lines) == 1 and lines[0].startswith(spec.columns[0].name), spec.name


def test_row_counts_match_what_was_seeded(factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_scope(factory) as session:
        seed(session)
        result = export_powerbi(session, tmp_path)
    counts = result.row_counts
    assert (
        counts["dim_organization"] == 1
        and counts["dim_topic"] == 1
        and counts["dim_source"] == 1  # one source, seen in both its records and its runs
    )
    assert (
        counts["fact_activity"] == 2
        and counts["fact_scores"] == 1
        and counts["fact_score_components"] == 2
    )
    assert counts["fact_forecasts"] == 1 and counts["fact_anomalies"] == 1
    assert (
        counts["fact_insights"] == 1
        and counts["fact_evidence"] == 1
        and counts["fact_ingestion_runs"] == 1
    )


# --------------------------------------------------------------------------------------
# Values and conventions
# --------------------------------------------------------------------------------------
def test_scores_are_flattened_with_category_and_qualification(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        ids = seed(session)
        export_powerbi(session, tmp_path)
    row = read_table(tmp_path, "fact_scores")[0]
    assert (
        row["score_value"] == "71.5"
        and row["category"] == "Emerging"
        and row["is_qualified"] == "1"
    )
    assert row["sample_size"] == "57.27"  # weighted record counts are not whole numbers
    assert row["topic_id"] == str(ids["topic"]) and row["organization_id"] == ""


def test_score_components_become_rows(factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    rows = {r["component"]: r for r in read_table(tmp_path, "fact_score_components")}
    assert rows["velocity"]["points"] == "40.0" and rows["velocity"]["is_available"] == "1"
    assert rows["patent_growth"]["raw_value"] == "" and rows["patent_growth"]["is_available"] == "0"


def test_an_entity_is_split_into_organization_or_topic(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        ids = seed(session)
        export_powerbi(session, tmp_path)
    anomaly = read_table(tmp_path, "fact_anomalies")[0]
    assert anomaly["organization_id"] == str(ids["org"]) and anomaly["topic_id"] == ""
    assert anomaly["explanation"] == "A spike."


def test_an_activity_total_across_all_topics_has_a_blank_topic_not_zero(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    rows = read_table(tmp_path, "fact_activity")
    org_total = next(r for r in rows if r["organization_id"])
    assert org_total["topic_id"] == "" and org_total["activity_count"] == "2"
    topic_row = next(r for r in rows if r["topic_id"])
    assert topic_row["organization_id"] == "" and topic_row["activity_count"] == "4"


def test_flags_are_only_one_or_zero(factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    for spec in TABLES:
        flag_columns = [c.name for c in spec.columns if c.type == "flag"]
        for row in read_table(tmp_path, spec.name):
            for name in flag_columns:
                assert row[name] in {"0", "1"}, (spec.name, name)


def test_the_topic_dimension_carries_its_therapeutic_area(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    row = read_table(tmp_path, "dim_topic")[0]
    assert (
        row["therapeutic_area"] == "Oncology"
        and row["is_active"] == "1"
        and row["is_ai_candidate"] == "0"
    )


def test_a_monitored_organization_is_marked_and_an_excluded_one_is_not(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        session.add(
            Organization(
                canonical_name="Excluded",
                normalized_name="excluded",
                discovered_automatically=True,
                manually_excluded=True,
            )
        )
        export_powerbi(session, tmp_path)
    monitored = {
        r["organization_name"]: r["is_monitored"] for r in read_table(tmp_path, "dim_organization")
    }
    assert monitored == {"Acme Bio": "1", "Excluded": "0"}


def test_the_ingestion_run_gets_a_duration_and_warning_count(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    row = read_table(tmp_path, "fact_ingestion_runs")[0]
    assert row["duration_seconds"] == "300.0" and row["warning_count"] == "1"
    assert row["start_time"] == "2026-08-02T01:00:00Z"


# --------------------------------------------------------------------------------------
# Flattening: no nested JSON anywhere
# --------------------------------------------------------------------------------------
def test_nested_metrics_are_flattened_to_dotted_paths() -> None:
    pairs = dict(
        flatten_metrics({"a": {"b": 1, "c": [1, 2]}, "d": "x", "e": [{"skip": 1}], "f": None})
    )
    assert pairs == {"a.b": 1, "a.c": "1; 2", "d": "x"}


def test_flattening_stops_at_a_sensible_depth() -> None:
    deep: dict[str, Any] = {}
    node = deep
    for _ in range(20):
        node["k"] = {}
        node = node["k"]
    node["leaf"] = 1
    assert list(flatten_metrics(deep)) == []


def test_evaluation_metrics_land_as_numbers_or_text(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    rows = {r["metric_path"]: r for r in read_table(tmp_path, "fact_evaluations")}
    assert float(rows["folds"]["value_number"]) == 12.0
    assert (
        rows["nested.passed"]["value_text"] == "true"
        and rows["nested.passed"]["value_number"] == ""
    )
    assert rows["warnings"]["value_text"] == "a; b"
    assert "fold_detail" not in rows  # a list of structures is detail, not a metric


def test_no_cell_holds_a_json_structure(factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    for spec in TABLES:
        for row in read_table(tmp_path, spec.name):
            for value in row.values():
                assert not value.startswith(("{", "[")), (spec.name, value)


def test_insight_evidence_is_flattened_into_its_own_table(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        ids = seed(session)
        export_powerbi(session, tmp_path)
    evidence = read_table(tmp_path, "fact_evidence")[0]
    assert evidence["insight_id"] == str(ids["insight"]) and evidence["source_identifier"] == "P1"
    assert evidence["published_date"] == "2026-07-15" and evidence["record_type"] == "patent"
    assert read_table(tmp_path, "fact_insights")[0]["evidence_count"] == "1"


def test_a_hostile_record_title_cannot_run_as_a_formula(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session, title='=HYPERLINK("http://evil","click")\nsecond line')
        export_powerbi(session, tmp_path)
    title = read_table(tmp_path, "fact_evidence")[0]["title"]
    assert title.startswith("'=") and "\n" not in title


# --------------------------------------------------------------------------------------
# Relationships: the model has to join up
# --------------------------------------------------------------------------------------
def test_every_key_in_a_fact_exists_in_its_dimension(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    orgs = {r["organization_id"] for r in read_table(tmp_path, "dim_organization")}
    topics = {r["topic_id"] for r in read_table(tmp_path, "dim_topic")}
    for name in (
        "fact_activity",
        "fact_scores",
        "fact_forecasts",
        "fact_anomalies",
        "fact_insights",
    ):
        for row in read_table(tmp_path, name):
            assert row["organization_id"] in orgs | {""}, name
            assert row["topic_id"] in topics | {""}, name


def test_every_date_in_a_fact_exists_in_the_date_dimension(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    dates = {r["date"] for r in read_table(tmp_path, "dim_date")}
    for name, columns in {
        "fact_activity": ["period"],
        "fact_scores": ["score_date"],
        "fact_forecasts": ["forecast_date", "target_period"],
        "fact_anomalies": ["anomaly_date"],
        "fact_insights": ["insight_date"],
        "fact_evidence": ["published_date"],
        "fact_evaluations": ["evaluation_date"],
    }.items():
        for row in read_table(tmp_path, name):
            for column in columns:
                assert row[column] in dates | {""}, (name, column, row[column])


def test_the_date_dimension_is_continuous_and_whole_months(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    rows = read_table(tmp_path, "dim_date")
    days = [date.fromisoformat(r["date"]) for r in rows]
    assert days == sorted(days) and len(days) == len(set(days))
    assert all((b - a).days == 1 for a, b in zip(days, days[1:], strict=False))
    assert days[0].day == 1  # starts on the first of a month
    assert (days[-1] + timedelta(days=1)).day == 1  # ends on the last day of a month
    assert rows[0]["month_name"] == "July" and rows[0]["is_month_start"] == "1"
    assert rows[-1]["date"] == "2026-09-30"


def test_every_score_component_belongs_to_a_score(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
    score_ids = {r["score_id"] for r in read_table(tmp_path, "fact_scores")}
    assert {r["score_id"] for r in read_table(tmp_path, "fact_score_components")} <= score_ids


# --------------------------------------------------------------------------------------
# Synthetic data is never anonymous
# --------------------------------------------------------------------------------------
def test_synthetic_data_is_flagged_on_every_fact_row_and_in_the_metadata(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session, synthetic=True)
        export_powerbi(session, tmp_path)
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["is_synthetic"] is True and "SYNTHETIC" in metadata["data_origin"]
    for spec in TABLES:
        if spec.kind == "fact" and any(c.name == "is_synthetic" for c in spec.columns):
            assert {r["is_synthetic"] for r in read_table(tmp_path, spec.name)} <= {"1"}, spec.name


def test_live_data_is_not_flagged(factory: sessionmaker[Session], tmp_path: Path) -> None:
    with session_scope(factory) as session:
        seed(session, synthetic=False)
        export_powerbi(session, tmp_path)
    assert (
        json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))["is_synthetic"] is False
    )
    assert {r["is_synthetic"] for r in read_table(tmp_path, "fact_scores")} == {"0"}


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------
def test_the_metadata_describes_every_table_with_typed_columns(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        result = export_powerbi(
            session, tmp_path, generated_at=datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
        )
    metadata = json.loads((tmp_path / METADATA_FILE).read_text(encoding="utf-8"))
    assert (
        metadata["schema_version"] == "1.0" and metadata["generated_at"] == "2026-09-01T12:00:00Z"
    )
    assert metadata["latest_score_date"] == "2026-08-01"
    assert set(metadata["tables"]) == set(TABLE_NAMES)
    scores = metadata["tables"]["fact_scores"]
    assert scores["rows"] == result.row_counts["fact_scores"]
    assert {"name": "score_value", "type": "float"} in scores["columns"]
    assert metadata["conventions"]["flags"] == "1 or 0"


# --------------------------------------------------------------------------------------
# Repeatability and failure behaviour
# --------------------------------------------------------------------------------------
def test_exporting_unchanged_data_twice_gives_identical_files(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path / "one")
        export_powerbi(session, tmp_path / "two")
    for spec in TABLES:
        assert (tmp_path / "one" / spec.filename).read_bytes() == (
            tmp_path / "two" / spec.filename
        ).read_bytes()


def test_a_second_export_replaces_the_first_rather_than_accumulating(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
        export_powerbi(session, tmp_path)
    assert len(read_table(tmp_path, "fact_scores")) == 1


def test_other_files_in_the_folder_are_left_alone(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    (tmp_path / "report.pbix").write_text("mine", encoding="utf-8")
    with session_scope(factory) as session:
        export_powerbi(session, tmp_path)
    assert (tmp_path / "report.pbix").read_text(encoding="utf-8") == "mine"


def test_a_failed_export_leaves_no_completion_marker(
    factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder without refresh_metadata.json must be read as incomplete, never as current."""
    with session_scope(factory) as session:
        seed(session)
        export_powerbi(session, tmp_path)
        assert (tmp_path / METADATA_FILE).exists()

        def broken(_session: Session) -> Iterator[dict[str, Any]]:
            yield {"score_id": "not a number"}

        patched = tuple(
            (
                spec
                if spec.name != "fact_scores"
                else powerbi_export.TableSpec(spec.name, spec.kind, spec.columns, broken)
            )
            for spec in TABLES
        )
        monkeypatch.setattr(powerbi_export, "TABLES", patched)
        with pytest.raises(ExportError):
            export_powerbi(session, tmp_path)
    assert not (tmp_path / METADATA_FILE).exists()
