"""Power BI export: a star schema written as CSV files.

CEWS does not generate a Power BI report file. It writes clean, typed tables that Power BI
Desktop loads from a folder (Get data > Folder or Text/CSV); the model, relationships and
measures are documented in ``dashboards/powerbi/``. Streamlit stays the primary dashboard.

The files, all under one output directory:

    dimensions   dim_date, dim_organization, dim_topic, dim_source
    facts        fact_activity, fact_scores, fact_score_components, fact_forecasts,
                 fact_anomalies, fact_insights, fact_evidence, fact_ingestion_runs,
                 fact_evaluations
    metadata     refresh_metadata.json  (written last: its arrival means the export is complete)

Choices that matter to whoever builds a report on top:

* **Nothing is computed here.** Every value is read from what the pipeline stored.
* **Synthetic data is never anonymous.** Every fact row carries ``is_synthetic`` and the
  metadata file states the data origin, so a report can show a warning banner.
* **No nested JSON.** Anything stored as a structure (score components, evaluation metrics,
  evidence) is flattened into rows and columns.
* **Deterministic.** Rows are ordered by key, so exporting unchanged data twice gives
  byte-identical CSV files.
* **Missing means blank.** A blank ``organization_id`` or ``topic_id`` means "not about a
  specific one" (an activity total across all topics, say), never zero.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import EntityType
from cews.dashboard.queries import data_origin, latest_score_date
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
from cews.exports.csv_export import Column, write_csv

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
METADATA_FILE = "refresh_metadata.json"
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)  # fmt: skip
MAX_METRIC_DEPTH = 6

Rows = Callable[[Session], Iterator[dict[str, Any]]]


def _c(*pairs: tuple[str, str]) -> tuple[Column, ...]:
    return tuple(Column(name, kind) for name, kind in pairs)


@dataclass(frozen=True)
class TableSpec:
    """One exported table: its file, its typed columns and the function that yields its rows."""

    name: str
    kind: str  # "dimension" or "fact"
    columns: tuple[Column, ...]
    rows: Rows

    @property
    def filename(self) -> str:
        """The CSV file this table is written to."""
        return f"{self.name}.csv"


@dataclass
class ExportResult:
    """What one export wrote."""

    output_directory: Path
    generated_at: datetime
    is_synthetic: bool
    data_origin: str
    row_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "output_directory": str(self.output_directory),
            "generated_at": self.generated_at.isoformat(),
            "is_synthetic": self.is_synthetic,
            "data_origin": self.data_origin,
            "row_counts": dict(self.row_counts),
        }


def _split_entity(entity_type: str, entity_id: int) -> tuple[int | None, int | None]:
    """(organization_id, topic_id): an entity is one or the other, so both columns can be
    related directly to their dimension without a bridge table."""
    if entity_type == EntityType.COMPETITOR.value:
        return entity_id, None
    if entity_type == EntityType.TOPIC.value:
        return None, entity_id
    return None, None


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    return value.date() if isinstance(value, datetime) else value


# ----------------------------------------------------------------------------------------
# Dimensions
# ----------------------------------------------------------------------------------------
def _date_bounds(session: Session) -> tuple[date, date] | None:
    """The first and last date that appears anywhere in the exported facts."""
    candidates: list[date] = []
    columns: tuple[Any, ...] = (
        ActivityAggregate.period,
        Score.score_date,
        Forecast.forecast_date,
        Forecast.target_period,
        Anomaly.anomaly_date,
        Insight.insight_date,
        EvaluationRun.evaluation_date,
        SourceRecord.published_at,
        IngestionRun.start_time,
    )
    for column in columns:
        low, high = session.execute(select(func.min(column), func.max(column))).one()
        candidates += [value for value in (_as_date(low), _as_date(high)) if value is not None]
    return (min(candidates), max(candidates)) if candidates else None


def _dim_date(session: Session) -> Iterator[dict[str, Any]]:
    bounds = _date_bounds(session)
    if bounds is None:
        return
    day = date(bounds[0].year, bounds[0].month, 1)
    last = (
        date(bounds[1].year, 12, 31)
        if bounds[1].month == 12
        else (date(bounds[1].year, bounds[1].month + 1, 1) - timedelta(days=1))
    )
    while day <= last:
        quarter = (day.month - 1) // 3 + 1
        yield {
            "date": day,
            "date_key": day.year * 10000 + day.month * 100 + day.day,
            "year": day.year,
            "quarter": quarter,
            "quarter_label": f"{day.year} Q{quarter}",
            "month_number": day.month,
            "month_name": MONTH_NAMES[day.month - 1],
            "year_month": f"{day.year}-{day.month:02d}",
            "month_start": date(day.year, day.month, 1),
            "day_of_month": day.day,
            "is_month_start": day.day == 1,
        }
        day += timedelta(days=1)


def _dim_organization(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(Organization).order_by(Organization.id)):
        included = row.manually_included or row.discovered_automatically
        yield {
            "organization_id": row.id,
            "organization_name": row.canonical_name,
            "organization_type": row.organization_type,
            "country": row.country,
            "website": row.website,
            "parent_organization_id": row.parent_id,
            "is_monitored": included and not row.manually_excluded,
            "is_manually_included": row.manually_included,
            "is_manually_excluded": row.manually_excluded,
            "is_synthetic": row.is_synthetic,
        }


def _dim_topic(session: Session) -> Iterator[dict[str, Any]]:
    areas = {row.id: row for row in session.scalars(select(TherapeuticArea))}
    for row in session.scalars(select(Topic).order_by(Topic.id)):
        area = areas.get(row.therapeutic_area_id) if row.therapeutic_area_id else None
        parent_area = areas.get(area.parent_id) if area and area.parent_id else None
        yield {
            "topic_id": row.id,
            "topic_key": row.key,
            "topic_name": row.canonical_name,
            "topic_type": row.topic_type,
            "parent_topic_id": row.parent_id,
            "therapeutic_area_key": area.key if area else None,
            "therapeutic_area": area.canonical_name if area else None,
            "parent_therapeutic_area": parent_area.canonical_name if parent_area else None,
            "status": row.status,
            "is_active": row.active,
            "is_ai_candidate": row.topic_type == "ai_candidate",
        }


def _dim_source(session: Session) -> Iterator[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"types": set(), "records": 0, "first": None, "last": None, "synthetic": False}
    )
    for source, record_type, count, first, last, synthetic in session.execute(
        select(
            SourceRecord.source,
            SourceRecord.record_type,
            func.count(),
            func.min(SourceRecord.published_at),
            func.max(SourceRecord.published_at),
            func.max(SourceRecord.is_synthetic),
        ).group_by(SourceRecord.source, SourceRecord.record_type)
    ):
        entry = stats[source]
        entry["types"].add(record_type)
        entry["records"] += count
        entry["synthetic"] = entry["synthetic"] or bool(synthetic)
        for key, value in (("first", first), ("last", last)):
            day = _as_date(value)
            if day is not None:
                pick = min if key == "first" else max
                entry[key] = day if entry[key] is None else pick(entry[key], day)
    for (source,) in session.execute(select(IngestionRun.source).distinct()):
        stats[source]  # a source that has run but yielded nothing still belongs in the list
    for source in sorted(stats):
        entry = stats[source]
        yield {
            "source_name": source,
            "record_types": "; ".join(sorted(entry["types"])),
            "records": entry["records"],
            "first_published": entry["first"],
            "last_published": entry["last"],
            "is_synthetic": entry["synthetic"],
        }


# ----------------------------------------------------------------------------------------
# Facts
# ----------------------------------------------------------------------------------------
def _fact_activity(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(
        select(ActivityAggregate).order_by(
            ActivityAggregate.period,
            ActivityAggregate.period_type,
            ActivityAggregate.organization_id,
            ActivityAggregate.topic_id,
            ActivityAggregate.source_type,
        )
    ):
        yield {
            "period": row.period,
            "period_type": row.period_type,
            "organization_id": row.organization_id,
            "topic_id": row.topic_id,
            "source_type": row.source_type,
            "activity_count": row.activity_count,
            "weighted_activity": row.weighted_activity,
            "unique_record_count": row.unique_record_count,
            "is_synthetic": row.is_synthetic,
        }


def _fact_scores(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(Score).order_by(Score.id)):
        organization_id, topic_id = _split_entity(row.entity_type, row.entity_id)
        breakdown = row.component_json or {}
        yield {
            "score_id": row.id,
            "score_date": row.score_date,
            "entity_type": row.entity_type,
            "organization_id": organization_id,
            "topic_id": topic_id,
            "context_key": row.context_key,
            "score_type": row.score_type,
            "score_value": row.score_value,
            "confidence_score": row.confidence_score,
            "category": breakdown.get("category"),
            "is_qualified": bool(breakdown.get("qualified", False)),
            "sample_size": breakdown.get("sample_size"),
            "explanation": breakdown.get("explanation"),
            "scoring_version": row.scoring_version,
            "is_synthetic": row.is_synthetic,
        }


def _fact_score_components(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(Score).order_by(Score.id)):
        components = (row.component_json or {}).get("components", {})
        for name in sorted(components):
            detail = components[name]
            if not isinstance(detail, dict):
                continue
            yield {
                "score_id": row.id,
                "component": name,
                "raw_value": detail.get("raw_value"),
                "normalized_value": detail.get("normalized"),
                "weight": detail.get("weight"),
                "points": detail.get("contribution"),
                "is_available": bool(detail.get("available", False)),
            }


def _fact_forecasts(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(Forecast).order_by(Forecast.id)):
        organization_id, topic_id = _split_entity(row.entity_type, row.entity_id)
        yield {
            "forecast_id": row.id,
            "forecast_date": row.forecast_date,
            "target_period": row.target_period,
            "entity_type": row.entity_type,
            "organization_id": organization_id,
            "topic_id": topic_id,
            "source_type": row.source_type,
            "predicted_value": row.predicted_value,
            "lower_bound": row.lower_bound,
            "upper_bound": row.upper_bound,
            "model_name": row.model_name,
            "backtest_metric_name": row.backtest_metric_name,
            "backtest_metric": row.backtest_metric,
            "training_months": row.training_months,
            "is_synthetic": row.is_synthetic,
        }


def _fact_anomalies(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(Anomaly).order_by(Anomaly.id)):
        organization_id, topic_id = _split_entity(row.entity_type, row.entity_id)
        yield {
            "anomaly_id": row.id,
            "anomaly_date": row.anomaly_date,
            "entity_type": row.entity_type,
            "organization_id": organization_id,
            "topic_id": topic_id,
            "metric": row.metric,
            "observed_value": row.observed_value,
            "expected_lower": row.expected_lower,
            "expected_upper": row.expected_upper,
            "deviation": row.deviation,
            "method": row.method,
            "anomaly_class": row.anomaly_class,
            "confidence": row.confidence,
            "explanation": (row.evidence_json or {}).get("explanation"),
            "is_synthetic": row.is_synthetic,
        }


def _fact_insights(session: Session) -> Iterator[dict[str, Any]]:
    counts: dict[int, int] = {
        insight_id: total
        for insight_id, total in session.execute(
            select(InsightEvidence.insight_id, func.count()).group_by(InsightEvidence.insight_id)
        ).all()
    }
    for row in session.scalars(select(Insight).order_by(Insight.id)):
        organization_id, topic_id = _split_entity(row.entity_type, row.entity_id)
        yield {
            "insight_id": row.id,
            "insight_date": row.insight_date,
            "insight_type": row.insight_type,
            "severity": row.severity,
            "entity_type": row.entity_type,
            "organization_id": organization_id,
            "topic_id": topic_id,
            "title": row.title,
            "observed_fact": row.observed_fact,
            "interpretation": row.interpretation,
            "recommended_review": row.recommended_review,
            "confidence_score": row.confidence_score,
            "status": row.status,
            "evidence_count": counts.get(row.id, 0),
            "is_synthetic": row.is_synthetic,
        }


def _fact_evidence(session: Session) -> Iterator[dict[str, Any]]:
    query = (
        select(InsightEvidence, SourceRecord)
        .join(SourceRecord, SourceRecord.id == InsightEvidence.source_record_id)
        .order_by(InsightEvidence.id)
    )
    for link, record in session.execute(query):
        yield {
            "evidence_id": link.id,
            "insight_id": link.insight_id,
            "source_record_id": record.id,
            "source_name": record.source,
            "record_type": record.record_type,
            "source_identifier": record.source_record_id,
            "title": record.title,
            "published_date": _as_date(record.published_at),
            "url": record.source_url,
            "is_synthetic": record.is_synthetic,
        }


def _fact_ingestion_runs(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(IngestionRun).order_by(IngestionRun.id)):
        seconds = (
            (row.end_time - row.start_time).total_seconds()
            if row.end_time is not None and row.start_time is not None
            else None
        )
        yield {
            "run_id": row.id,
            "job_id": row.job_id,
            "source_name": row.source,
            "start_time": row.start_time,
            "end_time": row.end_time,
            "duration_seconds": seconds,
            "status": row.status,
            "records_requested": row.records_requested,
            "records_received": row.records_received,
            "records_inserted": row.records_inserted,
            "records_updated": row.records_updated,
            "records_skipped": row.records_skipped,
            "duplicate_count": row.duplicate_count,
            "error_count": row.error_count,
            "warning_count": len(row.warnings_json or []),
            "collection_mode": row.collection_mode,
            "error_summary": row.error_summary,
            "is_synthetic": row.is_synthetic,
        }


def flatten_metrics(value: Any, path: str = "", depth: int = 0) -> Iterator[tuple[str, Any]]:
    """Turn nested evaluation metrics into (dotted path, scalar) pairs.

    Lists of plain values become one semicolon-joined text value; lists of structures are left
    out (they are detail that belongs in the evaluation report, not in a metrics table).
    """
    if depth > MAX_METRIC_DEPTH or value is None:
        return
    if isinstance(value, dict):
        for key in sorted(value):
            yield from flatten_metrics(value[key], f"{path}.{key}" if path else str(key), depth + 1)
    elif isinstance(value, list):
        if value and all(not isinstance(item, dict | list) for item in value):
            yield path, "; ".join(str(item) for item in value)
    else:
        yield path, value


def _fact_evaluations(session: Session) -> Iterator[dict[str, Any]]:
    for row in session.scalars(select(EvaluationRun).order_by(EvaluationRun.id)):
        for path, value in flatten_metrics(row.metrics_json or {}):
            is_number = isinstance(value, int | float) and not isinstance(value, bool)
            yield {
                "evaluation_id": row.evaluation_id,
                "evaluation_type": row.evaluation_type,
                "evaluation_date": row.evaluation_date,
                "period_start": row.period_start,
                "period_end": row.period_end,
                "metric_path": path,
                "value_number": value if is_number else None,
                "value_text": (
                    None
                    if is_number
                    else str(value).lower() if isinstance(value, bool) else str(value)
                ),
                "is_synthetic": row.is_synthetic,
            }


# ----------------------------------------------------------------------------------------
# The schema
# ----------------------------------------------------------------------------------------
TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "dim_date", "dimension",
        _c(("date", "date"), ("date_key", "int"), ("year", "int"), ("quarter", "int"),
           ("quarter_label", "text"), ("month_number", "int"), ("month_name", "text"),
           ("year_month", "text"), ("month_start", "date"), ("day_of_month", "int"),
           ("is_month_start", "flag")),
        _dim_date,
    ),
    TableSpec(
        "dim_organization", "dimension",
        _c(("organization_id", "int"), ("organization_name", "text"), ("organization_type", "text"),
           ("country", "text"), ("website", "text"), ("parent_organization_id", "int"),
           ("is_monitored", "flag"), ("is_manually_included", "flag"),
           ("is_manually_excluded", "flag"), ("is_synthetic", "flag")),
        _dim_organization,
    ),
    TableSpec(
        "dim_topic", "dimension",
        _c(("topic_id", "int"), ("topic_key", "text"), ("topic_name", "text"), ("topic_type", "text"),
           ("parent_topic_id", "int"), ("therapeutic_area_key", "text"), ("therapeutic_area", "text"),
           ("parent_therapeutic_area", "text"), ("status", "text"), ("is_active", "flag"),
           ("is_ai_candidate", "flag")),
        _dim_topic,
    ),
    TableSpec(
        "dim_source", "dimension",
        _c(("source_name", "text"), ("record_types", "text"), ("records", "int"),
           ("first_published", "date"), ("last_published", "date"), ("is_synthetic", "flag")),
        _dim_source,
    ),
    TableSpec(
        "fact_activity", "fact",
        _c(("period", "date"), ("period_type", "text"), ("organization_id", "int"),
           ("topic_id", "int"), ("source_type", "text"), ("activity_count", "int"),
           ("weighted_activity", "float"), ("unique_record_count", "int"), ("is_synthetic", "flag")),
        _fact_activity,
    ),
    TableSpec(
        "fact_scores", "fact",
        _c(("score_id", "int"), ("score_date", "date"), ("entity_type", "text"),
           ("organization_id", "int"), ("topic_id", "int"), ("context_key", "text"),
           ("score_type", "text"), ("score_value", "float"), ("confidence_score", "float"),
           ("category", "text"), ("is_qualified", "flag"), ("sample_size", "float"),
           ("explanation", "text"), ("scoring_version", "text"), ("is_synthetic", "flag")),
        _fact_scores,
    ),
    TableSpec(
        "fact_score_components", "fact",
        _c(("score_id", "int"), ("component", "text"), ("raw_value", "float"),
           ("normalized_value", "float"), ("weight", "float"), ("points", "float"),
           ("is_available", "flag")),
        _fact_score_components,
    ),
    TableSpec(
        "fact_forecasts", "fact",
        _c(("forecast_id", "int"), ("forecast_date", "date"), ("target_period", "date"),
           ("entity_type", "text"), ("organization_id", "int"), ("topic_id", "int"),
           ("source_type", "text"), ("predicted_value", "float"), ("lower_bound", "float"),
           ("upper_bound", "float"), ("model_name", "text"), ("backtest_metric_name", "text"),
           ("backtest_metric", "float"), ("training_months", "int"), ("is_synthetic", "flag")),
        _fact_forecasts,
    ),
    TableSpec(
        "fact_anomalies", "fact",
        _c(("anomaly_id", "int"), ("anomaly_date", "date"), ("entity_type", "text"),
           ("organization_id", "int"), ("topic_id", "int"), ("metric", "text"),
           ("observed_value", "float"), ("expected_lower", "float"), ("expected_upper", "float"),
           ("deviation", "float"), ("method", "text"), ("anomaly_class", "text"),
           ("confidence", "float"), ("explanation", "text"), ("is_synthetic", "flag")),
        _fact_anomalies,
    ),
    TableSpec(
        "fact_insights", "fact",
        _c(("insight_id", "int"), ("insight_date", "date"), ("insight_type", "text"),
           ("severity", "text"), ("entity_type", "text"), ("organization_id", "int"),
           ("topic_id", "int"), ("title", "text"), ("observed_fact", "text"),
           ("interpretation", "text"), ("recommended_review", "text"),
           ("confidence_score", "float"), ("status", "text"), ("evidence_count", "int"),
           ("is_synthetic", "flag")),
        _fact_insights,
    ),
    TableSpec(
        "fact_evidence", "fact",
        _c(("evidence_id", "int"), ("insight_id", "int"), ("source_record_id", "int"),
           ("source_name", "text"), ("record_type", "text"), ("source_identifier", "text"),
           ("title", "text"), ("published_date", "date"), ("url", "text"), ("is_synthetic", "flag")),
        _fact_evidence,
    ),
    TableSpec(
        "fact_ingestion_runs", "fact",
        _c(("run_id", "int"), ("job_id", "text"), ("source_name", "text"),
           ("start_time", "datetime"), ("end_time", "datetime"), ("duration_seconds", "float"),
           ("status", "text"), ("records_requested", "int"), ("records_received", "int"),
           ("records_inserted", "int"), ("records_updated", "int"), ("records_skipped", "int"),
           ("duplicate_count", "int"), ("error_count", "int"), ("warning_count", "int"),
           ("collection_mode", "text"), ("error_summary", "text"), ("is_synthetic", "flag")),
        _fact_ingestion_runs,
    ),
    TableSpec(
        "fact_evaluations", "fact",
        _c(("evaluation_id", "text"), ("evaluation_type", "text"), ("evaluation_date", "date"),
           ("period_start", "date"), ("period_end", "date"), ("metric_path", "text"),
           ("value_number", "float"), ("value_text", "text"), ("is_synthetic", "flag")),
        _fact_evaluations,
    ),
)  # fmt: skip
TABLE_NAMES: tuple[str, ...] = tuple(spec.name for spec in TABLES)


def export_powerbi(
    session: Session, output_directory: Path, *, generated_at: datetime | None = None
) -> ExportResult:
    """Write every table and the refresh metadata to ``output_directory``.

    The metadata file is written last and atomically, so a reader that sees it knows every table
    beside it is complete for this refresh. Files from an earlier export are replaced, not
    accumulated; other files in the directory are left alone.

    Raises:
        ExportError: if a value cannot be written as its column's declared type.
        OSError: if the directory cannot be written to.
    """
    moment = generated_at or datetime.now(UTC)
    origin = data_origin(session)
    result = ExportResult(
        output_directory=output_directory,
        generated_at=moment,
        is_synthetic=origin.is_demo,
        data_origin=origin.label,
    )
    latest = latest_score_date(session)
    # Remove the previous completion marker first: if this export fails part-way, the folder must
    # not still claim to be complete while holding a mixture of old and new tables.
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / METADATA_FILE).unlink(missing_ok=True)
    tables: dict[str, Any] = {}
    for spec in TABLES:
        count = write_csv(output_directory / spec.filename, spec.columns, spec.rows(session))
        result.row_counts[spec.name] = count
        tables[spec.name] = {
            "file": spec.filename,
            "kind": spec.kind,
            "rows": count,
            "columns": [{"name": column.name, "type": column.type} for column in spec.columns],
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_origin": origin.label,
        "is_synthetic": origin.is_demo,
        "latest_score_date": latest.isoformat() if latest else None,
        "conventions": {
            "dates": "YYYY-MM-DD",
            "timestamps": "ISO 8601, UTC",
            "flags": "1 or 0",
            "missing_value": "empty cell",
            "decimal_separator": ".",
            "encoding": "UTF-8 with byte-order mark",
        },
        "tables": tables,
    }
    temporary = output_directory / (METADATA_FILE + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    os.replace(temporary, output_directory / METADATA_FILE)
    LOGGER.info("power bi export: %s", result.as_dict())
    return result
