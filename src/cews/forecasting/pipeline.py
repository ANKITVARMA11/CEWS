"""Running the forecasting and anomaly pass.

For every topic and monitored competitor this takes the monthly activity, picks a forecasting
model by backtest, stores the forecast with its interval, and records any unusual months with
what kind of unusual they were.

Both outputs are written with their reasoning: the forecast keeps every candidate model's
backtest error and why the winner won, and each anomaly keeps the range it broke out of and the
months it was compared against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType, SourceType, TopicStatus
from cews.database.models import Anomaly as AnomalyRow
from cews.database.models import Forecast, Organization, Topic
from cews.discovery.anomaly_detection import Anomaly, detect_activity_anomalies
from cews.features.activity_counts import monthly_series
from cews.features.time_windows import add_months, month_range, month_start
from cews.forecasting.model_selection import ModelSelection, select_best_model
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

HISTORY_MONTHS = 30
DEFAULT_HORIZON = 6
ANOMALY_METRIC = "monthly_activity"


@dataclass
class EntityForecast:
    """What the pass produced for one topic or competitor."""

    entity_type: str
    entity_id: int
    name: str
    months: list[date]
    series: list[float]
    selection: ModelSelection
    anomalies: list[Anomaly] = field(default_factory=list)

    @property
    def forecast_months(self) -> list[date]:
        """The months the forecast covers."""
        if not self.selection.forecast or not self.months:
            return []
        return [
            add_months(self.months[-1], step)
            for step in range(1, self.selection.forecast.horizon + 1)
        ]


@dataclass
class ForecastRun:
    """The outcome of one forecasting pass."""

    as_of: date
    horizon: int
    entities: list[EntityForecast] = field(default_factory=list)
    forecasts_written: int = 0
    forecasts_updated: int = 0
    anomalies_written: int = 0
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def anomalies(self) -> list[tuple[EntityForecast, Anomaly]]:
        """Every anomaly found, newest month first."""
        pairs = [(entity, found) for entity in self.entities for found in entity.anomalies]
        return sorted(pairs, key=lambda pair: pair[1].period or date.min, reverse=True)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        models: dict[str, int] = {}
        for entity in self.entities:
            if entity.selection.model:
                models[entity.selection.model] = models.get(entity.selection.model, 0) + 1
        kinds: dict[str, int] = {}
        for _, found in self.anomalies():
            kinds[found.kind.value] = kinds.get(found.kind.value, 0) + 1
        return {
            "as_of": self.as_of.isoformat(),
            "horizon_months": self.horizon,
            "entities": len(self.entities),
            "models_chosen": models,
            "forecasts_written": self.forecasts_written,
            "forecasts_updated": self.forecasts_updated,
            "anomalies": kinds,
            "anomalies_written": self.anomalies_written,
            "skipped": list(self.skipped),
            "warnings": list(self.warnings),
        }


def _quiet_months(by_source: dict[str, list[float]], months: int) -> list[int]:
    """Months where every source reported nothing, which suggests a collection gap."""
    quiet: list[int] = []
    for index in range(months):
        if all(values[index] == 0 for values in by_source.values()) and any(
            sum(values) > 0 for values in by_source.values()
        ):
            quiet.append(index)
    return quiet


def _store_forecast(
    session: Session,
    entity: EntityForecast,
    forecast_date: date,
    *,
    is_synthetic: bool,
    run: ForecastRun,
) -> None:
    selection = entity.selection
    if selection.forecast is None:
        return
    existing = {
        row.target_period: row
        for row in session.scalars(
            select(Forecast).where(
                Forecast.entity_type == entity.entity_type,
                Forecast.entity_id == entity.entity_id,
                Forecast.source_type == "all",
                Forecast.forecast_date == forecast_date,
            )
        )
    }
    metric = selection.metric
    for period, value, low, high in zip(
        entity.forecast_months,
        selection.forecast.predictions,
        selection.forecast.lower,
        selection.forecast.upper,
        strict=True,
    ):
        row = existing.pop(period, None)
        if row is None:
            session.add(
                Forecast(
                    entity_type=entity.entity_type,
                    entity_id=entity.entity_id,
                    source_type="all",
                    forecast_date=forecast_date,
                    target_period=period,
                    predicted_value=round(value, 4),
                    lower_bound=round(low, 4),
                    upper_bound=round(high, 4),
                    model_name=selection.model,
                    backtest_metric_name="mase",
                    backtest_metric=None if metric is None else round(metric, 4),
                    training_months=selection.forecast.training_months,
                    is_synthetic=is_synthetic,
                )
            )
            run.forecasts_written += 1
        else:
            row.predicted_value = round(value, 4)
            row.lower_bound = round(low, 4)
            row.upper_bound = round(high, 4)
            row.model_name = selection.model
            row.backtest_metric = None if metric is None else round(metric, 4)
            row.training_months = selection.forecast.training_months
            run.forecasts_updated += 1
    for row in existing.values():
        session.delete(row)


def _store_anomalies(
    session: Session, entity: EntityForecast, *, is_synthetic: bool, run: ForecastRun
) -> None:
    for found in entity.anomalies:
        if found.period is None:
            continue
        existing = session.scalars(
            select(AnomalyRow).where(
                AnomalyRow.entity_type == entity.entity_type,
                AnomalyRow.entity_id == entity.entity_id,
                AnomalyRow.anomaly_date == found.period,
                AnomalyRow.metric == ANOMALY_METRIC,
            )
        ).first()
        if existing is None:
            session.add(
                AnomalyRow(
                    anomaly_date=found.period,
                    entity_type=entity.entity_type,
                    entity_id=entity.entity_id,
                    metric=ANOMALY_METRIC,
                    observed_value=round(found.observed, 4),
                    expected_lower=round(found.expected_lower, 4),
                    expected_upper=round(found.expected_upper, 4),
                    deviation=round(found.deviation, 4),
                    method=found.method,
                    anomaly_class=found.kind.value,
                    confidence=round(found.confidence, 2),
                    evidence_json=found.as_dict(),
                    is_synthetic=is_synthetic,
                )
            )
            run.anomalies_written += 1
        else:
            existing.observed_value = round(found.observed, 4)
            existing.expected_lower = round(found.expected_lower, 4)
            existing.expected_upper = round(found.expected_upper, 4)
            existing.deviation = round(found.deviation, 4)
            existing.anomaly_class = found.kind.value
            existing.confidence = round(found.confidence, 2)
            existing.evidence_json = found.as_dict()


def run_forecasting(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime | date | None = None,
    horizon: int = DEFAULT_HORIZON,
    history_months: int = HISTORY_MONTHS,
    store: bool = True,
    is_synthetic: bool = False,
    detect_anomalies: bool = True,
) -> ForecastRun:
    """Forecast every topic and competitor, and record unusual months.

    Args:
        as_of: the last complete month to use (default: now; the month in progress is excluded).
        horizon: how many months ahead to forecast.
        history_months: how much history to read.
        store: write the results to the database.
        is_synthetic: mark stored rows as demo data.
        detect_anomalies: also look for unusual months.

    Raises:
        ValueError: for a horizon or history below 1.
    """
    if horizon < 1 or history_months < 1:
        raise ValueError("horizon and history_months must be positive")
    moment = as_of or datetime.now(UTC)
    last_complete = add_months(month_start(moment), -1)
    months = month_range(last_complete, history_months)
    run = ForecastRun(as_of=last_complete, horizon=horizon)

    topics = list(
        session.scalars(
            select(Topic).where(Topic.active.is_(True), Topic.status == TopicStatus.ACTIVE.value)
        )
    )
    competitors = list(
        session.scalars(
            select(Organization).where(
                (Organization.manually_included.is_(True))
                | (Organization.discovered_automatically.is_(True))
            )
        )
    )
    if not topics and not competitors:
        run.warnings.append("no topics or competitors to forecast; run: cews normalize")
        return run

    targets: list[tuple[str, int, str, dict[str, Any]]] = [
        (EntityType.TOPIC.value, topic.id, topic.canonical_name, {"topic_id": topic.id})
        for topic in topics
    ] + [
        (
            EntityType.COMPETITOR.value,
            organization.id,
            organization.canonical_name,
            {"organization_id": organization.id},
        )
        for organization in competitors
    ]

    for entity_type, entity_id, name, lookup in targets:
        by_source = monthly_series(
            session, months, source_types=[source.value for source in SourceType], **lookup
        )
        series = [
            sum(values[index] for values in by_source.values()) for index in range(len(months))
        ]
        if sum(series) <= 0:
            run.skipped.append(name)
            continue

        selection = select_best_model(series, horizon=horizon)
        entity = EntityForecast(
            entity_type=entity_type,
            entity_id=entity_id,
            name=name,
            months=list(months),
            series=series,
            selection=selection,
        )
        if selection.forecast is None:
            run.warnings.append(f"{name}: {selection.reason}")
        if detect_anomalies:
            entity.anomalies = detect_activity_anomalies(
                series, periods=months, quiet_months=_quiet_months(by_source, len(months))
            )
        run.entities.append(entity)

        if store:
            _store_forecast(session, entity, last_complete, is_synthetic=is_synthetic, run=run)
            _store_anomalies(session, entity, is_synthetic=is_synthetic, run=run)
    if store:
        session.flush()
    LOGGER.info("forecasting pass: %s", run.as_dict())
    return run
