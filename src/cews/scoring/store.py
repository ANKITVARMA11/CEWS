"""Saving scores, with the working shown.

A score row carries the number, its confidence, and the full breakdown that produced it:
component values, weights, contributions, which inputs were missing, the rules it passed or
failed, the inputs it was built from, and the scoring version. That is what lets the dashboard
answer "why is this 78?" and what makes an old score readable after the weights change.

Rows are keyed by date, entity, context, score type and scoring version, so re-running a day
rewrites its scores instead of stacking duplicates, while a run under different weights (a new
version) is kept alongside rather than overwriting history.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.database.models import Score
from cews.scoring.base import ScoreResult

LOGGER = logging.getLogger(__name__)

# (score_date, entity_type, entity_id, context_key, score_type, scoring_version)
ScoreKey = tuple[date, str, int, str, str, str]


@dataclass
class StoreSummary:
    """What one save did."""

    written: int = 0
    updated: int = 0
    removed: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "written": self.written,
            "updated": self.updated,
            "removed": self.removed,
            "warnings": list(self.warnings),
        }


def _key(row: Score) -> ScoreKey:
    return (
        row.score_date,
        row.entity_type,
        row.entity_id,
        row.context_key,
        row.score_type,
        row.scoring_version,
    )


def save_scores(
    session: Session,
    score_date: date,
    results: Sequence[ScoreResult],
    *,
    evidence: Mapping[tuple[str, int], Mapping[str, Any]] | None = None,
    is_synthetic: bool = False,
    remove_missing: bool = True,
) -> StoreSummary:
    """Write scores for one date, replacing what was there for the same version.

    Args:
        score_date: the date these scores describe.
        results: the scores to save.
        evidence: optional inputs per entity, stored inside the breakdown.
        is_synthetic: mark the rows as demo data.
        remove_missing: delete rows of the same types and version that this run did not produce,
            so an entity that no longer qualifies does not keep a stale score on the dashboard.

    Raises:
        ValueError: if a score or its confidence falls outside 0-100, which the database would
            reject anyway, but is caught here with a message naming the entity.
    """
    summary = StoreSummary()
    if not results:
        return summary

    versions = {result.scoring_version for result in results}
    types = {result.score_type for result in results}
    existing: dict[ScoreKey, Score] = {
        _key(row): row
        for row in session.scalars(
            select(Score).where(
                Score.score_date == score_date,
                Score.score_type.in_(types),
                Score.scoring_version.in_(versions),
            )
        )
    }

    for result in results:
        for label, value in (("score", result.value), ("confidence", result.confidence)):
            if not 0.0 <= value <= 100.0:
                raise ValueError(
                    f"{result.entity_name}: {result.score_type} {label} is {value:.2f}, "
                    "outside the 0-100 scale"
                )
        breakdown = result.as_component_json()
        inputs = (evidence or {}).get((result.entity_type, result.entity_id))
        if inputs is not None:
            breakdown["inputs"] = dict(inputs)
        breakdown["explanation"] = result.explain()

        key: ScoreKey = (
            score_date,
            result.entity_type,
            result.entity_id,
            result.context_key,
            result.score_type,
            result.scoring_version,
        )
        row = existing.pop(key, None)
        if row is None:
            session.add(
                Score(
                    score_date=score_date,
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    context_key=result.context_key,
                    score_type=result.score_type,
                    score_value=round(result.value, 4),
                    confidence_score=round(result.confidence, 4),
                    component_json=breakdown,
                    scoring_version=result.scoring_version,
                    is_synthetic=is_synthetic,
                )
            )
            summary.written += 1
        else:
            row.score_value = round(result.value, 4)
            row.confidence_score = round(result.confidence, 4)
            row.component_json = breakdown
            row.is_synthetic = is_synthetic
            summary.updated += 1

    if remove_missing:
        for row in existing.values():
            session.delete(row)
            summary.removed += 1
    elif existing:
        summary.warnings.append(
            f"{len(existing)} stored score(s) were not recomputed and were left in place"
        )
    session.flush()
    LOGGER.info("saved scores for %s: %s", score_date, summary.as_dict())
    return summary


def load_scores(
    session: Session,
    score_date: date,
    *,
    score_type: str | None = None,
    entity_type: str | None = None,
) -> list[Score]:
    """Read stored scores for a date, newest version first, highest score first."""
    query = select(Score).where(Score.score_date == score_date)
    if score_type is not None:
        query = query.where(Score.score_type == score_type)
    if entity_type is not None:
        query = query.where(Score.entity_type == entity_type)
    rows = list(session.scalars(query))
    rows.sort(key=lambda row: (row.scoring_version, row.score_value), reverse=True)
    return rows


def latest_score_date(session: Session, *, score_type: str | None = None) -> date | None:
    """The most recent date scores were stored for."""
    query = select(func.max(Score.score_date))
    if score_type is not None:
        query = query.where(Score.score_type == score_type)
    return session.scalar(query)
