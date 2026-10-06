"""A sheet for experts to mark up, and a way to read their verdicts back.

The export lists every scored topic with what the algorithm said and why (score, confidence,
plain-language explanation, and the source records behind it) next to two blank columns,
``expert_rating`` and ``expert_comment``. An expert fills those in offline, in any spreadsheet,
and :func:`summarize_expert_ratings` reads the completed file back.

The completed file *is* the record of the expert's judgement: this system does not store topic
ratings in its database, so the summary reports what is in the file and nothing more. It also
keeps expert validation apart from the algorithm's own numbers, by design: an expert agreeing
with the ranking is evidence about the ranking, not an input to it.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import AlertRating, ScoreType
from cews.database.models import Score, Topic
from cews.insights.evidence import record_ids_for_entity

COLUMNS = (
    "topic_id",
    "topic",
    "score",
    "confidence",
    "qualifies",
    "explanation",
    "evidence",
    "expert_rating",
    "expert_comment",
)
VALID_RATINGS = frozenset(member.value for member in AlertRating)
EVIDENCE_LIMIT = 5


class ExpertReviewError(ValueError):
    """Raised when an expert review file cannot be read as asked."""


@dataclass(frozen=True)
class ExpertSummary:
    """What a completed expert sheet says about the ranking."""

    rows: int
    rated: int
    unrated: int
    by_rating: dict[str, int]
    precision_at_k: float | None
    k: int
    mean_score_relevant: float | None
    mean_score_not_relevant: float | None
    invalid_ratings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 2)

        return {
            "rows": self.rows,
            "rated": self.rated,
            "unrated": self.unrated,
            "by_rating": dict(self.by_rating),
            "precision_at_k": (
                None if self.precision_at_k is None else round(self.precision_at_k, 4)
            ),
            "k": self.k,
            "mean_score_relevant": rounded(self.mean_score_relevant),
            "mean_score_not_relevant": rounded(self.mean_score_not_relevant),
            "invalid_ratings": list(self.invalid_ratings),
        }


def export_expert_review(
    session: Session, path: Path, *, score_date: date | None = None, limit: int | None = None
) -> int:
    """Write the review sheet, highest-scoring topic first. Returns the number of rows written.

    Args:
        score_date: which stored trend-score date to export (default: the most recent).
        limit: export only the top ``limit`` topics (default: all).

    The expert columns are written blank. Nothing is exported for a date with no stored scores.
    """
    query = select(Score).where(Score.score_type == ScoreType.TREND.value, Score.context_key == "")
    if score_date is not None:
        query = query.where(Score.score_date == score_date)
    else:
        newest = select(Score.score_date).order_by(Score.score_date.desc()).limit(1)
        query = query.where(Score.score_date == newest.scalar_subquery())
    scores = sorted(session.scalars(query), key=lambda row: row.score_value, reverse=True)
    if limit is not None:
        scores = scores[:limit]
    names = {row.id: row.canonical_name for row in session.scalars(select(Topic))}

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:  # BOM so Excel reads UTF-8
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in scores:
            breakdown = row.component_json or {}
            evidence = record_ids_for_entity(
                session, row.entity_type, row.entity_id, limit=EVIDENCE_LIMIT
            )
            writer.writerow(
                [
                    row.entity_id,
                    names.get(row.entity_id, f"topic {row.entity_id}"),
                    round(row.score_value, 1),
                    round(row.confidence_score, 1),
                    "yes" if breakdown.get("qualified") else "no",
                    breakdown.get("explanation", ""),
                    "; ".join(f"record {record_id}" for record_id in evidence),
                    "",
                    "",
                ]
            )
    return len(scores)


def summarize_expert_ratings(path: Path, *, k: int = 10) -> ExpertSummary:
    """Read a completed sheet and report how the expert's ratings line up with the ranking.

    Precision@K counts, among the ``k`` highest-scoring *rated* topics, the share the expert
    called relevant. A rating that is not one of the known values is reported, not guessed at,
    and left out of every calculation.

    Raises:
        ExpertReviewError: if the file is missing or lacks the expected columns, or ``k`` is
            not positive.
    """
    if k < 1:
        raise ExpertReviewError("k must be positive")
    if not path.is_file():
        raise ExpertReviewError(f"expert review file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"topic", "score", "expert_rating"} - set(reader.fieldnames or [])
        if missing:
            raise ExpertReviewError(f"{path}: missing column(s) {sorted(missing)}")
        rows = list(reader)

    rated: list[tuple[float, str]] = []
    invalid: list[str] = []
    for row in rows:
        rating = (row.get("expert_rating") or "").strip().lower().replace(" ", "_")
        if not rating:
            continue
        if rating not in VALID_RATINGS:
            invalid.append(rating)
            continue
        try:
            rated.append((float(row["score"]), rating))
        except (TypeError, ValueError):
            invalid.append(f"unreadable score for {row.get('topic')!r}")

    counts: dict[str, int] = {member.value: 0 for member in AlertRating}
    for _score, rating in rated:
        counts[rating] += 1
    top = sorted(rated, key=lambda item: item[0], reverse=True)[:k]
    relevant_scores = [score for score, rating in rated if rating == AlertRating.RELEVANT.value]
    not_relevant_scores = [
        score for score, rating in rated if rating == AlertRating.NOT_RELEVANT.value
    ]
    return ExpertSummary(
        rows=len(rows),
        rated=len(rated),
        unrated=len(rows) - len(rated) - len(invalid),
        by_rating=counts,
        precision_at_k=(
            sum(1 for _s, rating in top if rating == AlertRating.RELEVANT.value) / len(top)
            if top
            else None
        ),
        k=k,
        mean_score_relevant=statistics.fmean(relevant_scores) if relevant_scores else None,
        mean_score_not_relevant=(
            statistics.fmean(not_relevant_scores) if not_relevant_scores else None
        ),
        invalid_ratings=tuple(invalid),
    )
