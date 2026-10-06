"""Five deterministic rules that turn stored scores into insights.

Every rule reads scores, forecasts and modifiers that another pass already computed and stored;
nothing here recomputes a score. Given the same stored data, the same insights are produced every
time - there is no model call anywhere in this file.

Two rules that always apply, whatever the specific insight:

* **No evidence, no insight.** A candidate with nothing to point at is dropped before it is ever
  written, not stored with an empty evidence list.
* **The same fact is not reported twice.** Each insight gets a ``dedupe_key`` scoped to the month
  it was found in; if an insight with that key already exists, the new one is silently skipped.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType, InsightSeverity, InsightType, ScoreType, SourceType
from cews.database.models import Insight, InsightEvidence, Organization, Score, Topic
from cews.discovery.new_market_entry import detect_new_therapeutic_area_entry
from cews.insights.evidence import record_ids_for_entity
from cews.insights.templates import (
    InsightText,
    competitor_movement_text,
    emerging_trend_text,
    new_market_entry_text,
    opportunity_text,
    patent_surge_text,
)
from cews.scoring.config import load_scoring_config
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

MIN_RANK_MOVE = 2  # places; smaller moves are ordinary reshuffling, not worth an insight
PATENT_SURGE_MIN_PERCENTILE = 85.0
HIGH_SEVERITY_SCORE = 80.0
DEFAULT_EVIDENCE_LIMIT = 8


@dataclass(frozen=True)
class InsightCandidate:
    """One insight before it is checked for evidence and duplicates."""

    insight_type: InsightType
    entity_type: str
    entity_id: int
    severity: InsightSeverity
    text: InsightText
    confidence: float
    dedupe_key: str
    evidence_record_ids: list[int] = field(default_factory=list)


@dataclass
class InsightRun:
    """What one insight-generation pass produced."""

    insight_date: date
    candidates_found: int = 0
    rejected_no_evidence: int = 0
    below_confidence_floor: int = 0
    duplicates_suppressed: int = 0
    stored: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "insight_date": self.insight_date.isoformat(),
            "candidates_found": self.candidates_found,
            "rejected_no_evidence": self.rejected_no_evidence,
            "below_confidence_floor": self.below_confidence_floor,
            "duplicates_suppressed": self.duplicates_suppressed,
            "stored": self.stored,
            "warnings": list(self.warnings),
        }


def _severity_for_score(value: float) -> InsightSeverity:
    if value >= HIGH_SEVERITY_SCORE:
        return InsightSeverity.HIGH
    return InsightSeverity.WATCH


def _names(session: Session) -> dict[tuple[str, int], str]:
    names: dict[tuple[str, int], str] = {}
    for topic_id, name in session.execute(select(Topic.id, Topic.canonical_name)).all():
        names[(EntityType.TOPIC.value, int(topic_id))] = str(name)
    for organization_id, name in session.execute(
        select(Organization.id, Organization.canonical_name)
    ).all():
        names[(EntityType.COMPETITOR.value, int(organization_id))] = str(name)
    return names


def _scores(
    session: Session, score_type: str, score_date: date, *, context_key: str | None = ""
) -> list[Score]:
    query = select(Score).where(Score.score_date == score_date, Score.score_type == score_type)
    if context_key is not None:
        query = query.where(Score.context_key == context_key)
    return list(session.scalars(query))


def _month_key(value: date) -> str:
    return value.strftime("%Y-%m")


# --------------------------------------------------------------------------------------
# The five rules
# --------------------------------------------------------------------------------------
def find_emerging_trends(
    session: Session, score_date: date, names: dict[tuple[str, int], str]
) -> list[InsightCandidate]:
    """Topics that cleared every rule for being an emerging trend (see ``trend_score.py``)."""
    candidates = []
    for row in _scores(session, ScoreType.TREND.value, score_date):
        breakdown = row.component_json or {}
        if not breakdown.get("qualified"):
            continue
        name = names.get((row.entity_type, row.entity_id), f"topic {row.entity_id}")
        text = emerging_trend_text(
            name,
            score=row.score_value,
            confidence=row.confidence_score,
            supporting_sources=len(
                [c for c in breakdown.get("components", {}).values() if c.get("available")]
            ),
            sample_size=int(breakdown.get("sample_size", 0)),
        )
        candidates.append(
            InsightCandidate(
                insight_type=InsightType.EMERGING_TREND,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                severity=_severity_for_score(row.score_value),
                text=text,
                confidence=row.confidence_score,
                dedupe_key=f"emerging_trend:{row.entity_type}:{row.entity_id}:{_month_key(score_date)}",
                evidence_record_ids=record_ids_for_entity(
                    session, row.entity_type, row.entity_id, limit=DEFAULT_EVIDENCE_LIMIT
                ),
            )
        )
    return candidates


def find_competitor_movements(
    session: Session, score_date: date, names: dict[tuple[str, int], str], *, min_confidence: float
) -> list[InsightCandidate]:
    """Competitors whose innovation rank moved by enough places to be worth a look.

    Only produces anything from the second scoring run onward: a rank has nothing to move
    relative to on the first run, and that absence is a fact, not a zero.
    """
    candidates = []
    for row in _scores(session, ScoreType.INNOVATION.value, score_date):
        if row.confidence_score < min_confidence:
            continue
        rank_detail = (
            (row.component_json or {}).get("components", {}).get("rank", {}).get("detail", {})
        )
        rank_change = rank_detail.get("rank_change")
        if rank_change is None or abs(rank_change) < MIN_RANK_MOVE:
            continue
        current_rank = int(rank_detail["rank"])
        # innovation_score.assign_ranks defines rank_change as previous_rank - current_rank
        # (positive means the competitor moved up), so previous_rank is the sum of the two.
        previous_rank = current_rank + int(rank_change)
        name = names.get((row.entity_type, row.entity_id), f"competitor {row.entity_id}")
        text = competitor_movement_text(
            name, previous_rank=previous_rank, current_rank=current_rank, score=row.score_value
        )
        candidates.append(
            InsightCandidate(
                insight_type=InsightType.COMPETITOR_MOVEMENT,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                severity=InsightSeverity.WATCH,
                text=text,
                confidence=row.confidence_score,
                dedupe_key=(
                    f"competitor_movement:{row.entity_type}:{row.entity_id}:"
                    f"{_month_key(score_date)}:{current_rank}"
                ),
                evidence_record_ids=record_ids_for_entity(
                    session, row.entity_type, row.entity_id, limit=DEFAULT_EVIDENCE_LIMIT
                ),
            )
        )
    return candidates


def find_new_market_entries(
    session: Session, score_date: date, names: dict[tuple[str, int], str]
) -> list[InsightCandidate]:
    """Competitors whose threat score was boosted by a first move into a new area."""
    entries = detect_new_therapeutic_area_entry(
        session, as_of=datetime(score_date.year, score_date.month, score_date.day, tzinfo=UTC)
    )
    candidates = []
    for entry in entries:
        name = names.get(
            (EntityType.COMPETITOR.value, entry.organization_id), entry.organization_name
        )
        text = new_market_entry_text(
            name,
            area_name=entry.topic_name,
            records_in_window=entry.records_in_window,
            quiet_months=entry.quiet_months,
        )
        candidates.append(
            InsightCandidate(
                insight_type=InsightType.NEW_MARKET_ENTRY,
                entity_type=EntityType.COMPETITOR.value,
                entity_id=entry.organization_id,
                severity=InsightSeverity.WATCH,
                text=text,
                confidence=100.0,  # this is an observed count, not a modelled score
                dedupe_key=(
                    f"new_market_entry:{entry.organization_id}:{entry.topic_id}:{_month_key(score_date)}"
                ),
                evidence_record_ids=record_ids_for_entity(
                    session,
                    EntityType.COMPETITOR.value,
                    entry.organization_id,
                    limit=DEFAULT_EVIDENCE_LIMIT,
                ),
            )
        )
    return candidates


def find_patent_surges(
    session: Session, score_date: date, names: dict[tuple[str, int], str], *, min_confidence: float
) -> list[InsightCandidate]:
    """Topics or competitors whose patent growth is in the top tier for their cohort."""
    candidates = []
    for score_type, entity_kind_label in (
        (ScoreType.TREND.value, "topic"),
        (ScoreType.THREAT.value, "competitor"),
    ):
        context = "" if score_type == ScoreType.THREAT.value else None
        for row in _scores(session, score_type, score_date, context_key=context):
            if row.confidence_score < min_confidence:
                continue
            component = (row.component_json or {}).get("components", {}).get("patent_growth")
            if not component or not component.get("available"):
                continue
            if component["normalized"] < PATENT_SURGE_MIN_PERCENTILE:
                continue
            raw = component.get("raw_value") or 0.0
            growth_percent = (math.exp(raw) - 1.0) * 100.0
            if growth_percent <= 0:
                continue
            name = names.get(
                (row.entity_type, row.entity_id), f"{entity_kind_label} {row.entity_id}"
            )
            patent_ids = record_ids_for_entity(
                session,
                row.entity_type,
                row.entity_id,
                source_type=SourceType.PATENT.value,
                limit=DEFAULT_EVIDENCE_LIMIT,
            )
            text = patent_surge_text(
                name, entity_kind_label, growth_percent=growth_percent, sample_size=len(patent_ids)
            )
            candidates.append(
                InsightCandidate(
                    insight_type=InsightType.PATENT_SURGE,
                    entity_type=row.entity_type,
                    entity_id=row.entity_id,
                    severity=_severity_for_score(component["normalized"]),
                    text=text,
                    confidence=row.confidence_score,
                    dedupe_key=f"patent_surge:{row.entity_type}:{row.entity_id}:{_month_key(score_date)}",
                    evidence_record_ids=patent_ids,
                )
            )
    return candidates


def find_opportunities(
    session: Session, score_date: date, names: dict[tuple[str, int], str]
) -> list[InsightCandidate]:
    """Topics that cleared every rule for the opportunity score (see ``opportunity_score.py``)."""
    candidates = []
    for row in _scores(session, ScoreType.OPPORTUNITY.value, score_date):
        breakdown = row.component_json or {}
        if not breakdown.get("qualified"):
            continue
        component = breakdown.get("components", {}).get("low_competition", {})
        openness = component.get("normalized", 0.0)
        note = (
            "very few competitors are active in it"
            if openness >= 80
            else "the field is not yet crowded"
        )
        name = names.get((row.entity_type, row.entity_id), f"topic {row.entity_id}")
        text = opportunity_text(
            name, score=row.score_value, confidence=row.confidence_score, competition_note=note
        )
        candidates.append(
            InsightCandidate(
                insight_type=InsightType.OPPORTUNITY,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                severity=_severity_for_score(row.score_value),
                text=text,
                confidence=row.confidence_score,
                dedupe_key=f"opportunity:{row.entity_type}:{row.entity_id}:{_month_key(score_date)}",
                evidence_record_ids=record_ids_for_entity(
                    session, row.entity_type, row.entity_id, limit=DEFAULT_EVIDENCE_LIMIT
                ),
            )
        )
    return candidates


# --------------------------------------------------------------------------------------
# Orchestration: run every rule, then filter, dedupe, and store
# --------------------------------------------------------------------------------------
def generate_insights(
    session: Session,
    settings: Settings,
    *,
    score_date: date | None = None,
    store: bool = True,
    is_synthetic: bool = False,
) -> InsightRun:
    """Run every rule against the most recent (or given) score date and store what survives.

    Args:
        score_date: which date's scores to read (default: the most recent one stored).
        store: write surviving insights and their evidence links.
        is_synthetic: mark stored rows as demo data.

    Raises:
        ScoringConfigError: if the scoring configuration cannot be read.
    """
    config = load_scoring_config(settings)
    when = score_date or session.scalar(select(Score.score_date).order_by(Score.score_date.desc()))
    run = InsightRun(insight_date=when or date.today())
    if when is None:
        run.warnings.append("no scores stored yet; run: cews score")
        return run

    names = _names(session)
    min_confidence = config.emerging_trend.min_confidence
    all_candidates: list[InsightCandidate] = [
        *find_emerging_trends(session, when, names),
        *find_competitor_movements(session, when, names, min_confidence=min_confidence),
        *find_new_market_entries(session, when, names),
        *find_patent_surges(session, when, names, min_confidence=min_confidence),
        *find_opportunities(session, when, names),
    ]
    run.candidates_found = len(all_candidates)

    existing_keys = set(session.scalars(select(Insight.dedupe_key)))
    for candidate in all_candidates:
        if not candidate.evidence_record_ids:
            run.rejected_no_evidence += 1
            LOGGER.warning(
                "dropped %s insight with no evidence: %s",
                candidate.insight_type,
                candidate.text.title,
            )
            continue
        if candidate.dedupe_key in existing_keys:
            run.duplicates_suppressed += 1
            continue
        existing_keys.add(
            candidate.dedupe_key
        )  # guard against two candidates sharing a key this run
        if store:
            insight = Insight(
                insight_date=when,
                severity=candidate.severity.value,
                insight_type=candidate.insight_type.value,
                entity_type=candidate.entity_type,
                entity_id=candidate.entity_id,
                title=candidate.text.title,
                observed_fact=candidate.text.observed_fact,
                interpretation=candidate.text.interpretation,
                recommended_review=candidate.text.recommended_review,
                confidence_score=candidate.confidence,
                dedupe_key=candidate.dedupe_key,
                is_synthetic=is_synthetic,
            )
            session.add(insight)
            session.flush()
            session.add_all(
                InsightEvidence(insight_id=insight.id, source_record_id=record_id)
                for record_id in candidate.evidence_record_ids
            )
            run.stored += 1

    if store:
        session.flush()
    LOGGER.info("insight generation: %s", run.as_dict())
    return run
