"""Turning numbers into three separate sentences: fact, interpretation, and what to do.

Every insight keeps these apart, on purpose. The **observed fact** is something anyone could
verify by looking at the same records. The **interpretation** is what the rule concluded from
it, which is a judgement and could be wrong. The **recommended review** is a next step for a
person, never an instruction CEWS expects to be followed automatically.

No model writes any of this text. Every function here is a pure string template filled in from
numbers already computed elsewhere; the same inputs always produce the same sentences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InsightText:
    """The three required parts of one insight, plus a short title for a list view."""

    title: str
    observed_fact: str
    interpretation: str
    recommended_review: str


def emerging_trend_text(
    topic_name: str, *, score: float, confidence: float, supporting_sources: int, sample_size: int
) -> InsightText:
    """A topic that cleared every rule for being called an emerging trend."""
    return InsightText(
        title=f"{topic_name}: emerging trend",
        observed_fact=(
            f"{topic_name} scored {score:.0f}/100 for trend strength, backed by {sample_size} "
            f"record(s) across {supporting_sources} independent source(s)."
        ),
        interpretation=(
            "Activity in this topic is rising consistently enough, across enough independent "
            f"sources, to be treated as a genuine trend rather than noise (confidence {confidence:.0f}/100)."
        ),
        recommended_review=(
            "Have a subject-matter expert confirm this topic warrants closer monitoring or "
            "further investment review."
        ),
    )


def competitor_movement_text(
    competitor_name: str, *, previous_rank: int, current_rank: int, score: float
) -> InsightText:
    """A competitor's innovation ranking moved by a meaningful number of places."""
    moved_up = current_rank < previous_rank
    places = abs(previous_rank - current_rank)
    direction = "up" if moved_up else "down"
    return InsightText(
        title=f"{competitor_name}: moved {direction} the innovation ranking",
        observed_fact=(
            f"{competitor_name} moved {direction} from rank {previous_rank} to rank "
            f"{current_rank} on innovation output (score {score:.0f}/100)."
        ),
        interpretation=(
            f"{competitor_name}'s research output relative to the other monitored competitors "
            f"has {'increased' if moved_up else 'decreased'} by {places} place(s) since the "
            "last scoring run."
        ),
        recommended_review=(
            f"Review what changed in {competitor_name}'s patent, trial or publication activity "
            "over the period."
        ),
    )


def new_market_entry_text(
    competitor_name: str, *, area_name: str, records_in_window: int, quiet_months: int
) -> InsightText:
    """A competitor started activity in an area it had no history in."""
    return InsightText(
        title=f"{competitor_name}: entered {area_name}",
        observed_fact=(
            f"{competitor_name} recorded {records_in_window} record(s) in {area_name} after "
            f"{quiet_months} month(s) with none."
        ),
        interpretation=(
            f"This is a first move by {competitor_name} into {area_name}, not a continuation of "
            "existing work there."
        ),
        recommended_review=f"Assess whether {area_name} is a strategic priority and what {competitor_name}'s entry means for it.",
    )


def patent_surge_text(
    entity_name: str, entity_kind_label: str, *, growth_percent: float, sample_size: int
) -> InsightText:
    """Patent activity for a topic or competitor grew sharply."""
    return InsightText(
        title=f"{entity_name}: patent activity surge",
        observed_fact=(
            f"Patent activity for {entity_kind_label} {entity_name} grew by roughly "
            f"{growth_percent:.0f}% over the comparison period ({sample_size} patent record(s))."
        ),
        interpretation=(
            "A sharp rise in patent filings often precedes a competitor's public announcements "
            "by months, since filings happen before disclosure."
        ),
        recommended_review=f"Review the recent patent filings behind {entity_name} for the technology or claim areas involved.",
    )


def opportunity_text(
    topic_name: str, *, score: float, confidence: float, competition_note: str
) -> InsightText:
    """A topic is growing and the field is not yet crowded."""
    return InsightText(
        title=f"{topic_name}: opportunity for review",
        observed_fact=(
            f"{topic_name} scored {score:.0f}/100 for opportunity: it is trending and "
            f"{competition_note} (confidence {confidence:.0f}/100)."
        ),
        interpretation=(
            "This combination of growth and open competitive space is worth an expert's "
            "attention. This is a prioritization signal for review, not an investment, "
            "commercial or scientific recommendation."
        ),
        recommended_review=f"Have a subject-matter expert assess whether {topic_name} merits further investigation.",
    )
