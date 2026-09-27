"""Unit tests for competitor ranking: the arithmetic, the weights and the eligibility rules.

The database here is tiny and hand-built, so every expected score can be worked out on paper.
Behaviour on realistic data lives in ``tests/integration/test_normalization_pipeline.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RelationshipType, SourceType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Organization, RecordOrganization, SourceRecord
from cews.discovery.competitor_discovery import (
    DEFAULT_WEIGHTS,
    discover_competitors,
    load_discovery_weights,
)
from cews.scoring.normalization import clamp_score, percentile_normalize, renormalize_weights
from cews.settings import Settings, load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
INSIDE = AS_OF - timedelta(days=30)
OUTSIDE = AS_OF - timedelta(days=500)


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
            "scoring_config_file": SCORING_FILE,
            "competitor_mode": "AUTO",
            "min_competitor_evidence_count": 1,
            "top_competitors": 10,
        },
    )


def add_organization(session: Session, name: str, kind: str = "company") -> Organization:
    organization = Organization(
        canonical_name=name, normalized_name=name.casefold(), organization_type=kind
    )
    session.add(organization)
    session.flush()
    return organization


def add_records(
    session: Session,
    organization: Organization,
    record_type: SourceType,
    count: int,
    *,
    relationship: RelationshipType = RelationshipType.SPONSOR,
    confidence: float = 1.0,
    when: datetime = INSIDE,
) -> None:
    """Attach ``count`` records of one type to an organization."""
    for index in range(count):
        key = f"{organization.id}-{record_type.value}-{relationship.value}-{index}"
        record = SourceRecord(
            source="test",
            source_record_id=key,
            record_type=record_type.value,
            fetched_at=when,
            published_at=when,
            content_hash=f"{abs(hash(key)):064x}"[:64],
        )
        session.add(record)
        session.flush()
        session.add(
            RecordOrganization(
                source_record_id=record.id,
                organization_id=organization.id,
                relationship_type=relationship.value,
                confidence=confidence,
            )
        )
    session.flush()


def run(factory: sessionmaker[Session], settings: Settings, **kwargs: Any) -> Any:
    with session_scope(factory) as session:
        return discover_competitors(session, settings, as_of=AS_OF, persist=False, **kwargs)


# --------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------
def test_default_weights_match_the_specification() -> None:
    assert DEFAULT_WEIGHTS == {
        "trial": 0.30,
        "patent": 0.25,
        "publication": 0.20,
        "funding": 0.15,
        "announcement": 0.10,
    }
    assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)


def test_configured_weights_are_loaded_and_sum_to_one(settings: Settings) -> None:
    weights, months = load_discovery_weights(settings)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert set(weights) == set(DEFAULT_WEIGHTS)
    assert months == 12


def test_missing_components_share_their_weight_out() -> None:
    effective = renormalize_weights(DEFAULT_WEIGHTS, ["trial", "publication"])
    assert sum(effective.values()) == pytest.approx(1.0)
    # 0.30 and 0.20 rescaled: trials keep 60% of the total, publications 40%
    assert effective["trial"] == pytest.approx(0.6)
    assert effective["publication"] == pytest.approx(0.4)
    assert "patent" not in effective


# --------------------------------------------------------------------------------------
# Arithmetic, checked by hand
# --------------------------------------------------------------------------------------
def test_score_matches_a_hand_calculation(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """Three companies, trials only, so the score is the trial percentile alone.

    Percentile is (below + half the ties) / count: 2.5/3 = 83.33, 1.5/3 = 50, 0.5/3 = 16.67.
    """
    with session_scope(factory) as session:
        for name, trials in (("Busy", 10), ("Middle", 5), ("Quiet", 1)):
            add_records(session, add_organization(session, name), SourceType.CLINICAL_TRIAL, trials)

    result = run(factory, settings)
    scores = {entry.name: entry.score for entry in result.monitored}
    assert scores == {"Busy": 83.33, "Middle": 50.0, "Quiet": 16.67}
    assert result.weights == {"trial": 1.0}
    assert set(result.unavailable_sources) == {"patent", "publication", "funding", "announcement"}


def test_components_carry_their_weights_and_add_up(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """Two companies, trials and patents only, so the weights renormalize to 0.30/0.55 and 0.25/0.55."""
    with session_scope(factory) as session:
        leader = add_organization(session, "Leader")
        follower = add_organization(session, "Follower")
        add_records(session, leader, SourceType.CLINICAL_TRIAL, 8)
        add_records(session, leader, SourceType.PATENT, 6)
        add_records(session, follower, SourceType.CLINICAL_TRIAL, 2)
        add_records(session, follower, SourceType.PATENT, 1)

    by_name = {entry.name: entry for entry in run(factory, settings).monitored}
    assert by_name["Leader"].score == 75.0
    assert by_name["Follower"].score == 25.0
    leading = by_name["Leader"]
    total = sum(component.contribution for component in leading.components.values())
    assert total == pytest.approx(leading.score, abs=0.01)
    assert leading.components["trial"].weight == pytest.approx(0.3 / 0.55)
    assert leading.components["trial"].record_count == 8


def test_an_affiliation_counts_for_less_than_a_sponsorship(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """A paper by someone at a company is weaker evidence than sponsoring a trial.

    Normalization stores that judgement as the link confidence (an affiliation at 0.5); ranking
    sums those confidences, so the same number of records counts for less.
    """
    with session_scope(factory) as session:
        sponsor = add_organization(session, "Sponsor")
        employer = add_organization(session, "Employer")
        add_records(session, sponsor, SourceType.PUBLICATION, 4)
        add_records(
            session,
            employer,
            SourceType.PUBLICATION,
            4,
            relationship=RelationshipType.AFFILIATION,
            confidence=0.5,
        )

    by_name = {entry.name: entry for entry in run(factory, settings).monitored}
    assert by_name["Sponsor"].components["publication"].raw_activity == 4.0
    assert by_name["Employer"].components["publication"].raw_activity == 2.0
    assert by_name["Employer"].components["publication"].record_count == 4  # evidence still counted
    assert by_name["Sponsor"].score > by_name["Employer"].score


def test_every_score_stays_inside_the_scale(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        for index in range(6):
            organization = add_organization(session, f"Company {index}")
            add_records(session, organization, SourceType.CLINICAL_TRIAL, index * 20)
            add_records(session, organization, SourceType.PATENT, index)
    for entry in run(factory, settings).monitored:
        assert 0.0 <= entry.score <= 100.0
        for component in entry.components.values():
            assert 0.0 <= component.normalized <= 100.0


def test_a_lone_candidate_sits_in_the_middle(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    """With nobody to compare against, a percentile cannot say more than "average"."""
    with session_scope(factory) as session:
        add_records(session, add_organization(session, "Only"), SourceType.CLINICAL_TRIAL, 9)
    assert run(factory, settings).monitored[0].score == 50.0


def test_equally_active_companies_score_the_same(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        for name in ("A", "B", "C"):
            add_records(session, add_organization(session, name), SourceType.PATENT, 3)
    assert {entry.score for entry in run(factory, settings).monitored} == {50.0}


# --------------------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------------------
def test_records_outside_the_window_do_not_count(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        add_records(
            session, add_organization(session, "Recent"), SourceType.CLINICAL_TRIAL, 3, when=INSIDE
        )
        add_records(
            session,
            add_organization(session, "Historic"),
            SourceType.CLINICAL_TRIAL,
            30,
            when=OUTSIDE,
        )
    assert [entry.name for entry in run(factory, settings).monitored] == ["Recent"]


def test_thin_evidence_is_not_ranked(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        add_records(session, add_organization(session, "Established"), SourceType.PATENT, 12)
        add_records(session, add_organization(session, "Barely seen"), SourceType.PATENT, 2)

    strict = settings.model_copy(update={"min_competitor_evidence_count": 5})
    assert [entry.name for entry in run(factory, strict).monitored] == ["Established"]


@pytest.mark.parametrize("kind", ["university", "hospital", "government"])
def test_academic_and_public_bodies_are_not_competitors(
    factory: sessionmaker[Session], settings: Settings, kind: str
) -> None:
    with session_scope(factory) as session:
        add_records(session, add_organization(session, "Company"), SourceType.PATENT, 4)
        add_records(session, add_organization(session, "Other", kind), SourceType.PATENT, 40)

    assert [entry.name for entry in run(factory, settings).monitored] == ["Company"]
    both = {entry.name for entry in run(factory, settings, include_all_types=True).monitored}
    assert both == {"Company", "Other"}


def test_an_excluded_company_never_appears(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        add_records(session, add_organization(session, "Wanted"), SourceType.PATENT, 4)
        add_records(session, add_organization(session, "Unwanted"), SourceType.PATENT, 40)

    without = settings.model_copy(update={"competitor_exclude": "Unwanted"})
    assert [entry.name for entry in run(factory, without).monitored] == ["Wanted"]


def test_no_data_is_reported_rather_than_guessed(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    result = run(factory, settings)
    assert result.monitored == [] and result.considered == []
    assert any("nothing to rank" in warning for warning in result.warnings)


def test_ranking_is_repeatable(factory: sessionmaker[Session], settings: Settings) -> None:
    with session_scope(factory) as session:
        for name, trials, patents in (("A", 7, 2), ("B", 3, 9), ("C", 5, 5)):
            organization = add_organization(session, name)
            add_records(session, organization, SourceType.CLINICAL_TRIAL, trials)
            add_records(session, organization, SourceType.PATENT, patents)

    first = [(e.name, e.score) for e in run(factory, settings).monitored]
    second = [(e.name, e.score) for e in run(factory, settings).monitored]
    assert first == second


def test_evidence_is_kept_for_every_ranked_company(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with session_scope(factory) as session:
        add_records(session, add_organization(session, "Traceable"), SourceType.PATENT, 6)
    entry = run(factory, settings).monitored[0]
    assert entry.evidence_count == 6
    assert entry.evidence_record_ids  # a ranking can always be traced back to records
    assert entry.source_types == ("patent",)


# --------------------------------------------------------------------------------------
# Helpers used by the ranking
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], [16.6667, 50.0, 83.3333]),
        ([5.0], [50.0]),
        ([4.0, 4.0], [50.0, 50.0]),
        ([], []),
    ],
)
def test_percentile_normalize(values: list[float], expected: list[float]) -> None:
    assert percentile_normalize(values) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize(
    ("value", "expected"), [(-5.0, 0.0), (0.0, 0.0), (42.0, 42.0), (100.0, 100.0), (150.0, 100.0)]
)
def test_clamp_score(value: float, expected: float) -> None:
    assert clamp_score(value) == expected


def test_renormalize_weights_edge_cases() -> None:
    assert renormalize_weights(DEFAULT_WEIGHTS, list(DEFAULT_WEIGHTS)) == DEFAULT_WEIGHTS
    with pytest.raises(ValueError, match="no weighted components"):
        renormalize_weights(DEFAULT_WEIGHTS, [])
    assert renormalize_weights(DEFAULT_WEIGHTS, ["funding"]) == {"funding": pytest.approx(1.0)}
