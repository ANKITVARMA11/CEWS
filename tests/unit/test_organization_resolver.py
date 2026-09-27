"""Unit tests for resolving raw organization names to database rows."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.database.models import Organization, OrganizationAlias, ReviewQueueItem
from cews.normalization.resolver import PARENT_QUEUE, REVIEW_QUEUE, OrganizationResolver
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def session(factory: sessionmaker[Session]) -> Iterator[Session]:
    with factory() as active:
        yield active
        active.rollback()


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


def _resolver(session: Session, settings: Settings, **kwargs: object) -> OrganizationResolver:
    return OrganizationResolver(session, settings, **kwargs)  # type: ignore[arg-type]


def test_spelling_variants_resolve_to_one_organization(
    session: Session, settings: Settings
) -> None:
    resolver = _resolver(session, settings)
    names = [
        "Zentavia Pharma",
        "Zentavia Pharma, Inc.",
        "ZENTAVIA PHARMA INC",
        "Zentavia Pharmaceuticals Ltd",
    ]
    matches = [resolver.resolve(name) for name in names]
    assert all(match is not None for match in matches)
    assert len({match.organization.id for match in matches if match}) == 1
    assert resolver.stats.created == 1
    assert session.scalar(select(Organization).where(Organization.id == matches[0].organization.id))


def test_abbreviations_and_missing_spaces_resolve(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    first = resolver.resolve("Lumaris Therapeutics")
    assert first is not None
    assert resolver.resolve("Lumaris Tx").organization.id == first.organization.id
    second = resolver.resolve("Orvexa Bio")
    assert second is not None
    assert resolver.resolve("OrvexaBio").organization.id == second.organization.id
    assert resolver.resolve("Orvexa Biosciences").organization.id == second.organization.id


def test_lookalike_companies_stay_separate(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    bio = resolver.resolve("Orvexa Bio")
    labs = resolver.resolve("Orvexa Labs")
    assert bio is not None and labs is not None
    assert bio.organization.id != labs.organization.id


def test_a_subsidiary_is_not_merged_into_its_parent(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    parent = resolver.resolve("Zentavia Pharma")
    child = resolver.resolve("Zentavia Oncology Ltd")
    assert parent is not None and child is not None
    assert parent.organization.id != child.organization.id
    session.flush()
    queued = session.scalars(
        select(ReviewQueueItem).where(ReviewQueueItem.queue_type == PARENT_QUEUE)
    ).all()
    assert queued and "Zentavia" in str(queued[0].payload_json)
    assert child.organization.parent_id is None  # suggested, never applied automatically


def test_an_ambiguous_short_name_goes_to_review(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    resolver.resolve("Zentavia Pharma")
    resolver.resolve("Zentavia Oncology Ltd")
    match = resolver.resolve("Zentavia")
    assert match is not None and match.needs_review
    assert match.method == "ambiguous" and match.confidence < 0.5
    session.flush()
    item = session.scalars(
        select(ReviewQueueItem).where(ReviewQueueItem.queue_type == REVIEW_QUEUE)
    ).one()
    assert sorted(item.payload_json["candidates"]) == [
        "Zentavia Oncology Ltd",
        "Zentavia Pharma",
    ]


def test_a_short_name_with_one_candidate_is_linked_but_flagged(
    session: Session, settings: Settings
) -> None:
    resolver = _resolver(session, settings)
    parent = resolver.resolve("Zentavia Pharma")
    match = resolver.resolve("Zentavia")
    assert parent is not None and match is not None
    assert match.organization.id == parent.organization.id
    assert match.method == "prefix" and match.needs_review


def test_close_spellings_are_matched_by_similarity(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    first = resolver.resolve("Harrowgate University")
    match = resolver.resolve("Univ. of Harrowgate")
    assert first is not None and match is not None
    assert match.organization.id == first.organization.id
    assert match.method == "fuzzy" and match.confidence >= settings.ai_org_match_auto_threshold


def test_similar_but_not_certain_names_are_kept_apart_and_queued(
    session: Session, settings: Settings
) -> None:
    resolver = _resolver(session, settings)
    resolver.resolve("Meridian Therapeutics")
    match = resolver.resolve("Meridian Theraputics")  # typo, below the automatic threshold
    assert match is not None and match.created and match.needs_review
    session.flush()
    items = session.scalars(
        select(ReviewQueueItem).where(ReviewQueueItem.queue_type == REVIEW_QUEUE)
    ).all()
    assert any("similar_to" in (item.payload_json or {}) for item in items)


def test_fuzzy_matching_can_be_switched_off(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings, use_fuzzy=False)
    first = resolver.resolve("Harrowgate University")
    match = resolver.resolve("Univ. of Harrowgate")
    assert first is not None and match is not None
    assert match.organization.id != first.organization.id


def test_a_company_is_never_merged_into_a_university(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    university = resolver.resolve("Fixture University")
    company = resolver.resolve("Fixture Universal Therapeutics")
    assert university is not None and company is not None
    assert university.organization.id != company.organization.id


def test_the_fullest_spelling_becomes_the_display_name(
    session: Session, settings: Settings
) -> None:
    resolver = _resolver(session, settings)
    first = resolver.resolve("Nexoria Gen.")
    assert first is not None
    organization = first.organization
    assert organization.canonical_name == "Nexoria Gen."
    resolver.resolve("Nexoria Genetics Corp.")
    assert organization.canonical_name == "Nexoria Genetics Corp."
    assert organization.organization_type == "company"


@pytest.mark.parametrize("name", [None, "", "   ", "!!!"])
def test_unusable_names_resolve_to_nothing(
    session: Session, settings: Settings, name: str | None
) -> None:
    assert _resolver(session, settings).resolve(name) is None


def test_every_spelling_is_recorded_as_an_alias(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    resolver.resolve("Zentavia Pharma")
    resolver.resolve("Zentavia Pharma, Inc.")
    session.flush()
    aliases = session.scalars(select(OrganizationAlias)).all()
    assert aliases and all(0 <= alias.confidence <= 1 for alias in aliases)
    assert {alias.match_method for alias in aliases} <= {
        "exact",
        "normalized",
        "expanded",
        "compact",
        "expanded_compact",
        "prefix",
        "fuzzy",
    }


def test_resolving_the_same_name_twice_adds_nothing(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    for _ in range(3):
        resolver.resolve("Zentavia Pharma, Inc.")
    session.flush()
    assert len(session.scalars(select(Organization)).all()) == 1
    assert resolver.stats.created == 1


def test_a_new_resolver_reuses_what_is_already_stored(
    factory: sessionmaker[Session], settings: Settings
) -> None:
    with factory() as first_session:
        first = OrganizationResolver(first_session, settings).resolve("Zentavia Pharma, Inc.")
        assert first is not None
        first_session.commit()
        stored_id = first.organization.id
    with factory() as second_session:
        resolver = OrganizationResolver(second_session, settings)
        match = resolver.resolve("ZENTAVIA PHARMA INC")
        assert match is not None and match.organization.id == stored_id
        assert resolver.stats.created == 0


def test_statistics_are_reported(session: Session, settings: Settings) -> None:
    resolver = _resolver(session, settings)
    resolver.resolve("Zentavia Pharma")
    resolver.resolve("Zentavia Pharma, Inc.")
    stats = resolver.stats.as_dict()
    assert stats["resolved"] == 2 and stats["created"] == 1
    assert sum(stats["by_method"].values()) == 2
