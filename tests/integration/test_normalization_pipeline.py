"""Integration tests for the normalization pipeline and competitor discovery."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import ProcessingStatus, RelationshipType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import (
    Organization,
    OrganizationAlias,
    RecordOrganization,
    RecordTopic,
    ReviewQueueItem,
    SourceRecord,
    Topic,
)
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.normalization.pipeline import (
    deduplicate_records,
    normalize_records,
    reset_normalization,
)
from cews.normalization.topics import Taxonomy, load_taxonomy, sync_taxonomy
from cews.settings import Settings, load_settings
from support import TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy(TAXONOMY_FILE)


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
        overrides={"topic_taxonomy_file": TAXONOMY_FILE, "min_competitor_evidence_count": 5},
    )


@pytest.fixture
def loaded(factory: sessionmaker[Session], taxonomy: Taxonomy) -> sessionmaker[Session]:
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.2)))
    return factory


def _count(factory: sessionmaker[Session], model: type, **where: Any) -> int:
    with session_scope(factory) as session:
        query = select(func.count()).select_from(model)
        for name, value in where.items():
            query = query.where(getattr(model, name) == value)
        return int(session.scalar(query) or 0)


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
def test_records_are_linked_to_organizations_and_topics(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        summary = normalize_records(session, settings, taxonomy)
    assert summary.records_processed > 200 and summary.failures == 0
    assert summary.organization_links >= summary.records_processed
    assert summary.topic_links > 0
    assert _count(loaded, SourceRecord, processing_status=ProcessingStatus.NORMALIZED.value) == (
        summary.records_processed
    )
    # every demo company is found, and nothing is invented
    with session_scope(loaded) as session:
        names = {o.canonical_name for o in session.scalars(select(Organization))}
    assert "Zentavia Pharmaceuticals Ltd" in names or "Zentavia Pharma" in names
    assert any("Orvexa" in name for name in names)


def test_running_twice_adds_nothing(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        first = normalize_records(session, settings, taxonomy)
    links = _count(loaded, RecordOrganization)
    with session_scope(loaded) as session:
        second = normalize_records(session, settings, taxonomy)
    assert second.records_processed == 0
    assert _count(loaded, RecordOrganization) == links
    assert first.records_processed > 0


def test_reprocessing_reuses_the_existing_organizations(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    organizations = _count(loaded, Organization)
    links = _count(loaded, RecordOrganization)
    with session_scope(loaded) as session:
        reset_normalization(session)
        again = normalize_records(session, settings, taxonomy, reprocess=True)
    assert again.records_processed > 0 and again.organizations_created == 0
    assert _count(loaded, Organization) == organizations
    assert _count(loaded, RecordOrganization) == links


def test_a_limit_processes_only_part_of_the_backlog(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        summary = normalize_records(session, settings, taxonomy, limit=25)
    assert summary.records_processed == 25
    assert _count(loaded, SourceRecord, processing_status=ProcessingStatus.NEW.value) > 0


def test_relationship_types_and_confidence_reflect_the_evidence(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    with session_scope(loaded) as session:
        rows = session.execute(
            select(
                RecordOrganization.relationship_type,
                func.min(RecordOrganization.confidence),
                func.max(RecordOrganization.confidence),
            ).group_by(RecordOrganization.relationship_type)
        ).all()
    by_type = {name: (low, high) for name, low, high in rows}
    assert RelationshipType.SPONSOR.value in by_type
    assert by_type[RelationshipType.SPONSOR.value][1] >= 0.9  # sponsoring is strong evidence
    if RelationshipType.AFFILIATION.value in by_type:
        # an author's affiliation is weak evidence and must never outrank sponsorship
        assert by_type[RelationshipType.AFFILIATION.value][1] <= 0.55


def test_topic_assignments_carry_their_confidence(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    with session_scope(loaded) as session:
        rows = session.execute(
            select(Topic.key, func.count(RecordTopic.id))
            .join(RecordTopic, RecordTopic.topic_id == Topic.id)
            .group_by(Topic.key)
        ).all()
        confidences = session.scalars(select(RecordTopic.confidence)).all()
    assigned = {key: count for key, count in rows}
    assert "crispr_gene_editing" in assigned
    assert all(0 < value <= 0.95 for value in confidences)


def test_records_matching_no_topic_are_counted_not_forced(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        summary = normalize_records(session, settings, taxonomy)
    # the demo contains a topic the taxonomy deliberately lacks; those records stay untagged
    assert summary.records_without_topic > 0


def test_a_broken_record_does_not_stop_the_pass(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        record = session.scalars(select(SourceRecord).limit(1)).one()
        record.raw_payload_json = {"protocolSection": "not a mapping"}
        record.title = None
    with session_scope(loaded) as session:
        summary = normalize_records(session, settings, taxonomy)
    assert summary.records_processed > 0
    assert summary.failures + summary.records_processed > 200


def test_uncertain_names_are_queued_for_review(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    with session_scope(loaded) as session:
        items = session.scalars(select(ReviewQueueItem)).all()
        pairs = [i.payload_json.get("organizations") for i in items if i.payload_json]
    assert items
    # the look-alike companies are surfaced for a human, not merged
    assert any(pair and "Orvexa" in str(pair) for pair in pairs)


def test_duplicate_detection_is_idempotent(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    with session_scope(loaded) as session:
        first = deduplicate_records(session)
        second = deduplicate_records(session)
    assert first.duplicates == second.duplicates
    assert second.cleared == 0


# --------------------------------------------------------------------------------------
# Competitor discovery
# --------------------------------------------------------------------------------------
def _discover(factory: sessionmaker[Session], settings: Settings, **kwargs: Any) -> Any:
    with session_scope(factory) as session:
        return discover_competitors(session, settings, as_of=AS_OF, **kwargs)


def test_discovery_ranks_companies_with_explainable_components(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    result = _discover(loaded, settings.model_copy(update={"competitor_mode": "AUTO"}))
    assert result.monitored
    top = result.monitored[0]
    assert 0 <= top.score <= 100 and top.evidence_count > 0
    assert set(top.components) <= {"trial", "patent", "publication", "funding", "announcement"}
    assert abs(sum(c.contribution for c in top.components.values()) - top.score) < 0.5
    assert top.evidence_record_ids  # every ranking can be traced to records
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


def test_universities_and_hospitals_are_not_competitors(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    auto = settings.model_copy(update={"competitor_mode": "AUTO", "top_competitors": 20})
    names = {entry.name for entry in _discover(loaded, auto).monitored}
    assert not any("University" in name or "Medical Center" in name for name in names)
    with_all = _discover(loaded, auto, include_all_types=True)
    assert any("University" in entry.name for entry in with_all.monitored)


def test_manual_competitors_always_appear_and_exclusions_never_do(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
        leader = discover_competitors(
            session,
            settings.model_copy(update={"competitor_mode": "AUTO"}),
            as_of=AS_OF,
            persist=False,
        ).monitored[0]
    hybrid = settings.model_copy(
        update={
            "competitor_mode": "HYBRID",
            "competitor_include": "Fixture Unknown Pharma",
            "competitor_exclude": leader.name,
            "top_competitors": 5,
        }
    )
    result = _discover(loaded, hybrid)
    names = {entry.name for entry in result.monitored}
    assert leader.name not in names  # excluded, however active it is
    assert any("Fixture Unknown Pharma" in name for name in names)  # included, with no data
    assert next(e for e in result.monitored if "Fixture Unknown" in e.name).score == 0.0


def test_a_configured_competitor_appears_whether_or_not_the_run_persists(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    """A preview must show the same monitored list a real run would."""
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    hybrid = settings.model_copy(
        update={
            "competitor_mode": "HYBRID",
            "competitor_include": "Fixture Unknown Pharma",
            "top_competitors": 3,
        }
    )
    preview = {entry.name for entry in _discover(loaded, hybrid, persist=False).monitored}
    stored = {entry.name for entry in _discover(loaded, hybrid, persist=True).monitored}
    assert "Fixture Unknown Pharma" in preview
    assert preview == stored


def test_a_named_organization_is_ranked_even_when_its_type_is_excluded(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    """Universities are not competitors by default, but naming one overrides that."""
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    without = {
        e.name
        for e in _discover(
            loaded, settings.model_copy(update={"competitor_mode": "AUTO"})
        ).monitored
    }
    assert not any("Harrowgate" in name for name in without)

    named = settings.model_copy(
        update={"competitor_mode": "MANUAL", "competitor_include": "Harrowgate University"}
    )
    monitored = _discover(loaded, named).monitored
    assert [entry.name for entry in monitored] == ["Harrowgate University"]
    # ranked on its real activity, not shown as a zero
    assert monitored[0].score > 0 and monitored[0].evidence_count > 0


def test_manual_mode_monitors_only_the_configured_list(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    manual = settings.model_copy(
        update={"competitor_mode": "MANUAL", "competitor_include": "Zentavia Pharma"}
    )
    result = _discover(loaded, manual)
    assert [entry.manually_included for entry in result.monitored] == [True]
    assert "Zentavia" in result.monitored[0].name


def test_hybrid_fills_the_remaining_slots(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    hybrid = settings.model_copy(
        update={
            "competitor_mode": "HYBRID",
            "competitor_include": "Zentavia Pharma",
            "top_competitors": 4,
        }
    )
    result = _discover(loaded, hybrid)
    assert len(result.monitored) == 4
    assert sum(1 for entry in result.monitored if entry.manually_included) == 1


def test_missing_source_types_share_their_weight(
    factory: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        dataset = generate_demo_dataset(DemoConfig(scale=0.2))
        dataset.records = [r for r in dataset.records if r.record_type in ("publication", "patent")]
        load_demo_dataset(session, dataset)
        normalize_records(session, settings, taxonomy)
    result = _discover(factory, settings.model_copy(update={"competitor_mode": "AUTO"}))
    assert set(result.unavailable_sources) >= {"trial", "funding", "announcement"}
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    assert any("weight was shared" in warning for warning in result.warnings)


def test_discovery_records_who_is_monitored(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy)
    auto = settings.model_copy(update={"competitor_mode": "AUTO", "top_competitors": 3})
    result = _discover(loaded, auto)
    with session_scope(loaded) as session:
        flagged = {
            o.canonical_name
            for o in session.scalars(
                select(Organization).where(Organization.discovered_automatically.is_(True))
            )
        }
    assert flagged == {entry.name for entry in result.monitored}


def test_discovery_without_data_says_so(
    factory: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
    result = _discover(factory, settings.model_copy(update={"competitor_mode": "AUTO"}))
    assert result.monitored == []
    assert any("nothing to rank" in warning for warning in result.warnings)


# --------------------------------------------------------------------------------------
# Committing in batches: an interrupted run keeps what it finished
# --------------------------------------------------------------------------------------
BATCH = 100


def _fingerprint(factory: sessionmaker[Session]) -> dict[str, Any]:
    """Everything normalization produces, in a form that can be compared between runs."""
    with session_scope(factory) as session:
        return {
            "statuses": {
                status: count
                for status, count in session.execute(
                    select(SourceRecord.processing_status, func.count()).group_by(
                        SourceRecord.processing_status
                    )
                ).all()
            },
            "organizations": sorted(
                (o.canonical_name, o.organization_type)
                for o in session.scalars(select(Organization))
            ),
            "org_links": session.scalar(select(func.count()).select_from(RecordOrganization)),
            "topic_links": session.scalar(select(func.count()).select_from(RecordTopic)),
            "review": sorted(
                (q.queue_type, q.subject_ref) for q in session.scalars(select(ReviewQueueItem))
            ),
        }


class Interrupted(Exception):
    """Stands in for the process dying part-way through."""


def test_committing_each_batch_gives_exactly_the_same_result(
    settings: Settings, taxonomy: Taxonomy
) -> None:
    def run(commit: bool) -> dict[str, Any]:
        engine = create_memory_engine()
        upgrade_database(engine)
        factory = create_session_factory(engine)
        with session_scope(factory) as session:
            sync_taxonomy(session, taxonomy)
            load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.2)))
        with session_scope(factory) as session:
            normalize_records(
                session, settings, taxonomy, batch_size=BATCH, commit_each_batch=commit
            )
        result = _fingerprint(factory)
        engine.dispose()
        return result

    assert run(True) == run(False)


def test_an_interrupted_run_keeps_the_batches_it_finished_and_the_next_run_completes_it(
    settings: Settings, taxonomy: Taxonomy
) -> None:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.2)))
    total = _count(factory, SourceRecord)
    assert total > 3 * BATCH  # enough records for the interruption to fall mid-run

    def die_after_two_batches(done: int, _total: int) -> None:
        if done >= 2 * BATCH:
            raise Interrupted

    with pytest.raises(Interrupted), session_scope(factory) as session:
        normalize_records(
            session, settings, taxonomy, batch_size=BATCH, commit_each_batch=True,
            progress=die_after_two_batches,
        )  # fmt: skip

    after_crash = _fingerprint(factory)
    finished = after_crash["statuses"].get(ProcessingStatus.NORMALIZED.value, 0)
    assert finished == 2 * BATCH  # the two finished batches survived the failure
    assert after_crash["statuses"].get(ProcessingStatus.NEW.value, 0) == total - finished

    with session_scope(factory) as session:  # the "next run": carries on, does not start over
        summary = normalize_records(
            session, settings, taxonomy, batch_size=BATCH, commit_each_batch=True
        )
    assert summary.records_processed == total - finished

    # and the end state is what one uninterrupted run gives
    clean_engine = create_memory_engine()
    upgrade_database(clean_engine)
    clean = create_session_factory(clean_engine)
    with session_scope(clean) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.2)))
    with session_scope(clean) as session:
        normalize_records(session, settings, taxonomy, batch_size=BATCH)
    assert _fingerprint(factory) == _fingerprint(clean)
    engine.dispose()
    clean_engine.dispose()


def test_without_the_option_an_interruption_still_loses_everything(
    settings: Settings, taxonomy: Taxonomy
) -> None:
    """The default is unchanged: the caller owns the transaction, so a failure rolls it all back."""
    engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.2)))

    def die(done: int, _total: int) -> None:
        if done >= 2 * BATCH:
            raise Interrupted

    with pytest.raises(Interrupted), session_scope(factory) as session:
        normalize_records(session, settings, taxonomy, batch_size=BATCH, progress=die)
    assert _fingerprint(factory)["statuses"].get(ProcessingStatus.NORMALIZED.value, 0) == 0
    engine.dispose()


def test_committing_in_batches_does_not_keep_every_record_in_memory(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy, batch_size=BATCH, commit_each_batch=True)
        held = sum(1 for obj in session.identity_map.values() if isinstance(obj, SourceRecord))
        total = session.scalar(select(func.count()).select_from(SourceRecord)) or 0
    assert total > 3 * BATCH
    assert held <= BATCH, f"{held} of {total} records were still held in memory"


def test_names_added_by_the_resolver_are_still_saved_after_records_are_released(
    loaded: sessionmaker[Session], settings: Settings, taxonomy: Taxonomy
) -> None:
    """Releasing records must not detach the organizations the resolver keeps updating."""
    with session_scope(loaded) as session:
        normalize_records(session, settings, taxonomy, batch_size=BATCH, commit_each_batch=True)
    with session_scope(loaded) as session:
        mentions = session.execute(select(func.sum(OrganizationAlias.mention_count))).scalar() or 0
    assert mentions > 0
