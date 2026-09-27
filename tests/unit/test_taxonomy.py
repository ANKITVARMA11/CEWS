"""Unit tests for taxonomy loading and database synchronization."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import TopicStatus
from cews.database.models import TherapeuticArea, Topic, TopicAlias
from cews.normalization.topics import TaxonomyError, load_taxonomy, sync_taxonomy
from support import TAXONOMY_FILE

pytestmark = pytest.mark.unit

VALID = """
therapeutic_areas:
  - {id: onc, name: Oncology, parent: null, synonyms: [cancer]}
  - {id: vax, name: Vaccines, parent: onc, synonyms: [vaccine]}
topics:
  - {id: t1, name: Topic One, type: modality, area: vax, synonyms: [alpha, Beta]}
  - {id: t2, name: Topic Two, area: null}
matching: {default_method: keyword, min_confidence: 0.6}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_taxonomy_loads() -> None:
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    assert len(taxonomy.areas) == 8
    assert len(taxonomy.topics) == 8
    assert taxonomy.default_method == "keyword"
    keys = {t.key for t in taxonomy.topics}
    assert "crispr_gene_editing" in keys
    assert "targeted_protein_degradation" not in keys  # deliberately absent (AI discovery demo)


def test_valid_custom_taxonomy(tmp_path: Path) -> None:
    taxonomy = load_taxonomy(_write(tmp_path, VALID))
    assert [a.key for a in taxonomy.areas] == ["onc", "vax"]
    assert taxonomy.areas[1].parent == "onc"
    assert taxonomy.topics[0].synonyms == ("alpha", "Beta")
    assert taxonomy.topics[1].topic_type == "topic"
    assert taxonomy.min_confidence == 0.6


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (VALID.replace("id: t2", "id: t1"), "duplicate topic"),
        (VALID.replace("id: vax", "id: onc"), "duplicate therapeutic area"),
        (VALID.replace("parent: onc", "parent: ghost"), "unknown parent"),
        (VALID.replace("area: vax", "area: ghost"), "unknown area"),
        (VALID.replace("name: Oncology", "name: ''"), "missing or empty 'name'"),
        (VALID.replace("[alpha, Beta]", "alpha"), "list of non-empty strings"),
        (VALID.replace("min_confidence: 0.6", "min_confidence: 3"), "min_confidence"),
        ("- just\n- a list\n", "mapping"),
        ("topics: [unclosed", "cannot read"),
    ],
)
def test_invalid_taxonomies_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(TaxonomyError, match=message):
        load_taxonomy(_write(tmp_path, text))


def test_area_cycle_is_rejected(tmp_path: Path) -> None:
    cyclic = """
therapeutic_areas:
  - {id: a, name: A, parent: b}
  - {id: b, name: B, parent: a}
"""
    with pytest.raises(TaxonomyError, match="cycle"):
        load_taxonomy(_write(tmp_path, cyclic))


def test_missing_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(tmp_path / "absent.yaml")


def test_sync_creates_rows_and_is_idempotent(tmp_path: Path, session: Session) -> None:
    taxonomy = load_taxonomy(_write(tmp_path, VALID))
    first = sync_taxonomy(session, taxonomy)
    # 2 areas, and a topic row for each area as well as each topic
    assert (first.areas_created, first.topics_created) == (2, 4)
    assert first.aliases_created == 6  # 2 topic synonyms + each area's name and synonym
    second = sync_taxonomy(session, taxonomy)
    assert (second.areas_created, second.topics_created, second.aliases_created) == (0, 0, 0)
    assert session.scalar(select(func.count()).select_from(TherapeuticArea)) == 2
    assert session.scalar(select(func.count()).select_from(Topic)) == 4
    assert session.scalar(select(func.count()).select_from(TopicAlias)) == 6


def test_each_therapeutic_area_becomes_a_topic(tmp_path: Path, session: Session) -> None:
    sync_taxonomy(session, load_taxonomy(_write(tmp_path, VALID)))
    area_topic = session.scalar(select(Topic).where(Topic.key == "area:vax"))
    parent_topic = session.scalar(select(Topic).where(Topic.key == "area:onc"))
    assert area_topic is not None and parent_topic is not None
    assert area_topic.topic_type == "therapeutic_area"
    assert area_topic.parent_id == parent_topic.id  # the area hierarchy is mirrored


def test_sync_sets_hierarchy_and_updates_changes(tmp_path: Path, session: Session) -> None:
    sync_taxonomy(session, load_taxonomy(_write(tmp_path, VALID)))
    vax = session.scalar(select(TherapeuticArea).where(TherapeuticArea.key == "vax"))
    onc = session.scalar(select(TherapeuticArea).where(TherapeuticArea.key == "onc"))
    assert vax is not None and onc is not None and vax.parent_id == onc.id
    topic = session.scalar(select(Topic).where(Topic.key == "t1"))
    assert topic is not None and topic.therapeutic_area_id == vax.id

    renamed = VALID.replace("name: Topic One", "name: Renamed Topic").replace(
        "[alpha, Beta]", "[alpha, Beta, gamma]"
    )
    summary = sync_taxonomy(session, load_taxonomy(_write(tmp_path, renamed)))
    assert summary.topics_updated == 1 and summary.aliases_created == 1
    session.refresh(topic)
    assert topic.canonical_name == "Renamed Topic"


def test_sync_leaves_ai_discovered_topics_alone(tmp_path: Path, session: Session) -> None:
    session.add(
        Topic(
            key="ai_1",
            canonical_name="Candidate",
            topic_type="ai_discovered",
            status=TopicStatus.PENDING_REVIEW.value,
            active=False,
        )
    )
    session.flush()
    sync_taxonomy(session, load_taxonomy(_write(tmp_path, VALID)))
    candidate = session.scalar(select(Topic).where(Topic.key == "ai_1"))
    assert candidate is not None and candidate.status == "pending_review"
