"""Tests for embedding-based topic discovery.

The real embedding model needs a network download, so every test injects a controlled synthetic
embedder instead: text belonging to the same underlying topic maps to a shared centre plus a
small amount of noise, exactly the geometry a real sentence-transformer would produce for
genuinely related sentences. This exercises the clustering and novelty logic on its own merits,
never on whether a downloaded model happens to agree.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.ai.topic_discovery import (
    TopicCandidate,
    _top_terms,
    discover_topic_candidates,
)
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ReviewQueueItem, Topic, TopicAlias
from cews.settings import load_settings

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(0)
DIMENSIONS = 16


def _synthetic_embedder(group_of: dict[str, str]) -> tuple[object, dict[str, np.ndarray]]:
    """Build a fake embedder: one random centre per group name, noise added per call."""
    centres = {
        group: RNG.normal(size=DIMENSIONS) for group in set(group_of.values()) if group != "noise"
    }

    def embed(texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            group = group_of[text]
            if group == "noise":
                vectors.append(RNG.normal(size=DIMENSIONS) * 4)
            else:
                vectors.append(centres[group] + RNG.normal(size=DIMENSIONS) * 0.05)
        return np.array(vectors)

    return embed, centres


NOVEL_TEXTS = [
    "PROTAC molecule design",
    "protein degrader trial",
    "molecular glue mechanism",
    "targeted degradation platform",
    "PROTAC therapeutics",
    "degrader drug candidate",
]
CRISPR_LIKE_TEXTS = [f"CRISPR editing therapy {letter}" for letter in "ABCDEF"]
NOISE_TEXTS = ["random noise one", "random noise two"]


CRISPR_REFERENCE_TEXT = "CRISPR gene editing CRISPR"  # what _existing_topic_texts assembles


def _group_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    mapping.update({text: "novel" for text in NOVEL_TEXTS})
    mapping.update({text: "crispr" for text in CRISPR_LIKE_TEXTS})
    mapping.update({text: "noise" for text in NOISE_TEXTS})
    # discover_topic_candidates also embeds each existing topic's own name+aliases, to measure
    # novelty against; that reference text belongs to the same semantic group as the CRISPR-like
    # cluster above, so it must map to the same synthetic centre.
    mapping[CRISPR_REFERENCE_TEXT] = "crispr"
    return mapping


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture
def settings() -> object:
    return load_settings(env_file=None)


def _seed_crispr_topic(session: Session) -> Topic:
    topic = Topic(
        key="crispr_gene_editing",
        canonical_name="CRISPR gene editing",
        topic_type="technology",
        active=True,
    )
    session.add(topic)
    session.flush()
    session.add(TopicAlias(topic_id=topic.id, alias="CRISPR"))
    return topic


def _texts_by_record() -> dict[int, str]:
    texts_by_record: dict[int, str] = {}
    record_id = 1
    for text in NOVEL_TEXTS + CRISPR_LIKE_TEXTS + NOISE_TEXTS:
        texts_by_record[record_id] = text
        record_id += 1
    return texts_by_record


# --------------------------------------------------------------------------------------
# The core behaviour: novel clusters are found, existing ones are not duplicated
# --------------------------------------------------------------------------------------
def test_a_genuinely_novel_cluster_is_reported(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        run = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    assert len(run.candidates) == 1
    candidate = run.candidates[0]
    assert len(candidate.record_ids) == 6
    assert candidate.nearest_existing_topic == "CRISPR gene editing"
    assert run.topics_created == 1


def test_a_cluster_resembling_an_existing_topic_is_not_duplicated(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        run = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    assert run.clusters_matching_existing_topics == 1
    assert all("CRISPR" not in candidate.label for candidate in run.candidates)


def test_noise_records_never_form_a_cluster(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        run = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    clustered_ids = {
        record_id for candidate in run.candidates for record_id in candidate.record_ids
    }
    noise_ids = set(range(13, 15))  # the last two records are the noise texts
    assert not (clustered_ids & noise_ids)
    assert run.records_clustered < run.records_considered


def test_a_discovered_topic_is_created_inactive_and_pending_review(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    with session_scope(factory) as session:
        candidate_topic = session.scalar(select(Topic).where(Topic.topic_type == "ai_candidate"))
    assert candidate_topic is not None
    assert candidate_topic.active is False
    assert candidate_topic.status == "pending_review"


def test_a_candidate_is_queued_for_human_review(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    with session_scope(factory) as session:
        item = session.scalar(
            select(ReviewQueueItem).where(ReviewQueueItem.queue_type == "ai_topic_candidate")
        )
    assert item is not None
    assert item.status == "pending"
    assert item.payload_json["record_count"] == 6


# --------------------------------------------------------------------------------------
# Idempotency: this is the bug that was found and fixed
# --------------------------------------------------------------------------------------
def test_rerunning_discovery_does_not_crash_or_duplicate(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        first = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    with session_scope(factory) as session:
        second = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    assert first.topics_created == 1
    assert second.topics_created == 0  # nothing new; must not be recreated
    with session_scope(factory) as session:
        topic_count = session.scalar(select(func.count()).select_from(Topic))
        queue_count = session.scalar(select(func.count()).select_from(ReviewQueueItem))
    assert topic_count == 2  # the seeded CRISPR topic plus exactly one candidate
    assert queue_count == 1


def test_a_rerun_refreshes_the_queued_evidence(
    factory: sessionmaker[Session], settings: object
) -> None:
    """The queue entry is updated in place, not silently left stale."""
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
        discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    with session_scope(factory) as session:
        item = session.scalar(
            select(ReviewQueueItem).where(ReviewQueueItem.queue_type == "ai_topic_candidate")
        )
    assert item is not None and item.payload_json["record_count"] == 6


# --------------------------------------------------------------------------------------
# Guardrails: nothing is scored until a person approves it
# --------------------------------------------------------------------------------------
def test_nothing_is_written_to_the_taxonomy_file(
    factory: sessionmaker[Session], settings: object, tmp_path: object
) -> None:
    """Only the database changes; config/topic_taxonomy.yaml is never touched."""

    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,  # type: ignore[arg-type]
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
        )
    # the assertion is structural: discover_topic_candidates never imports or opens any yaml file
    import inspect

    from cews.ai import topic_discovery

    source = inspect.getsource(topic_discovery)
    assert "yaml" not in source and "taxonomy_file" not in source


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------
def test_too_few_unmatched_records_is_reported_not_guessed(
    factory: sessionmaker[Session], settings: object
) -> None:
    with session_scope(factory) as session:
        run = discover_topic_candidates(
            session,
            settings,
            {1: "one lonely record"},
            embed=lambda texts: np.zeros((len(texts), 4)),
            min_cluster_size=5,
        )
    assert run.candidates == []
    assert any("at least 5" in warning for warning in run.warnings)


def test_no_cluster_reaching_minimum_size_is_reported(
    factory: sessionmaker[Session], settings: object
) -> None:
    """Five completely unrelated texts should never be forced into one cluster."""
    texts = {index: f"utterly unrelated sentence number {index}" for index in range(1, 6)}

    def scattered(items: list[str]) -> np.ndarray:
        return RNG.normal(size=(len(items), DIMENSIONS)) * 10  # push everything far apart

    with session_scope(factory) as session:
        run = discover_topic_candidates(
            session,
            settings,
            texts,
            embed=scattered,
            min_cluster_size=5,
            max_distance=0.01,
        )
    assert run.candidates == []


def test_embedding_unavailable_is_reported_not_raised(
    factory: sessionmaker[Session], settings: object
) -> None:
    from cews.ai.embeddings import EmbeddingUnavailableError

    def broken(texts: list[str]) -> np.ndarray:
        raise EmbeddingUnavailableError("no model available in this environment")

    with session_scope(factory) as session:
        run = discover_topic_candidates(
            session,
            settings,
            {index: f"text {index}" for index in range(1, 8)},
            embed=broken,
        )
    assert run.candidates == []
    assert any("topic discovery skipped" in warning for warning in run.warnings)


@pytest.mark.parametrize(("min_cluster_size", "max_distance"), [(0, 0.3), (5, 0.0), (5, -0.1)])
def test_invalid_parameters_are_refused(
    factory: sessionmaker[Session], settings: object, min_cluster_size: int, max_distance: float
) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        discover_topic_candidates(
            session,
            settings,
            {1: "a", 2: "b", 3: "c", 4: "d", 5: "e"},
            embed=lambda texts: np.zeros((len(texts), 4)),
            min_cluster_size=min_cluster_size,
            max_distance=max_distance,
        )


def test_store_false_finds_candidates_without_writing_them(
    factory: sessionmaker[Session], settings: object
) -> None:
    embed, _ = _synthetic_embedder(_group_map())
    with session_scope(factory) as session:
        _seed_crispr_topic(session)
        run = discover_topic_candidates(
            session,
            settings,
            _texts_by_record(),
            embed=embed,
            min_cluster_size=5,
            max_distance=0.3,
            novelty_threshold=0.5,
            store=False,
        )
    assert len(run.candidates) == 1
    assert run.topics_created == 0
    with session_scope(factory) as session:
        assert session.scalar(select(func.count()).select_from(ReviewQueueItem)) == 0


def test_a_candidate_is_json_friendly() -> None:
    import json

    candidate = TopicCandidate(
        label="Targeted Protein Degradation",
        terms=("protac", "degrader"),
        record_ids=(1, 2, 3),
        novelty=1.2,
        nearest_existing_topic="CRISPR gene editing",
        nearest_existing_similarity=-0.2,
    )
    json.dumps(candidate.as_dict())
    assert candidate.as_dict()["record_count"] == 3


# --------------------------------------------------------------------------------------
# _top_terms
# --------------------------------------------------------------------------------------
def test_top_terms_picks_distinctive_words() -> None:
    terms = _top_terms(
        [
            "PROTAC molecule design for cancer",
            "PROTAC degrader trial results",
            "molecular glue degrader mechanism",
        ]
    )
    assert any("protac" in term or "degrader" in term for term in terms)


def test_top_terms_handles_a_single_text() -> None:
    assert _top_terms(["one two three"]) == ("one", "two", "three")


def test_top_terms_handles_no_text() -> None:
    assert _top_terms([]) == ()


def test_top_terms_handles_only_stop_words() -> None:
    assert _top_terms(["the a an", "of in on"]) == ()
