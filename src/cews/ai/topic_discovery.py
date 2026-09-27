"""Finding topics the taxonomy does not have a name for.

Deterministic keyword matching only finds what someone already thought to write down. This
clusters the text of unmatched records with embeddings, and reports any cluster that sits far
from every existing topic as a **candidate** — never written to the taxonomy automatically, and
never scored or trended until a person promotes it.

The clustering itself (DBSCAN over cosine distance) needs no cluster count chosen in advance and
lets records that fit nowhere stay unclustered rather than being forced into the nearest group.
Cluster "names" are drawn from the cluster's own text (the most distinctive terms in it, by
TF-IDF) rather than invented by a model, so a name can always be traced back to the words that
produced it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.ai.embeddings import EmbeddingCache, EmbeddingUnavailableError, embed_texts
from cews.constants import TopicStatus
from cews.database.models import ReviewQueueItem, Topic, TopicAlias
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

CANDIDATE_TOPIC_TYPE = "ai_candidate"
QUEUE_TYPE = "ai_topic_candidate"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MAX_DISTANCE = 0.35  # DBSCAN eps in cosine-distance space
TOP_TERMS_PER_CLUSTER = 6
EmbedFn = Callable[[list[str]], np.ndarray]


@dataclass(frozen=True)
class TopicCandidate:
    """One cluster of unmatched text that does not resemble any existing topic."""

    label: str
    terms: tuple[str, ...]
    record_ids: tuple[int, ...]
    novelty: float
    """``1 - cosine_similarity`` to the nearest existing topic. Cosine similarity ranges -1 to 1,
    so novelty ranges 0 (identical to something that exists) to 2 (as different as two vectors
    can be); it is a distance, not a probability, and is not clamped to 0-1."""
    nearest_existing_topic: str | None
    nearest_existing_similarity: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored as the review-queue payload."""
        return {
            "label": self.label,
            "terms": list(self.terms),
            "record_count": len(self.record_ids),
            "record_ids": list(self.record_ids)[:20],
            "novelty": round(self.novelty, 4),
            "nearest_existing_topic": self.nearest_existing_topic,
            "nearest_existing_similarity": round(self.nearest_existing_similarity, 4),
        }


@dataclass
class DiscoveryRun:
    """What one discovery pass found."""

    candidates: list[TopicCandidate] = field(default_factory=list)
    records_considered: int = 0
    records_clustered: int = 0
    clusters_matching_existing_topics: int = 0
    topics_created: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "records_considered": self.records_considered,
            "records_clustered": self.records_clustered,
            "clusters_matching_existing_topics": self.clusters_matching_existing_topics,
            "candidates_found": len(self.candidates),
            "topics_created": self.topics_created,
            "warnings": list(self.warnings),
        }


def _default_embed(settings: Settings, cache: EmbeddingCache | None) -> EmbedFn:
    def call(texts: list[str]) -> np.ndarray:
        return embed_texts(texts, model_name=settings.embedding_model, cache=cache)

    return call


def _top_terms(texts: Sequence[str], *, count: int = TOP_TERMS_PER_CLUSTER) -> tuple[str, ...]:
    """The most distinctive words in a small set of texts, by TF-IDF.

    A name drawn from the cluster's own words can always be traced back to what produced it,
    unlike a name a model invented.
    """
    if len(texts) < 2:
        return tuple(texts[0].split()[:count]) if texts else ()
    vectorizer = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:  # every text was pure stop-words or empty after filtering
        return ()
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = {index: term for term, index in vectorizer.vocabulary_.items()}
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return tuple(vocab[index] for index in ranked[:count] if index in vocab)


def discover_topic_candidates(
    session: Session,
    settings: Settings,
    texts_by_record: dict[int, str],
    *,
    existing_topic_texts: dict[str, str] | None = None,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    max_distance: float = DEFAULT_MAX_DISTANCE,
    novelty_threshold: float | None = None,
    embed: EmbedFn | None = None,
    cache: EmbeddingCache | None = None,
    store: bool = True,
) -> DiscoveryRun:
    """Cluster unmatched record text and report clusters that resemble no existing topic.

    Args:
        texts_by_record: source_record_id -> text (title plus abstract, already assembled by the
            caller) for records that matched nothing in the deterministic taxonomy pass.
        existing_topic_texts: topic name -> representative text (name plus synonyms), used to
            decide whether a cluster is genuinely new. Read from the database if not given.
        min_cluster_size: DBSCAN's minimum points to form a cluster.
        max_distance: DBSCAN's eps, in cosine distance (0 identical, 2 opposite).
        novelty_threshold: a cluster within this cosine similarity of an existing topic is
            treated as something the keyword matcher simply missed, not something new
            (default: ``settings.ai_topic_novelty_threshold``).
        embed: override the embedding function (tests inject a deterministic fake here so no
            network or model download is needed to exercise the clustering logic).
        cache: an :class:`EmbeddingCache` to reuse across runs.
        store: write discovered candidates as pending-review topics.

    Nothing is written to the taxonomy file, and no candidate becomes a scored topic until it is
    promoted through the review queue.

    Raises:
        ValueError: if ``min_cluster_size`` or ``max_distance`` is not positive.
    """
    if min_cluster_size < 1 or max_distance <= 0:
        raise ValueError("min_cluster_size and max_distance must be positive")
    threshold = (
        novelty_threshold if novelty_threshold is not None else settings.ai_topic_novelty_threshold
    )
    run = DiscoveryRun(records_considered=len(texts_by_record))
    if len(texts_by_record) < min_cluster_size:
        run.warnings.append(
            f"only {len(texts_by_record)} unmatched record(s); at least {min_cluster_size} are "
            "needed to form a cluster"
        )
        return run

    embed_fn = embed or _default_embed(settings, cache)
    record_ids = sorted(texts_by_record)
    texts = [texts_by_record[record_id] for record_id in record_ids]

    try:
        vectors = embed_fn(texts)
    except EmbeddingUnavailableError as exc:
        run.warnings.append(f"topic discovery skipped: {exc}")
        return run

    labels = DBSCAN(eps=max_distance, min_samples=min_cluster_size, metric="cosine").fit_predict(
        vectors
    )
    clusters: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(labels):
        if cluster_id == -1:  # DBSCAN's label for "does not belong to any cluster"
            continue
        clusters.setdefault(int(cluster_id), []).append(index)
    run.records_clustered = sum(len(members) for members in clusters.values())
    if not clusters:
        run.warnings.append("no cluster reached the minimum size; nothing looked like a group")
        return run

    reference_names, reference_texts = _existing_topic_texts(session, existing_topic_texts)
    reference_vectors = embed_fn(reference_texts) if reference_texts else None

    existing_labels = {row.alias.casefold() for row in session.scalars(select(TopicAlias))}
    for cluster_id, member_indices in sorted(clusters.items()):
        member_ids = tuple(record_ids[index] for index in member_indices)
        member_texts = [texts[index] for index in member_indices]
        centroid = vectors[member_indices].mean(axis=0, keepdims=True)

        nearest_name: str | None = None
        nearest_similarity = 0.0
        if reference_vectors is not None:
            similarities = cosine_similarity(centroid, reference_vectors)[0]
            best = int(np.argmax(similarities))
            nearest_name, nearest_similarity = reference_names[best], float(similarities[best])

        if nearest_similarity >= threshold:
            run.clusters_matching_existing_topics += 1
            continue  # the keyword matcher missed this, but the topic already exists

        terms = _top_terms(member_texts)
        label = ", ".join(terms[:3]).title() if terms else f"Unnamed cluster {cluster_id}"
        if label.casefold() in existing_labels:
            run.clusters_matching_existing_topics += 1
            continue
        candidate = TopicCandidate(
            label=label,
            terms=terms,
            record_ids=member_ids,
            novelty=1.0 - nearest_similarity,
            nearest_existing_topic=nearest_name,
            nearest_existing_similarity=nearest_similarity,
        )
        run.candidates.append(candidate)
        if store and _store_candidate(session, candidate):
            run.topics_created += 1

    if store:
        session.flush()
    LOGGER.info("topic discovery: %s", run.as_dict())
    return run


def _existing_topic_texts(
    session: Session, given: dict[str, str] | None
) -> tuple[list[str], list[str]]:
    if given is not None:
        return list(given), list(given.values())
    rows = session.execute(
        select(Topic.canonical_name, Topic.id).where(Topic.topic_type != CANDIDATE_TOPIC_TYPE)
    ).all()
    if not rows:
        return [], []
    alias_by_topic: dict[int, list[str]] = {}
    for topic_id, alias in session.execute(select(TopicAlias.topic_id, TopicAlias.alias)).all():
        alias_by_topic.setdefault(topic_id, []).append(str(alias))
    names = [str(name) for name, _ in rows]
    texts = [f"{name} {' '.join(alias_by_topic.get(topic_id, []))}" for name, topic_id in rows]
    return names, texts


def _store_candidate(session: Session, candidate: TopicCandidate) -> bool:
    """Write one candidate as a pending-review topic, linked but not yet scored.

    A candidate topic is created inactive and in ``pending_review``: it exists so its records can
    be linked to it, but the scoring and feature passes only ever look at active topics, so a
    candidate contributes nothing to a score until a person reviews and activates it.

    Returns False, and touches nothing, when this exact candidate is already sitting in the
    review queue: discovery is meant to be re-run on every pass over still-unmatched records, and
    the same cluster reappearing must never be reported as newly created twice.
    """
    key = "ai_candidate_" + "_".join(term.replace(" ", "_") for term in candidate.terms[:3])[:100]
    key = key or f"ai_candidate_{abs(hash(candidate.label))}"
    existing_topic = session.scalar(select(Topic).where(Topic.key == key))
    if existing_topic is not None:
        topic = existing_topic
    else:
        topic = Topic(
            key=key,
            canonical_name=candidate.label,
            topic_type=CANDIDATE_TOPIC_TYPE,
            status=TopicStatus.PENDING_REVIEW.value,
            active=False,
        )
        session.add(topic)
        session.flush()

    subject_ref = f"topic:{topic.id}"
    already_queued = session.scalar(
        select(ReviewQueueItem).where(
            ReviewQueueItem.queue_type == QUEUE_TYPE,
            ReviewQueueItem.subject_ref == subject_ref,
            ReviewQueueItem.status == "pending",
        )
    )
    if already_queued is not None:
        # The cluster is unchanged since the last pass; refresh the evidence (record count,
        # novelty) rather than leaving a stale payload, but do not add a second entry.
        already_queued.payload_json = candidate.as_dict()
        return False

    session.add(
        ReviewQueueItem(
            queue_type=QUEUE_TYPE,
            subject_ref=subject_ref,
            payload_json=candidate.as_dict(),
            status="pending",
            created_at=datetime.now(UTC),
        )
    )
    return True
