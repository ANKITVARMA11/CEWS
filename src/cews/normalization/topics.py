"""Topic taxonomy: loading, database synchronization and keyword assignment.

:func:`load_taxonomy` validates ``config/topic_taxonomy.yaml`` and :func:`sync_taxonomy` mirrors
it into the ``therapeutic_areas``, ``topics`` and ``topic_aliases`` tables. :class:`TopicMatcher`
then assigns topics to records by matching the taxonomy's names and synonyms against their text.

Matching is deterministic and explainable: every assignment records which terms matched, in
which field, and a confidence that reflects both. A record can belong to several topics.
Each therapeutic area is stored as a topic too (key ``area:<id>``), so a record about a
monitored area that has no narrower topic of its own still counts towards it.
Embedding-based discovery of topics the taxonomy does not contain arrives in Phase 9.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import TopicStatus
from cews.database.models import TherapeuticArea, Topic, TopicAlias

LOGGER = logging.getLogger(__name__)


class TaxonomyError(ValueError):
    """Raised when the taxonomy file is missing or inconsistent."""


@dataclass(frozen=True)
class TaxonomyArea:
    """A therapeutic area (optionally nested under a parent area)."""

    key: str
    name: str
    parent: str | None
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class TaxonomyTopic:
    """A research topic, optionally attached to a therapeutic area."""

    key: str
    name: str
    topic_type: str
    area: str | None
    synonyms: tuple[str, ...]


@dataclass(frozen=True)
class Taxonomy:
    """The parsed taxonomy plus matching defaults."""

    version: int
    areas: tuple[TaxonomyArea, ...]
    topics: tuple[TaxonomyTopic, ...]
    default_method: str
    min_confidence: float


AREA_TOPIC_PREFIX = "area:"
AREA_TOPIC_TYPE = "therapeutic_area"


def area_topic_key(area_key: str) -> str:
    """The topic key that represents a therapeutic area itself."""
    return f"{AREA_TOPIC_PREFIX}{area_key}"


@dataclass(frozen=True)
class SyncSummary:
    """Counts of rows created or updated by :func:`sync_taxonomy`."""

    areas_created: int = 0
    areas_updated: int = 0
    topics_created: int = 0
    topics_updated: int = 0
    aliases_created: int = 0


def _synonyms(entry: dict[str, Any], label: str) -> tuple[str, ...]:
    raw = entry.get("synonyms") or []
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise TaxonomyError(f"{label}: 'synonyms' must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in raw))


def _require(entry: Any, key: str, label: str) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get(key), str) or not entry[key].strip():
        raise TaxonomyError(f"{label}: missing or empty '{key}'")
    return str(entry[key]).strip()


def load_taxonomy(path: Path) -> Taxonomy:
    """Load and validate a taxonomy YAML file.

    Checks: file exists, keys are unique, parents and areas exist, and the area hierarchy
    has no cycles.

    Raises:
        TaxonomyError: for a missing file or any inconsistency.
    """
    if not path.is_file():
        raise TaxonomyError(f"taxonomy file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise TaxonomyError(f"cannot read taxonomy {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TaxonomyError(f"{path} must contain a mapping")

    areas: list[TaxonomyArea] = []
    for index, entry in enumerate(data.get("therapeutic_areas") or []):
        label = f"therapeutic_areas[{index}]"
        key = _require(entry, "id", label)
        parent = entry.get("parent")
        areas.append(
            TaxonomyArea(
                key, _require(entry, "name", label), parent or None, _synonyms(entry, label)
            )
        )
    area_keys = [a.key for a in areas]
    if len(area_keys) != len(set(area_keys)):
        raise TaxonomyError("duplicate therapeutic area ids in taxonomy")
    parents = {a.key: a.parent for a in areas}
    for key, parent in parents.items():
        if parent is not None and parent not in parents:
            raise TaxonomyError(f"therapeutic area '{key}' has unknown parent '{parent}'")
    for key in parents:
        seen = {key}
        current = parents[key]
        while current is not None:
            if current in seen:
                raise TaxonomyError(f"cycle in therapeutic area hierarchy at '{key}'")
            seen.add(current)
            current = parents[current]

    topics: list[TaxonomyTopic] = []
    for index, entry in enumerate(data.get("topics") or []):
        label = f"topics[{index}]"
        key = _require(entry, "id", label)
        area = entry.get("area")
        if area is not None and area not in parents:
            raise TaxonomyError(f"topic '{key}' refers to unknown area '{area}'")
        topics.append(
            TaxonomyTopic(
                key,
                _require(entry, "name", label),
                str(entry.get("type") or "topic"),
                area,
                _synonyms(entry, label),
            )
        )
    topic_keys = [t.key for t in topics]
    if len(topic_keys) != len(set(topic_keys)):
        raise TaxonomyError("duplicate topic ids in taxonomy")

    matching = data.get("matching") or {}
    min_confidence = float(matching.get("min_confidence", 0.5))
    if not 0 <= min_confidence <= 1:
        raise TaxonomyError("matching.min_confidence must be between 0 and 1")
    return Taxonomy(
        version=int(data.get("taxonomy_version", 1)),
        areas=tuple(areas),
        topics=tuple(topics),
        default_method=str(matching.get("default_method", "keyword")),
        min_confidence=min_confidence,
    )


def sync_taxonomy(session: Session, taxonomy: Taxonomy) -> SyncSummary:
    """Mirror the taxonomy into the database. Existing rows are updated, never duplicated.

    Topics discovered by the AI layer (status ``pending_review``) and any topic not in the
    taxonomy are left untouched.
    """
    areas_created = areas_updated = topics_created = topics_updated = aliases_created = 0

    existing_areas = {a.key: a for a in session.scalars(select(TherapeuticArea))}
    for area in taxonomy.areas:
        row = existing_areas.get(area.key)
        if row is None:
            row = TherapeuticArea(key=area.key, canonical_name=area.name)
            session.add(row)
            existing_areas[area.key] = row
            areas_created += 1
        elif row.canonical_name != area.name:
            row.canonical_name = area.name
            areas_updated += 1
    session.flush()
    for area in taxonomy.areas:
        row = existing_areas[area.key]
        parent_id = existing_areas[area.parent].id if area.parent else None
        if row.parent_id != parent_id:
            row.parent_id = parent_id
    session.flush()

    existing_topics = {t.key: t for t in session.scalars(select(Topic))}
    # Each therapeutic area is also a topic, so a record about "CAR-T" (an area with no
    # narrower topic in the taxonomy) still counts towards that area.
    for area in taxonomy.areas:
        key = area_topic_key(area.key)
        row_area = existing_topics.get(key)
        area_id = existing_areas[area.key].id
        if row_area is None:
            row_area = Topic(
                key=key,
                canonical_name=area.name,
                topic_type=AREA_TOPIC_TYPE,
                therapeutic_area_id=area_id,
                status=TopicStatus.ACTIVE.value,
                active=True,
            )
            session.add(row_area)
            existing_topics[key] = row_area
            topics_created += 1
        elif row_area.canonical_name != area.name or row_area.therapeutic_area_id != area_id:
            row_area.canonical_name = area.name
            row_area.therapeutic_area_id = area_id
            topics_updated += 1
    session.flush()
    for area in taxonomy.areas:
        row_area = existing_topics[area_topic_key(area.key)]
        parent_key = area_topic_key(area.parent) if area.parent else None
        row_area.parent_id = existing_topics[parent_key].id if parent_key else None
    session.flush()

    for topic in taxonomy.topics:
        topic_area_id = existing_areas[topic.area].id if topic.area else None
        row_t = existing_topics.get(topic.key)
        if row_t is None:
            row_t = Topic(
                key=topic.key,
                canonical_name=topic.name,
                topic_type=topic.topic_type,
                therapeutic_area_id=topic_area_id,
                status=TopicStatus.ACTIVE.value,
                active=True,
            )
            session.add(row_t)
            existing_topics[topic.key] = row_t
            topics_created += 1
        else:
            changed = (
                row_t.canonical_name != topic.name
                or row_t.topic_type != topic.topic_type
                or row_t.therapeutic_area_id != topic_area_id
            )
            if changed:
                row_t.canonical_name = topic.name
                row_t.topic_type = topic.topic_type
                row_t.therapeutic_area_id = topic_area_id
                topics_updated += 1
    session.flush()

    known_aliases = {(a.topic_id, a.alias.casefold()) for a in session.scalars(select(TopicAlias))}
    for area in taxonomy.areas:
        area_topic_id = existing_topics[area_topic_key(area.key)].id
        for alias in (area.name, *area.synonyms):
            marker = (area_topic_id, alias.casefold())
            if marker not in known_aliases:
                session.add(
                    TopicAlias(
                        topic_id=area_topic_id,
                        alias=alias,
                        matching_method="keyword",
                        confidence=1.0,
                    )
                )
                known_aliases.add(marker)
                aliases_created += 1
    for topic in taxonomy.topics:
        topic_id = existing_topics[topic.key].id
        for alias in topic.synonyms:
            marker = (topic_id, alias.casefold())
            if marker not in known_aliases:
                session.add(
                    TopicAlias(
                        topic_id=topic_id, alias=alias, matching_method="keyword", confidence=1.0
                    )
                )
                known_aliases.add(marker)
                aliases_created += 1
    session.flush()

    summary = SyncSummary(
        areas_created, areas_updated, topics_created, topics_updated, aliases_created
    )
    LOGGER.info("taxonomy synced: %s", summary)
    return summary


# --------------------------------------------------------------------------------------
# Keyword assignment
# --------------------------------------------------------------------------------------
# How much weight a match carries, by the field it was found in.
FIELD_CONFIDENCE: dict[str, float] = {
    "title": 0.85,
    "keywords": 0.8,
    "conditions": 0.8,
    "mesh_terms": 0.8,
    "interventions": 0.75,
    "abstract": 0.65,
}
DEFAULT_FIELD_CONFIDENCE = 0.6
EXTRA_TERM_BONUS = 0.05
MAX_CONFIDENCE = 0.95


@dataclass(frozen=True)
class TopicMatch:
    """One topic assigned to a record, with the evidence for it."""

    topic_key: str
    confidence: float
    method: str
    matched_terms: tuple[str, ...]
    matched_fields: tuple[str, ...]


def _term_pattern(term: str) -> re.Pattern[str]:
    """A pattern for one term: separator-insensitive, plural-tolerant, whole words only."""
    parts = [re.escape(part) for part in re.split(r"[\s-]+", term.strip()) if part]
    if not parts:
        raise ValueError("empty term")
    body = r"[\s-]+".join(parts)
    if not parts[-1].lower().endswith("s"):
        body += "s?"
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.IGNORECASE)


class TopicMatcher:
    """Assigns taxonomy topics to record text by keyword and phrase matching."""

    def __init__(self, taxonomy: Taxonomy, *, min_confidence: float | None = None) -> None:
        """Compile one pattern per taxonomy term."""
        self.taxonomy = taxonomy
        self.min_confidence = taxonomy.min_confidence if min_confidence is None else min_confidence
        self._patterns: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        entries: list[tuple[str, list[str]]] = [
            (area_topic_key(area.key), [area.name, *area.synonyms]) for area in taxonomy.areas
        ]
        entries += [(topic.key, [topic.name, *topic.synonyms]) for topic in taxonomy.topics]
        for key, terms in entries:
            compiled = []
            for term in dict.fromkeys(terms):
                try:
                    compiled.append((term, _term_pattern(term)))
                except (ValueError, re.error):  # pragma: no cover - guarded by taxonomy checks
                    LOGGER.warning("skipping unusable taxonomy term %r", term)
            self._patterns[key] = compiled

    def match(self, fields: Mapping[str, Any]) -> list[TopicMatch]:
        """Return the topics found in ``fields`` (field name -> text or list of texts).

        Field names drive confidence: a hit in a title or in structured metadata counts for
        more than one in an abstract. Matching several distinct terms raises confidence a
        little. Results are sorted by confidence, then topic key.
        """
        texts: list[tuple[str, str]] = []
        for field_name, value in fields.items():
            if value is None:
                continue
            if isinstance(value, str):
                texts.append((field_name, value))
            elif isinstance(value, (list, tuple)):
                joined = " ; ".join(str(item) for item in value if item)
                if joined:
                    texts.append((field_name, joined))
        if not texts:
            return []

        matches: list[TopicMatch] = []
        for topic_key, patterns in self._patterns.items():
            terms: dict[str, None] = {}
            found_fields: dict[str, None] = {}
            best_field_confidence = 0.0
            for term, pattern in patterns:
                for field_name, text in texts:
                    if pattern.search(text):
                        terms[term] = None
                        found_fields[field_name] = None
                        best_field_confidence = max(
                            best_field_confidence,
                            FIELD_CONFIDENCE.get(field_name, DEFAULT_FIELD_CONFIDENCE),
                        )
            if not terms:
                continue
            confidence = min(
                MAX_CONFIDENCE, best_field_confidence + EXTRA_TERM_BONUS * (len(terms) - 1)
            )
            if confidence < self.min_confidence:
                continue
            matches.append(
                TopicMatch(
                    topic_key=topic_key,
                    confidence=round(confidence, 3),
                    method=self.taxonomy.default_method,
                    matched_terms=tuple(terms),
                    matched_fields=tuple(found_fields),
                )
            )
        matches.sort(key=lambda m: (-m.confidence, m.topic_key))
        return matches
