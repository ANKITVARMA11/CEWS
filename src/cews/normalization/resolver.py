"""Resolving raw organization names to rows in the ``organizations`` table.

The resolver never guesses silently. It tries, in order:

1. the deterministic match keys from :mod:`cews.normalization.organizations` (exact, normalized,
   abbreviation-expanded, and space-free forms), which is how most spelling variants are caught;
2. a **prefix** rule for short names (``Zentavia`` against ``Zentavia Pharmaceuticals``), used
   only when exactly one organization fits;
3. **fuzzy** similarity, and only when the two names do not differ by a word that marks a
   different business (``Orvexa Biosciences`` vs ``Orvexa Laboratories``).

Anything below the automatic threshold creates a **new** organization and a review-queue item
instead of merging, so uncertain names are never folded into the wrong company. Possible
parent/subsidiary relationships are suggested for review too, never applied automatically.
Every link records the method, the confidence and the reason.
"""

from __future__ import annotations

import logging
from bisect import bisect_left, insort
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from rapidfuzz import fuzz, process
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import OrganizationType
from cews.database.models import Organization, OrganizationAlias, ReviewQueueItem
from cews.normalization.organizations import (
    DIVISION_WORDS,
    NormalizedName,
    classify_organization_type,
    display_quality,
    normalize_organization_name,
    shares_distinguishing_words,
)
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

REVIEW_QUEUE = "organization_match"
PARENT_QUEUE = "organization_parent"
MAX_PREFIX_TOKENS = 2


def sorted_words(text: str) -> str:
    """The words of ``text`` in alphabetical order, single-spaced.

    ``token_sort_ratio(a, b)`` is by definition the plain ratio of these two strings, so sorting a
    name once when it is stored (instead of again inside every comparison) gives the same scores
    much faster.
    """
    return " ".join(sorted(text.split()))


KEY_CONFIDENCE = {
    "exact": 1.0,
    "normalized": 0.98,
    "expanded": 0.95,
    "compact": 0.92,
    "expanded_compact": 0.9,
    "alias": 0.95,
    "prefix": 0.75,
}
STRONG_TYPE_CONFIDENCE = 0.8


@dataclass(frozen=True)
class OrganizationMatch:
    """How a raw name was resolved."""

    organization: Organization
    method: str
    confidence: float
    created: bool
    needs_review: bool = False
    reason: str = ""


@dataclass
class ResolverStats:
    """Counters for one normalization run."""

    resolved: int = 0
    created: int = 0
    review_items: int = 0
    by_method: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        return {
            "resolved": self.resolved,
            "created": self.created,
            "review_items": self.review_items,
            "by_method": dict(self.by_method),
        }


class OrganizationResolver:
    """Matches raw names to organizations, creating them and review items as needed.

    One instance is used for a whole normalization run: it keeps an in-memory index of the
    organizations and aliases it has seen, so repeated names cost nothing.

    Every lookup that could otherwise scan all known organizations for each new name (prefix
    matches, related-company suggestions, the alias rows of one organization) goes through an
    index instead, and fuzzy matching runs in rapidfuzz's compiled loop with a cutoff. The results
    are identical to scanning everything, which the test suite checks against the plain scans;
    the point is that a run over tens of thousands of distinct organizations no longer takes time
    proportional to the square of that number.
    """

    def __init__(self, session: Session, settings: Settings, *, use_fuzzy: bool = True) -> None:
        """Load the existing organizations and aliases into the match index."""
        self.session = session
        self.settings = settings
        self.use_fuzzy = use_fuzzy
        self.auto_threshold = settings.ai_org_match_auto_threshold
        self.review_threshold = settings.ai_org_match_review_threshold
        self.stats = ResolverStats()
        self._index: dict[str, set[int]] = defaultdict(set)
        self._aliases: set[tuple[int, str]] = set()
        self._alias_rows: dict[tuple[int, str], OrganizationAlias] = {}
        self._organizations: dict[int, Organization] = {}
        self._names: dict[int, NormalizedName] = {}
        self._queued: set[tuple[str, str]] = set()
        # Indexes over organization names (see the class docstring). Id lists are kept ascending.
        self._token_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self._prefix_index: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self._leading_index: dict[str, list[int]] = defaultdict(list)
        self._fuzzy_ids: list[int] = []
        self._fuzzy_text: list[str] = []
        self._fuzzy_position: dict[int, int] = {}
        self._rows_by_org: dict[int, dict[tuple[int, str], OrganizationAlias]] = defaultdict(dict)
        self._load()

    # ---- index ---------------------------------------------------------------------------
    def _load(self) -> None:
        for organization in self.session.scalars(select(Organization)):
            self._remember(organization)
        for alias in self.session.scalars(select(OrganizationAlias)):
            self._index[alias.normalized_alias].add(alias.organization_id)
            self._aliases.add((alias.organization_id, alias.normalized_alias))
            self._alias_rows[(alias.organization_id, alias.normalized_alias)] = alias
            self._rows_by_org[alias.organization_id][
                (alias.organization_id, alias.normalized_alias)
            ] = alias
        for item in self.session.scalars(
            select(ReviewQueueItem).where(
                ReviewQueueItem.queue_type.in_([REVIEW_QUEUE, PARENT_QUEUE])
            )
        ):
            self._queued.add((item.queue_type, item.subject_ref))

    def _remember(self, organization: Organization) -> None:
        self._organizations[organization.id] = organization
        name = normalize_organization_name(organization.canonical_name)
        self._names[organization.id] = name
        self._index_name(organization.id, name)
        for _, key in name.match_keys():
            self._index[key].add(organization.id)
        self._index[organization.normalized_name].add(organization.id)

    # ---- name indexes --------------------------------------------------------------------
    def _index_name(self, organization_id: int, name: NormalizedName) -> None:
        tokens = tuple(name.expanded_tokens)
        if tokens:
            insort(self._token_index[tokens], organization_id)
            for length in range(1, len(tokens)):
                insort(self._prefix_index[tokens[:length]], organization_id)
            if len(tokens) > 1:
                insort(self._leading_index[tokens[0]], organization_id)
        if name.expanded:
            words = sorted_words(name.expanded)
            if organization_id in self._fuzzy_position:
                self._fuzzy_text[self._fuzzy_position[organization_id]] = words
            else:
                self._fuzzy_position[organization_id] = len(self._fuzzy_ids)
                self._fuzzy_ids.append(organization_id)
                self._fuzzy_text.append(words)

    def _unindex_name(self, organization_id: int, name: NormalizedName) -> None:
        """Take a name's token entries out of the indexes (the fuzzy list is updated in place)."""

        def drop(bucket: dict[Any, list[int]], key: Any) -> None:
            ids = bucket.get(key)
            if ids:
                position = bisect_left(ids, organization_id)
                if position < len(ids) and ids[position] == organization_id:
                    del ids[position]

        tokens = tuple(name.expanded_tokens)
        if tokens:
            drop(self._token_index, tokens)
            for length in range(1, len(tokens)):
                drop(self._prefix_index, tokens[:length])
            if len(tokens) > 1:
                drop(self._leading_index, tokens[0])

    def _add_alias(self, organization: Organization, name: NormalizedName, method: str) -> None:
        """Record the spellings of this name against the organization (one row per key)."""
        for _, key in name.match_keys():
            marker = (organization.id, key[:255])
            if marker in self._aliases:
                existing = self._alias_rows.get(marker)
                if existing is not None and display_quality(name.original) > display_quality(
                    existing.alias
                ):
                    existing.alias = name.original[
                        :255
                    ]  # a better-presented spelling of the same name
                continue
            self._aliases.add(marker)
            row = OrganizationAlias(
                organization_id=organization.id,
                alias=name.original[:255],
                normalized_alias=key[:255],
                alias_type="variant",
                match_method=method,
                confidence=KEY_CONFIDENCE.get(method, 0.9),
                mention_count=0,
            )
            self.session.add(row)
            self._alias_rows[marker] = row
            self._rows_by_org[organization.id][marker] = row
            self._index[key].add(organization.id)

    # ---- matching ------------------------------------------------------------------------
    def _by_keys(self, name: NormalizedName) -> tuple[str, list[int]]:
        for method, key in name.match_keys():
            candidates = sorted(self._index.get(key, ()))
            if candidates:
                return method, candidates
        return "", []

    def _by_prefix(self, name: NormalizedName) -> list[int]:
        """Organizations whose name starts with this (short) name and is longer than it."""
        if not 1 <= len(name.expanded_tokens) <= MAX_PREFIX_TOKENS:
            return []
        return list(self._prefix_index.get(tuple(name.expanded_tokens), ()))

    def _by_fuzzy(self, name: NormalizedName) -> tuple[int | None, float]:
        """The most similar known organization, if it is similar enough to matter.

        Anything below the lower of the two thresholds is never acted on by the caller, so it is
        not returned: the search stops early on candidates that cannot reach it. Ties go to the
        organization seen first, as a plain scan would.
        """
        if not name.expanded or not self._fuzzy_text:
            return None, 0.0
        floor = min(self.auto_threshold, self.review_threshold)
        hit = process.extractOne(
            sorted_words(name.expanded),
            self._fuzzy_text,
            scorer=fuzz.ratio,
            score_cutoff=max(floor * 100 - 1e-6, 0.0),
        )
        if hit is None:
            return None, 0.0
        score = hit[1] / 100
        if score < floor:  # the cutoff above is slightly loose; the exact comparison is here
            return None, 0.0
        return self._fuzzy_ids[hit[2]], score

    def _compatible(self, name: NormalizedName, organization: Organization) -> bool:
        """Companies are never fuzzily merged into universities, hospitals or agencies."""
        candidate = classify_organization_type(name.original)
        existing = organization.organization_type
        if candidate.organization_type is OrganizationType.UNKNOWN or existing == "unknown":
            return True
        if candidate.confidence < STRONG_TYPE_CONFIDENCE:
            return True
        return candidate.organization_type.value == existing

    def _prefer_common_name(self, organization: Organization, name: NormalizedName) -> None:
        """Show the spelling seen most often, so a rare legal variant does not win.

        Sources write the same company many ways. Each spelling's mentions are counted on its
        alias row, and the most-mentioned one becomes the display name ("Orvexa Bio" rather than
        "Orvexa Biosciences"). Presentation only breaks ties, so the name cannot flip back and
        forth between two spellings as records arrive; counts live in the database, so the choice
        is the same on the next run.
        """
        current = self._names[organization.id]
        if name.expanded != current.expanded:
            return  # a different company or a subsidiary; never rename across names
        self._count_mention(organization, name)

        best_row: OrganizationAlias | None = None
        best = (self._mentions_for(organization, current), display_quality(current.original))
        for row in self._rows_by_org[organization.id].values():
            candidate = (row.mention_count or 0, display_quality(row.alias))
            if candidate > best:
                best_row, best = row, candidate
        if best_row is None or best_row.alias == organization.canonical_name:
            return
        parsed = normalize_organization_name(best_row.alias)
        if parsed.expanded != current.expanded:
            return
        organization.canonical_name = best_row.alias[:255]
        self._unindex_name(organization.id, current)
        self._names[organization.id] = parsed
        self._index_name(organization.id, parsed)
        classification = classify_organization_type(best_row.alias)
        if classification.organization_type is not OrganizationType.UNKNOWN:
            organization.organization_type = classification.organization_type.value

    def _count_mention(self, organization: Organization, name: NormalizedName) -> int:
        """Record that this exact spelling was seen once more; returns the new count."""
        keys = [key for _, key in name.match_keys()]
        for key in keys:
            row = self._alias_rows.get((organization.id, key[:255]))
            if row is not None:
                row.mention_count = (row.mention_count or 0) + 1
                return row.mention_count
        return 0

    def _mentions_for(self, organization: Organization, name: NormalizedName) -> int:
        for _, key in name.match_keys():
            row = self._alias_rows.get((organization.id, key[:255]))
            if row is not None:
                return row.mention_count or 0
        return 0

    # ---- review queue --------------------------------------------------------------------
    def _queue(self, queue_type: str, subject: str, payload: dict[str, Any]) -> bool:
        marker = (queue_type, subject[:255])
        if marker in self._queued:
            return False
        self._queued.add(marker)
        self.session.add(
            ReviewQueueItem(
                queue_type=queue_type,
                subject_ref=subject[:255],
                payload_json=payload,
                status="pending",
                created_at=datetime.now(UTC),
            )
        )
        self.stats.review_items += 1
        return True

    def _suggest_relationship(self, organization: Organization, name: NormalizedName) -> None:
        """Queue a possible parent/subsidiary or related-company link, for a human to confirm.

        Two shapes are suggested: one name is a strict prefix of the other (``Zentavia`` and
        ``Zentavia Pharmaceuticals``), or both start with the same distinctive word but differ
        afterwards (``Zentavia Pharmaceuticals`` and ``Zentavia Oncology``). The second shape
        also catches unrelated companies with similar names, which is exactly why it is only
        ever a suggestion: CEWS never sets ``parent_id`` on its own.
        """
        tokens = tuple(name.expanded_tokens)
        if not tokens:
            return
        own = organization.id

        def first_other(ids: list[int] | None) -> int | None:
            for candidate_id in ids or ():
                if candidate_id != own:
                    return candidate_id
            return None

        # The first other organization (lowest id) that is related in either of the two ways,
        # found through the indexes instead of by looking at every organization.
        found = [
            first_other(self._token_index.get(tokens[:length])) for length in range(1, len(tokens))
        ]
        found.append(first_other(self._prefix_index.get(tokens)))
        leading = tokens[0]
        if len(tokens) > 1 and leading not in DIVISION_WORDS:
            found.append(first_other(self._leading_index.get(leading)))
        candidates = [candidate_id for candidate_id in found if candidate_id is not None]
        if not candidates:
            return

        other_id = min(candidates)
        other = self._names[other_id]
        shorter, longer = sorted((other.expanded_tokens, name.expanded_tokens), key=len)
        if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
            reason = "one name is the start of the other"
        else:
            reason = "both names start with the same word"
        other_organization = self._organizations[other_id]
        pair = sorted([organization.canonical_name, other_organization.canonical_name])
        self._queue(
            PARENT_QUEUE,
            f"{pair[0]}|{pair[1]}",
            {
                "organizations": pair,
                "reason": reason,
                "action": (
                    "decide whether these are parent and subsidiary, the same company, or "
                    "unrelated companies with similar names; CEWS never links them itself"
                ),
            },
        )

    # ---- public --------------------------------------------------------------------------
    def create_organization(
        self,
        name: NormalizedName,
        *,
        discovered: bool = True,
        synthetic: bool = False,
    ) -> Organization:
        """Create an organization from a normalized name and index it."""
        classification = classify_organization_type(name.original)
        organization = Organization(
            canonical_name=name.original[:255],
            normalized_name=(name.expanded or name.normalized)[:255],
            organization_type=classification.organization_type.value,
            discovered_automatically=discovered,
            is_synthetic=synthetic,
            created_at=datetime.now(UTC),
        )
        self.session.add(organization)
        self.session.flush()
        self._remember(organization)
        self.stats.created += 1
        LOGGER.debug(
            "created organization %s (%s, %s)",
            organization.canonical_name,
            classification.organization_type.value,
            classification.reason,
        )
        return organization

    def resolve(self, raw_name: str | None, *, synthetic: bool = False) -> OrganizationMatch | None:
        """Resolve one raw name, creating an organization and review items when needed.

        Returns None when the name is empty or unusable.
        """
        if not raw_name or not raw_name.strip():
            return None
        try:
            name = normalize_organization_name(raw_name)
        except ValueError:
            LOGGER.debug("skipping unusable organization name %r", str(raw_name)[:80])
            return None
        if name.is_empty:
            return None

        method, candidates = self._by_keys(name)
        if len(candidates) == 1:
            organization = self._organizations[candidates[0]]
            self._add_alias(organization, name, method)
            self._prefer_common_name(organization, name)
            return self._done(
                OrganizationMatch(organization, method, KEY_CONFIDENCE.get(method, 0.9), False)
            )
        if len(candidates) > 1:
            organization = self.create_organization(name, synthetic=synthetic)
            self._queue(
                REVIEW_QUEUE,
                name.expanded or name.normalized,
                {
                    "raw_name": name.original,
                    "reason": "the same name key matches several organizations",
                    "candidates": [self._organizations[c].canonical_name for c in candidates],
                },
            )
            return self._done(
                OrganizationMatch(organization, "ambiguous", 0.4, True, True, "ambiguous key")
            )

        prefix_candidates = self._by_prefix(name)
        if len(prefix_candidates) == 1:
            organization = self._organizations[prefix_candidates[0]]
            if self._compatible(name, organization):
                self._add_alias(organization, name, "prefix")
                self._queue(
                    REVIEW_QUEUE,
                    name.expanded or name.normalized,
                    {
                        "raw_name": name.original,
                        "reason": "short name matched by prefix",
                        "linked_to": organization.canonical_name,
                        "action": "confirm the link or split it",
                    },
                )
                return self._done(
                    OrganizationMatch(
                        organization,
                        "prefix",
                        KEY_CONFIDENCE["prefix"],
                        False,
                        True,
                        "prefix match",
                    )
                )
        elif len(prefix_candidates) > 1:
            organization = self.create_organization(name, synthetic=synthetic)
            self._queue(
                REVIEW_QUEUE,
                name.expanded or name.normalized,
                {
                    "raw_name": name.original,
                    "reason": "short name could belong to several organizations",
                    "candidates": [
                        self._organizations[c].canonical_name for c in prefix_candidates
                    ],
                },
            )
            return self._done(
                OrganizationMatch(organization, "ambiguous", 0.4, True, True, "ambiguous prefix")
            )

        if self.use_fuzzy:
            best_id, score = self._by_fuzzy(name)
            if best_id is not None:
                organization = self._organizations[best_id]
                mergeable = shares_distinguishing_words(name, self._names[best_id])
                if (
                    score >= self.auto_threshold
                    and mergeable
                    and self._compatible(name, organization)
                ):
                    self._add_alias(organization, name, "fuzzy")
                    return self._done(OrganizationMatch(organization, "fuzzy", score, False))
                if score >= self.review_threshold:
                    created = self.create_organization(name, synthetic=synthetic)
                    self._queue(
                        REVIEW_QUEUE,
                        name.expanded or name.normalized,
                        {
                            "raw_name": name.original,
                            "reason": "similar to an existing organization but not merged",
                            "similar_to": organization.canonical_name,
                            "similarity": round(score, 3),
                            "blocked_by": None if mergeable else "names differ by a division word",
                        },
                    )
                    self._suggest_relationship(created, name)
                    return self._done(
                        OrganizationMatch(created, "created", 0.6, True, True, "near match")
                    )

        organization = self.create_organization(name, synthetic=synthetic)
        self._suggest_relationship(organization, name)
        return self._done(OrganizationMatch(organization, "created", 0.9, True))

    def _done(self, match: OrganizationMatch) -> OrganizationMatch:
        self.stats.resolved += 1
        self.stats.by_method[match.method] += 1
        return match
