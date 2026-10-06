"""The indexed resolver must give exactly the answers a plain scan of every organization gives.

The resolver's lookups were rewritten to use indexes, because scanning every known organization for
every new name made a run over tens of thousands of organizations take time proportional to the
square of that number. This file keeps the original scans, word for word, as a reference and runs
randomized corpora through both: every decision, every stored row and every review item must match.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

import pytest
from rapidfuzz import fuzz
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import OrganizationType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Organization, OrganizationAlias, ReviewQueueItem
from cews.normalization.organizations import (
    DIVISION_WORDS,
    NormalizedName,
    classify_organization_type,
    display_quality,
    normalize_organization_name,
)
from cews.normalization.resolver import (
    MAX_PREFIX_TOKENS,
    PARENT_QUEUE,
    OrganizationResolver,
    sorted_words,
)
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit


class ScanningResolver(OrganizationResolver):
    """The resolver as it was before the indexes: it looks at every organization each time."""

    def _by_prefix(self, name: NormalizedName) -> list[int]:
        if not 1 <= len(name.expanded_tokens) <= MAX_PREFIX_TOKENS:
            return []
        prefix = name.expanded_tokens
        return sorted(
            organization_id
            for organization_id, other in self._names.items()
            if len(other.expanded_tokens) > len(prefix)
            and other.expanded_tokens[: len(prefix)] == prefix
        )

    def _by_fuzzy(self, name: NormalizedName) -> tuple[int | None, float]:
        best_id, best_score = None, 0.0
        for organization_id, other in self._names.items():
            if not other.expanded or not name.expanded:
                continue
            score = fuzz.token_sort_ratio(name.expanded, other.expanded) / 100
            if score > best_score:
                best_id, best_score = organization_id, score
        # The original returned the best score whatever it was; the caller only acts on scores at
        # or above a threshold, so hits below the lower threshold are reported as no hit.
        floor = min(self.auto_threshold, self.review_threshold)
        return (best_id, best_score) if best_score >= floor else (None, 0.0)

    def _prefer_common_name(self, organization: Organization, name: NormalizedName) -> None:
        current = self._names[organization.id]
        if name.expanded != current.expanded:
            return
        self._count_mention(organization, name)

        best_row: OrganizationAlias | None = None
        best = (self._mentions_for(organization, current), display_quality(current.original))
        for (organization_id, _), row in self._alias_rows.items():  # every alias row of every org
            if organization_id != organization.id:
                continue
            candidate = (row.mention_count or 0, display_quality(row.alias))
            if candidate > best:
                best_row, best = row, candidate
        if best_row is None or best_row.alias == organization.canonical_name:
            return
        parsed = normalize_organization_name(best_row.alias)
        if parsed.expanded != current.expanded:
            return
        organization.canonical_name = best_row.alias[:255]
        self._names[organization.id] = parsed
        classification = classify_organization_type(best_row.alias)
        if classification.organization_type is not OrganizationType.UNKNOWN:
            organization.organization_type = classification.organization_type.value

    def _suggest_relationship(self, organization: Organization, name: NormalizedName) -> None:
        if not name.expanded_tokens:
            return
        for other_id, other in sorted(self._names.items()):
            if other_id == organization.id or not other.expanded_tokens:
                continue
            shorter, longer = sorted((other.expanded_tokens, name.expanded_tokens), key=len)
            leading = name.expanded_tokens[0]
            if len(shorter) < len(longer) and longer[: len(shorter)] == shorter:
                reason = "one name is the start of the other"
            elif (
                other.expanded_tokens[0] == leading
                and leading not in DIVISION_WORDS
                and len(other.expanded_tokens) > 1
                and len(name.expanded_tokens) > 1
            ):
                reason = "both names start with the same word"
            else:
                continue
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
            return


# --------------------------------------------------------------------------------------
# A corpus that exercises every path
# --------------------------------------------------------------------------------------
STEMS = [
    "Arden",
    "Bright",
    "Calder",
    "Dunmore",
    "Elan",
    "Fenwick",
    "Garrow",
    "Halden",
    "Ivor",
    "Jarrow",
    "Kestrel",
    "Lorne",
    "Marden",
    "Norwick",
    "Orvane",
    "Pellham",
    "Quillon",
    "Ridgemont",
    "Solmere",
    "Talvora",
    "Varethyn",
    "Zentavia",
]
KINDS = [
    "Pharmaceuticals",
    "Pharma",
    "Biosciences",
    "Bio",
    "Therapeutics",
    "Oncology",
    "Laboratories",
    "Diagnostics",
    "Genetics",
    "Medical",
]
SUFFIXES = ["Inc", "Inc.", "Ltd", "GmbH", "LLC", "Corp", "Corporation", "Co", "AG", ""]
INSTITUTIONS = [
    "University of {}",
    "{} University",
    "{} Hospital",
    "{} Institute",
    "{} Research Council",
]


def typo(text: str, rng: random.Random) -> str:
    if len(text) < 5:
        return text
    i = rng.randrange(1, len(text) - 1)
    return rng.choice(
        [
            text[:i] + text[i + 1 :],
            text[:i] + text[i] + text[i:],
            text[:i] + text[i + 1] + text[i] + text[i + 2 :],
        ]
    )


def corpus(seed: int, size: int) -> list[str]:
    rng = random.Random(seed)
    bases = [
        f"{rng.choice(STEMS)} {rng.choice(KINDS)} {rng.choice(SUFFIXES)}".strip()
        for _ in range(size // 3)
    ]
    out: list[str] = []
    for _ in range(size):
        roll = rng.random()
        base = rng.choice(bases)
        if roll < 0.20:
            out.append(base)  # an exact repeat
        elif roll < 0.35:
            out.append(rng.choice([base.upper(), base.lower(), base + ".", "  " + base + "  "]))
        elif roll < 0.50:
            out.append(typo(base, rng))  # spelling errors: the fuzzy path
        elif roll < 0.60:
            out.append(base.split()[0])  # a short name: the prefix path
        elif roll < 0.72:
            out.append(f"{base.split()[0]} {rng.choice(KINDS)}")  # a sibling: shared leading word
        elif roll < 0.80:
            out.append(rng.choice(INSTITUTIONS).format(rng.choice(STEMS)))
        elif roll < 0.86:
            out.append(f"{base} {rng.choice(['Europe', 'Oncology', 'Holdings'])}")  # a longer name
        elif roll < 0.90:
            out.append(rng.choice(["", "   ", "Inc", "The", "N/A"]))  # unusable or tiny
        else:
            out.append(f"{rng.choice(STEMS)}{rng.choice(STEMS).lower()} {rng.choice(KINDS)}")
    return out


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


def run(
    resolver_class: type[OrganizationResolver],
    settings: Settings,
    names: list[str],
    *,
    preload: list[str] | None = None,
) -> dict[str, Any]:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
    decisions: list[tuple[Any, ...] | None] = []
    with session_scope(factory) as session:
        if preload:  # organizations that already exist when the run starts (the _load path)
            seed_resolver = OrganizationResolver(session, settings)
            for name in preload:
                seed_resolver.resolve(name, synthetic=True)
            session.flush()
        resolver = resolver_class(session, settings)
        for raw in names:
            match = resolver.resolve(raw, synthetic=True)
            decisions.append(
                None
                if match is None
                else (
                    match.organization.canonical_name,
                    match.method,
                    round(match.confidence, 9),
                    match.created,
                    match.needs_review,
                    match.reason,
                )
            )
        session.flush()
        state = {
            "decisions": decisions,
            "stats": resolver.stats.as_dict(),
            "organizations": sorted(
                (o.canonical_name, o.normalized_name, o.organization_type, o.parent_id)
                for o in session.scalars(select(Organization))
            ),
            "aliases": sorted(
                (a.organization_id, a.alias, a.normalized_alias, a.match_method, a.mention_count)
                for a in session.scalars(select(OrganizationAlias))
            ),
            "review": sorted(
                (q.queue_type, q.subject_ref, str(sorted((q.payload_json or {}).items())), q.status)
                for q in session.scalars(select(ReviewQueueItem))
            ),
        }
    engine.dispose()
    return state


# --------------------------------------------------------------------------------------
# The equivalence itself
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_the_indexed_resolver_makes_exactly_the_same_decisions(
    settings: Settings, seed: int
) -> None:
    names = corpus(seed, 400)
    assert run(OrganizationResolver, settings, names) == run(ScanningResolver, settings, names)


@pytest.mark.parametrize("seed", [11, 12])
def test_the_same_holds_when_organizations_already_exist_at_the_start(
    settings: Settings, seed: int
) -> None:
    preload = corpus(seed, 150)
    names = corpus(seed + 100, 300)
    assert run(OrganizationResolver, settings, names, preload=preload) == run(
        ScanningResolver, settings, names, preload=preload
    )


@pytest.mark.parametrize(
    ("auto", "review"), [(0.92, 0.80), (0.95, 0.70), (0.90, 0.89), (0.99, 0.50)]
)
def test_the_same_holds_for_other_valid_thresholds(auto: float, review: float) -> None:
    custom = load_settings(
        env_file=None,
        overrides={"ai_org_match_auto_threshold": auto, "ai_org_match_review_threshold": review},
    )
    names = corpus(21, 300)
    assert run(OrganizationResolver, custom, names) == run(ScanningResolver, custom, names)


def spelling_corpus(seed: int) -> list[str]:
    """Companies seen under two spellings, one of which turns out to be far more common."""
    rng = random.Random(seed)
    out: list[str] = []
    for stem in STEMS:
        rare, common = f"{stem} Biosciences", f"{stem} Bio"
        out.append(rare)  # seen first, so it starts as the display name
        out.extend([common] * rng.randrange(3, 9))
        out.extend([rare.upper(), common + "."])
    rng.shuffle(out)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_most_common_spelling_becomes_the_display_name_exactly_as_before(
    settings: Settings, seed: int
) -> None:
    names = spelling_corpus(seed)
    new, old = run(OrganizationResolver, settings, names), run(ScanningResolver, settings, names)
    assert new == old
    displayed = {row[0] for row in new["organizations"]}
    assert any(name.endswith(" Bio") for name in displayed), displayed  # the rename really happened


def test_the_corpus_really_exercises_every_path(settings: Settings) -> None:
    """A guard against the comparison above passing only because nothing interesting happened."""
    state = run(OrganizationResolver, settings, corpus(1, 400))
    methods = state["stats"]["by_method"]
    assert methods.get("fuzzy", 0) > 0, methods
    assert methods.get("prefix", 0) > 0, methods
    assert methods.get("ambiguous", 0) > 0 or state["stats"]["review_items"] > 0
    payloads = " ".join(payload for _, _, payload, _ in state["review"])
    assert "one name is the start of the other" in payloads
    assert "both names start with the same word" in payloads
    assert any(q[0] == PARENT_QUEUE for q in state["review"])
    assert state["stats"]["created"] > 50


# --------------------------------------------------------------------------------------
# The pieces
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("zentavia pharma", "pharma zentavia"),
        ("zentavia pharmaceuticals", "zentavja pharmaceuticals"),
        ("a b c", "c b a"),
        ("  spaced   out  ", "out spaced"),
        ("élan biothérapeutique", "biothérapeutique élan"),
        ("", "x"),
        ("same", "same"),
        ("北京 医药", "医药 北京"),
    ],
)
def test_sorting_the_words_once_gives_the_same_score_as_token_sort_ratio(a: str, b: str) -> None:
    assert fuzz.ratio(sorted_words(a), sorted_words(b)) == pytest.approx(
        fuzz.token_sort_ratio(a, b)
    )


def test_sorting_the_words_matches_token_sort_ratio_on_random_strings() -> None:
    rng = random.Random(3)
    alphabet = "abcde fgh"
    for _ in range(2000):
        a = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 25)))
        b = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 25)))
        assert fuzz.ratio(sorted_words(a), sorted_words(b)) == pytest.approx(
            fuzz.token_sort_ratio(a, b)
        )


def test_two_equally_similar_organizations_resolve_to_the_first_seen(settings: Settings) -> None:
    """A tie must go to the earlier organization, exactly as a plain scan would."""
    names = ["Zentavia Pharmaceuticals", "Zentavja Pharmaceuticals", "Zentavka Pharmaceuticals"]
    assert run(OrganizationResolver, settings, names) == run(ScanningResolver, settings, names)


def test_the_indexes_stay_consistent_with_the_organizations(settings: Settings) -> None:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        resolver = OrganizationResolver(session, settings)
        for raw in corpus(5, 300):
            resolver.resolve(raw, synthetic=True)
        listed: dict[int, int] = defaultdict(int)
        for ids in resolver._token_index.values():
            assert ids == sorted(ids) and len(ids) == len(set(ids))
            for identifier in ids:
                listed[identifier] += 1
        with_tokens = {i for i, n in resolver._names.items() if n.expanded_tokens}
        assert set(listed) == with_tokens and all(count == 1 for count in listed.values())
        assert (
            len(resolver._fuzzy_ids) == len(resolver._fuzzy_text) == len(resolver._fuzzy_position)
        )
        for identifier, position in resolver._fuzzy_position.items():
            assert resolver._fuzzy_ids[position] == identifier
            assert resolver._fuzzy_text[position] == sorted_words(
                resolver._names[identifier].expanded
            )
    engine.dispose()


def test_a_run_over_many_organizations_finishes_quickly_enough() -> None:
    """A regression guard for the quadratic behaviour, not a benchmark: 3,000 distinct names took
    over ten seconds before the indexes and now take a few."""
    import time

    settings = load_settings(env_file=None)
    rng = random.Random(9)
    names = list(
        {
            f"{rng.choice(STEMS)}{rng.choice(STEMS).lower()} {rng.choice(KINDS)} {rng.choice(KINDS)} {rng.choice(SUFFIXES)}".strip()
            for _ in range(6000)
        }
    )[:3000]
    engine = create_memory_engine()
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        resolver = OrganizationResolver(session, settings)
        started = time.monotonic()
        for raw in names:
            resolver.resolve(raw, synthetic=True)
        elapsed = time.monotonic() - started
    engine.dispose()
    assert elapsed < 12, f"resolving {len(names)} distinct names took {elapsed:.1f}s"
