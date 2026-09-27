"""Deterministic synthetic demo data for CEWS.

Everything produced here is SYNTHETIC. Companies, universities and agencies are invented, the
activity patterns are artificial and do not describe real-world research, and every record is
labelled (source names start with ``synthetic_``, identifiers start with ``SYN-``, abstracts
start with ``[SYNTHETIC]``, and source URLs use the reserved ``.invalid`` domain).

The dataset is designed to exercise every part of the pipeline:

* a sustained multi-source trend, a declining topic, a seasonal topic;
* a one-time spike (an anomaly that must not be called a trend);
* a high-growth, low-competition opportunity;
* a high-threat competitor that also enters a new therapeutic area;
* a low-confidence trend built on very few records;
* a topic that is NOT in the taxonomy (for AI topic discovery);
* messy organization-name variants and a look-alike organization (for entity resolution).

The same ``DemoConfig`` always yields byte-identical output. Ground-truth labels are kept in
each record's payload under ``synthetic_ground_truth`` and must never be read by the
production pipeline; they exist only so evaluations can check the pipeline's answers.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from cews.constants import DEMO_RANDOM_SEED, SYNTHETIC_SOURCE_PREFIX, SourceType

DEMO_END_MONTH = date(2026, 8, 1)
DEMO_MONTHS = 30
DEMO_FETCHED_AT = datetime(2026, 9, 1, tzinfo=UTC)
SYNTHETIC_URL_BASE = "https://example.invalid/synthetic"

SOURCE_NAMES: dict[SourceType, str] = {
    SourceType.CLINICAL_TRIAL: f"{SYNTHETIC_SOURCE_PREFIX}clinical_trials",
    SourceType.PUBLICATION: f"{SYNTHETIC_SOURCE_PREFIX}publications",
    SourceType.PATENT: f"{SYNTHETIC_SOURCE_PREFIX}patents",
    SourceType.FUNDING: f"{SYNTHETIC_SOURCE_PREFIX}funding",
    SourceType.ANNOUNCEMENT: f"{SYNTHETIC_SOURCE_PREFIX}announcements",
}
ID_PREFIX: dict[SourceType, str] = {
    SourceType.CLINICAL_TRIAL: "SYN-CT",
    SourceType.PUBLICATION: "SYN-PUB",
    SourceType.PATENT: "SYN-PAT",
    SourceType.FUNDING: "SYN-AWD",
    SourceType.ANNOUNCEMENT: "SYN-ANN",
}
SOURCE_ORDER: tuple[SourceType, ...] = (
    SourceType.PUBLICATION,
    SourceType.CLINICAL_TRIAL,
    SourceType.PATENT,
    SourceType.FUNDING,
    SourceType.ANNOUNCEMENT,
)


# --------------------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoOrg:
    """An invented organization with several messy spellings."""

    key: str
    canonical: str
    org_type: str
    variants: tuple[str, ...]
    parent: str | None = None
    country: str = "US"


DEMO_ORGS: tuple[DemoOrg, ...] = (
    DemoOrg(
        "zentavia",
        "Zentavia Pharma",
        "company",
        (
            "Zentavia Pharma",
            "Zentavia Pharma, Inc.",
            "ZENTAVIA PHARMA INC",
            "Zentavia Pharmaceuticals Ltd",
            "Zentavia",
        ),
    ),
    DemoOrg(
        "zentavia_onc",
        "Zentavia Oncology Ltd",
        "company",
        ("Zentavia Oncology Ltd", "Zentavia Oncology Limited"),
        parent="zentavia",
    ),
    DemoOrg(
        "orvexa",
        "Orvexa Bio",
        "company",
        ("Orvexa Bio", "Orvexa Bio Inc.", "OrvexaBio", "Orvexa Biosciences"),
    ),
    DemoOrg(
        "orvexa_labs",
        "Orvexa Labs",
        "company",
        ("Orvexa Labs", "Orvexa Laboratories"),
        country="GB",
    ),
    DemoOrg(
        "lumaris",
        "Lumaris Therapeutics",
        "company",
        (
            "Lumaris Therapeutics",
            "Lumaris Therapeutics, Inc.",
            "Lumaris Tx",
            "LUMARIS THERAPEUTICS PLC",
        ),
        country="GB",
    ),
    DemoOrg(
        "halcyra",
        "Halcyra Biosciences",
        "company",
        ("Halcyra Biosciences", "Halcyra Bio", "Halcyra Biosciences GmbH"),
        country="DE",
    ),
    DemoOrg(
        "nexoria",
        "Nexoria Genetics",
        "company",
        ("Nexoria Genetics", "Nexoria Genetics Corp.", "Nexoria Gen."),
    ),
    DemoOrg(
        "varethyn",
        "Varethyn Labs",
        "company",
        ("Varethyn Labs", "Varethyn Laboratories", "Varethyn Labs Ltd"),
    ),
    DemoOrg(
        "harrowgate_univ",
        "Harrowgate University",
        "university",
        ("Harrowgate University", "Univ. of Harrowgate", "Harrowgate Univ."),
        country="GB",
    ),
    DemoOrg(
        "st_aldric",
        "St. Aldric Medical Center",
        "hospital",
        ("St. Aldric Medical Center", "Saint Aldric Medical Ctr"),
    ),
    DemoOrg(
        "riverbend",
        "Riverbend Research Institute",
        "nonprofit",
        ("Riverbend Research Institute", "Riverbend Research Inst."),
    ),
)
_ORG_BY_KEY: dict[str, DemoOrg] = {org.key: org for org in DEMO_ORGS}
ACADEMIC_KEYS: tuple[str, ...] = ("harrowgate_univ", "st_aldric", "riverbend")
FUNDING_AGENCIES: tuple[str, ...] = (
    "National Institute for Synthetic Health",
    "Synthetic Research Council",
)


@dataclass(frozen=True)
class DemoTopic:
    """A demo topic and the phrases used in synthetic text (taxonomy synonyms)."""

    key: str
    name: str
    phrases: tuple[str, ...]
    in_taxonomy: bool = True


DEMO_TOPICS: tuple[DemoTopic, ...] = (
    DemoTopic(
        "crispr_gene_editing",
        "CRISPR gene editing",
        ("CRISPR", "Cas9", "base editing", "prime editing"),
    ),
    DemoTopic(
        "immune_checkpoint",
        "Immune checkpoint inhibitors",
        ("PD-1", "PD-L1", "checkpoint inhibitor"),
    ),
    DemoTopic("aav_vectors", "AAV gene delivery", ("AAV", "adeno-associated virus")),
    DemoTopic(
        "ai_drug_discovery",
        "AI-assisted drug discovery",
        ("AI drug discovery", "machine learning drug discovery", "generative chemistry"),
    ),
    DemoTopic(
        "bispecific_antibodies", "Bispecific antibodies", ("bispecific antibody", "T-cell engager")
    ),
    DemoTopic(
        "mrna_therapeutics", "mRNA therapeutics", ("mRNA vaccine", "messenger RNA", "mRNA-based")
    ),
    DemoTopic(
        "neurodegeneration",
        "Neurodegeneration",
        ("amyloid beta", "tau protein", "neurodegenerative disease"),
    ),
    DemoTopic("sirna_rnai", "siRNA and RNA interference", ("siRNA", "RNAi", "RNA interference")),
    DemoTopic(
        "targeted_protein_degradation",
        "Targeted protein degradation",
        ("targeted protein degradation", "PROTAC", "molecular glue degrader"),
        in_taxonomy=False,
    ),
)
_TOPIC_BY_KEY: dict[str, DemoTopic] = {t.key: t for t in DEMO_TOPICS}

Shape = Callable[[int, int], float]


def _lin(start: float, end: float) -> Shape:
    return lambda m, n: start + (end - start) * m / max(n - 1, 1)


def _flat(level: float) -> Shape:
    return lambda m, n: level


def _seasonal(level: float, amplitude: float) -> Shape:
    return lambda m, n: level * (1 + amplitude * math.sin(2 * math.pi * m / 12))


P, T, A, F, N = (
    SourceType.PUBLICATION,
    SourceType.CLINICAL_TRIAL,
    SourceType.PATENT,
    SourceType.FUNDING,
    SourceType.ANNOUNCEMENT,
)

# Expected records per month for each topic and source type (before Poisson noise).
PROFILES: dict[str, dict[SourceType, Shape]] = {
    "crispr_gene_editing": {
        P: _lin(8, 22),
        T: _lin(2, 6),
        A: _lin(3, 10),
        F: _lin(1, 4),
        N: _lin(1, 3),
    },
    "immune_checkpoint": {
        P: _lin(20, 8),
        T: _lin(6, 2),
        A: _lin(8, 3),
        F: _lin(3, 1),
        N: _lin(3, 1),
    },
    "aav_vectors": {P: _flat(8), T: _flat(2), A: _flat(3), F: _flat(1), N: _flat(1)},
    "ai_drug_discovery": {
        P: _lin(2, 14),
        T: _lin(0.3, 2.5),
        A: _lin(0.5, 3),
        F: _lin(0.3, 2),
        N: _lin(0.5, 3),
    },
    "bispecific_antibodies": {
        P: _lin(6, 12),
        T: _lin(2, 5),
        A: _lin(3, 9),
        F: _lin(1, 2),
        N: _lin(1, 3),
    },
    "mrna_therapeutics": {
        P: _seasonal(12, 0.3),
        T: _flat(3),
        A: _flat(4),
        F: _flat(1.5),
        N: _flat(2),
    },
    "neurodegeneration": {P: _lin(6, 9), T: _lin(2, 3), A: _lin(2, 3), F: _flat(2), N: _flat(1)},
    "targeted_protein_degradation": {
        P: _lin(1, 10),
        T: _lin(0.2, 1.5),
        A: _lin(0.5, 4),
        F: _lin(0.2, 1),
        N: _lin(0.3, 2),
    },
    "sirna_rnai": {},
}

# Relative activity weight of each competitor within a topic.
TOPIC_COMPETITORS: dict[str, dict[str, float]] = {
    "crispr_gene_editing": {
        "orvexa": 3,
        "nexoria": 3,
        "lumaris": 2,
        "zentavia": 1,
        "orvexa_labs": 1,
    },
    "immune_checkpoint": {"lumaris": 3, "orvexa": 2, "halcyra": 2, "zentavia": 2, "varethyn": 1},
    "aav_vectors": {"nexoria": 3, "varethyn": 2, "halcyra": 1},
    "ai_drug_discovery": {"nexoria": 3, "halcyra": 1},
    "bispecific_antibodies": {"zentavia": 1, "lumaris": 2, "orvexa": 2, "halcyra": 1},
    "mrna_therapeutics": {"halcyra": 3, "varethyn": 2, "orvexa": 1, "lumaris": 1, "orvexa_labs": 1},
    "neurodegeneration": {"varethyn": 3, "halcyra": 2, "lumaris": 1},
    "targeted_protein_degradation": {"orvexa": 2, "halcyra": 1, "nexoria": 1},
    "sirna_rnai": {"varethyn": 1},
}
ACADEMIC_SHARE: dict[SourceType, float] = {P: 0.3, T: 0.1, A: 0.05, F: 0.7, N: 0.0}
NO_ACADEMIC_TOPICS = frozenset({"sirna_rnai"})

# Scenario timing is expressed relative to the last month so it holds for any length.
ZENTAVIA_ENTRY_TOPIC = "neurodegeneration"
ENTRY_EXTRA_RATES: dict[SourceType, float] = {T: 1.5, A: 1.0, P: 1.0}
SPIKE_TOPIC, SPIKE_SOURCE, SPIKE_SIZE = "aav_vectors", A, 25
FALSE_TREND_TOPIC = "sirna_rnai"

CONDITIONS = (
    "solid tumors",
    "hematologic malignancies",
    "rare genetic disorders",
    "autoimmune disease",
    "neurological disorders",
    "metabolic disease",
    "inflammatory disease",
)
STUDY_TYPES = (
    "preclinical study",
    "systematic review",
    "cohort analysis",
    "mechanistic study",
    "retrospective analysis",
)
JOURNALS = (
    "Journal of Synthetic Biology Research",
    "Synthetic Medicine Letters",
    "Annals of Illustrative Science",
    "Demo Journal of Therapeutics",
)
COUNTRIES = ("US", "GB", "DE", "FR", "JP", "IN", "CA", "AU")
CPC_CODES: dict[str, tuple[str, ...]] = {
    "crispr_gene_editing": ("C12N15/90", "C12N9/22"),
    "immune_checkpoint": ("C07K16/28", "A61K39/395"),
    "aav_vectors": ("C12N15/86", "A61K48/00"),
    "ai_drug_discovery": ("G16C20/70", "G16B40/00"),
    "bispecific_antibodies": ("C07K16/46", "C07K16/28"),
    "mrna_therapeutics": ("C12N15/113", "A61K31/7115"),
    "neurodegeneration": ("A61K38/17", "C07K14/47"),
    "sirna_rnai": ("C12N15/113", "C12N2310/14"),
    "targeted_protein_degradation": ("C07D401/04", "A61K47/54"),
}
ANNOUNCEMENT_TYPES = (
    "partnership",
    "licensing",
    "acquisition",
    "clinical_milestone",
    "funding",
    "regulatory",
)
ANNOUNCEMENT_WEIGHTS = (0.25, 0.2, 0.08, 0.27, 0.12, 0.08)


# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DemoConfig:
    """Parameters of a demo dataset. The defaults give the standard 30-month dataset."""

    seed: int = DEMO_RANDOM_SEED
    end_month: date = DEMO_END_MONTH
    months: int = DEMO_MONTHS
    scale: float = 1.0

    def validate(self) -> None:
        """Raise ``ValueError`` for parameters outside the supported range."""
        if not 12 <= self.months <= 120:
            raise ValueError("months must be between 12 and 120")
        if not 0.05 <= self.scale <= 10:
            raise ValueError("scale must be between 0.05 and 10")
        if self.end_month.day != 1:
            raise ValueError("end_month must be the first day of a month")


@dataclass(frozen=True)
class DemoRecord:
    """One synthetic source record and its type-specific details."""

    source: str
    source_record_id: str
    record_type: str
    title: str
    abstract: str
    source_url: str
    published_at: datetime
    payload: dict[str, Any]
    detail: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (dates and datetimes become ISO strings)."""
        return json.loads(json.dumps(self.__dict__, default=_json_default))


@dataclass
class DemoDataset:
    """A complete synthetic dataset plus its ground-truth scenario description."""

    config: DemoConfig
    months: list[date]
    records: list[DemoRecord] = field(default_factory=list)
    scenarios: list[dict[str, Any]] = field(default_factory=list)

    @property
    def organizations(self) -> tuple[DemoOrg, ...]:
        """The invented organizations (ground truth for entity resolution)."""
        return DEMO_ORGS

    @property
    def topics(self) -> tuple[DemoTopic, ...]:
        """The demo topics."""
        return DEMO_TOPICS

    def count_by_type(self) -> dict[str, int]:
        """Number of records per record type."""
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.record_type] = counts.get(record.record_type, 0) + 1
        return counts


def _json_default(value: Any) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def add_months(day: date, months: int) -> date:
    """Return the first day of the month ``months`` after (or before) ``day``'s month."""
    index = day.year * 12 + (day.month - 1) + months
    return date(index // 12, index % 12 + 1, 1)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 60:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    threshold = math.exp(-lam)
    count, product = 0, 1.0
    while True:
        product *= rng.random()
        if product <= threshold:
            return count
        count += 1


def _choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _variant(rng: random.Random, org: DemoOrg) -> str:
    base_weights = [5, 2, 2, 1, 1]
    weights = base_weights[: len(org.variants)]
    return rng.choices(list(org.variants), weights=weights)[0]


def _cap(text: str) -> str:
    return text[0].upper() + text[1:]


def _when(rng: random.Random, month: date) -> datetime:
    return datetime(month.year, month.month, rng.randint(1, 28), 12, 0, tzinfo=UTC)


class _IdCounter:
    def __init__(self) -> None:
        self._counts: dict[SourceType, int] = {}

    def next(self, source_type: SourceType) -> str:
        self._counts[source_type] = self._counts.get(source_type, 0) + 1
        return f"{ID_PREFIX[source_type]}-{self._counts[source_type]:05d}"


# --------------------------------------------------------------------------------------
# Record builders
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Ctx:
    """Everything needed to build one record."""

    rng: random.Random
    source_type: SourceType
    topic: DemoTopic
    org: DemoOrg
    raw_name: str
    month: date
    index: int
    total: int
    record_id: str


def _phrases(ctx: _Ctx) -> tuple[str, str]:
    first = ctx.rng.choice(ctx.topic.phrases)
    rest = [p for p in ctx.topic.phrases if p != first] or [first]
    return first, ctx.rng.choice(rest)


def _base_payload(ctx: _Ctx, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "synthetic": True,
        "raw_organization_name": ctx.raw_name,
        "synthetic_ground_truth": {"topic": ctx.topic.key, "organization": ctx.org.canonical},
    }
    payload.update(extra)
    return payload


def _build_trial(ctx: _Ctx) -> tuple[str, str, dict[str, Any], dict[str, Any], datetime]:
    rng = ctx.rng
    phrase, other = _phrases(ctx)
    condition = rng.choice(CONDITIONS)
    zentavia_bispecific = ctx.topic.key == "bispecific_antibodies" and ctx.org.key.startswith(
        "zentavia"
    )
    if zentavia_bispecific:
        progress = ctx.index / max(ctx.total - 1, 1)
        if progress < 0.35:
            phase = "PHASE1"
        elif progress < 0.85:
            phase = rng.choice(["PHASE1", "PHASE2", "PHASE2"])
        else:
            phase = rng.choice(["PHASE2", "PHASE3"])
        enrollment = int(60 + progress * 500 + rng.randint(-20, 20))
    else:
        phase = _choice(rng, {"PHASE1": 0.35, "PHASE2": 0.4, "PHASE3": 0.2, "PHASE4": 0.05})
        enrollment = rng.randint(30, 400)
    ratio = ctx.index / max(ctx.total - 1, 1)
    status = _choice(
        rng,
        {
            "COMPLETED": 0.55 * (1 - ratio) + 0.01,
            "ACTIVE_NOT_RECRUITING": 0.2,
            "RECRUITING": 0.15 + 0.4 * ratio,
            "NOT_YET_RECRUITING": 0.05 + 0.1 * ratio,
            "TERMINATED": 0.05,
        },
    )
    published = _when(rng, ctx.month)
    start = (published + timedelta(days=rng.randint(-60, 60))).date()
    completion = start + timedelta(days=30 * rng.randint(12, 48))
    number = phase.replace("PHASE", "")
    title = f"A Phase {number} Study of a {phrase} Candidate in Participants With {condition}"
    abstract = (
        f"[SYNTHETIC] Interventional study evaluating a {phrase} candidate in {condition}. "
        f"Related terms: {other}. Sponsor: {ctx.raw_name}. Artificial demo data."
    )
    payload = _base_payload(ctx, sponsor_name=ctx.raw_name, phase=phase, enrollment=enrollment)
    detail = {
        "trial_identifier": ctx.record_id,
        "sponsor_name": ctx.raw_name,
        "phase": phase,
        "status": status,
        "enrollment": enrollment,
        "start_date": start,
        "completion_date": completion,
        "intervention": f"{_cap(phrase)} candidate",
        "condition": condition,
        "countries": sorted(rng.sample(COUNTRIES, rng.randint(1, 3))),
    }
    return title, abstract, payload, detail, published


def _build_publication(ctx: _Ctx) -> tuple[str, str, dict[str, Any], dict[str, Any], datetime]:
    rng = ctx.rng
    phrase, other = _phrases(ctx)
    condition = rng.choice(CONDITIONS)
    study = rng.choice(STUDY_TYPES)
    templates = (
        f"{_cap(phrase)} for {condition}: a {study}",
        f"Advances in {phrase} targeting {condition}",
        f"{_cap(phrase)}-based strategies in {condition}: a {study}",
        f"Evaluating {phrase} in {condition}",
    )
    title = rng.choice(templates)
    abstract = (
        f"[SYNTHETIC] This illustrative record discusses {phrase} in {condition}. It also mentions "
        f"{other}. Affiliation: {ctx.raw_name}. The content is artificial demo data."
    )
    published = _when(rng, ctx.month)
    authors = [f"Synthetic Author {rng.randint(1, 500)}" for _ in range(rng.randint(2, 5))]
    payload = _base_payload(ctx, affiliations=[ctx.raw_name])
    detail = {
        "publication_identifier": ctx.record_id,
        "journal": rng.choice(JOURNALS),
        "publication_date": published.date(),
        "authors": authors,
        "affiliations": [ctx.raw_name],
        "citation_count": rng.randint(0, 40),
        "publication_type": "journal-article",
    }
    return title, abstract, payload, detail, published


def _build_patent(ctx: _Ctx) -> tuple[str, str, dict[str, Any], dict[str, Any], datetime]:
    rng = ctx.rng
    phrase, other = _phrases(ctx)
    condition = rng.choice(CONDITIONS)
    published = _when(rng, ctx.month)
    application = published.date() - timedelta(days=30 * 18 + rng.randint(-30, 30))
    granted = rng.random() < 0.25
    title = f"Compositions and methods for {phrase} in {condition}"
    abstract = (
        f"[SYNTHETIC] Patent application describing {phrase} compositions for {condition}. "
        f"Related terms: {other}. Assignee: {ctx.raw_name}. Artificial demo data."
    )
    payload = _base_payload(ctx, assignee_name=ctx.raw_name)
    detail = {
        "patent_identifier": ctx.record_id,
        "application_date": application,
        "publication_date": published.date(),
        "grant_date": (
            (published.date() + timedelta(days=rng.randint(180, 720))) if granted else None
        ),
        "assignee_name": ctx.raw_name,
        "inventors": [
            f"Synthetic Inventor {rng.randint(1, 500)}" for _ in range(rng.randint(1, 4))
        ],
        "patent_classifications": list(CPC_CODES[ctx.topic.key]),
        "patent_family_id": f"SYN-FAM-{ctx.record_id[-5:]}",
        "legal_status": "granted" if granted else "pending",
    }
    return title, abstract, payload, detail, published


def _build_funding(ctx: _Ctx) -> tuple[str, str, dict[str, Any], dict[str, Any], datetime]:
    rng = ctx.rng
    phrase, _ = _phrases(ctx)
    condition = rng.choice(CONDITIONS)
    published = _when(rng, ctx.month)
    start = published.date()
    end = start + timedelta(days=365 * rng.randint(2, 4))
    agency = rng.choice(FUNDING_AGENCIES)
    amount = float(round(rng.uniform(150_000, 2_500_000), -3))
    title = f"{_cap(phrase)} research for {condition}"
    abstract = (
        f"[SYNTHETIC] Grant supporting {phrase} research in {condition}. Recipient: {ctx.raw_name}. "
        f"Agency: {agency}. Artificial demo data."
    )
    payload = _base_payload(ctx, recipient_name=ctx.raw_name, agency=agency)
    detail = {
        "award_identifier": ctx.record_id,
        "recipient_name": ctx.raw_name,
        "agency": agency,
        "amount": amount,
        "currency": "USD",
        "start_date": start,
        "end_date": end,
    }
    return title, abstract, payload, detail, published


def _build_announcement(
    ctx: _Ctx,
) -> tuple[str, str, dict[str, Any], dict[str, Any], datetime]:
    rng = ctx.rng
    phrase, _ = _phrases(ctx)
    kind = rng.choices(ANNOUNCEMENT_TYPES, weights=ANNOUNCEMENT_WEIGHTS)[0]
    partner_pool = [o for o in DEMO_ORGS if o.org_type == "company" and o.key != ctx.org.key]
    if rng.random() < 0.3:
        partner_pool = [_ORG_BY_KEY[k] for k in ACADEMIC_KEYS]
    partner = rng.choice(partner_pool)
    partner_name = _variant(rng, partner)
    who = ctx.raw_name
    number = rng.choice(["1", "2", "3"])
    bodies = {
        "partnership": (
            f"{who} and {partner_name} announce a research collaboration",
            f"{who} and {partner_name} today announced a research collaboration to advance {phrase} programs.",
        ),
        "licensing": (
            f"{who} enters exclusive license agreement with {partner_name}",
            f"{who} entered an exclusive license agreement with {partner_name} for {phrase} technology.",
        ),
        "acquisition": (
            f"{who} to acquire {partner_name}",
            f"{who} announced it will acquire {partner_name}, adding {phrase} capabilities.",
        ),
        "clinical_milestone": (
            f"{who} doses first patient in Phase {number} trial",
            f"{who} announced the first patient was dosed in a Phase {number} trial of a {phrase} candidate.",
        ),
        "funding": (
            f"{who} announces financing round",
            f"{who} announced a financing round to advance its {phrase} pipeline.",
        ),
        "regulatory": (
            f"{who} receives clearance to begin clinical study",
            f"{who} received regulatory clearance to begin a clinical study of a {phrase} therapy.",
        ),
    }
    title, body = bodies[kind]
    abstract = f"[SYNTHETIC] {body} This is an artificial press release for demonstration."
    published = _when(rng, ctx.month)
    has_partner = kind in {"partnership", "licensing", "acquisition"}
    payload = _base_payload(ctx, announcer_name=who)
    payload["synthetic_ground_truth"] = {
        **payload["synthetic_ground_truth"],
        "announcement_type": kind,
        "partner": partner.canonical if has_partner else None,
    }
    detail = {
        "organization_name": who,
        "announcement_type": None,
        "announcement_date": published.date(),
        "partner_organizations": None,
        "detected_topics": None,
    }
    return title, abstract, payload, detail, published


_BUILDERS = {
    SourceType.CLINICAL_TRIAL: _build_trial,
    SourceType.PUBLICATION: _build_publication,
    SourceType.PATENT: _build_patent,
    SourceType.FUNDING: _build_funding,
    SourceType.ANNOUNCEMENT: _build_announcement,
}


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
def _month_org_weights(topic_key: str, source_type: SourceType, m: int, n: int) -> dict[str, float]:
    weights = dict(TOPIC_COMPETITORS[topic_key])
    if topic_key == "bispecific_antibodies" and source_type in {T, A, P}:
        weights["zentavia"] = 1 + 0.6 * m  # steadily dominating the topic
    return weights


def _pick_org(
    rng: random.Random, topic_key: str, source_type: SourceType, m: int, n: int
) -> DemoOrg:
    if topic_key not in NO_ACADEMIC_TOPICS and rng.random() < ACADEMIC_SHARE[source_type]:
        academic = ACADEMIC_KEYS if source_type in {P, F} else ACADEMIC_KEYS[:2]
        return _ORG_BY_KEY[rng.choice(academic)]
    key = _choice(rng, _month_org_weights(topic_key, source_type, m, n))
    if key == "zentavia" and topic_key == "bispecific_antibodies" and rng.random() < 0.15:
        key = "zentavia_onc"
    return _ORG_BY_KEY[key]


def generate_demo_dataset(config: DemoConfig | None = None) -> DemoDataset:
    """Generate the synthetic dataset for ``config`` (deterministic for a given config).

    Raises:
        ValueError: if the configuration is out of range.
    """
    config = config or DemoConfig()
    config.validate()
    rng = random.Random(config.seed)
    n = config.months
    first_month = add_months(config.end_month, -(n - 1))
    months = [add_months(first_month, i) for i in range(n)]
    ids = _IdCounter()
    dataset = DemoDataset(config=config, months=months)

    spike_month = n - 4
    entry_month = n - 6
    false_trend_counts = {n - 18: 1, n - 3: 1, n - 2: 2, n - 1: 4}

    for m, month in enumerate(months):
        for topic in DEMO_TOPICS:
            for source_type in SOURCE_ORDER:
                jobs: list[tuple[int, str | None]] = []  # (count, forced organization key)
                if topic.key == FALSE_TREND_TOPIC:
                    if source_type is P:
                        jobs.append((false_trend_counts.get(m, 0), None))
                else:
                    shape = PROFILES[topic.key].get(source_type)
                    if shape is not None:
                        jobs.append((_poisson(rng, shape(m, n) * config.scale), None))
                    if (
                        topic.key == SPIKE_TOPIC
                        and source_type is SPIKE_SOURCE
                        and m == spike_month
                    ):
                        jobs.append((max(1, round(SPIKE_SIZE * config.scale)), "varethyn"))
                    if topic.key == ZENTAVIA_ENTRY_TOPIC and m >= entry_month:
                        extra = ENTRY_EXTRA_RATES.get(source_type)
                        if extra:
                            jobs.append((_poisson(rng, extra * config.scale), "zentavia"))
                for count, forced in jobs:
                    for _ in range(count):
                        org = (
                            _ORG_BY_KEY[forced]
                            if forced
                            else _pick_org(rng, topic.key, source_type, m, n)
                        )
                        record_id = ids.next(source_type)
                        ctx = _Ctx(
                            rng=rng,
                            source_type=source_type,
                            topic=topic,
                            org=org,
                            raw_name=_variant(rng, org),
                            month=month,
                            index=m,
                            total=n,
                            record_id=record_id,
                        )
                        title, abstract, payload, detail, published = _BUILDERS[source_type](ctx)
                        dataset.records.append(
                            DemoRecord(
                                source=SOURCE_NAMES[source_type],
                                source_record_id=record_id,
                                record_type=source_type.value,
                                title=title,
                                abstract=abstract,
                                source_url=f"{SYNTHETIC_URL_BASE}/{SOURCE_NAMES[source_type]}/{record_id}",
                                published_at=published,
                                payload=payload,
                                detail=detail,
                            )
                        )
    dataset.scenarios = _describe_scenarios(months, spike_month, entry_month)
    return dataset


def _describe_scenarios(
    months: list[date], spike_month: int, entry_month: int
) -> list[dict[str, Any]]:
    return [
        {
            "name": "sustained_trend",
            "topic": "crispr_gene_editing",
            "description": "Steady growth across publications, trials, patents, funding and announcements.",
        },
        {
            "name": "declining_topic",
            "topic": "immune_checkpoint",
            "description": "Activity falls across all source types.",
        },
        {
            "name": "one_time_anomaly",
            "topic": SPIKE_TOPIC,
            "source_type": SPIKE_SOURCE.value,
            "month": months[spike_month].isoformat(),
            "description": "Flat activity with one month of patents from a single organization.",
        },
        {
            "name": "high_growth_low_competition",
            "topic": "ai_drug_discovery",
            "description": "Strong multi-source growth with only two active competitors.",
        },
        {
            "name": "high_threat_competitor",
            "organization": "Zentavia Pharma",
            "topic": "bispecific_antibodies",
            "description": "Rapidly rising trials (with phase progression and larger enrollment), patents and publications.",
        },
        {
            "name": "new_market_entry",
            "organization": "Zentavia Pharma",
            "topic": ZENTAVIA_ENTRY_TOPIC,
            "first_month": months[entry_month].isoformat(),
            "description": "First significant activity in a new therapeutic area.",
        },
        {
            "name": "low_confidence_false_trend",
            "topic": FALSE_TREND_TOPIC,
            "description": "Large relative growth from about eight publications by one organization.",
        },
        {
            "name": "seasonal_topic",
            "topic": "mrna_therapeutics",
            "description": "Publication counts follow a yearly cycle.",
        },
        {
            "name": "topic_missing_from_taxonomy",
            "topic": "targeted_protein_degradation",
            "description": "Growing topic that config/topic_taxonomy.yaml does not contain (AI topic discovery target).",
        },
        {
            "name": "messy_organization_names",
            "organizations": [o.canonical for o in DEMO_ORGS],
            "description": "Several spellings per organization, a subsidiary (Zentavia Oncology Ltd) and a "
            "look-alike (Orvexa Labs vs Orvexa Bio) that must not be merged automatically.",
        },
    ]


# --------------------------------------------------------------------------------------
# Fixture files
# --------------------------------------------------------------------------------------
def _dump(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_fixtures(dataset: DemoDataset, directory: Path) -> list[Path]:
    """Write the dataset as JSON Lines (one file per source) plus reference JSON files.

    Output is deterministic, so unchanged data produces identical files.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source_type in SOURCE_ORDER:
        path = directory / f"{SOURCE_NAMES[source_type]}.jsonl"
        lines = [
            json.dumps(record.to_jsonable(), sort_keys=True, ensure_ascii=False)
            for record in dataset.records
            if record.record_type == source_type.value
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\n")
        written.append(path)
    references: dict[str, Any] = {
        "scenarios.json": dataset.scenarios,
        "organizations.json": [o.__dict__ for o in DEMO_ORGS],
        "topics.json": [t.__dict__ for t in DEMO_TOPICS],
        "config.json": {
            "seed": dataset.config.seed,
            "end_month": dataset.config.end_month,
            "months": dataset.config.months,
            "scale": dataset.config.scale,
            "synthetic": True,
        },
    }
    for name, data in references.items():
        path = directory / name
        _dump(path, data)
        written.append(path)
    return written
