"""Deciding which organizations to monitor as competitors.

Discovery ranks the organizations seen in collected records and combines that with the manual
lists from ``.env``:

* ``MANUAL`` - only the organizations you listed;
* ``HYBRID`` (default) - your list first, then the highest-ranked others fill the remaining
  slots up to ``TOP_COMPETITORS``;
* ``AUTO`` - discovery fills every slot.

Organizations named in ``COMPETITOR_EXCLUDE`` never appear, in any mode.

The ranking score (weights from ``config/scoring_weights.yaml``):

    100 x (0.30 trials + 0.25 patents + 0.20 publications + 0.15 funding + 0.10 announcements)

Each component is the organization's percentile rank among the candidates for that source type,
so counts from different sources are comparable. Activity sums the *link confidence* recorded
during normalization rather than plain record counts, so weaker evidence counts for less: a
sponsored trial is stored at full confidence, while an author's affiliation is stored at half,
because a paper written by someone at a company is not proof that the company ran the work.
Source types with no data anywhere have their weight shared out rather than counted as zero.

Universities, hospitals and government bodies are left out unless ``include_all_types`` is set,
and an organization needs ``MIN_COMPETITOR_EVIDENCE_COUNT`` records before it can be ranked.
Every score keeps its components, raw counts and example records, so any ranking can be traced
back to the data behind it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import OrganizationType, SourceType
from cews.database.models import Organization, RecordOrganization, SourceRecord
from cews.normalization.organizations import (
    classify_organization_type,
    normalize_organization_name,
)
from cews.scoring.normalization import clamp_score, percentile_normalize, renormalize_weights
from cews.settings import CompetitorPlan, Settings, resolve_competitor_mode

LOGGER = logging.getLogger(__name__)

DEFAULT_WINDOW_MONTHS = 12
DEFAULT_WEIGHTS: dict[str, float] = {
    "trial": 0.30,
    "patent": 0.25,
    "publication": 0.20,
    "funding": 0.15,
    "announcement": 0.10,
}
SOURCE_TYPE_BY_COMPONENT: dict[str, SourceType] = {
    "trial": SourceType.CLINICAL_TRIAL,
    "patent": SourceType.PATENT,
    "publication": SourceType.PUBLICATION,
    "funding": SourceType.FUNDING,
    "announcement": SourceType.ANNOUNCEMENT,
}
COMPETITOR_TYPES = frozenset({OrganizationType.COMPANY.value, OrganizationType.UNKNOWN.value})
EVIDENCE_SAMPLE = 5


@dataclass(frozen=True)
class ComponentScore:
    """One source type's contribution to an organization's ranking."""

    name: str
    raw_activity: float
    record_count: int
    normalized: float
    weight: float

    @property
    def contribution(self) -> float:
        """Points this component adds to the final score."""
        return round(self.normalized * self.weight, 3)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        return {
            "raw_activity": round(self.raw_activity, 3),
            "record_count": self.record_count,
            "normalized": round(self.normalized, 2),
            "weight": round(self.weight, 4),
            "contribution": self.contribution,
        }


@dataclass
class CompetitorScore:
    """An organization's discovery score with everything behind it."""

    organization_id: int
    name: str
    organization_type: str
    score: float
    evidence_count: int
    components: dict[str, ComponentScore]
    source_types: tuple[str, ...]
    evidence_record_ids: tuple[int, ...] = ()
    manually_included: bool = False
    discovered: bool = True

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy, suitable for an audit trail."""
        return {
            "organization": self.name,
            "organization_id": self.organization_id,
            "organization_type": self.organization_type,
            "score": round(self.score, 2),
            "evidence_count": self.evidence_count,
            "source_types": list(self.source_types),
            "components": {name: c.as_dict() for name, c in self.components.items()},
            "evidence_record_ids": list(self.evidence_record_ids),
            "manually_included": self.manually_included,
            "discovered": self.discovered,
        }


@dataclass
class DiscoveryResult:
    """The monitored competitor list and how it was reached."""

    monitored: list[CompetitorScore] = field(default_factory=list)
    considered: list[CompetitorScore] = field(default_factory=list)
    plan: CompetitorPlan | None = None
    window_start: datetime | None = None
    window_end: datetime | None = None
    weights: dict[str, float] = field(default_factory=dict)
    unavailable_sources: tuple[str, ...] = ()
    missing_manual_names: tuple[str, ...] = ()
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly summary."""
        return {
            "monitored": [c.as_dict() for c in self.monitored],
            "mode": self.plan.mode.value if self.plan else None,
            "window": {
                "start": self.window_start.isoformat() if self.window_start else None,
                "end": self.window_end.isoformat() if self.window_end else None,
            },
            "weights": self.weights,
            "unavailable_sources": list(self.unavailable_sources),
            "missing_manual_names": list(self.missing_manual_names),
            "warnings": self.warnings,
        }


def load_discovery_weights(settings: Settings) -> tuple[dict[str, float], int]:
    """Read the discovery weights and window from ``scoring_weights.yaml``.

    Falls back to the documented defaults when the file cannot be read.
    """
    weights = dict(DEFAULT_WEIGHTS)
    window = DEFAULT_WINDOW_MONTHS
    try:
        data = yaml.safe_load(settings.scoring_config_file.read_text(encoding="utf-8")) or {}
        configured = (data.get("weight_sets") or {}).get("competitor_discovery")
        if isinstance(configured, Mapping):
            parsed = {str(k): float(v) for k, v in configured.items()}
            if parsed and set(parsed) <= set(DEFAULT_WEIGHTS):
                weights = parsed
        discovery = data.get("competitor_discovery") or {}
        months = discovery.get("window_months") if isinstance(discovery, Mapping) else None
        if isinstance(months, int) and months > 0:
            window = months
    except (OSError, yaml.YAMLError, TypeError, ValueError) as exc:
        LOGGER.warning("using default discovery weights: %s", exc)
    return weights, window


def _activity(
    session: Session, start: datetime, end: datetime
) -> tuple[dict[int, dict[str, float]], dict[int, dict[str, int]], dict[int, list[int]]]:
    """Weighted activity, record counts and example records per organization and source type."""
    rows = session.execute(
        select(
            RecordOrganization.organization_id,
            SourceRecord.record_type,
            func.sum(RecordOrganization.confidence),
            func.count(func.distinct(SourceRecord.id)),
        )
        .join(SourceRecord, SourceRecord.id == RecordOrganization.source_record_id)
        .where(
            SourceRecord.published_at.is_not(None),
            SourceRecord.published_at >= start,
            SourceRecord.published_at < end,
            SourceRecord.duplicate_of_id.is_(None),
        )
        .group_by(RecordOrganization.organization_id, SourceRecord.record_type)
    ).all()
    weighted: dict[int, dict[str, float]] = {}
    counts: dict[int, dict[str, int]] = {}
    for organization_id, record_type, confidence_sum, record_count in rows:
        weighted.setdefault(organization_id, {})[record_type] = float(confidence_sum or 0)
        counts.setdefault(organization_id, {})[record_type] = int(record_count or 0)

    examples: dict[int, list[int]] = {}
    for organization_id in weighted:
        examples[organization_id] = [
            int(record_id)
            for record_id in session.scalars(
                select(SourceRecord.id)
                .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
                .where(
                    RecordOrganization.organization_id == organization_id,
                    SourceRecord.published_at >= start,
                    SourceRecord.published_at < end,
                    SourceRecord.duplicate_of_id.is_(None),
                )
                .order_by(SourceRecord.published_at.desc())
                .limit(EVIDENCE_SAMPLE)
            )
        ]
    return weighted, counts, examples


def _match_manual_names(session: Session, names: Sequence[str]) -> tuple[dict[str, int], list[str]]:
    """Map configured competitor names to organizations, reporting the ones not seen yet."""
    index: dict[str, int] = {}
    for organization in session.scalars(select(Organization)):
        parsed = normalize_organization_name(organization.canonical_name)
        for _, key in parsed.match_keys():
            index.setdefault(key, organization.id)
        index.setdefault(organization.normalized_name, organization.id)

    found: dict[str, int] = {}
    missing: list[str] = []
    for name in names:
        parsed = normalize_organization_name(name)
        match = next((index[key] for _, key in parsed.match_keys() if key in index), None)
        if match is None:
            missing.append(name)
        else:
            found[name] = match
    return found, missing


def discover_competitors(
    session: Session,
    settings: Settings,
    *,
    as_of: datetime | None = None,
    window_months: int | None = None,
    include_all_types: bool = False,
    persist: bool = True,
) -> DiscoveryResult:
    """Rank organizations and return the competitor list to monitor.

    Args:
        as_of: end of the activity window (default: now).
        window_months: length of the window (default: from ``scoring_weights.yaml``).
        include_all_types: also rank universities, hospitals and government bodies.
        persist: record the include/discovered flags on the organizations.

    Raises:
        SettingsError: if the competitor configuration is unusable (for example MANUAL mode
            with no usable names).
    """
    plan = resolve_competitor_mode(settings)
    weights, configured_months = load_discovery_weights(settings)
    months = window_months or configured_months
    end = as_of or datetime.now(UTC)
    start = end - timedelta(days=30 * months)

    result = DiscoveryResult(plan=plan, window_start=start, window_end=end)
    result.warnings.extend(plan.warnings)

    weighted, counts, examples = _activity(session, start, end)
    organizations = {
        organization.id: organization
        for organization in session.scalars(select(Organization))
        if organization.id in weighted
    }
    manual_ids, missing = _match_manual_names(session, plan.include)
    excluded_ids = set(_match_manual_names(session, plan.exclude)[0].values())
    result.missing_manual_names = tuple(missing)
    if missing:
        # A configured competitor is monitored whether or not any record mentions it yet, so
        # its absence is visible on the dashboard rather than silently dropped.
        result.warnings.append(
            f"no records collected yet for: {', '.join(missing)} (monitored with no activity)"
        )
        placeholder_id = 0
        for name in missing:
            parsed = normalize_organization_name(name)
            organization = Organization(
                canonical_name=parsed.original[:255],
                normalized_name=(parsed.expanded or parsed.normalized)[:255],
                organization_type=classify_organization_type(name).organization_type.value,
                manually_included=True,
                discovered_automatically=False,
                created_at=datetime.now(UTC),
            )
            if persist:
                session.add(organization)
                session.flush()
            else:
                # A preview must show the same list a real run would, so the organization is
                # given a temporary id instead of a database row.
                placeholder_id -= 1
                organization.id = placeholder_id
            manual_ids[name] = organization.id
            organizations[organization.id] = organization

    manual_id_set = set(manual_ids.values())
    candidates: list[int] = []
    for organization_id, organization in organizations.items():
        if organization_id in excluded_ids or organization.manually_excluded:
            continue
        if (
            not include_all_types
            and organization.organization_type not in COMPETITOR_TYPES
            and organization_id not in manual_id_set
        ):
            continue  # asking for an organization by name overrides the type filter
        evidence = sum(counts.get(organization_id, {}).values())
        if evidence == 0:
            continue  # nothing to rank; configured competitors still appear, with score 0
        if (
            evidence < settings.min_competitor_evidence_count
            and organization_id not in manual_id_set
        ):
            continue
        candidates.append(organization_id)

    available = [
        component
        for component, source_type in SOURCE_TYPE_BY_COMPONENT.items()
        if any(weighted.get(oid, {}).get(source_type.value) for oid in candidates)
    ]
    result.unavailable_sources = tuple(c for c in SOURCE_TYPE_BY_COMPONENT if c not in available)
    if not available:
        result.warnings.append("no activity in the selected window; nothing to rank")
        return _finish(session, result, plan, manual_ids, organizations, persist)

    effective = renormalize_weights(weights, available)
    result.weights = {name: round(weight, 4) for name, weight in effective.items()}
    if result.unavailable_sources:
        result.warnings.append(
            "no data for: "
            + ", ".join(sorted(result.unavailable_sources))
            + "; their weight was shared across the available sources"
        )

    normalized: dict[str, dict[int, float]] = {}
    for component in available:
        source_type = SOURCE_TYPE_BY_COMPONENT[component].value
        values = [weighted.get(oid, {}).get(source_type, 0.0) for oid in candidates]
        normalized[component] = dict(zip(candidates, percentile_normalize(values), strict=True))

    scored: list[CompetitorScore] = []
    for organization_id in candidates:
        organization = organizations[organization_id]
        components = {
            component: ComponentScore(
                name=component,
                raw_activity=weighted.get(organization_id, {}).get(
                    SOURCE_TYPE_BY_COMPONENT[component].value, 0.0
                ),
                record_count=counts.get(organization_id, {}).get(
                    SOURCE_TYPE_BY_COMPONENT[component].value, 0
                ),
                normalized=normalized[component][organization_id],
                weight=effective[component],
            )
            for component in available
        }
        scored.append(
            CompetitorScore(
                organization_id=organization_id,
                name=organization.canonical_name,
                organization_type=organization.organization_type,
                score=round(clamp_score(sum(c.contribution for c in components.values())), 2),
                evidence_count=sum(counts.get(organization_id, {}).values()),
                components=components,
                source_types=tuple(sorted(counts.get(organization_id, {}))),
                evidence_record_ids=tuple(examples.get(organization_id, ())),
                manually_included=organization_id in manual_id_set,
                discovered=organization_id not in manual_id_set,
            )
        )
    scored.sort(key=lambda c: (-c.score, c.name))
    result.considered = scored
    return _finish(session, result, plan, manual_ids, organizations, persist)


def _finish(
    session: Session,
    result: DiscoveryResult,
    plan: CompetitorPlan,
    manual_ids: Mapping[str, int],
    organizations: Mapping[int, Organization],
    persist: bool,
) -> DiscoveryResult:
    """Apply the competitor mode to the ranking and record the outcome."""
    by_id = {score.organization_id: score for score in result.considered}
    monitored: list[CompetitorScore] = []
    seen: set[int] = set()

    for name, organization_id in manual_ids.items():
        if organization_id in seen:
            continue
        score = by_id.get(organization_id)
        if score is None:
            organization = organizations.get(organization_id)
            score = CompetitorScore(
                organization_id=organization_id,
                name=organization.canonical_name if organization else name,
                organization_type=organization.organization_type if organization else "unknown",
                score=0.0,
                evidence_count=0,
                components={},
                source_types=(),
            )
        score.manually_included = True
        score.discovered = False
        seen.add(organization_id)
        monitored.append(score)

    filled = 0
    for score in result.considered:
        if filled >= plan.auto_slots:
            break
        if score.organization_id in seen:
            continue
        seen.add(score.organization_id)
        monitored.append(score)
        filled += 1

    monitored.sort(key=lambda c: (-c.score, c.name))
    result.monitored = monitored

    if persist:
        manual_id_set = set(manual_ids.values())
        monitored_ids = {score.organization_id for score in monitored}
        for organization in session.scalars(select(Organization)):
            organization.manually_included = organization.id in manual_id_set
            organization.discovered_automatically = (
                organization.id in monitored_ids and organization.id not in manual_id_set
            )
        session.flush()
    LOGGER.info(
        "discovery: %d monitored of %d candidate(s), mode %s",
        len(result.monitored),
        len(result.considered),
        plan.mode.value,
    )
    return result
