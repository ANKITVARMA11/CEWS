"""Reading and checking the scoring configuration.

Every weight, threshold, category and rule lives in ``config/scoring_weights.yaml`` so the way
CEWS scores things can be changed without touching code. This module loads that file, checks it
adds up, and hands the rest of the scoring engine a validated object.

The checks matter: a weight set that does not sum to 1 would silently produce scores that cannot
reach 100, and a typo in a component name would silently drop that component from every score.
Both are refused at load time with a message naming the problem.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

WEIGHT_TOLERANCE = 1e-6
DEFAULT_VERSION = "1.0.0"

# The components each weight set must describe, so a typo cannot quietly remove one.
EXPECTED_COMPONENTS: dict[str, frozenset[str]] = {
    "composite_activity": frozenset(
        {"publication", "clinical_trial", "patent", "funding", "announcement"}
    ),
    "trend_score": frozenset(
        {"velocity", "momentum", "patent_growth", "funding_growth", "consistency"}
    ),
    "innovation_score": frozenset({"patent", "clinical_trial", "publication", "funding"}),
    "threat_score": frozenset({"trial_growth", "patent_growth", "publication_growth"}),
    "opportunity_score": frozenset({"trend", "low_competition", "source_agreement"}),
    "confidence_score": frozenset(
        {"sample", "source_agreement", "completeness", "freshness", "model_stability"}
    ),
    "competitor_discovery": frozenset(
        {"trial", "patent", "publication", "funding", "announcement"}
    ),
}
# Sets where leaving a component out is a deliberate choice rather than a mistake.
OPTIONAL_SETS = frozenset({"composite_activity"})


class ScoringConfigError(ValueError):
    """Raised when the scoring configuration cannot be used as written."""


@dataclass(frozen=True)
class Category:
    """A named band of a 0-100 score."""

    label: str
    minimum: float
    maximum: float

    def contains(self, value: float) -> bool:
        """True when ``value`` falls in this band."""
        return self.minimum <= value <= self.maximum


@dataclass(frozen=True)
class EmergingTrendRules:
    """What a topic must satisfy before it may be called an emerging trend.

    A high score on its own is never enough: the spec requires confidence, a minimum sample, and
    support from more than one independent source, and it excludes a single isolated spike.
    """

    min_trend_score: float = 60.0
    min_confidence: float = 60.0
    min_source_types: int = 2
    min_sample_size: int = 10
    exclude_single_spike: bool = True


@dataclass(frozen=True)
class ThreatModifier:
    """One capped adjustment to a threat score."""

    name: str
    max_points: float


@dataclass(frozen=True)
class ThreatModifiers:
    """The optional threat adjustments, individually and collectively capped."""

    enabled: bool = True
    total_cap_points: float = 15.0
    items: tuple[ThreatModifier, ...] = ()

    def cap_for(self, name: str) -> float:
        """The cap for one modifier, or 0 when it is not configured."""
        return next((item.max_points for item in self.items if item.name == name), 0.0)


@dataclass(frozen=True)
class ScoringConfig:
    """The validated scoring configuration."""

    version: str
    weight_sets: Mapping[str, Mapping[str, float]]
    categories: Mapping[str, tuple[Category, ...]]
    emerging_trend: EmergingTrendRules
    threat_modifiers: ThreatModifiers
    source: Path | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def weights(self, name: str) -> dict[str, float]:
        """The weight set called ``name``.

        Raises:
            ScoringConfigError: if that set is not configured.
        """
        if name not in self.weight_sets:
            raise ScoringConfigError(f"no weight set named {name!r} in the scoring configuration")
        return dict(self.weight_sets[name])

    def categories_for(self, score_type: str) -> tuple[Category, ...]:
        """The bands for a score type, or an empty tuple when none are configured."""
        return self.categories.get(score_type, ())


def validate_weights(weights: Mapping[str, float], *, name: str = "weights") -> dict[str, float]:
    """Check one weight set and return it as plain floats.

    Weights must be finite, not negative, at least one must be positive, and they must sum to 1
    (within a small tolerance), because every score is meant to reach 0-100.

    Raises:
        ScoringConfigError: describing what is wrong.
    """
    if not weights:
        raise ScoringConfigError(f"{name}: no components configured")
    clean: dict[str, float] = {}
    for component, value in weights.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ScoringConfigError(f"{name}.{component}: {value!r} is not a number") from exc
        if not math.isfinite(number) or number < 0:
            raise ScoringConfigError(f"{name}.{component}: weight must be finite and not negative")
        clean[str(component)] = number
    total = sum(clean.values())
    if total <= 0:
        raise ScoringConfigError(f"{name}: at least one weight must be above zero")
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ScoringConfigError(
            f"{name}: weights must sum to 1, but sum to {total:.4f} "
            f"({', '.join(f'{k}={v}' for k, v in sorted(clean.items()))})"
        )
    return clean


def _parse_categories(raw: Any, label: str) -> tuple[Category, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ScoringConfigError(f"categories.{label}: expected a list of bands")
    bands: list[Category] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or "label" not in entry:
            raise ScoringConfigError(f"categories.{label}[{index}]: needs a label, min and max")
        try:
            minimum = float(entry.get("min", 0))
            maximum = float(entry.get("max", 100))
        except (TypeError, ValueError) as exc:
            raise ScoringConfigError(
                f"categories.{label}[{index}]: min/max must be numbers"
            ) from exc
        if minimum > maximum:
            raise ScoringConfigError(f"categories.{label}[{index}]: min is greater than max")
        bands.append(Category(str(entry["label"]), minimum, maximum))
    ordered = tuple(sorted(bands, key=lambda band: band.minimum))
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if later.minimum <= earlier.maximum:
            raise ScoringConfigError(
                f"categories.{label}: bands '{earlier.label}' and '{later.label}' overlap"
            )
    return ordered


def _parse_threat_modifiers(raw: Any) -> ThreatModifiers:
    if not isinstance(raw, Mapping):
        return ThreatModifiers()
    items: list[ThreatModifier] = []
    for name, entry in (raw.get("items") or {}).items():
        points = entry.get("max_points", 0) if isinstance(entry, Mapping) else entry
        try:
            value = float(points)
        except (TypeError, ValueError) as exc:
            raise ScoringConfigError(
                f"threat_modifiers.{name}: max_points must be a number"
            ) from exc
        if value < 0:
            raise ScoringConfigError(f"threat_modifiers.{name}: max_points cannot be negative")
        items.append(ThreatModifier(str(name), value))
    total_cap = float(raw.get("total_cap_points", 15.0))
    if total_cap < 0:
        raise ScoringConfigError("threat_modifiers.total_cap_points cannot be negative")
    return ThreatModifiers(
        enabled=bool(raw.get("enabled", True)),
        total_cap_points=total_cap,
        items=tuple(sorted(items, key=lambda item: item.name)),
    )


def load_scoring_config(settings: Settings, *, path: Path | None = None) -> ScoringConfig:
    """Load and validate ``config/scoring_weights.yaml``.

    Raises:
        ScoringConfigError: if the file is missing, unreadable, or does not add up.
    """
    source = path or settings.scoring_config_file
    if not source.is_file():
        raise ScoringConfigError(f"scoring configuration not found: {source}")
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ScoringConfigError(f"cannot read {source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ScoringConfigError(f"{source}: expected a mapping at the top level")

    raw_sets = document.get("weight_sets")
    if not isinstance(raw_sets, Mapping) or not raw_sets:
        raise ScoringConfigError(f"{source}: no weight_sets configured")

    warnings: list[str] = []
    weight_sets: dict[str, dict[str, float]] = {}
    for name, weights in raw_sets.items():
        if not isinstance(weights, Mapping):
            raise ScoringConfigError(
                f"weight_sets.{name}: expected a mapping of component to weight"
            )
        checked = validate_weights(weights, name=f"weight_sets.{name}")
        expected = EXPECTED_COMPONENTS.get(str(name))
        if expected is not None:
            unknown = sorted(set(checked) - expected)
            if unknown:
                raise ScoringConfigError(
                    f"weight_sets.{name}: unknown component(s) {unknown}; expected {sorted(expected)}"
                )
            missing = sorted(expected - set(checked))
            if missing and str(name) not in OPTIONAL_SETS:
                raise ScoringConfigError(
                    f"weight_sets.{name}: missing component(s) {missing}; every component must be "
                    "given a weight, even if it is zero"
                )
            if missing:
                warnings.append(
                    f"weight_sets.{name} leaves out {', '.join(missing)}; those sources do not "
                    "contribute to this signal"
                )
        weight_sets[str(name)] = checked

    raw_categories = document.get("categories") or {}
    if not isinstance(raw_categories, Mapping):
        raise ScoringConfigError(f"{source}: categories must be a mapping")
    categories = {
        str(label): _parse_categories(bands, str(label)) for label, bands in raw_categories.items()
    }

    rules_raw = document.get("emerging_trend_rules") or {}
    if not isinstance(rules_raw, Mapping):
        raise ScoringConfigError(f"{source}: emerging_trend_rules must be a mapping")
    rules = EmergingTrendRules(
        min_trend_score=float(rules_raw.get("min_trend_score", 60.0)),
        min_confidence=float(rules_raw.get("min_confidence", 60.0)),
        min_source_types=int(rules_raw.get("min_source_types", 2)),
        min_sample_size=int(rules_raw.get("min_sample_size", settings.min_topic_sample_size)),
        exclude_single_spike=bool(rules_raw.get("exclude_single_spike", True)),
    )
    for field_name, value in (
        ("min_trend_score", rules.min_trend_score),
        ("min_confidence", rules.min_confidence),
    ):
        if not 0 <= value <= 100:
            raise ScoringConfigError(f"emerging_trend_rules.{field_name} must be between 0 and 100")
    if rules.min_source_types < 1:
        raise ScoringConfigError("emerging_trend_rules.min_source_types must be at least 1")

    config = ScoringConfig(
        version=str(document.get("scoring_version", DEFAULT_VERSION)),
        weight_sets=weight_sets,
        categories=categories,
        emerging_trend=rules,
        threat_modifiers=_parse_threat_modifiers(document.get("threat_modifiers")),
        source=source,
        warnings=tuple(warnings),
    )
    LOGGER.debug("loaded scoring configuration version %s from %s", config.version, source)
    return config


def available_components(
    weights: Mapping[str, float], present: Sequence[str] | set[str]
) -> list[str]:
    """The components of a weight set that actually have data, in configured order."""
    return [component for component in weights if component in set(present)]
