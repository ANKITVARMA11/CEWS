"""Unit tests for loading and checking the scoring configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from cews.scoring.config import (
    EXPECTED_COMPONENTS,
    ScoringConfigError,
    available_components,
    load_scoring_config,
    validate_weights,
)
from cews.settings import Settings, load_settings
from support import SCORING_FILE

pytestmark = pytest.mark.unit


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None, overrides={"scoring_config_file": SCORING_FILE})


def write_config(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "scoring.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def base_document() -> dict[str, Any]:
    return yaml.safe_load(SCORING_FILE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# validate_weights
# --------------------------------------------------------------------------------------
def test_weights_that_sum_to_one_are_accepted() -> None:
    assert validate_weights({"a": 0.6, "b": 0.4}) == {"a": 0.6, "b": 0.4}


def test_weights_that_do_not_sum_to_one_are_refused() -> None:
    """Otherwise a score silently cannot reach 100."""
    with pytest.raises(ScoringConfigError, match="must sum to 1"):
        validate_weights({"a": 0.6, "b": 0.3}, name="trend_score")


@pytest.mark.parametrize(
    "weights",
    [{}, {"a": -0.5, "b": 1.5}, {"a": 0.0, "b": 0.0}, {"a": "half", "b": 0.5}, {"a": float("inf")}],
)
def test_invalid_weights_are_refused(weights: dict[str, Any]) -> None:
    with pytest.raises(ScoringConfigError):
        validate_weights(weights)


def test_a_zero_weight_is_allowed() -> None:
    """Turning one component off is a legitimate configuration choice."""
    assert validate_weights({"a": 1.0, "b": 0.0})["b"] == 0.0


# --------------------------------------------------------------------------------------
# The shipped configuration
# --------------------------------------------------------------------------------------
def test_the_shipped_configuration_loads(settings: Settings) -> None:
    config = load_scoring_config(settings)
    assert config.version == "1.0.0"
    assert config.weights("trend_score") == {
        "velocity": 0.30,
        "momentum": 0.25,
        "patent_growth": 0.20,
        "funding_growth": 0.15,
        "consistency": 0.10,
    }
    assert sum(config.weights("confidence_score").values()) == pytest.approx(1.0)


def test_every_weight_set_sums_to_one(settings: Settings) -> None:
    config = load_scoring_config(settings)
    for name in config.weight_sets:
        assert sum(config.weights(name).values()) == pytest.approx(1.0), name


def test_the_deliberate_composite_choice_is_surfaced(settings: Settings) -> None:
    """Patents and funding are left out on purpose; that must be visible, not silent."""
    warnings = " ".join(load_scoring_config(settings).warnings)
    assert "composite_activity" in warnings and "funding" in warnings and "patent" in warnings


def test_categories_are_ordered_bands(settings: Settings) -> None:
    bands = load_scoring_config(settings).categories_for("trend")
    assert [band.label for band in bands][0] == "Low activity or declining"
    assert bands[-1].maximum == 100.0
    assert all(bands[i].maximum < bands[i + 1].minimum for i in range(len(bands) - 1))


def test_emerging_trend_rules_are_loaded(settings: Settings) -> None:
    rules = load_scoring_config(settings).emerging_trend
    assert rules.min_source_types >= 2  # one source cannot corroborate itself
    assert rules.exclude_single_spike is True
    assert 0 <= rules.min_confidence <= 100


def test_threat_modifiers_are_capped(settings: Settings) -> None:
    modifiers = load_scoring_config(settings).threat_modifiers
    assert modifiers.total_cap_points > 0
    assert all(item.max_points > 0 for item in modifiers.items)
    assert modifiers.cap_for("phase_progression") > 0
    assert modifiers.cap_for("not_a_modifier") == 0.0


# --------------------------------------------------------------------------------------
# Broken configurations
# --------------------------------------------------------------------------------------
def test_a_missing_file_is_reported(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(ScoringConfigError, match="not found"):
        load_scoring_config(settings, path=tmp_path / "absent.yaml")


def test_unreadable_yaml_is_reported(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("weight_sets: [unclosed\n", encoding="utf-8")
    with pytest.raises(ScoringConfigError, match="cannot read"):
        load_scoring_config(settings, path=path)


def test_a_weight_set_that_does_not_add_up_is_refused(settings: Settings, tmp_path: Path) -> None:
    document = base_document()
    document["weight_sets"]["trend_score"]["velocity"] = 0.9
    with pytest.raises(ScoringConfigError, match="weight_sets.trend_score"):
        load_scoring_config(settings, path=write_config(tmp_path, document))


def test_a_misspelled_component_is_refused(settings: Settings, tmp_path: Path) -> None:
    """A typo would otherwise drop that component from every score, silently."""
    document = base_document()
    weights = document["weight_sets"]["trend_score"]
    weights["velocty"] = weights.pop("velocity")
    with pytest.raises(ScoringConfigError, match="unknown component"):
        load_scoring_config(settings, path=write_config(tmp_path, document))


def test_a_missing_component_is_refused(settings: Settings, tmp_path: Path) -> None:
    document = base_document()
    weights = document["weight_sets"]["innovation_score"]
    del weights["funding"]
    weights["patent"] = round(weights["patent"] + 0.10, 4)
    with pytest.raises(ScoringConfigError, match="missing component"):
        load_scoring_config(settings, path=write_config(tmp_path, document))


def test_overlapping_categories_are_refused(settings: Settings, tmp_path: Path) -> None:
    document = base_document()
    document["categories"]["trend"][1]["min"] = 10
    with pytest.raises(ScoringConfigError, match="overlap"):
        load_scoring_config(settings, path=write_config(tmp_path, document))


@pytest.mark.parametrize(
    ("field", "value"), [("min_trend_score", 150), ("min_confidence", -5), ("min_source_types", 0)]
)
def test_impossible_rules_are_refused(
    settings: Settings, tmp_path: Path, field: str, value: float
) -> None:
    document = base_document()
    document["emerging_trend_rules"][field] = value
    with pytest.raises(ScoringConfigError, match=field):
        load_scoring_config(settings, path=write_config(tmp_path, document))


def test_no_weight_sets_at_all(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(ScoringConfigError, match="no weight_sets"):
        load_scoring_config(settings, path=write_config(tmp_path, {"scoring_version": "2"}))


def test_expected_components_cover_every_shipped_set(settings: Settings) -> None:
    """Every set in the file is checked against a known component list."""
    assert set(load_scoring_config(settings).weight_sets) <= set(EXPECTED_COMPONENTS)


def test_available_components_keeps_configured_order() -> None:
    weights = {"velocity": 0.3, "momentum": 0.25, "consistency": 0.1}
    assert available_components(weights, {"consistency", "velocity"}) == ["velocity", "consistency"]
