"""Small helpers shared by test modules (importable because ``tests`` is on pythonpath)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAXONOMY_FILE = REPO_ROOT / "config" / "topic_taxonomy.yaml"
SCORING_FILE = REPO_ROOT / "config" / "scoring_weights.yaml"
REGISTRY_FILE = REPO_ROOT / "config" / "source_registry.yaml"
