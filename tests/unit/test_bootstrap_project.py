"""Unit tests for scripts/bootstrap_project.py (Phase 1)."""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest

import bootstrap_project as bp

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def manifest() -> bp.Manifest:
    """The manifest is pure data, so build it once."""
    return bp.build_manifest()


@pytest.fixture
def scaffold(tmp_path: Path) -> Path:
    """A freshly bootstrapped project root."""
    report = bp.bootstrap(tmp_path)
    assert report.failed() == []
    return tmp_path


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def test_manifest_paths_are_unique_relative_and_safe(manifest: bp.Manifest) -> None:
    paths = [f.path for f in manifest.files]
    assert len(paths) == len(set(paths))
    for path in paths + [d.path for d in manifest.directories]:
        assert not path.startswith("/")
        assert ".." not in path.split("/")


def test_manifest_contains_required_root_files(manifest: bp.Manifest) -> None:
    paths = {f.path for f in manifest.files}
    required = {
        ".env.example",
        ".gitignore",
        "README.md",
        "LICENSE",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "Makefile",
        "docker-compose.yml",
        "pytest.ini",
        "alembic.ini",
        "config/logging.yaml",
        "config/scoring_weights.yaml",
        "config/topic_taxonomy.yaml",
        "config/source_registry.yaml",
        "src/cews/__init__.py",
        "src/cews/settings.py",
        "src/cews/ai/topic_discovery.py",
        "migrations/sql/create_views.sql",
        "dashboards/powerbi/dax_measures.md",
        "dashboards/streamlit/app.py",
        "tests/conftest.py",
        "docs/architecture.md",
    }
    assert required <= paths


def test_manifest_does_not_include_the_script_or_its_own_test(manifest: bp.Manifest) -> None:
    paths = {f.path for f in manifest.files}
    assert "scripts/bootstrap_project.py" not in paths
    assert "tests/unit/test_bootstrap_project.py" not in paths


def test_root_and_config_files_are_critical(manifest: bp.Manifest) -> None:
    critical = {f.path for f in manifest.files if f.critical}
    assert {"pyproject.toml", ".env.example", "Makefile", "config/scoring_weights.yaml"} <= critical
    assert "docs/troubleshooting.md" not in critical


def test_every_source_package_has_an_init(manifest: bp.Manifest) -> None:
    paths = {f.path for f in manifest.files}
    for package in bp.SRC_MODULES:
        assert f"{package}/__init__.py" in paths


# --------------------------------------------------------------------------------------
# Creation, idempotency, safety
# --------------------------------------------------------------------------------------
def test_bootstrap_creates_every_manifest_item(scaffold: Path, manifest: bp.Manifest) -> None:
    assert bp.validate_scaffold(scaffold, manifest) == []
    for spec in manifest.files:
        assert (scaffold / spec.path).is_file()


def test_second_run_creates_nothing_and_skips_everything(scaffold: Path) -> None:
    second = bp.bootstrap(scaffold)
    assert second.select(bp.Status.CREATED) == []
    assert second.select(bp.Status.OVERWRITTEN) == []
    assert second.failed() == []
    assert len(second.select(bp.Status.SKIPPED)) == len(second.results)


def test_non_empty_file_is_not_overwritten_without_force(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("my own readme", encoding="utf-8")
    report = bp.bootstrap(tmp_path)
    assert readme.read_text(encoding="utf-8") == "my own readme"
    result = next(r for r in report.results if r.path == "README.md")
    assert result.status is bp.Status.SKIPPED


def test_force_overwrites_non_empty_file(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("my own readme", encoding="utf-8")
    report = bp.bootstrap(tmp_path, force=True)
    assert "CEWS" in readme.read_text(encoding="utf-8")
    result = next(r for r in report.results if r.path == "README.md")
    assert result.status is bp.Status.OVERWRITTEN


def test_existing_empty_file_is_populated(tmp_path: Path) -> None:
    env_example = tmp_path / ".env.example"
    env_example.write_text("", encoding="utf-8")
    bp.bootstrap(tmp_path)
    assert "FETCH_INTERVAL_MINUTES=120" in env_example.read_text(encoding="utf-8")


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    report = bp.bootstrap(tmp_path, dry_run=True)
    assert list(tmp_path.iterdir()) == []
    assert report.count(bp.Status.CREATED, bp.Kind.FILE) > 100


def test_missing_root_is_created(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "cews"
    report = bp.bootstrap(root)
    assert report.failed() == []
    assert (root / "pyproject.toml").is_file()


def test_failure_is_isolated_and_reported(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "architecture.md").mkdir()  # a directory where a file belongs
    report = bp.bootstrap(tmp_path)
    failed = {r.path: r for r in report.failed()}
    assert "docs/architecture.md" in failed
    assert failed["docs/architecture.md"].critical is False
    assert (tmp_path / "pyproject.toml").is_file()  # everything else still created


def test_safe_join_rejects_traversal_and_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        bp._safe_join(tmp_path, "../outside.txt")
    with pytest.raises(ValueError):
        bp._safe_join(tmp_path, "/etc/passwd")
    assert bp._safe_join(tmp_path, "a/b.txt").is_relative_to(tmp_path.resolve())


# --------------------------------------------------------------------------------------
# Exit codes and CLI
# --------------------------------------------------------------------------------------
def test_main_returns_zero_on_success(tmp_path: Path) -> None:
    assert bp.main(["--root", str(tmp_path)]) == bp.EXIT_OK


def test_main_returns_one_when_critical_item_fails(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").mkdir()  # critical file blocked by a directory
    assert bp.main(["--root", str(tmp_path)]) == bp.EXIT_CRITICAL_FAILURE


def test_main_returns_zero_when_only_non_critical_item_fails(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "troubleshooting.md").mkdir()
    assert bp.main(["--root", str(tmp_path)]) == bp.EXIT_OK


def test_validate_mode_exit_codes(tmp_path: Path) -> None:
    assert bp.main(["--root", str(tmp_path), "--validate"]) == bp.EXIT_CRITICAL_FAILURE
    bp.bootstrap(tmp_path)
    assert bp.main(["--root", str(tmp_path), "--validate"]) == bp.EXIT_OK


def test_validate_detects_deleted_file(scaffold: Path, manifest: bp.Manifest) -> None:
    (scaffold / "config" / "scoring_weights.yaml").unlink()
    problems = bp.validate_scaffold(scaffold, manifest)
    assert [p.path for p in problems] == ["config/scoring_weights.yaml"]
    assert problems[0].critical is True


def test_help_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        bp.main(["--help"])
    assert excinfo.value.code == 0


def test_resolve_root_rules(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit"
    assert bp.resolve_root(str(explicit), Path("/x/scripts/b.py"), tmp_path) == explicit.resolve()
    scripts_dir = tmp_path / "repo" / "scripts"
    scripts_dir.mkdir(parents=True)
    assert bp.resolve_root(None, scripts_dir / "b.py", tmp_path) == (tmp_path / "repo").resolve()
    elsewhere = tmp_path / "downloads"
    elsewhere.mkdir()
    assert bp.resolve_root(None, elsewhere / "b.py", tmp_path) == (tmp_path / "cews").resolve()


def test_summary_mentions_created_skipped_and_failed(
    scaffold: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bp.print_summary(bp.bootstrap(scaffold))
    out = capsys.readouterr().out
    assert "created" in out and "skipped" in out and "failed" in out


def test_tree_lists_expected_entries_and_hides_markers(scaffold: Path) -> None:
    tree = bp.render_tree(scaffold, ascii_only=True)
    assert "pyproject.toml" in tree
    assert "scoring_weights.yaml" in tree
    assert ".gitkeep" not in tree
    assert "+-- " in tree or "`-- " in tree


# --------------------------------------------------------------------------------------
# Content of generated files
# --------------------------------------------------------------------------------------
def test_all_python_files_parse_and_have_docstrings(scaffold: Path) -> None:
    py_files = list(scaffold.rglob("*.py"))
    assert len(py_files) > 100
    for path in py_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert ast.get_docstring(tree), f"missing module docstring: {path}"


def test_placeholder_tests_are_marked_skipped(scaffold: Path) -> None:
    test_files = [p for p in (scaffold / "tests").rglob("test_*.py")]
    assert len(test_files) == sum(len(names) for names in bp.TEST_FILES.values())
    for path in test_files:
        assert "pytest.mark.skip" in path.read_text(encoding="utf-8"), path


def test_phase_mapping() -> None:
    assert bp.phase_for("src/cews/ingestion/adapters/pubmed.py") == 4
    assert bp.phase_for("src/cews/ingestion/base.py") == 3
    assert bp.phase_for("src/cews/scoring/trend_score.py") == 7
    assert bp.phase_for("src/cews/discovery/competitor_discovery.py") == 5
    assert bp.phase_for("src/cews/ai/embeddings.py") == 9
    assert bp.phase_for("tests/unit/test_growth.py") == 6
    assert bp.phase_for("docs/architecture.md") == bp.DEFAULT_PHASE


def _parse_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key] = value
    return values


def test_env_example_defaults(scaffold: Path) -> None:
    env = _parse_env((scaffold / ".env.example").read_text(encoding="utf-8"))
    assert env["FETCH_INTERVAL_MINUTES"] == "120"
    assert env["COMPETITOR_MODE"] == "HYBRID"
    assert env["ENABLE_SCHEDULER"] == "true"
    assert env["TIMEZONE"] == "Asia/Kolkata"
    assert env["DATABASE_BACKEND"] == "sqlite"
    for flag in (
        "ENABLE_AI_TOPIC_DISCOVERY",
        "ENABLE_AI_ANNOUNCEMENT_EXTRACTION",
        "ENABLE_AI_ORG_MATCHING",
    ):
        assert env[flag] == "false"


def test_env_example_has_no_inline_comments_or_secrets(scaffold: Path) -> None:
    env = _parse_env((scaffold / ".env.example").read_text(encoding="utf-8"))
    for key, value in env.items():
        assert "#" not in value, f"inline comment in {key}"
        if (
            key.endswith(("_KEY", "_SECRET"))
            or key.endswith("PASSWORD")
            and key != "POSTGRES_PASSWORD"
        ):
            assert value == "", f"{key} must be blank in the template"


def test_gitignore_protects_env_files(scaffold: Path) -> None:
    lines = (scaffold / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in lines
    assert "!.env.example" in lines


def test_makefile_recipes_use_tabs(scaffold: Path) -> None:
    text = (scaffold / "Makefile").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert not any(line.startswith("    ") for line in lines)
    assert any(line.startswith("\t") for line in lines)
    phony = next(line for line in lines if line.startswith(".PHONY"))
    for target in (
        "setup",
        "db-up",
        "db-init",
        "seed",
        "analyze",
        "evaluate",
        "export",
        "api",
        "dashboard",
        "test",
    ):
        assert target in phony.split()


def test_requirements_are_bounded_and_match_pyproject(scaffold: Path) -> None:
    for name in ("requirements.txt", "requirements-dev.txt", "requirements-ai.txt"):
        for line in (scaffold / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "-r")):
                continue
            assert re.search(r"(==|<)", line), f"{name}: {line} has no upper bound"
    pyproject = tomllib.loads((scaffold / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert list(bp.RUNTIME_DEPS) == pyproject["project"]["dependencies"]
    assert list(bp.DEV_DEPS) == pyproject["project"]["optional-dependencies"]["dev"]


def test_theme_json_is_valid(scaffold: Path) -> None:
    theme = json.loads(
        (scaffold / "dashboards/powerbi/powerbi_theme.json").read_text(encoding="utf-8")
    )
    assert len(theme["dataColors"]) >= 6


def test_yaml_configs_parse_and_weights_sum_to_one(scaffold: Path) -> None:
    yaml = pytest.importorskip("yaml")
    for path in list((scaffold / "config").glob("*.yaml")) + [scaffold / "docker-compose.yml"]:
        assert yaml.safe_load(path.read_text(encoding="utf-8")), f"empty or invalid YAML: {path}"
    scoring = yaml.safe_load((scaffold / "config/scoring_weights.yaml").read_text(encoding="utf-8"))
    for name, weights in scoring["weight_sets"].items():
        assert sum(weights.values()) == pytest.approx(1.0), f"{name} weights must sum to 1"


def test_taxonomy_references_resolve(scaffold: Path) -> None:
    yaml = pytest.importorskip("yaml")
    taxonomy = yaml.safe_load((scaffold / "config/topic_taxonomy.yaml").read_text(encoding="utf-8"))
    area_ids = [a["id"] for a in taxonomy["therapeutic_areas"]]
    assert len(area_ids) == len(set(area_ids))
    for area in taxonomy["therapeutic_areas"]:
        assert area["parent"] is None or area["parent"] in area_ids
    topic_ids = [t["id"] for t in taxonomy["topics"]]
    assert len(topic_ids) == len(set(topic_ids))
    for topic in taxonomy["topics"]:
        assert topic["area"] is None or topic["area"] in area_ids


def test_source_registry_has_no_hardcoded_company_feeds(scaffold: Path) -> None:
    yaml = pytest.importorskip("yaml")
    registry = yaml.safe_load(
        (scaffold / "config/source_registry.yaml").read_text(encoding="utf-8")
    )
    rss = next(s for s in registry["sources"] if s["id"] == "generic_rss")
    assert rss["feeds"] == []
    assert len({s["id"] for s in registry["sources"]}) == len(registry["sources"])
