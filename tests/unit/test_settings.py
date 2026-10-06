"""Unit tests for cews.settings (configuration loading, validation, competitor modes)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from cews import settings as settings_module
from cews.constants import DEFAULT_ENV_FILE, CompetitorMode, DatabaseBackend, NormalizationMethod
from cews.settings import (
    Settings,
    SettingsError,
    load_competitor_config_file,
    load_settings,
    name_key,
    parse_competitor_list,
    resolve_competitor_mode,
    resolve_env_file,
    validate_settings,
)
from support import REPO_ROOT, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.unit


def _load(**overrides: object) -> Settings:
    return load_settings(env_file=None, overrides=overrides)


# --------------------------------------------------------------------------------------
# Defaults and overrides
# --------------------------------------------------------------------------------------
def test_default_fetch_interval_is_120_minutes() -> None:
    settings = _load()
    assert settings.fetch_interval_minutes == 120
    assert settings.fetch_interval == timedelta(minutes=120)


def test_defaults_match_spec() -> None:
    settings = _load()
    assert settings.competitor_mode is CompetitorMode.HYBRID
    assert settings.database_backend is DatabaseBackend.SQLITE
    assert settings.normalization_method is NormalizationMethod.PERCENTILE
    assert settings.top_competitors == 20
    assert settings.timezone == "Asia/Kolkata"
    assert settings.competitor_include_list == [
        "Pfizer",
        "Roche",
        "Novartis",
        "Merck",
        "AstraZeneca",
    ]
    assert settings.competitor_exclude_list == []
    assert settings.enable_scheduler is True
    assert settings.run_fetch_on_startup is False
    assert settings.enable_ai_topic_discovery is False


def test_environment_variable_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "45")
    monkeypatch.setenv("TOP_COMPETITORS", "8")
    settings = _load()
    assert settings.fetch_interval_minutes == 45
    assert settings.top_competitors == 8


def test_env_file_is_read_and_real_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FETCH_INTERVAL_MINUTES=30\nTOP_COMPETITORS=12\n", encoding="utf-8")
    assert load_settings(env_file=env_file).fetch_interval_minutes == 30
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "60")
    settings = load_settings(env_file=env_file)
    assert settings.fetch_interval_minutes == 60
    assert settings.top_competitors == 12


def test_explicit_overrides_beat_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "60")
    assert _load(fetch_interval_minutes=15).fetch_interval_minutes == 15


def test_env_file_with_byte_order_mark_is_parsed(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_bytes(b"\xef\xbb\xbfFETCH_INTERVAL_MINUTES=15\n")
    assert load_settings(env_file=env_file).fetch_interval_minutes == 15


def test_unknown_keys_in_env_file_are_ignored(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("POSTGRES_USER=cews\nSOMETHING_ELSE=1\n", encoding="utf-8")
    assert load_settings(env_file=env_file).fetch_interval_minutes == 120


def test_explicit_missing_env_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="not found"):
        load_settings(env_file=tmp_path / "missing.env")


def test_env_example_matches_defaults() -> None:
    """The template and the code defaults must never drift apart."""
    from_example = load_settings(env_file=REPO_ROOT / ".env.example")
    defaults = _load()
    assert from_example.model_dump() == defaults.model_dump()


def test_enumerations_are_case_insensitive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COMPETITOR_MODE", "manual")
    monkeypatch.setenv("LOG_LEVEL", "debug")
    monkeypatch.setenv("DATABASE_BACKEND", "PostgreSQL")
    monkeypatch.setenv("NORMALIZATION_METHOD", "Robust_ZScore")
    settings = _load()
    assert settings.competitor_mode is CompetitorMode.MANUAL
    assert settings.log_level == "DEBUG"
    assert settings.database_backend is DatabaseBackend.POSTGRESQL
    assert settings.normalization_method is NormalizationMethod.ROBUST_ZSCORE


def test_blank_optional_values_become_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCBI_API_KEY", "")
    monkeypatch.setenv("COMPETITOR_CONFIG_FILE", "  ")
    settings = _load()
    assert settings.ncbi_api_key is None
    assert settings.competitor_config_file is None


def test_relative_paths_resolve_against_project_root(tmp_path: Path) -> None:
    settings = _load(project_root=tmp_path, sqlite_path="./db/x.db", export_directory="out")
    assert settings.sqlite_path == tmp_path / "db" / "x.db"
    assert settings.export_directory == tmp_path / "out"
    absolute = tmp_path / "elsewhere" / "y.db"
    assert _load(project_root=tmp_path, sqlite_path=absolute).sqlite_path == absolute


def test_enabled_sources_follow_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _load().enabled_sources == ("clinical_trials_gov", "pubmed", "europe_pmc", "generic_rss")
    monkeypatch.setenv("ENABLE_PUBMED", "false")
    monkeypatch.setenv("ENABLE_OPENALEX", "true")
    assert _load().enabled_sources == (
        "clinical_trials_gov",
        "europe_pmc",
        "openalex",
        "generic_rss",
    )


# --------------------------------------------------------------------------------------
# Invalid values
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("FETCH_INTERVAL_MINUTES", "0"),
        ("FETCH_INTERVAL_MINUTES", "abc"),
        ("FETCH_INTERVAL_MINUTES", "99999"),
        ("ENABLE_SCHEDULER", "maybe"),
        ("LOG_LEVEL", "LOUD"),
        ("APP_ENV", "staging"),
        ("COMPETITOR_MODE", "SOMETIMES"),
        ("DATABASE_BACKEND", "mysql"),
        ("TIMEZONE", "Mars/Olympus_Mons"),
        ("TOP_COMPETITORS", "0"),
        ("NORMALIZATION_METHOD", "zscore"),
        ("ALERT_TREND_THRESHOLD", "101"),
        ("SOURCE_MAX_RETRIES", "-1"),
        ("SOURCE_REQUEST_TIMEOUT_SECONDS", "0"),
        ("LLM_BASE_URL", "localhost:11434"),
        ("MIN_TOPIC_SAMPLE_SIZE", "0"),
        ("BACKTEST_HORIZON_MONTHS", "0"),
        ("AI_TOPIC_NOVELTY_THRESHOLD", "1.5"),
    ],
)
def test_invalid_values_raise_clear_errors(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(SettingsError) as excinfo:
        _load()
    assert variable in str(excinfo.value)
    assert any(problem.startswith(variable) for problem in excinfo.value.problems)


def test_all_problems_are_reported_together(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "0")
    monkeypatch.setenv("LOG_LEVEL", "LOUD")
    with pytest.raises(SettingsError) as excinfo:
        _load()
    assert len(excinfo.value.problems) == 2


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"normalization_winsor_lower": 0.9, "normalization_winsor_upper": 0.1}, "WINSOR"),
        ({"incremental_lookback_days": 400, "default_lookback_days": 30}, "LOOKBACK"),
        ({"ai_org_match_review_threshold": 0.95, "ai_org_match_auto_threshold": 0.9}, "REVIEW"),
        ({"confidence_min_score": 90, "confidence_max_score": 10}, "CONFIDENCE"),
        ({"database_backend": "postgresql", "database_url": "mysql://u:p@h/db"}, "DATABASE_URL"),
    ],
)
def test_cross_field_validation(overrides: dict[str, object], fragment: str) -> None:
    with pytest.raises(SettingsError, match=fragment):
        _load(**overrides)


def test_secrets_are_masked() -> None:
    settings = _load(ncbi_api_key="super-secret-key", openalex_api_key="another-secret")
    assert "super-secret-key" not in repr(settings)
    assert "super-secret-key" not in str(settings.safe_dump())
    assert sorted(settings.secret_values()) == ["another-secret", "super-secret-key"]


def test_secret_value_never_appears_in_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCBI_API_KEY", "leaky-secret-value")
    monkeypatch.setenv("FETCH_INTERVAL_MINUTES", "0")
    with pytest.raises(SettingsError) as excinfo:
        _load()
    assert "leaky-secret-value" not in str(excinfo.value)


# --------------------------------------------------------------------------------------
# parse_competitor_list and name_key
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, []),
        ("", []),
        ("   ", []),
        ("Pfizer", ["Pfizer"]),
        ("Pfizer, Roche ,Novartis", ["Pfizer", "Roche", "Novartis"]),
        ("Pfizer,,Roche,", ["Pfizer", "Roche"]),
        ("Pfizer, pfizer, PFIZER", ["Pfizer"]),
        ("Pfizer\nRoche\r\nNovartis", ["Pfizer", "Roche", "Novartis"]),
        ("Merck & Co., Inc.; Pfizer", ["Merck & Co., Inc.", "Pfizer"]),
        ("  Eli   Lilly  ", ["Eli Lilly"]),
        (["Pfizer", " Roche ", "pfizer"], ["Pfizer", "Roche"]),
        ([], []),
    ],
)
def test_parse_competitor_list(raw: str | list[str] | None, expected: list[str]) -> None:
    assert parse_competitor_list(raw) == expected


def test_parse_competitor_list_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        parse_competitor_list(42)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        parse_competitor_list(["ok", 3])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="longer than"):
        parse_competitor_list("x" * 500)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Pfizer", "Pfizer Inc."),
        ("Pfizer", "PFIZER, INC"),
        ("Merck & Co., Inc.", "Merck"),
        ("Novartis AG", "novartis"),
        ("AstraZeneca PLC", "AstraZeneca"),
    ],
)
def test_name_key_ignores_case_punctuation_and_legal_suffixes(left: str, right: str) -> None:
    assert name_key(left) == name_key(right)


def test_name_key_keeps_different_companies_apart() -> None:
    assert name_key("Orvexa Bio") != name_key("Orvexa Labs")
    assert name_key("Co") == "co"  # a name that is only a suffix is preserved


# --------------------------------------------------------------------------------------
# resolve_competitor_mode
# --------------------------------------------------------------------------------------
def test_hybrid_mode_keeps_includes_and_fills_remaining_slots() -> None:
    plan = resolve_competitor_mode(_load())
    assert plan.mode is CompetitorMode.HYBRID
    assert plan.include == ("Pfizer", "Roche", "Novartis", "Merck", "AstraZeneca")
    assert plan.auto_slots == 15
    assert plan.top_n == 20
    assert plan.warnings == ()


def test_manual_include_always_remains_and_exclude_always_disappears() -> None:
    plan = resolve_competitor_mode(
        _load(competitor_include="Pfizer,Roche,Novartis", competitor_exclude="roche inc.")
    )
    assert plan.include == ("Pfizer", "Novartis")
    assert plan.exclude == ("roche inc.",)
    assert any("Roche" in warning for warning in plan.warnings)


def test_hybrid_never_drops_includes_even_beyond_top_n() -> None:
    plan = resolve_competitor_mode(_load(competitor_include="A,B,C", top_competitors=2))
    assert plan.include == ("A", "B", "C")
    assert plan.auto_slots == 0
    assert plan.warnings


def test_hybrid_with_no_includes_uses_all_slots_for_discovery() -> None:
    plan = resolve_competitor_mode(_load(competitor_include="", top_competitors=7))
    assert plan.include == ()
    assert plan.auto_slots == 7


def test_manual_mode_uses_only_includes() -> None:
    plan = resolve_competitor_mode(_load(competitor_mode="MANUAL", competitor_include="A,B"))
    assert plan.include == ("A", "B")
    assert plan.auto_slots == 0


@pytest.mark.parametrize("include", ["", "Pfizer"])
def test_manual_mode_without_usable_includes_is_an_error(include: str) -> None:
    with pytest.raises(SettingsError, match="MANUAL"):
        resolve_competitor_mode(
            _load(competitor_mode="MANUAL", competitor_include=include, competitor_exclude="Pfizer")
        )


def test_auto_mode_ignores_includes_but_honors_excludes() -> None:
    plan = resolve_competitor_mode(
        _load(
            competitor_mode="AUTO",
            competitor_include="A,B",
            competitor_exclude="Z",
            top_competitors=9,
        )
    )
    assert plan.include == ()
    assert plan.exclude == ("Z",)
    assert plan.auto_slots == 9
    assert any("ignored" in warning for warning in plan.warnings)


def test_config_file_is_merged_with_environment_lists(tmp_path: Path) -> None:
    config = tmp_path / "competitors.yaml"
    config.write_text(
        "include:\n  - Sanofi\n  - Pfizer\nexclude: 'Bayer; Roche'\n", encoding="utf-8"
    )
    plan = resolve_competitor_mode(
        _load(competitor_include="Pfizer,Roche", competitor_config_file=config)
    )
    assert plan.include == ("Pfizer", "Sanofi")  # Roche excluded by the file, Pfizer not duplicated
    assert set(plan.exclude) == {"Bayer", "Roche"}


def test_config_file_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(SettingsError, match="not found"):
        load_competitor_config_file(tmp_path / "nope.yaml")
    bad_keys = tmp_path / "bad.yaml"
    bad_keys.write_text("include: [A]\nmonitor: [B]\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="unknown keys"):
        load_competitor_config_file(bad_keys)
    not_mapping = tmp_path / "list.yaml"
    not_mapping.write_text("- A\n- B\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="mapping"):
        load_competitor_config_file(not_mapping)
    broken = tmp_path / "broken.yaml"
    broken.write_text("include: [unclosed\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="cannot read"):
        load_competitor_config_file(broken)
    wrong_type = tmp_path / "types.yaml"
    wrong_type.write_text("include: [1, 2]\n", encoding="utf-8")
    with pytest.raises(SettingsError, match="invalid list"):
        load_competitor_config_file(wrong_type)


def test_empty_config_file_is_allowed(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_competitor_config_file(empty) == ([], [])


# --------------------------------------------------------------------------------------
# validate_settings
# --------------------------------------------------------------------------------------
def test_validate_settings_accepts_the_shipped_configuration(settings: Settings) -> None:
    assert validate_settings(settings) == []


def test_validate_settings_reports_missing_files(tmp_path: Path) -> None:
    settings = _load(
        topic_taxonomy_file=tmp_path / "no_taxonomy.yaml", scoring_config_file=SCORING_FILE
    )
    with pytest.raises(SettingsError, match="TOPIC_TAXONOMY_FILE"):
        validate_settings(settings)


def test_validate_settings_warns_about_risky_configuration(settings: Settings) -> None:
    risky = settings.model_copy(
        update={
            "fetch_interval_minutes": 5,
            "enable_openalex": True,
            "enable_patents": True,
        }
    )
    warnings = " | ".join(validate_settings(risky))
    assert "FETCH_INTERVAL_MINUTES=5" in warnings
    assert "OPENALEX_API_KEY" in warnings
    assert "USPTO_ODP_API_KEY" in warnings


def test_validate_settings_warns_when_no_source_is_enabled(settings: Settings) -> None:
    quiet = settings.model_copy(
        update={
            "enable_clinical_trials_gov": False,
            "enable_pubmed": False,
            "enable_europe_pmc": False,
            "enable_generic_rss": False,
        }
    )
    assert any("No data sources" in warning for warning in validate_settings(quiet))


def test_validate_settings_warns_when_embedding_library_is_missing(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings_module.importlib.util, "find_spec", lambda name: None)
    enabled = settings.model_copy(update={"enable_ai_topic_discovery": True})
    assert any("sentence-transformers" in warning for warning in validate_settings(enabled))


def test_llm_provider_defaults_to_none(settings: Settings) -> None:
    assert settings.llm_provider.value == "none"
    assert settings.llm_api_key is None


def test_llm_provider_can_be_switched_to_a_local_or_hosted_endpoint() -> None:
    # model_copy(update=...) writes the raw value without running validators/coercion, so
    # switching providers is exercised through load_settings, exactly as .env does it.
    local = load_settings(
        env_file=None,
        overrides={"llm_provider": "ollama", "llm_base_url": "http://localhost:11434/v1"},
    )
    assert local.llm_provider.value == "ollama"
    hosted = load_settings(
        env_file=None,
        overrides={
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://integrate.api.nvidia.com/v1",
            "llm_model": "meta/llama-3.1-8b-instruct",
            "llm_api_key": "sk-abc",
        },
    )
    assert hosted.llm_provider.value == "openai_compatible"
    assert hosted.llm_api_key is not None
    assert hosted.llm_api_key.get_secret_value() == "sk-abc"


def test_llm_api_key_is_not_shown_in_repr() -> None:
    """A SecretStr must never leak into logs, tracebacks or repr output."""
    hosted = load_settings(
        env_file=None,
        overrides={"llm_provider": "openai_compatible", "llm_api_key": "sk-do-not-leak"},
    )
    assert hosted.llm_api_key is not None
    assert "sk-do-not-leak" not in repr(hosted.llm_api_key)
    assert "sk-do-not-leak" not in str(hosted.llm_api_key)


@pytest.mark.parametrize("bad_url", ["not-a-url", "ftp://x/v1", "localhost:11434"])
def test_llm_base_url_must_be_http(bad_url: str) -> None:
    with pytest.raises(SettingsError, match="LLM_BASE_URL"):
        load_settings(env_file=None, overrides={"llm_base_url": bad_url})


def test_an_empty_llm_api_key_in_env_is_none_not_an_empty_secret() -> None:
    """LLM_API_KEY= with nothing after it must mean "unset", like every other secret field."""
    settings = load_settings(env_file=REPO_ROOT / ".env.example")
    assert settings.llm_api_key is None


def test_openai_compatible_without_a_key_is_flagged() -> None:
    unkeyed = load_settings(
        env_file=None,
        overrides={"enable_ai_announcement_extraction": True, "llm_provider": "openai_compatible"},
    )
    warnings = " | ".join(validate_settings(unkeyed))
    assert "LLM_API_KEY" in warnings and "NVIDIA NIM" in warnings


def test_openai_compatible_with_a_key_is_not_flagged() -> None:
    keyed = load_settings(
        env_file=None,
        overrides={
            "enable_ai_announcement_extraction": True,
            "llm_provider": "openai_compatible",
            "llm_api_key": "sk-x",
        },
    )
    assert not any("LLM_API_KEY" in warning for warning in validate_settings(keyed))


def test_a_disabled_llm_provider_is_never_flagged() -> None:
    disabled = load_settings(
        env_file=None, overrides={"enable_ai_announcement_extraction": True, "llm_provider": "none"}
    )
    assert not any("LLM_API_KEY" in warning for warning in validate_settings(disabled))


def test_ollama_never_needs_an_api_key_warning() -> None:
    local = load_settings(
        env_file=None,
        overrides={"enable_ai_announcement_extraction": True, "llm_provider": "ollama"},
    )
    assert not any("LLM_API_KEY" in warning for warning in validate_settings(local))


def test_shipped_config_files_exist() -> None:
    assert TAXONOMY_FILE.is_file() and SCORING_FILE.is_file()


# --------------------------------------------------------------------------------------
# resolve_env_file: one shared policy for "nothing was specified", used by the CLI and the
# dashboard alike, so the two can never again silently disagree about what an empty value means
# --------------------------------------------------------------------------------------
def test_nothing_specified_resolves_to_the_default_env_file() -> None:
    assert resolve_env_file(None) == DEFAULT_ENV_FILE
    assert resolve_env_file("") == DEFAULT_ENV_FILE


def test_a_real_path_is_returned_unchanged() -> None:
    assert resolve_env_file("/tmp/custom.env") == "/tmp/custom.env"
    custom = Path("/tmp/custom.env")
    assert resolve_env_file(custom) == custom


def test_resolving_nothing_behaves_exactly_like_omitting_the_argument(tmp_path: Path) -> None:
    """The regression this guards: passing env_file=None explicitly to load_settings means
    "skip dotenv files entirely" - a different, stricter thing than not passing it at all, which
    tries the project's own .env. A caller that means "nothing was specified" must resolve to
    the latter, not accidentally send the former."""
    omitted = load_settings()
    resolved = load_settings(env_file=resolve_env_file(None))
    assert resolved.model_dump() == omitted.model_dump()


def test_an_empty_box_still_picks_up_a_real_env_file(tmp_path: Path) -> None:
    """A closer simulation of the dashboard's own bug: with DEFAULT_ENV_FILE pointed at a real
    file (as it would be in production, pointed at the project root), an empty string - exactly
    what a never-touched text box holds - must load it, not silently skip it."""
    real_env = tmp_path / ".env"
    real_env.write_text("LLM_PROVIDER=ollama\nLLM_MODEL=test-model\n", encoding="utf-8")
    original = settings_module.DEFAULT_ENV_FILE
    settings_module.DEFAULT_ENV_FILE = real_env
    try:
        settings = load_settings(env_file=resolve_env_file(""))
    finally:
        settings_module.DEFAULT_ENV_FILE = original
    assert settings.llm_provider.value == "ollama" and settings.llm_model == "test-model"
