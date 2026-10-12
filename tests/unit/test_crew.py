"""Unit tests for the CrewAI package's own logic - no real MCP connection or network call.

The real connection (discovering the actual 13 tools, building real Agents against them) is
covered by tests/integration/test_crew.py instead, against a real subprocess, the same way
tests/integration/test_agent_cli.py covers cews agent's own equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cews.crew.llm import build_crew_llm
from cews.crew.mcp import tools_for
from cews.settings import load_settings

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------------------
# build_crew_llm: mapping CEWS settings onto CrewAI's LLM the same way LLMClient already does
# --------------------------------------------------------------------------------------
def test_refuses_when_no_llm_is_configured() -> None:
    settings = load_settings(env_file=None)
    with pytest.raises(ValueError, match="no LLM is configured"):
        build_crew_llm(settings)


def test_maps_base_url_and_model_the_same_way_for_ollama_and_openai_compatible() -> None:
    """LLMClient treats every CEWS provider as "some OpenAI-shaped endpoint" - same base_url,
    same optional Bearer header, no provider-specific request shape. This must not invent a
    second convention."""
    for provider in ("ollama", "openai_compatible"):
        settings = load_settings(
            env_file=None,
            overrides={"llm_provider": provider, "llm_base_url": "http://x:1/v1", "llm_model": "m"},
        )
        llm = build_crew_llm(settings)
        assert llm.base_url == "http://x:1/v1"


def test_the_model_is_routed_through_litellms_openai_shape() -> None:
    settings = load_settings(
        env_file=None,
        overrides={
            "llm_provider": "ollama",
            "llm_base_url": "http://x/v1",
            "llm_model": "qwen2.5:3b",
        },
    )
    llm = build_crew_llm(settings)
    assert "qwen2.5:3b" in llm.model


def test_no_api_key_does_not_crash_the_builder() -> None:
    settings = load_settings(
        env_file=None,
        overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_api_key": None},
    )
    build_crew_llm(settings)  # must not raise


def test_an_api_key_is_passed_through() -> None:
    settings = load_settings(
        env_file=None,
        overrides={
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://x/v1",
            "llm_api_key": "sk-real",
        },
    )
    llm = build_crew_llm(settings)
    assert llm.api_key == "sk-real"


# --------------------------------------------------------------------------------------
# tools_for: splitting one shared tool collection by name, without a real MCP connection
# --------------------------------------------------------------------------------------
@dataclass
class FakeTool:
    name: str


def test_returns_only_the_named_tools() -> None:
    tools = [FakeTool("get_overview"), FakeTool("list_trends"), FakeTool("get_insight")]
    chosen = tools_for(tools, frozenset({"get_overview", "get_insight"}))  # type: ignore[arg-type]
    assert {t.name for t in chosen} == {"get_overview", "get_insight"}


def test_an_empty_name_set_returns_nothing() -> None:
    tools = [FakeTool("get_overview")]
    assert tools_for(tools, frozenset()) == []  # type: ignore[arg-type]


def test_a_name_the_server_does_not_publish_raises_clearly() -> None:
    tools = [FakeTool("get_overview")]
    with pytest.raises(ValueError, match="not_a_real_tool"):
        tools_for(tools, frozenset({"get_overview", "not_a_real_tool"}))  # type: ignore[arg-type]


def test_every_requested_name_missing_is_named_not_just_the_first() -> None:
    tools: list[FakeTool] = []
    with pytest.raises(ValueError) as exc_info:
        tools_for(tools, frozenset({"a", "b"}))  # type: ignore[arg-type]
    assert "'a'" in str(exc_info.value) and "'b'" in str(exc_info.value)
