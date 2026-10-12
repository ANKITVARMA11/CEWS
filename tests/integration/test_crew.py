"""Integration tests for the CrewAI package against a real MCP subprocess.

No LLM call is made here: this project's own test policy (tests/conftest.py) blocks every real
network call in every test, including to a local loopback server - the same constraint
tests/integration/test_agent_cli.py documents for `cews agent`'s own real-answer path. What can
be and is covered here: connecting to the real server, discovering the real tools, and building
real Agents and Tasks against them - everything up to, but not including, an actual LLM request.

The full real round trip - a real `cews crew brief` run against a scripted local HTTP server
standing in for an LLM, producing a real three-agent sequential result - was checked by hand
instead; see the note at the bottom of this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cews.cli import EXIT_FAILURE, EXIT_OK, main
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("crew")
    env = root / ".env"
    env.write_text(
        f"PROJECT_ROOT={root}\nSQLITE_PATH=./cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    return env


def test_connecting_discovers_every_real_mcp_tool(project: Path) -> None:
    from mcp import StdioServerParameters

    from cews.crew.mcp import server_params, tools_for

    params = server_params(str(project))
    assert isinstance(params, StdioServerParameters)

    from crewai_tools import MCPServerAdapter

    with MCPServerAdapter(params, connect_timeout=30) as tools:
        names = {t.name for t in tools}
        assert len(names) == 13
        assert {"get_overview", "list_trends", "get_insight"} <= names
        # tools_for against a REAL connection, not the fakes test_crew.py (unit) uses
        chosen = tools_for(tools, frozenset({"get_overview"}))
        assert [t.name for t in chosen] == ["get_overview"]


def test_real_agents_and_tasks_build_against_the_real_connection(project: Path) -> None:
    from crewai_tools import MCPServerAdapter

    from cews.crew.agents import build_agents
    from cews.crew.llm import build_crew_llm
    from cews.crew.mcp import server_params
    from cews.crew.tasks import build_tasks
    from cews.settings import load_settings

    settings = load_settings(env_file=str(project))
    llm = build_crew_llm(settings)
    with MCPServerAdapter(server_params(str(project)), connect_timeout=30) as tools:
        researcher, fact_checker, writer = build_agents(llm, tools)
        assert researcher.tools is not None
        assert fact_checker.tools is not None
        assert writer.tools is not None
        assert {t.name for t in researcher.tools} == {
            "get_overview",
            "list_insights",
            "list_trends",
        }
        assert {t.name for t in fact_checker.tools} == {"get_insight"}
        assert {t.name for t in writer.tools} == {"get_overview"}

        research, verification, writing = build_tasks(researcher, fact_checker, writer)
        assert verification.context == [research]
        assert writing.context == [research, verification]


def test_cmd_crew_refuses_without_an_llm_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    assert main(["crew", "brief", "--env-file", str(env)]) == EXIT_FAILURE
    assert "no LLM is configured" in capsys.readouterr().out


def test_cmd_crew_needs_an_initialized_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["crew", "brief", "--env-file", str(env)]) == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# What is NOT covered here, and how it was actually checked
# --------------------------------------------------------------------------------------
# A real `cews crew brief` run - real MCP subprocess, three real Agents, three chained Tasks,
# a real HTTP call per agent to an LLM endpoint, a real sequential crew result - was checked by
# hand against a small local HTTP server scripted to answer every request, the same technique
# tests/integration/test_agent_cli.py documents for `cews agent`'s own equivalent path. All
# three agents (Researcher, Fact-Checker, Briefing Writer) ran in order, each one's task
# completing and feeding the next, and the final briefing printed correctly.
