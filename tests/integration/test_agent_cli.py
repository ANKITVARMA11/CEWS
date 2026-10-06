"""Integration tests for `cews agent`.

This project's own test policy (`tests/conftest.py`) blocks every real network call in every
test, including to a local loopback server, so what these tests can cover is bounded: every path
that fails *before* an LLM connection is attempted (bad arguments, no LLM configured, database
not initialized) is covered here, against a real MCP subprocess. The "reaches a real LLM
endpoint and gets a real answer" path cannot be exercised by an automated test under that policy
without a network call, so it is not; see the note further down for how it was actually checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cews.cli import EXIT_FAILURE, EXIT_OK, main
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("agent_cli")
    env = root / ".env"
    env.write_text(
        f"PROJECT_ROOT={root}\nSQLITE_PATH=./data/cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "ENABLE_CLINICAL_TRIALS_GOV=false\nENABLE_PUBMED=false\nENABLE_EUROPE_PMC=false\n"
        # An LLM provider is declared (never actually contacted by the tests below - argument
        # validation happens before any connection is attempted) so those tests get past the
        # "is an LLM configured at all" check and reach the check they are actually about.
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    return env


def test_agent_ask_needs_a_question(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agent", "ask", "--env-file", str(project)]) == EXIT_FAILURE
    assert "usage: cews agent ask" in capsys.readouterr().out


def test_agent_verify_needs_an_id(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["agent", "verify", "--env-file", str(project)]) == EXIT_FAILURE
    assert "usage: cews agent verify" in capsys.readouterr().out


def test_agent_refuses_without_an_llm_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No LLM configured means no connection of any kind - MCP included - is even attempted."""
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n",
        encoding="utf-8",
    )
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    assert main(["agent", "ask", "anything", "--env-file", str(env)]) == EXIT_FAILURE
    assert "no LLM is configured" in capsys.readouterr().out


def test_agent_needs_an_initialized_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = tmp_path / ".env"
    env.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\nTOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\n"
        f"SCORING_CONFIG_FILE={SCORING_FILE}\nSOURCE_REGISTRY_FILE={REGISTRY_FILE}\n"
        "LLM_PROVIDER=ollama\nLLM_BASE_URL=http://127.0.0.1:1/v1\n",
        encoding="utf-8",
    )
    assert main(["agent", "ask", "anything", "--env-file", str(env)]) == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# What is NOT covered here, and how it was actually checked
# --------------------------------------------------------------------------------------
# A real MCP subprocess launch, a real HTTP call to an LLM endpoint, a real tool call and a real
# final answer, all through this exact `cews agent ask` entry point, were checked by hand rather
# than by an automated test in this file:
#
#   1. `cews agent ask "..."` against a deliberately unreachable LLM_BASE_URL: the MCP subprocess
#      launched correctly, and the failure was a clean "error: could not reach ollama:
#      Connection refused" - not the "ModuleNotFoundError: No module named 'cews'" it produced
#      before `_default_server_params` was fixed to pass `env=dict(os.environ)` to the child
#      process, and not the "fileno" crash it produced before `_stdio_session` was fixed to pass
#      an explicit, real `errlog`.
#   2. The same command against a small real HTTP server (`http.server`, scripted to return a
#      tool call and then a final answer) printed the expected answer and the tool it checked.
#
# Both fixes are covered here by the failure-path tests above only insofar as those tests confirm
# the code that would otherwise be exercised by a live LLM call is reached and its errors are
# handled cleanly; the agent's own tool-calling logic (the part that (1) and (2) additionally
# prove) has thorough automated coverage elsewhere with a mocked transport, dependency-injected
# in-process: see tests/unit/test_agent_core.py, test_agent_session.py, and
# tests/dashboard/test_chat_widget.py.
