"""Tests for persona definitions: what each one is allowed to touch, and what it is told.

The module docstring in ``cews.agents.personas`` claims each persona's tool set is checked
against every tool the real MCP server publishes - this file is what makes that claim true,
against a real, in-process server (the same one the app actually runs), not a hand-typed list of
tool names that could quietly drift out of sync with it.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from sqlalchemy.orm import Session, sessionmaker

from cews.agents.personas import ANALYST, BRIEFING_WRITER, FACT_CHECKER, PERSONAS, Persona
from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.mcp_server.server import build_server
from cews.settings import load_settings

pytestmark = pytest.mark.unit


def real_server_tool_names() -> frozenset[str]:
    engine = create_memory_engine()
    upgrade_database(engine)
    factory: sessionmaker[Session] = create_session_factory(engine)
    settings = load_settings(env_file=None)
    server = build_server(settings, factory)

    async def list_names() -> frozenset[str]:
        async with create_connected_server_and_client_session(server) as session:
            tools = (await session.list_tools()).tools
            return frozenset(tool.name for tool in tools)

    names = asyncio.run(list_names())
    engine.dispose()
    return names


# --------------------------------------------------------------------------------------
# Every persona's allowed tools are real, published MCP tools
# --------------------------------------------------------------------------------------
def test_every_personas_tools_actually_exist_on_the_real_server() -> None:
    published = real_server_tool_names()
    for persona in PERSONAS.values():
        unknown = persona.allowed_tools - published
        assert not unknown, f"{persona.name} lists tool(s) the server does not have: {unknown}"


def test_no_persona_is_left_with_an_empty_toolset() -> None:
    for persona in PERSONAS.values():
        assert persona.allowed_tools, f"{persona.name} has no tools at all"


# --------------------------------------------------------------------------------------
# Least privilege: each persona only gets what its own job description says it needs
# --------------------------------------------------------------------------------------
def test_the_fact_checker_can_only_read_one_insight() -> None:
    assert FACT_CHECKER.allowed_tools == frozenset({"get_insight"})


def test_the_briefing_writer_cannot_rank_competitors_or_trends() -> None:
    assert "list_competitors" not in BRIEFING_WRITER.allowed_tools
    assert "list_trends" not in BRIEFING_WRITER.allowed_tools


def test_the_analyst_has_the_widest_toolset_of_the_three() -> None:
    assert FACT_CHECKER.allowed_tools < ANALYST.allowed_tools
    assert BRIEFING_WRITER.allowed_tools < ANALYST.allowed_tools


def test_no_persona_can_change_anything() -> None:
    """Every real tool name starts with get_ or list_ - see the MCP server's own test for the
    stronger guarantee that no write-shaped tool exists on the server at all; this just confirms
    no persona's list contains something that looks like one, as an extra check close to home."""
    for persona in PERSONAS.values():
        for name in persona.allowed_tools:
            assert name.startswith(("get_", "list_")), f"{persona.name}: {name!r} is not read-only"


# --------------------------------------------------------------------------------------
# PERSONAS registry consistency
# --------------------------------------------------------------------------------------
def test_the_registry_key_matches_each_personas_own_name() -> None:
    for key, persona in PERSONAS.items():
        assert key == persona.name


def test_the_three_shipped_personas_are_all_registered() -> None:
    assert PERSONAS == {
        "analyst": ANALYST,
        "fact_checker": FACT_CHECKER,
        "briefing_writer": BRIEFING_WRITER,
    }


# --------------------------------------------------------------------------------------
# Every persona is told to ground its answer in a real tool call, not recall
# --------------------------------------------------------------------------------------
GROUNDING_PHRASE = "you already know the answer"


@pytest.mark.parametrize("persona", list(PERSONAS.values()), ids=lambda p: p.name)
def test_every_persona_is_told_to_call_a_tool_before_answering(persona: Persona) -> None:
    assert GROUNDING_PHRASE in persona.system_prompt


def test_the_analyst_is_told_this_as_the_first_instruction() -> None:
    """Put early in the prompt on purpose: models weight an instruction's position, and this is
    the rule most worth a model not skimming past - confirmed missing in a real exchange with a
    real model before this line was strengthened and moved here."""
    first_paragraph = ANALYST.system_prompt.split("\n\n")[0]
    assert "Call a tool first" in first_paragraph


def test_the_analyst_is_told_not_to_answer_from_memory() -> None:
    assert "do not answer from memory" in ANALYST.system_prompt.lower()


def test_the_grounding_rule_allows_a_real_exception_for_non_data_questions() -> None:
    """The rule must not be so absolute that "what can you do" or a question about an attached
    document forces a pointless tool call - see cews.agents.core's turn limit, which such a loop
    would eventually hit."""
    assert "not about cews's data at all" in ANALYST.system_prompt.lower()


# --------------------------------------------------------------------------------------
# The document-handling rule (a different, earlier fix) is still intact
# --------------------------------------------------------------------------------------
def test_every_persona_is_told_an_attached_document_needs_no_tool_call() -> None:
    for persona in PERSONAS.values():
        assert (
            "no tool call can read them" in persona.system_prompt
            or "no tool call is needed" in (persona.system_prompt)
        )
