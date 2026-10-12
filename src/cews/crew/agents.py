"""The three agents in the Weekly Briefing Crew.

Deliberately not a re-derivation of what each role should and should not touch: the tool sets
and the grounding rules both come straight from :mod:`cews.agents.personas` - the same ones the
single-agent Analyst/Fact-checker/Briefing-writer already use. One set of decisions about what a
"researcher" or "fact-checker" role is allowed to see, shared by both agent systems, rather than
two copies that could quietly drift apart.
"""

from __future__ import annotations

from crewai import LLM, Agent
from crewai.tools import BaseTool
from crewai_tools.adapters.tool_collection import ToolCollection

from cews.agents.personas import ANALYST, BRIEFING_WRITER, COMMON_RULES, FACT_CHECKER
from cews.crew.mcp import tools_for


def build_agents(llm: LLM, tools: ToolCollection[BaseTool]) -> tuple[Agent, Agent, Agent]:
    """The Researcher, Fact-checker and Writer, wired to their share of ``tools``."""
    researcher = Agent(
        role="CEWS Research Analyst",
        goal="Find the newest, best-supported findings CEWS has stored, with real tool calls",
        backstory=(
            "You read CEWS's own stored data for a living, never your own general knowledge "
            "about real companies.\n\n" + COMMON_RULES
        ),
        tools=tools_for(
            tools,
            ANALYST.allowed_tools & frozenset({"get_overview", "list_insights", "list_trends"}),
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    fact_checker = Agent(
        role="CEWS Fact-Checker",
        goal="Check every claim the researcher reports against that insight's own evidence",
        backstory=(
            "You check the claims in front of you against their cited evidence - nothing else. "
            "You do not add outside knowledge, and you say plainly when evidence is thin.\n\n"
            + COMMON_RULES
        ),
        tools=tools_for(tools, FACT_CHECKER.allowed_tools),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    writer = Agent(
        role="CEWS Briefing Writer",
        goal="Turn verified findings into a short briefing a leadership audience can act on",
        backstory=(
            "You write only from what the researcher and fact-checker already verified - never "
            "from your own knowledge, and never a claim neither of them actually reported.\n\n"
            + COMMON_RULES
        ),
        tools=tools_for(tools, BRIEFING_WRITER.allowed_tools & frozenset({"get_overview"})),
        llm=llm,
        allow_delegation=False,
        verbose=True,
    )
    return researcher, fact_checker, writer
