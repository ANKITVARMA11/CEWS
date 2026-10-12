"""Assembling and running the Weekly Briefing Crew end to end.

The one public entry point this package exists for: open one MCP connection, build the three
agents and their chained tasks, run them sequentially, close the connection, return the result.
"""

from __future__ import annotations

from crewai import Crew, Process

from cews.crew.agents import build_agents
from cews.crew.llm import build_crew_llm
from cews.crew.mcp import server_params
from cews.crew.tasks import build_tasks
from cews.settings import Settings


def run_briefing_crew(settings: Settings, env_file: str | None, *, verbose: bool = True) -> str:
    """Run the Researcher -> Fact-checker -> Writer crew once and return the final briefing.

    Raises:
        ValueError: if no LLM is configured (from :func:`cews.crew.llm.build_crew_llm`).
        ImportError: if ``crewai``/``crewai-tools[mcp]`` are not installed - see
            ``requirements-crew.txt``.
    """
    from crewai_tools import MCPServerAdapter  # imported here: optional dependency, see __init__

    llm = build_crew_llm(settings)
    with MCPServerAdapter(server_params(env_file), connect_timeout=30) as tools:
        researcher, fact_checker, writer = build_agents(llm, tools)
        research, verification, writing = build_tasks(researcher, fact_checker, writer)
        crew = Crew(
            agents=[researcher, fact_checker, writer],
            tasks=[research, verification, writing],
            process=Process.sequential,
            verbose=verbose,
        )
        result = crew.kickoff()
    return str(result)
