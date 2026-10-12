"""The three tasks of the Weekly Briefing Crew, chained sequentially.

Each task's ``context`` names the task(s) before it, which is what makes this genuinely
*collaborative* rather than three unrelated single-agent calls: CrewAI feeds the researcher's
actual output into the fact-checker's prompt, and both of theirs into the writer's, automatically.
"""

from __future__ import annotations

from crewai import Agent, Task


def build_tasks(researcher: Agent, fact_checker: Agent, writer: Agent) -> tuple[Task, Task, Task]:
    research = Task(
        description=(
            "Call a tool to check whether CEWS's data is synthetic or real, then find this "
            "week's newest, highest-confidence insights. List each one with its id, its score "
            "and confidence, and whether it is synthetic data."
        ),
        expected_output=(
            "A short list of insights, each with its id, headline claim, score, confidence, "
            "and a note on whether the underlying data is synthetic."
        ),
        agent=researcher,
    )
    verification = Task(
        description=(
            "For each insight id the researcher reported, fetch it and check its claims "
            "against its own cited evidence. Say plainly which claims are well supported and "
            "which are not."
        ),
        expected_output=(
            "For each insight id: supported / partially supported / not supported, with a one "
            "line reason citing the evidence actually found."
        ),
        agent=fact_checker,
        context=[research],
    )
    writing = Task(
        description=(
            "Using only what the researcher found and the fact-checker verified, write a short "
            "briefing for a leadership audience. Lead with whether the data is synthetic if it "
            "is. Drop anything the fact-checker could not support."
        ),
        expected_output=(
            "A short, plain-language briefing: a one-line summary, then each supported finding "
            "with its score, confidence, and insight id for reference."
        ),
        agent=writer,
        context=[research, verification],
    )
    return research, verification, writing
