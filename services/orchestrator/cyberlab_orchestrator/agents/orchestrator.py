"""Agent 1: Chief Research Orchestrator. Turns the research brief into a plan and search queries."""

from __future__ import annotations

from typing import Any

from cyberlab_common.db import Project
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ResearchSpec

SYSTEM = (
    "You are the chief research orchestrator of an AI-for-cybersecurity research lab. "
    "Plan a rigorous, reproducible study. You cannot run code or access the internet; other "
    "agents do that. Never invent results or citations. Output JSON with keys: "
    "'objective' (string), 'search_queries' (3-5 short scholarly search strings), "
    "'tasks' (list of {agent, task}), 'risks' (list of strings)."
)


def orchestrator_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        assert project is not None
        spec = ResearchSpec.model_validate(project.spec)
    plan = ctx.llm.complete_json(
        "orchestrator.plan",
        SYSTEM,
        f"Research title: {spec.title}\nGoal: {spec.goal}\nDataset: {spec.dataset_hint}\n"
        f"Constraints: {spec.constraints}",
        project_id=pid,
    )
    queries = [q for q in plan.get("search_queries", []) if isinstance(q, str)][:5]
    if not queries:
        queries = ["CIC-IDS2017 intrusion detection machine learning", "explainable intrusion detection"]
    plan["search_queries"] = queries
    with ctx.db.session() as s:
        ctx.note(s, pid, "orchestrator", "plan", plan)
    ctx.set_stage(pid, "planned", "Proposed")
    return {"plan": plan, "search_queries": queries}
