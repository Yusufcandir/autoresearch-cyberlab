"""Agent 3: Research Gap and Novelty. Output is always labelled Proposed, never a novelty claim."""

from __future__ import annotations

from typing import Any

from cyberlab_llm.gateway import as_list
from cyberlab_orchestrator.context import LabContext
from cyberlab_orchestrator.datasets import project_adapter

SYSTEM = (
    "You are a research gap analyst for intrusion detection research. From the literature "
    "summary, propose gaps that a study on {dataset} could address with classical and "
    "gradient-boosted models, plus honest caveats ({dataset} is widely studied, so novelty "
    "is limited). Output JSON: {'gaps': [{'gap', 'why_it_matters', 'testable_with'}], "
    "'novelty_caveats': [string]}. These are proposals, not established facts."
)


def gap_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    result = ctx.llm.complete_json(
        "gap.analysis",
        SYSTEM.replace("{dataset}", project_adapter(ctx, pid).display),
        f"Literature summary:\n{state.get('literature_summary', '')}\n\nPlan: {state.get('plan', {})}",
        project_id=pid,
    )
    for key in ("gaps", "novelty_caveats"):
        result[key] = as_list(result.get(key))
    with ctx.db.session() as s:
        ctx.note(s, pid, "research_gap", "gaps", {**result, "label": "Proposed"})
    ctx.set_stage(pid, "gaps_proposed", "Proposed")
    return {"gaps": result.get("gaps", [])}
