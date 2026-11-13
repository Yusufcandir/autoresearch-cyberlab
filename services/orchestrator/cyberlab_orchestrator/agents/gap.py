"""Agent 3: Research Gap and Novelty. Output is always labelled Proposed, never a novelty claim."""

from __future__ import annotations

from typing import Any

from cyberlab_orchestrator.context import LabContext

SYSTEM = (
    "You are a research gap analyst for intrusion detection research. From the literature "
    "summary, propose gaps that a study on CIC-IDS2017 could address with classical and "
    "gradient-boosted models, plus honest caveats (CIC-IDS2017 is widely studied, so novelty "
    "is limited). Output JSON: {'gaps': [{'gap', 'why_it_matters', 'testable_with'}], "
    "'novelty_caveats': [string]}. These are proposals, not established facts."
)


def gap_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    result = ctx.llm.complete_json(
        "gap.analysis",
        SYSTEM,
        f"Literature summary:\n{state.get('literature_summary', '')}\n\nPlan: {state.get('plan', {})}",
        project_id=pid,
    )
    with ctx.db.session() as s:
        ctx.note(s, pid, "research_gap", "gaps", {**result, "label": "Proposed"})
    ctx.set_stage(pid, "gaps_proposed", "Proposed")
    return {"gaps": result.get("gaps", [])}
