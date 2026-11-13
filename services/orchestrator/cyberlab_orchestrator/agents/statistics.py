"""Agent 8: Statistical Analysis. Numbers are deterministic; the LLM only interprets them."""

from __future__ import annotations

import json
from typing import Any

from cyberlab_manuscript.consistency import accuracy_gap_models
from cyberlab_orchestrator.context import LabContext
from cyberlab_qa.analysis import analyze_project

SYSTEM = (
    "You are a statistical analysis agent. Interpret the analysis JSON cautiously: effect sizes, "
    "uncertainty, multiple-seed variability, class imbalance and leakage flags. Do not restate "
    "numbers you were not given and do not overclaim. Every caveat must name the model(s) it "
    "applies to and must follow from the numbers given; a 'high accuracy hides weak minority-class "
    "performance' caveat applies only to models listed in "
    "'models_where_accuracy_exceeds_balanced_accuracy'. Output JSON: {'interpretation': str, "
    "'caveats': [str], 'recommended_followups': [str]}."
)


def statistics_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    ctx.authorize("statistics", "compute_statistics", "T1", pid)
    with ctx.db.session() as s:
        analysis = analyze_project(s, ctx.store, pid, state["protocol_version"])
    compact = {
        "models": {n: m["aggregate"] for n, m in analysis["models"].items()},
        "hypotheses": analysis["hypotheses"],
        "qa_flags": analysis["qa_flags"],
        # computed, not guessed: where accuracy is much higher than balanced accuracy
        "models_where_accuracy_exceeds_balanced_accuracy": accuracy_gap_models(analysis),
    }
    interp = ctx.llm.complete_json(
        "statistics.interpret", SYSTEM, json.dumps(compact, default=str)[:12000], project_id=pid
    )
    with ctx.db.session() as s:
        ctx.note(s, pid, "statistics", "interpretation", {**interp, "label": "Proposed"})
    label = "Verified" if analysis["metrics_recomputed_match"] else "Measured"
    ctx.set_stage(pid, "analyzed", label)
    return {"analysis_sha256": analysis["artifact_sha256"], "interpretation": interp}
