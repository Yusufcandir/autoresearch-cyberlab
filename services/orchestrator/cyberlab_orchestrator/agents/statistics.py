"""Agent 8: Statistical Analysis. Numbers are deterministic; the LLM only interprets them."""

from __future__ import annotations

import json
import logging
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
    report = run_validation(ctx, pid, state["protocol_version"], analysis)
    _run_feedback(ctx, pid, state["protocol_version"], analysis, report)
    if not report["manuscript_allowed"]:
        blocked = [
            f"{c['check']}: {c['detail']}" for c in report["checks"] if c["status"] == "BLOCK"
        ]
        ctx.set_stage(
            pid,
            "validation_blocked",
            "Blocked",
            "scientific validation blocked the manuscript: " + " | ".join(blocked)[:900],
        )
        return {
            "analysis_sha256": analysis["artifact_sha256"],
            "interpretation": interp,
            "validation_blocked": True,
        }
    label = "Verified" if analysis["metrics_recomputed_match"] else "Measured"
    ctx.set_stage(pid, "analyzed", label)
    return {
        "analysis_sha256": analysis["artifact_sha256"],
        "interpretation": interp,
        "validation_blocked": False,
    }


def _run_feedback(
    ctx: LabContext,
    pid: str,
    version: int,
    analysis: dict[str, Any],
    report: dict[str, Any],
) -> None:
    """Rules A-E on the validated evidence; decisions go to the research decision log."""
    from sqlalchemy import select

    from cyberlab_common.db import Project, ProtocolVersion
    from cyberlab_data import get_adapter
    from cyberlab_orchestrator.agents.feedback_controller import decide, record_decisions

    with ctx.db.session() as s:
        pv = s.scalar(
            select(ProtocolVersion).where(
                ProtocolVersion.project_id == pid, ProtocolVersion.version == version
            )
        )
        body = dict(pv.body) if pv else {}
        p = s.get(Project, pid)
        minutes = float(p.gpu_minutes_used) if p else 0.0
    groups = list(report.get("groups") or [])
    decisions = decide(report, analysis, body, get_adapter(body.get("dataset_id")), minutes, groups)
    if decisions:
        record_decisions(ctx, pid, decisions)


def run_validation(
    ctx: LabContext, pid: str, protocol_version: int, analysis: dict[str, Any]
) -> dict[str, Any]:
    """Scientific validation before writing (professor's plan, Phase 1): stored as an evidence
    artifact and an agent note; a BLOCK stops the workflow before the manuscript."""
    import tempfile
    from pathlib import Path

    from sqlalchemy import select

    from cyberlab_common.db import ProtocolVersion, RunRecord
    from cyberlab_data import get_adapter, prepared_dir_for
    from cyberlab_qa.scientific_validator import generate_validation_report
    from cyberlab_schemas import SplitSpec

    with ctx.db.session() as s:
        pv = s.scalar(
            select(ProtocolVersion).where(
                ProtocolVersion.project_id == pid, ProtocolVersion.version == protocol_version
            )
        )
        body = dict(pv.body) if pv else {}
        runs = [
            {"model": r.model_name, "seed": r.seed, "status": r.status}
            for r in s.scalars(select(RunRecord).where(RunRecord.project_id == pid))
            if r.experiment_id.startswith(f"p{protocol_version}-")
        ]
    adapter = get_adapter(body.get("dataset_id"))
    split = SplitSpec.model_validate(body.get("split", {}))
    prepared = prepared_dir_for(adapter.root(ctx.settings.dataset_root), split)
    if prepared.exists() and not (prepared / "data_preprocessing_audit.json").exists():
        # prepared by an older version without the stage-by-stage audit: prepare this exact
        # split again (deterministic, same rows) so the audit can be evaluated
        from cyberlab_data import prepare_split

        try:
            prepare_split(
                adapter,
                ctx.settings.dataset_root,
                split,
                benign_cap=ctx.extra.get("benign_cap", ctx.settings.benign_row_cap),
            )
        except Exception as exc:  # noqa: BLE001 - the report then says NOT_EVALUATED
            logging.getLogger(__name__).warning("re-preparing for the audit failed: %s", exc)
    report = generate_validation_report(analysis, body, runs, prepared)
    with tempfile.TemporaryDirectory() as tmp, ctx.db.session() as s:
        path = Path(tmp) / "scientific_validation_report.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        rec = ctx.store.put(s, pid, path, name="scientific_validation_report.json")
        summary = {
            "status": report["status"],
            "counts": report["counts"],
            "artifact_sha256": rec.sha256,
            "checks": [
                {"check": c["check"], "status": c["status"], "detail": c["detail"]}
                for c in report["checks"]
            ],
        }
        ctx.note(s, pid, "statistics", "scientific_validation", summary)
    analysis["validation"] = summary
    return report
