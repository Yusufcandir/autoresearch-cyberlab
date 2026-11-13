"""Durable research workflow (spec Step 10), built with LangGraph.

Planning (LLM) is separated from side effects (deterministic services). The graph stops for
humans at gate A (protocol and data rights) and gate D (manuscript release), and parks at
`wait_jobs` while the broker runs experiments. State is checkpointed per project
(thread_id = project id), so a restart resumes exactly where it stopped.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select

from cyberlab_common.db import ApprovalRecord, ManuscriptRecord, ProtocolVersion, RunRecord
from cyberlab_orchestrator.agents import (
    coding_agent,
    dataset_agent,
    experiment_agent,
    gap_agent,
    literature_agent,
    methodology_agent,
    orchestrator_agent,
    reviewer_agent,
    statistics_agent,
    writing_agent,
)
from cyberlab_orchestrator.agents.experiment import all_jobs_done, jobs_status
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ApprovalGate

MAX_REVISIONS = 1


class LabState(TypedDict, total=False):
    project_id: str
    plan: dict[str, Any]
    search_queries: list[str]
    verified_citation_keys: list[str]
    literature_summary: str
    gaps: list[dict[str, Any]]
    protocol_version: int
    protocol_sha256: str
    protocol_feedback: str
    dataset_status: str
    dataset_summary: dict[str, Any]
    blocked_reason: str
    compile_error: list[str]
    experiments: list[dict[str, Any]]
    job_ids: list[str]
    jobs_pending: bool
    analysis_sha256: str
    interpretation: dict[str, Any]
    manuscript_version: int
    evidence_blockers: list[dict[str, str]]
    review: dict[str, Any]
    review_feedback: list[str]
    revisions: int
    gate_a: str
    released: bool


def build_graph(ctx: LabContext, checkpointer: Any) -> Any:
    def node(fn: Any) -> Any:
        def run(state: LabState) -> dict[str, Any]:
            return dict(fn(ctx, dict(state)))

        run.__name__ = fn.__name__
        return run

    def gate_a(state: LabState) -> dict[str, Any]:
        pid, version = state["project_id"], state["protocol_version"]
        ctx.set_stage(pid, "awaiting_protocol_approval", "Proposed")
        answer = interrupt(
            {
                "gate": "A",
                "title": "Approve protocol and data rights",
                "protocol_version": version,
                "protocol_sha256": state["protocol_sha256"],
                "dataset": state.get("dataset_summary", {}),
            }
        )
        with ctx.db.session() as s:
            pv = s.scalar(
                select(ProtocolVersion).where(
                    ProtocolVersion.project_id == pid, ProtocolVersion.version == version
                )
            )
            rec = s.scalar(
                select(ApprovalRecord)
                .where(
                    ApprovalRecord.project_id == pid,
                    ApprovalRecord.gate == ApprovalGate.A_PROTOCOL.value,
                )
                .order_by(ApprovalRecord.created_at.desc())
            )
            valid = (
                pv is not None
                and rec is not None
                and rec.subject_version == version
                and rec.subject_sha256 == pv.sha256
            )
            decision = rec.decision if valid and rec is not None else "invalid"
            comment = rec.comment if rec is not None else ""
        if decision == "approved":
            ctx.set_stage(pid, "protocol_approved", "Approved")
            return {"gate_a": "approved"}
        return {"gate_a": decision, "protocol_feedback": comment or str(answer)}

    def wait_jobs(state: LabState) -> dict[str, Any]:
        pid = state["project_id"]
        if all_jobs_done(ctx, pid):
            return {"jobs_pending": False}
        interrupt({"gate": "jobs", "title": "Experiments running", "jobs": jobs_status(ctx, pid)})
        return {"jobs_pending": not all_jobs_done(ctx, pid)}

    def check_runs(state: LabState) -> dict[str, Any]:
        pid = state["project_id"]
        with ctx.db.session() as s:
            ok = s.scalar(
                select(RunRecord).where(
                    RunRecord.project_id == pid, RunRecord.status == "succeeded"
                )
            )
        if ok is None:
            ctx.set_stage(pid, "experiments_failed", "Blocked", "no experiment succeeded")
            return {"blocked_reason": "no experiment succeeded"}
        return {}

    def gate_d(state: LabState) -> dict[str, Any]:
        pid, version = state["project_id"], state["manuscript_version"]
        ctx.set_stage(
            pid, "awaiting_release", "Unverified" if state.get("evidence_blockers") else "Measured"
        )
        interrupt(
            {
                "gate": "D",
                "title": "Release manuscript",
                "manuscript_version": version,
                "evidence_blockers": state.get("evidence_blockers", []),
            }
        )
        with ctx.db.session() as s:
            ms = s.scalar(
                select(ManuscriptRecord).where(
                    ManuscriptRecord.project_id == pid, ManuscriptRecord.version == version
                )
            )
            released = ms is not None and ms.status == "released"
        if released:
            ctx.set_stage(pid, "released", "Approved")
        return {"released": released}

    g = StateGraph(LabState)
    g.add_node("orchestrator", node(orchestrator_agent))
    g.add_node("literature", node(literature_agent))
    g.add_node("research_gap", node(gap_agent))
    g.add_node("methodology", node(methodology_agent))
    g.add_node("dataset", node(dataset_agent))
    g.add_node("gate_a", gate_a)
    g.add_node("coding", node(coding_agent))
    g.add_node("experiment", node(experiment_agent))
    g.add_node("wait_jobs", wait_jobs)
    g.add_node("check_runs", check_runs)
    g.add_node("statistics", node(statistics_agent))
    g.add_node("writing", node(writing_agent))
    g.add_node("reviewer", node(reviewer_agent))
    g.add_node("gate_d", gate_d)

    g.add_edge(START, "orchestrator")
    g.add_edge("orchestrator", "literature")
    g.add_edge("literature", "research_gap")
    g.add_edge("research_gap", "methodology")
    g.add_edge("methodology", "dataset")
    g.add_conditional_edges(
        "dataset", lambda s: "gate_a" if s.get("dataset_status") == "verified" else END
    )
    g.add_conditional_edges(
        "gate_a",
        lambda s: (
            "coding"
            if s.get("gate_a") == "approved"
            else ("methodology" if s.get("gate_a") == "rejected" else "gate_a")
        ),
    )
    g.add_conditional_edges("coding", lambda s: END if s.get("compile_error") else "experiment")
    g.add_edge("experiment", "wait_jobs")
    g.add_conditional_edges(
        "wait_jobs", lambda s: "wait_jobs" if s.get("jobs_pending") else "check_runs"
    )
    g.add_conditional_edges(
        "check_runs", lambda s: END if s.get("blocked_reason") else "statistics"
    )
    g.add_edge("statistics", "writing")
    g.add_edge("writing", "reviewer")

    def after_review(s: LabState) -> str:
        if s.get("evidence_blockers") and s.get("revisions", 0) < MAX_REVISIONS:
            return "revise"
        return "gate_d"

    def revise(state: LabState) -> dict[str, Any]:
        return {"revisions": state.get("revisions", 0) + 1}

    g.add_node("revise", revise)
    g.add_conditional_edges("reviewer", after_review)
    g.add_edge("revise", "writing")
    g.add_conditional_edges("gate_d", lambda s: END if s.get("released") else "gate_d")
    return g.compile(checkpointer=checkpointer)
