"""Experiment feedback controller (professor's plan, Phase 3).

Plan -> Execute -> Analyze -> Validate -> Critique -> Revise. After the scientific validation,
explicit deterministic rules decide whether the evidence answers the research question. Each
rule writes a decision to the research decision log; a decision that needs new experiments
carries a concrete protocol change, which runs only after the PI approves it (it becomes a new
protocol version and goes through gate A like any other). No LLM decides; no shell command is
generated; the number of follow-up cycles and their compute are capped.

Rules
  A  insufficient class support      -> uncertainty warning, conservative claims, more data
  B  suspected leakage / weak split  -> follow-up with a stricter (group / time / official) split
  C  inappropriate statistical test  -> recompute the statistics (no new experiment)
  D  unstable results across seeds   -> follow-up with more seeds
  E  surprising or unsupported claim -> audit and confirm before claiming it
"""

from __future__ import annotations

import uuid
from typing import Any

from cyberlab_orchestrator.context import LabContext

MAX_FOLLOWUPS = 3


def _decision(
    rule: str,
    observed: str,
    problem: str,
    action: str,
    justification: str,
    benefit: str,
    change: dict[str, Any] | None,
    cost_minutes: float | None,
) -> dict[str, Any]:
    return {
        "decision_id": f"D-{uuid.uuid4().hex[:8]}",
        "rule": rule,
        "observed_evidence": observed,
        "identified_problem": problem,
        "proposed_action": action,
        "scientific_justification": justification,
        "expected_benefit": benefit,
        "computational_cost_estimate_minutes": cost_minutes,
        "protocol_change": change,
        "approval_status": "pending_pi" if change else "not_required",
        "final_outcome": None,
    }


def stricter_split(
    adapter: Any, protocol: dict[str, Any], groups: list[str]
) -> dict[str, Any] | None:
    """A defensible alternative split the dataset supports."""
    current = (protocol.get("split") or {}).get("strategy", "stratified_dedup")
    options = list(adapter.split_strategies)
    if "official" in options and current != "official":
        return {"strategy": "official"}
    if "by_day" in options and current != "by_day":
        return {"strategy": "by_day", "test_days": ["Thursday", "Friday"]}
    if "by_group" in options and current != "by_group" and groups:
        k = max(1, len(groups) // 4)
        return {"strategy": "by_group", "test_days": sorted(groups)[-k:]}
    if "grouped" in options and current != "grouped":
        return {"strategy": "grouped"}
    return None


def decide(
    validation: dict[str, Any],
    analysis: dict[str, Any],
    protocol: dict[str, Any],
    adapter: Any,
    runtime_minutes: float,
    groups: list[str] | None = None,
) -> list[dict[str, Any]]:
    checks = {c["check"]: c for c in validation.get("checks", [])}
    out: list[dict[str, Any]] = []
    rare = checks.get("rare_class_reliability")
    if rare and rare["status"] == "WARNING":
        out.append(
            _decision(
                "A",
                rare["detail"],
                "per-class results for the small classes rest on too few test rows to generalise",
                "state the uncertainty with exact intervals and interpret these classes "
                "conservatively; evaluate on a second dataset with more examples if one exists",
                "repeated resampling of the same rows does not create independent evidence",
                "claims about rare attacks match the strength of the evidence",
                None,
                None,
            )
        )
    weak = [
        c
        for name in (
            "capture_groups",
            "temporal_leakage",
            "near_duplicate_flows",
            "exact_feature_overlap",
        )
        if (c := checks.get(name)) and c["status"] in ("WARNING", "BLOCK")
    ]
    if weak:
        change = stricter_split(adapter, protocol, groups or [])
        out.append(
            _decision(
                "B",
                " | ".join(f"{c['check']}: {c['detail']}" for c in weak),
                "the test set may be too close to the training data to show generalisation",
                (
                    f"repeat the experiments with a {change['strategy']} split and compare with "
                    "the current results"
                )
                if change
                else "state the limitation; this dataset offers no stricter split",
                "a group- or time-aware split removes shared captures between training and test",
                "shows whether the scores survive a harder, more realistic test",
                {"split": change} if change else None,
                runtime_minutes if change else None,
            )
        )
    stats = checks.get("statistical_tests")
    if stats and stats["status"] == "BLOCK":
        out.append(
            _decision(
                "C",
                stats["detail"],
                "a statistical test does not match its hypothesis",
                "recompute the statistics with the paired permutation test on the hypothesis "
                "metric (use Rewrite draft; no new experiment)",
                "McNemar tests paired correctness, not a difference in the hypothesis metric",
                "hypothesis results that the test actually supports",
                None,
                0.0,
            )
        )
    cons = checks.get("result_consistency")
    if cons and cons["status"] == "WARNING" and cons.get("unstable"):
        seeds = list(protocol.get("seeds", []))
        more = [42 + i for i in range(min(10, max(len(seeds) * 2, 5)))]
        out.append(
            _decision(
                "D",
                cons["detail"],
                "performance varies substantially between seeds",
                f"repeat with {len(more)} seeds to estimate the variation reliably",
                "a mean over few unstable runs is not a reliable estimate",
                "a stable mean and standard deviation",
                {"seeds": more},
                runtime_minutes * len(more) / max(1, len(seeds)),
            )
        )
    if cons and cons["status"] == "WARNING" and not cons.get("unstable"):
        out.append(
            _decision(
                "E",
                cons["detail"],
                "an unusually large improvement must be confirmed before it is claimed",
                "audit the split and the single-feature shortcuts; confirm on a stricter split",
                "surprising gains are often caused by leakage or dataset artefacts",
                "the claim is either confirmed or withdrawn",
                None,
                None,
            )
        )
    return out


def record_decisions(ctx: LabContext, pid: str, decisions: list[dict[str, Any]]) -> None:
    from sqlalchemy import select

    from cyberlab_common.db import AgentNote

    def same(d: dict[str, Any]) -> tuple[Any, ...]:
        return (d.get("rule"), d.get("identified_problem"), d.get("proposed_action"))

    with ctx.db.session() as s:
        known = {
            same(n.content)
            for n in s.scalars(
                select(AgentNote).where(
                    AgentNote.project_id == pid, AgentNote.kind == "research_decision"
                )
            )
        }
        for d in decisions:
            if same(d) in known:  # a rewrite recomputes the statistics: no duplicate entry
                continue
            known.add(same(d))
            ctx.note(s, pid, "feedback_controller", "research_decision", d)
