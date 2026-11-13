"""Agent 4: Methodology Design. Proposes a protocol; the schema and allowlists have the last word."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select

from cyberlab_common.db import ProtocolVersion, audit
from cyberlab_data.cicids2017 import DATASET_ID
from cyberlab_orchestrator.context import LabContext
from cyberlab_protocol import default_protocol
from cyberlab_schemas import Hypothesis, ModelSpec, Protocol

SYSTEM = (
    "You are a methodology design agent. Refine the baseline experimental protocol for the "
    "study. You may only use model templates: logistic_regression, random_forest, xgboost, mlp. "
    "Allowed params: logistic_regression{C,max_iter}, random_forest{n_estimators,max_depth,"
    "min_samples_leaf}, xgboost{n_estimators,max_depth,learning_rate,subsample}, "
    "mlp{hidden_layer_sizes,alpha,max_iter}. Hypotheses compare model names with '>'. "
    "Output JSON: {'research_questions': [str], 'hypotheses': [{'id':'H1','statement','comparison'}], "
    "'models': [{'name','template','params','role'}], 'rationale': str}."
)


def methodology_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    base = default_protocol(pid, DATASET_ID)
    feedback = state.get("protocol_feedback", "")
    proposal = ctx.llm.complete_json(
        "methodology.protocol",
        SYSTEM,
        f"Baseline protocol:\n{base.model_dump_json(indent=1)}\n\nProposed gaps: {state.get('gaps', [])}\n"
        f"PI feedback on the previous version (if any): {feedback}",
        project_id=pid,
    )
    protocol, rationale, problems = _merge(base, proposal)
    with ctx.db.session() as s:
        version = (s.scalar(select(func.max(ProtocolVersion.version)).where(ProtocolVersion.project_id == pid)) or 0) + 1
        protocol = protocol.model_copy(update={"version": version})
        s.add(
            ProtocolVersion(
                project_id=pid,
                version=version,
                sha256=protocol.sha256,
                body=protocol.model_dump(mode="json"),
                rationale=rationale,
                created_by="methodology_agent",
            )
        )
        ctx.note(s, pid, "methodology", "protocol_proposal", {"version": version, "rationale": rationale,
                                                              "validation_problems": problems})
        audit(s, "methodology_agent", "protocol.drafted", pid, {"version": version, "sha256": protocol.sha256})
    ctx.set_stage(pid, "protocol_drafted", "Proposed")
    return {"protocol_version": version, "protocol_sha256": protocol.sha256}


def _merge(base: Protocol, proposal: dict[str, Any]) -> tuple[Protocol, str, list[str]]:
    """Accept the agent's proposal only where it validates; otherwise keep the baseline."""
    problems: list[str] = []
    update: dict[str, Any] = {}
    try:
        models = [ModelSpec.model_validate(m) for m in proposal.get("models", [])]
        if models:
            update["models"] = models
    except ValidationError as exc:
        problems.append(f"models rejected: {exc.errors()[:2]}")
    try:
        hyps = [Hypothesis.model_validate(h) for h in proposal.get("hypotheses", [])]
        if hyps:
            update["hypotheses"] = hyps
    except ValidationError as exc:
        problems.append(f"hypotheses rejected: {exc.errors()[:2]}")
    rqs = [q for q in proposal.get("research_questions", []) if isinstance(q, str) and q.strip()]
    if rqs:
        update["research_questions"] = rqs
    candidate = base.model_copy(update=update)
    try:
        candidate = Protocol.model_validate(json.loads(candidate.model_dump_json()))
        names = {m.name for m in candidate.models}
        for h in candidate.hypotheses:
            if not all(x.strip() in names for x in h.comparison.split(">")):
                raise ValueError(f"{h.id} compares unknown models")
        from cyberlab_tabular.templates import ALLOWED_PARAMS

        for m in candidate.models:
            extra = set(m.params) - ALLOWED_PARAMS[m.template]
            if extra:
                raise ValueError(f"{m.name}: params not allowed {sorted(extra)}")
    except (ValidationError, ValueError) as exc:
        problems.append(f"proposal rejected, baseline kept: {exc}")
        candidate = base
    return candidate, str(proposal.get("rationale", "")), problems
