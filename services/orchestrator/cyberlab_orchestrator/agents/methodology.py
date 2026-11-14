"""Agent 4: Methodology Design. Proposes a protocol; the schema and allowlists have the last word."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select

from cyberlab_common.db import Project, ProtocolVersion, audit
from cyberlab_data.cicids2017 import DATASET_ID
from cyberlab_orchestrator.context import LabContext
from cyberlab_protocol import default_protocol
from cyberlab_protocol.intent import (
    PRETTY,
    hypothesis_statement,
    research_intent,
    split_from_goal,
    testable_questions,
)
from cyberlab_schemas import Hypothesis, ModelSpec, Protocol, ResearchSpec, SplitSpec

SYSTEM = (
    "You are a methodology design agent. Design the experimental protocol for THIS study's "
    "research goal: its research questions and hypotheses must test what the goal asks, with "
    "the goal's primary metric. You may only use model templates: logistic_regression, "
    "random_forest, xgboost, mlp. Allowed params: logistic_regression{C,max_iter}, "
    "random_forest{n_estimators,max_depth,min_samples_leaf}, xgboost{n_estimators,max_depth,"
    "learning_rate,subsample}, mlp{hidden_layer_sizes,alpha,max_iter}. Hypotheses compare model "
    "names with '>' and give primary_metric in macro_f1, accuracy, balanced_accuracy, mcc. "
    "Only ask research questions these experiments can answer (no runtime, scalability, "
    "deployment or other-dataset questions). Output JSON: {'research_questions': [str], "
    "'hypotheses': [{'id':'H1','statement','comparison','primary_metric'}], "
    "'models': [{'name','template','params','role'}], 'rationale': str}."
)


def methodology_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    base = default_protocol(pid, DATASET_ID)
    feedback = state.get("protocol_feedback", "")
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        spec = ResearchSpec.model_validate(project.spec) if project else None
    goal = spec.goal if spec else ""
    constraints = spec.constraints if spec else []
    intent = research_intent(goal, constraints)
    proposal = ctx.llm.complete_json(
        "methodology.protocol",
        SYSTEM,
        f"STUDY TITLE: {spec.title if spec else ''}\nRESEARCH GOAL (the protocol must test "
        f"exactly this): {goal}\nCONSTRAINTS: {constraints}\n"
        f"Hypotheses stated in the goal: {intent['hypotheses']}\n"
        f"Primary metric stated in the goal: {intent['primary_metric']}\n\n"
        f"Baseline protocol (a template only; replace its research questions and hypotheses "
        f"with ones for this goal):\n{base.model_dump_json(indent=1)}\n\n"
        f"Proposed gaps: {state.get('gaps', [])}\n"
        f"PI feedback on the previous version (if any): {feedback}",
        project_id=pid,
    )
    protocol, rationale, problems = _merge(base, proposal)
    protocol, fixed = _fixed_params(protocol, feedback)
    problems += fixed
    protocol, enforced = _enforce_intent(protocol, intent, goal)
    problems += enforced
    with ctx.db.session() as s:
        version = (
            s.scalar(
                select(func.max(ProtocolVersion.version)).where(ProtocolVersion.project_id == pid)
            )
            or 0
        ) + 1
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
        ctx.note(
            s,
            pid,
            "methodology",
            "protocol_proposal",
            {"version": version, "rationale": rationale, "validation_problems": problems},
        )
        audit(
            s,
            "methodology_agent",
            "protocol.drafted",
            pid,
            {"version": version, "sha256": protocol.sha256},
        )
    ctx.set_stage(pid, "protocol_drafted", "Proposed")
    return {"protocol_version": version, "protocol_sha256": protocol.sha256}


def pi_param_overrides(feedback: str) -> dict[str, dict[str, Any]]:
    """Settings the PI asked for in a 'Request changes' comment, e.g.
    'random_forest max_depth 20' or 'mlp hidden_layer_sizes [128, 64]'."""
    import ast

    from cyberlab_tabular.templates import ALLOWED_PARAMS

    out: dict[str, dict[str, Any]] = {}
    for tmpl, allowed in ALLOWED_PARAMS.items():
        for p in allowed:
            m = re.search(
                rf"{tmpl}\W+(?:[\w ]*?\b)?{p}\s*[=:]?\s*(\[[^\]]*\]|[\d.]+|None)", feedback, re.I
            )
            if m:
                try:
                    out.setdefault(tmpl, {})[p] = ast.literal_eval(m.group(1))
                except (ValueError, SyntaxError):
                    continue
    return out


def _fixed_params(protocol: Protocol, feedback: str) -> tuple[Protocol, list[str]]:
    """Every model gets the reviewed default settings; only the PI's explicit comment changes
    them. The agent's own parameter choices are ignored (they can decide a hypothesis)."""
    from cyberlab_tabular.templates import DEFAULT_PARAMS

    overrides = pi_param_overrides(feedback or "")
    models = []
    for mdl in protocol.models:
        params = {**DEFAULT_PARAMS[mdl.template], **overrides.get(mdl.template, {})}
        models.append(mdl.model_copy(update={"params": params}))
    notes = ["model settings: reviewed defaults"]
    if overrides:
        notes.append(f"PI-requested settings applied: {overrides}")
    updated = protocol.model_copy(update={"models": models})
    return Protocol.model_validate(json.loads(updated.model_dump_json())), notes


def _enforce_intent(
    protocol: Protocol, intent: dict[str, Any], goal: str
) -> tuple[Protocol, list[str]]:
    """The researcher's own hypotheses and primary metric win over the agent's proposal, and
    research questions no experiment can answer are removed."""
    notes: list[str] = []
    names = {m.name for m in protocol.models}
    metric = intent.get("primary_metric")
    stated = [h for h in intent.get("hypotheses", []) if h[0] in names and h[1] in names]
    hyps = list(protocol.hypotheses)
    if stated:
        hyps = [
            Hypothesis(
                id=f"H{i}",
                statement=hypothesis_statement(a, b, m),
                comparison=f"{a} > {b}",
                primary_metric=m,
            )
            for i, (a, b, m) in enumerate(stated, 1)
        ]
        notes.append(
            "hypotheses taken from the research goal: "
            + ", ".join(f"{a} > {b} ({m})" for a, b, m in stated)
        )
    elif metric:
        hyps = [
            h.model_copy(
                update={
                    "primary_metric": metric,
                    "statement": hypothesis_statement(
                        *[x.strip() for x in h.comparison.split(">")], metric
                    ),
                }
            )
            for h in hyps
        ]
        notes.append(f"primary metric set from the research goal: {metric}")
    rqs, dropped = testable_questions(list(protocol.research_questions))
    if dropped:
        notes.append(f"research questions removed (no experiment measures them): {dropped}")
    if stated and (not rqs or _is_default_rq(rqs)):
        rqs = [goal_question(stated)]
    elif not rqs:
        first = re.split(r"(?<=[.?!])\s", goal.strip(), maxsplit=1)[0] if goal.strip() else ""
        rqs = [first.rstrip(".") + "?"] if first else list(protocol.research_questions)
    update: dict[str, Any] = {"hypotheses": hyps, "research_questions": rqs}
    wanted_split = split_from_goal(goal)
    if wanted_split:
        update["split"] = SplitSpec(**wanted_split)
        notes.append(f"split taken from the research goal: {wanted_split}")
    updated = protocol.model_copy(update=update)
    return Protocol.model_validate(json.loads(updated.model_dump_json())), notes


def _is_default_rq(rqs: list[str]) -> bool:
    return any(q.startswith("How do classical and gradient-boosted models") for q in rqs)


def goal_question(stated: list[tuple[str, str, str]]) -> str:
    a, b, m = stated[0]
    q = f"Does {PRETTY.get(a, a)} achieve higher {PRETTY.get(m, m)} than {PRETTY.get(b, b)}"
    if len(stated) > 1:
        a2, b2, m2 = stated[1]
        q += f", and does {PRETTY.get(a2, a2)} achieve higher {PRETTY.get(m2, m2)} than {PRETTY.get(b2, b2)}"
    return q + "?"


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
