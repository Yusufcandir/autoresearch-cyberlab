"""Agent 4: Methodology Design. Proposes a protocol; the schema and allowlists have the last word."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError
from sqlalchemy import func, select

from cyberlab_common.db import Project, ProtocolVersion, audit
from cyberlab_orchestrator.context import LabContext
from cyberlab_orchestrator.datasets import project_adapter
from cyberlab_protocol import default_protocol
from cyberlab_protocol.intent import (
    BASELINES,
    PRETTY,
    hypothesis_statement,
    min_support_from_text,
    models_named,
    research_intent,
    seeds_from_text,
    split_from_goal,
    testable_questions,
    tuning_from_text,
)
from cyberlab_schemas import Hypothesis, ModelSpec, Protocol, ResearchSpec, SplitSpec, TuningSpec

SYSTEM = (
    "You are a methodology design agent. Design the experimental protocol for THIS study's "
    "research goal: its research questions and hypotheses must test what the goal asks, with "
    "the goal's primary metric. You may only use these model templates (settings are fixed "
    "by the platform or tuned on the validation split, so leave params empty, except: an "
    "ablation model (role 'ablation') may set params {'drop_features': [feature names or name "
    "prefixes]} and a robustness model may set {'test_noise': standard deviations}): "
    "logistic_regression, random_forest, xgboost, lightgbm, mlp (classical); resnet_mlp, "
    "ft_transformer, cnn1d, tabnet (deep, PyTorch). Use the models the goal names; if it names "
    "none, use logistic_regression, random_forest, xgboost and mlp. Hypotheses compare model "
    "names with '>' and give primary_metric in macro_f1, accuracy, balanced_accuracy, mcc. "
    "Only ask research questions these experiments can answer (no runtime, scalability, "
    "deployment or other-dataset questions). Output JSON: {'research_questions': [str], "
    "'hypotheses': [{'id':'H1','statement','comparison','primary_metric'}], "
    "'models': [{'name','template','params','role'}], 'rationale': str}."
)


def methodology_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    adapter = project_adapter(ctx, pid)
    base = _dataset_baseline(
        default_protocol(pid, adapter.dataset_id, dataset_name=adapter.display), adapter
    )
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
    protocol, enforced = _enforce_intent(protocol, intent, goal, adapter.split_strategies)
    problems += enforced
    protocol, design = _design_from_text(protocol, " ".join([goal, *constraints, feedback or ""]))
    problems += design
    protocol, fit = _fit_modality(protocol, adapter, base)
    problems += fit
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
        # ablation / robustness settings are part of the design and are kept
        common = {k: v for k, v in mdl.params.items() if k in ("drop_features", "test_noise")}
        params = {**DEFAULT_PARAMS[mdl.template], **common, **overrides.get(mdl.template, {})}
        models.append(mdl.model_copy(update={"params": params}))
    notes = ["model settings: reviewed defaults"]
    if overrides:
        notes.append(f"PI-requested settings applied: {overrides}")
    updated = protocol.model_copy(update={"models": models})
    return Protocol.model_validate(json.loads(updated.model_dump_json())), notes


def _enforce_intent(
    protocol: Protocol,
    intent: dict[str, Any],
    goal: str,
    split_strategies: tuple[str, ...] = ("stratified_dedup", "by_day"),
) -> tuple[Protocol, list[str]]:
    """The researcher's own hypotheses and primary metric win over the agent's proposal, and
    research questions no experiment can answer are removed."""
    notes: list[str] = []
    named = models_named(goal)
    if len(named) >= 2 and sorted(named) != sorted(m.name for m in protocol.models):
        # the researcher's own model line-up wins over the agent's proposal
        from cyberlab_tabular.templates import DEFAULT_PARAMS

        models = [
            ModelSpec(
                name=n,
                template=n,
                params=dict(DEFAULT_PARAMS[n]),
                role="baseline" if n in BASELINES else "candidate",
            )
            for n in named
        ]
        keep = [
            h
            for h in protocol.hypotheses
            if all(x.strip() in named for x in h.comparison.split(">"))
        ]
        if not keep:
            base = next((n for n in named if n in BASELINES), named[-1])
            cand = next(n for n in named if n != base)
            m0 = intent.get("primary_metric") or "macro_f1"
            keep = [
                Hypothesis(
                    id="H1",
                    statement=hypothesis_statement(cand, base, m0),
                    comparison=f"{cand} > {base}",
                    primary_metric=m0,
                )
            ]
        protocol = protocol.model_copy(update={"models": models, "hypotheses": keep})
        notes.append(f"models taken from the research goal: {', '.join(named)}")
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
                        h.comparison.split(">")[0].strip(),
                        h.comparison.split(">")[-1].strip(),
                        metric,
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
    if wanted_split and wanted_split["strategy"] in split_strategies:
        update["split"] = SplitSpec(**wanted_split)
        notes.append(f"split taken from the research goal: {wanted_split}")
        if wanted_split["strategy"] == "cross_dataset":
            # attack families differ between datasets: compare attack versus benign
            update["label_scheme"] = "binary"
            notes.append("cross-dataset test: binary labels (attack versus BENIGN)")
    elif wanted_split:
        notes.append(
            f"split in the goal not available for this dataset: {wanted_split['strategy']}"
        )
    if protocol.split.strategy not in split_strategies and "split" not in update:
        update["split"] = SplitSpec()
        notes.append(
            f"split {protocol.split.strategy} not available for this dataset; default used"
        )
    updated = protocol.model_copy(update=update)
    return Protocol.model_validate(json.loads(updated.model_dump_json())), notes


def _design_from_text(protocol: Protocol, text: str) -> tuple[Protocol, list[str]]:
    """Tuning, number of seeds and the small-class threshold, when the researcher (goal,
    constraints) or the PI (review comment) asks for them. Decided before any result exists."""
    notes: list[str] = []
    update: dict[str, Any] = {}
    trials = tuning_from_text(text)
    metric = protocol.hypotheses[0].primary_metric if protocol.hypotheses else "macro_f1"
    if trials:
        update["tuning"] = TuningSpec(trials=trials, metric=metric)
        if protocol.split.val_fraction <= 0:
            update["split"] = protocol.split.model_copy(update={"val_fraction": 0.1})
        notes.append(f"tuning on the validation split: {trials} trials per model, {metric}")
    seeds = seeds_from_text(text) or (5 if trials else None)
    if seeds and seeds != len(protocol.seeds):
        update["seeds"] = [42 + i for i in range(max(1, min(seeds, 10)))]
        notes.append(f"{len(update['seeds'])} seeds")
    support = min_support_from_text(text)
    if support:
        update["min_test_support"] = support
        notes.append(f"classes under {support} test rows reported separately with exact intervals")
    if not update:
        return protocol, notes
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
        from cyberlab_tabular.templates import ALLOWED_PARAMS, COMMON_PARAMS

        for m in candidate.models:
            extra = set(m.params) - ALLOWED_PARAMS[m.template] - COMMON_PARAMS
            if extra:
                raise ValueError(f"{m.name}: params not allowed {sorted(extra)}")
    except (ValidationError, ValueError) as exc:
        problems.append(f"proposal rejected, baseline kept: {exc}")
        candidate = base
    return candidate, str(proposal.get("rationale", "")), problems


def _dataset_baseline(base: Protocol, adapter: Any) -> Protocol:
    """A dataset's own default line-up and split (e.g. text models and a split grouped by
    website for the phishing URL dataset)."""
    from cyberlab_tabular.templates import DEFAULT_PARAMS

    update: dict[str, Any] = {}
    if adapter.default_models:
        names = list(adapter.default_models)
        update["models"] = [
            ModelSpec(
                name=n,
                template=n,
                params=dict(DEFAULT_PARAMS[n]),
                role="baseline" if i == 0 else "candidate",
            )
            for i, n in enumerate(names)
        ]
        update["hypotheses"] = [
            Hypothesis(
                id="H1",
                statement=hypothesis_statement(names[-1], names[0], "macro_f1"),
                comparison=f"{names[-1]} > {names[0]}",
            )
        ]
        update["research_questions"] = [
            f"How well do {PRETTY.get(names[0], names[0])} and "
            f"{PRETTY.get(names[-1], names[-1])} detect attacks in {adapter.display}?"
        ]
    if adapter.default_split != "stratified_dedup":
        update["split"] = SplitSpec(strategy=adapter.default_split)
    if not update:
        return base
    return Protocol.model_validate(json.loads(base.model_copy(update=update).model_dump_json()))


def _fit_modality(protocol: Protocol, adapter: Any, base: Protocol) -> tuple[Protocol, list[str]]:
    """Text datasets take text templates only, tabular datasets tabular ones; anything else
    is removed here rather than failing later at compile time."""
    from cyberlab_tabular.templates import TEXT_TEMPLATES

    text = adapter.modality == "text"
    keep = [m for m in protocol.models if (m.template in TEXT_TEMPLATES) == text]
    if len(keep) == len(protocol.models):
        return protocol, []
    if len(keep) < 2:
        keep = list(base.models)
    names = {m.name for m in keep}
    hyps = [
        h for h in protocol.hypotheses if all(x.strip() in names for x in h.comparison.split(">"))
    ] or list(base.hypotheses)
    updated = protocol.model_copy(update={"models": keep, "hypotheses": hyps})
    return Protocol.model_validate(json.loads(updated.model_dump_json())), [
        f"models that do not fit the {adapter.modality} dataset were removed"
    ]
