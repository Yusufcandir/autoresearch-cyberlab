"""Compile an approved Protocol into typed, allowlisted ExperimentManifests.

The compiler is deterministic. It refuses: unapproved or stale protocols (approval must match
the exact protocol hash), unverified datasets, non-allowlisted templates, T3 risk (isolation
required) and budgets beyond the configured caps.
"""

from __future__ import annotations

from cyberlab_schemas import (
    Approval,
    ApprovalGate,
    DatasetManifest,
    ExperimentManifest,
    Hypothesis,
    ModelSpec,
    Protocol,
    SplitSpec,
)

ALLOWED_TEMPLATES = {"logistic_regression", "random_forest", "xgboost", "mlp"}
GPU_TEMPLATES = {"xgboost"}


class CompileError(ValueError):
    def __init__(self, reasons: list[str]):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def default_protocol(
    project_id: str, dataset_id: str, version: int = 1, dataset_name: str = "CIC-IDS2017"
) -> Protocol:
    """A sound baseline protocol that agents refine and the PI approves."""
    return Protocol(
        project_id=project_id,
        version=version,
        research_questions=[
            "How do classical and gradient-boosted models compare for multiclass intrusion "
            f"detection on de-duplicated {dataset_name}?",
        ],
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="Gradient boosting achieves higher macro-F1 than logistic regression.",
                comparison="xgboost > logistic_regression",
            ),
            Hypothesis(
                id="H2",
                statement="Random forest achieves higher macro-F1 than logistic regression.",
                comparison="random_forest > logistic_regression",
            ),
        ],
        dataset_id=dataset_id,
        split=SplitSpec(),
        models=[
            ModelSpec(name="logistic_regression", template="logistic_regression", role="baseline"),
            ModelSpec(name="random_forest", template="random_forest", role="baseline"),
            ModelSpec(name="xgboost", template="xgboost", role="candidate"),
            ModelSpec(name="mlp", template="mlp", role="candidate"),
        ],
        seeds=[42, 43, 44],
    )


def compile_protocol(
    protocol: Protocol,
    approval: Approval | None,
    dataset: DatasetManifest | None,
    *,
    max_job_minutes: int,
    use_gpu: bool,
) -> list[ExperimentManifest]:
    reasons: list[str] = []
    if approval is None:
        reasons.append("protocol has no PI approval (gate A)")
    else:
        if approval.gate != ApprovalGate.A_PROTOCOL or approval.decision != "approved":
            reasons.append("approval is not an approved gate-A decision")
        if (
            approval.subject_version != protocol.version
            or approval.subject_sha256 != protocol.sha256
        ):
            reasons.append(
                "approval is for a different protocol version; the protocol changed after approval"
            )
    if dataset is None:
        reasons.append(f"dataset {protocol.dataset_id} is not registered")
    elif dataset.status != "verified":
        reasons.append(f"dataset blocked: {dataset.blocked_reasons}")
    if protocol.risk_tier == "T3":
        reasons.append("BLOCKED: ISOLATION_REQUIRED (T3 protocols need an isolated host)")
    for m in protocol.models:
        if m.template not in ALLOWED_TEMPLATES:
            reasons.append(f"template {m.template} not allowlisted")
    names = [m.name for m in protocol.models]
    if len(set(names)) != len(names):
        reasons.append("model names must be unique")
    for h in protocol.hypotheses:
        for side in h.comparison.split(">"):
            if side.strip() not in names:
                reasons.append(f"{h.id} compares unknown model {side.strip()!r}")
    if reasons:
        raise CompileError(reasons)
    return [
        ExperimentManifest(
            experiment_id=f"p{protocol.version}-{m.name}-s{seed}",
            project_id=protocol.project_id,
            protocol_version=protocol.version,
            protocol_sha256=protocol.sha256,
            dataset_id=protocol.dataset_id,
            model=m,
            seed=seed,
            label_scheme=protocol.label_scheme,
            split=protocol.split,
            risk_tier=protocol.risk_tier,
            max_minutes=min(protocol.gpu_budget_minutes, max_job_minutes),
            use_gpu=use_gpu and m.template in GPU_TEMPLATES,
        )
        for m in protocol.models
        for seed in protocol.seeds
    ]
