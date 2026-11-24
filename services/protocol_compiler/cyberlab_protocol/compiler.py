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

ALLOWED_TEMPLATES = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "mlp",
    "lightgbm",
    "resnet_mlp",
    "ft_transformer",
    "cnn1d",
    "tabnet",
    "tfidf_logreg",
    "tfidf_linear_svm",
}
TEXT_TEMPLATES = {"tfidf_logreg", "tfidf_linear_svm"}
GPU_TEMPLATES = {"xgboost", "resnet_mlp", "ft_transformer", "cnn1d", "tabnet"}


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
    extra_datasets: list[tuple[str, SplitSpec, DatasetManifest | None]] | None = None,
) -> list[ExperimentManifest]:
    """Benchmark protocols also get one experiment per extra dataset; their ids end in
    '@<dataset id>' so every dataset's runs stay apart."""
    reasons: list[str] = []
    for ds_id, _split, man in extra_datasets or []:
        if man is None:
            reasons.append(f"benchmark dataset {ds_id} is not prepared")
        elif man.status != "verified":
            reasons.append(f"benchmark dataset {ds_id} blocked: {man.blocked_reasons}")
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
    if dataset is not None and dataset.features:
        text_data = all(f.startswith("text:") for f in dataset.features)
        for m in protocol.models:
            if (m.template in TEXT_TEMPLATES) != text_data:
                kind = "text" if text_data else "tabular"
                reasons.append(
                    f"UNAVAILABLE_FEATURE_VIEW: template {m.template} does not fit this {kind} "
                    "dataset"
                )
    names = [m.name for m in protocol.models]
    if len(set(names)) != len(names):
        reasons.append("model names must be unique")
    for h in protocol.hypotheses:
        for side in h.comparison.split(">"):
            if side.strip() not in names:
                reasons.append(f"{h.id} compares unknown model {side.strip()!r}")
    if reasons:
        raise CompileError(reasons)
    targets: list[tuple[str, SplitSpec, str]] = [(protocol.dataset_id, protocol.split, "")]
    targets += [(d, sp, f"@{d}") for d, sp, _ in extra_datasets or []]
    transfer: list[ExperimentManifest] = []
    if protocol.transfer and extra_datasets:
        # generalization: trained on each dataset (attack vs BENIGN, reviewed default settings)
        # and tested on every dataset; one run per model, seed and training dataset
        all_ds = [d for d, _, _ in targets]
        transfer = [
            ExperimentManifest(
                experiment_id=f"p{protocol.version}-{m.name}-s{seed}@xfer-{ds_id}",
                project_id=protocol.project_id,
                protocol_version=protocol.version,
                protocol_sha256=protocol.sha256,
                dataset_id=ds_id,
                model=m,
                seed=seed,
                label_scheme="binary",
                split=split,
                risk_tier=protocol.risk_tier,
                max_minutes=min(protocol.gpu_budget_minutes, max_job_minutes),
                use_gpu=use_gpu and m.template in GPU_TEMPLATES,
                tuning=None,
                transfer_targets=[d for d in all_ds if d != ds_id],
            )
            for ds_id, split, _ in targets
            for m in protocol.models
            if m.role != "ablation"
            for seed in protocol.seeds
        ]
    return transfer + [
        ExperimentManifest(
            experiment_id=f"p{protocol.version}-{m.name}-s{seed}{suffix}",
            project_id=protocol.project_id,
            protocol_version=protocol.version,
            protocol_sha256=protocol.sha256,
            dataset_id=ds_id,
            model=m,
            seed=seed,
            label_scheme=protocol.label_scheme,
            split=split,
            risk_tier=protocol.risk_tier,
            max_minutes=min(protocol.gpu_budget_minutes, max_job_minutes),
            use_gpu=use_gpu and m.template in GPU_TEMPLATES,
            tuning=protocol.tuning,
        )
        for ds_id, split, suffix in targets
        for m in protocol.models
        for seed in protocol.seeds
    ]
