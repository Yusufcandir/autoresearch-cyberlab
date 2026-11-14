"""Deterministic project analysis: aggregate runs, test hypotheses, publish evidence claims."""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from cyberlab_common.db import ClaimRecord, DatasetRecord, ProtocolVersion, RunRecord, audit
from cyberlab_data import dataset_record_id
from cyberlab_evidence import ArtifactStore
from cyberlab_qa.metrics import (
    QAFlag,
    bootstrap_ci,
    compute_metrics,
    leakage_audit,
    mcnemar,
    recompute_matches,
)
from cyberlab_schemas import EvidenceClaim, Protocol, StatusLabel

HEADLINE = (
    "macro_f1",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "false_positive_rate",
    "detection_rate",
)


def analyze_project(
    s: Session, store: ArtifactStore, project_id: str, protocol_version: int
) -> dict[str, Any]:
    pv = s.scalar(
        select(ProtocolVersion).where(
            ProtocolVersion.project_id == project_id, ProtocolVersion.version == protocol_version
        )
    )
    if pv is None:
        raise ValueError("protocol version not found")
    protocol = Protocol.model_validate(pv.body)
    runs = list(
        s.scalars(
            select(RunRecord).where(
                RunRecord.project_id == project_id, RunRecord.status == "succeeded"
            )
        )
    )
    runs = [r for r in runs if r.experiment_id.startswith(f"p{protocol_version}-")]
    if not runs:
        raise ValueError("no successful runs to analyze")

    preds: dict[tuple[str, int], pd.DataFrame] = {}
    recompute_ok = True
    for r in runs:
        path = store.path_for(s, project_id, r.receipt["artifacts"]["predictions.csv"])
        df = pd.read_csv(path)
        preds[(r.model_name, r.seed)] = df
        recompute_ok &= recompute_matches(r.metrics, df)

    models: dict[str, Any] = {}
    for m in protocol.models:
        seeds = sorted(seed for (name, seed) in preds if name == m.name)
        if not seeds:
            continue
        per_seed = {seed: compute_metrics(preds[(m.name, seed)]) for seed in seeds}
        agg: dict[str, Any] = {}
        for k in HEADLINE:
            vals = [per_seed[sd][k] for sd in seeds if k in per_seed[sd]]
            if vals:
                agg[k] = {
                    "mean": statistics.fmean(vals),
                    "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "n_seeds": len(vals),
                }
        first = per_seed[seeds[0]]
        models[m.name] = {
            "role": m.role,
            "template": m.template,
            "params": dict(m.params),
            "seeds": seeds,
            "aggregate": agg,
            "ci_seed": seeds[0],
            "macro_f1_ci": bootstrap_ci(preds[(m.name, seeds[0])], "macro_f1", n_boot=500),
            "per_class": first["per_class"],
            "confusion_matrix": first["confusion_matrix"],
        }
        # CIs also for every metric a hypothesis is tested on (e.g. balanced accuracy)
        for metric in {h.primary_metric for h in protocol.hypotheses} - {"macro_f1"}:
            models[m.name][f"{metric}_ci"] = bootstrap_ci(
                preds[(m.name, seeds[0])], metric, n_boot=500
            )

    hypotheses: list[dict[str, Any]] = []
    for h in protocol.hypotheses:
        a, b = (x.strip() for x in h.comparison.split(">"))
        common = sorted({sd for (n, sd) in preds if n == a} & {sd for (n, sd) in preds if n == b})
        if not common:
            hypotheses.append({"id": h.id, "statement": h.statement, "result": "not tested"})
            continue
        sd0 = common[0]
        test = mcnemar(preds[(a, sd0)], preds[(b, sd0)])
        ma = models[a]["aggregate"][h.primary_metric]["mean"]
        mb = models[b]["aggregate"][h.primary_metric]["mean"]
        supported = ma > mb and test["p_value"] < 0.05
        hypotheses.append(
            {
                "id": h.id,
                "statement": h.statement,
                "metric": h.primary_metric,
                "a": a,
                "b": b,
                "a_mean": ma,
                "b_mean": mb,
                "mcnemar_seed": sd0,
                "mcnemar": test,
                "result": "supported" if supported else "not supported",
            }
        )

    ds = s.get(DatasetRecord, dataset_record_id(protocol.dataset_id, protocol.split))
    overlap = int(ds.manifest.get("split_overlap_rows", 0)) if ds else 0
    audit_input = {
        n: {"macro_f1": m["aggregate"]["macro_f1"]["mean"], "per_class": m["per_class"]}
        for n, m in models.items()
    }
    flags = leakage_audit(audit_input, overlap)
    if not recompute_ok:
        flags.append(
            QAFlag("blocker", "RECOMPUTE_MISMATCH", "stored metrics differ from recomputation")
        )
    analysis: dict[str, Any] = {
        "project_id": project_id,
        "protocol_version": protocol_version,
        "protocol_sha256": protocol.sha256,
        "dataset": ds.manifest if ds else None,
        "models": models,
        "hypotheses": hypotheses,
        "metrics_recomputed_match": recompute_ok,
        "qa_flags": [f.__dict__ for f in flags],
        "run_ids": sorted(r.id for r in runs),
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "analysis.json"
        p.write_text(json.dumps(analysis, indent=2, default=str))
        rec = store.put(s, project_id, p, name="analysis.json")
    analysis["artifact_sha256"] = rec.sha256
    _publish_claims(s, project_id, analysis, rec.sha256)
    audit(
        s,
        "scientific_qa",
        "evidence.published",
        project_id,
        {"analysis_sha256": rec.sha256, "flags": len(flags), "recompute_ok": recompute_ok},
    )
    return analysis


def _publish_claims(s: Session, project_id: str, analysis: dict[str, Any], sha: str) -> None:
    s.execute(delete(ClaimRecord).where(ClaimRecord.project_id == project_id))
    verified = analysis["metrics_recomputed_match"] and not any(
        f["severity"] == "blocker" for f in analysis["qa_flags"]
    )
    status = StatusLabel.VERIFIED if verified else StatusLabel.MEASURED
    for name, m in analysis["models"].items():
        for metric, v in m["aggregate"].items():
            claim = EvidenceClaim(
                claim_id=f"{name}.{metric}",
                project_id=project_id,
                text=f"{name} {metric} mean over {v['n_seeds']} seeds",
                kind="metric",
                metric=metric,
                value=v["mean"],
                artifact_sha256=sha,
                status=status,
            )
            s.add(
                ClaimRecord(
                    id=f"{project_id[:8]}:{claim.claim_id}",
                    project_id=project_id,
                    body=claim.model_dump(mode="json"),
                    status=claim.status.value,
                )
            )
    for h in analysis["hypotheses"]:
        claim = EvidenceClaim(
            claim_id=f"hypothesis.{h['id']}",
            project_id=project_id,
            text=f"{h['id']}: {h['result']}",
            kind="comparison",
            artifact_sha256=sha,
            status=status,
        )
        s.add(
            ClaimRecord(
                id=f"{project_id[:8]}:{claim.claim_id}",
                project_id=project_id,
                body=claim.model_dump(mode="json"),
                status=claim.status.value,
            )
        )
