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
from cyberlab_data import benchmark_split, dataset_record_id
from cyberlab_evidence import ArtifactStore
from cyberlab_qa.metrics import (
    QAFlag,
    bootstrap_ci_seeds,
    bootstrap_diff_seeds,
    clopper_pearson,
    compute_metrics,
    friedman_nemenyi,
    holm,
    leakage_audit,
    mcnemar,
    paired_permutation_test,
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
    bench_runs = [r for r in runs if "@" in r.experiment_id]
    runs = [r for r in runs if "@" not in r.experiment_id]
    if not runs:
        raise ValueError("no successful runs to analyze")

    preds: dict[tuple[str, int], pd.DataFrame] = {}
    run_info: dict[str, dict[str, Any]] = {}
    recompute_ok = True
    for r in runs:
        arts = r.receipt.get("artifacts", {})
        if r.model_name not in run_info and "model_info.json" in arts:
            info = json.loads(store.path_for(s, project_id, arts["model_info.json"]).read_text())
            if "tuning.json" in arts:
                tj = json.loads(store.path_for(s, project_id, arts["tuning.json"]).read_text())
                info["tuning"] = {
                    k: tj.get(k)
                    for k in (
                        "metric",
                        "trials_requested",
                        "trials_run",
                        "tuning_rows",
                        "best_value",
                        "label",
                    )
                }
            run_info[r.model_name] = info
        path = store.path_for(s, project_id, r.receipt["artifacts"]["predictions.csv"])
        df = pd.read_csv(path)
        preds[(r.model_name, r.seed)] = df
        recompute_ok &= recompute_matches(r.metrics, df)

    models = _summaries(protocol, preds, run_info)

    hypotheses = _test_hypotheses(protocol, preds, models)

    ds = s.get(DatasetRecord, dataset_record_id(protocol.dataset_id, protocol.split))
    overlap = int(ds.manifest.get("split_overlap_rows", 0)) if ds else 0
    audit_input = {
        n: {
            "macro_f1": m["aggregate"]["macro_f1"]["mean"],
            "per_class": m["per_class"],
            "role": m["role"],
        }
        for n, m in models.items()
    }
    flags = leakage_audit(audit_input, overlap)
    benchmark, bench_ok = _benchmark(s, store, project_id, protocol, models, bench_runs, flags)
    recompute_ok = recompute_ok and bench_ok
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
        "min_test_support": protocol.min_test_support,
        "followup": _followup_comparison(s, store, project_id, protocol),
        "tuning": protocol.tuning.model_dump() if protocol.tuning else None,
        "metrics_recomputed_match": recompute_ok,
        "qa_flags": [f.__dict__ for f in flags],
        "run_ids": sorted(r.id for r in [*runs, *bench_runs]),
    }
    if benchmark:
        analysis["benchmark"] = benchmark
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


def _load_preds(
    s: Session, store: ArtifactStore, project_id: str, runs: list[Any]
) -> tuple[dict[tuple[str, int], pd.DataFrame], dict[str, dict[str, Any]], bool]:
    preds: dict[tuple[str, int], pd.DataFrame] = {}
    run_info: dict[str, dict[str, Any]] = {}
    ok = True
    for r in runs:
        arts = r.receipt.get("artifacts", {})
        if r.model_name not in run_info and "model_info.json" in arts:
            run_info[r.model_name] = json.loads(
                store.path_for(s, project_id, arts["model_info.json"]).read_text()
            )
        df = pd.read_csv(store.path_for(s, project_id, arts["predictions.csv"]))
        preds[(r.model_name, r.seed)] = df
        ok &= recompute_matches(r.metrics, df)
    return preds, run_info, ok


def _benchmark(
    s: Session,
    store: ArtifactStore,
    project_id: str,
    protocol: Protocol,
    main_models: dict[str, Any],
    bench_runs: list[Any],
    flags: list[QAFlag],
) -> tuple[dict[str, Any] | None, bool]:
    """Every benchmark dataset analysed like the main one (same summaries, same tests), plus
    a Friedman test and Nemenyi critical difference over the per-dataset mean scores."""
    if not protocol.benchmark_datasets:
        return None, True
    metric = protocol.hypotheses[0].primary_metric if protocol.hypotheses else "macro_f1"
    per_dataset: dict[str, Any] = {protocol.dataset_id: {"models": main_models}}
    all_ok = True
    tests: list[dict[str, Any]] = []
    for ds_id in protocol.benchmark_datasets:
        mine = [r for r in bench_runs if r.experiment_id.endswith(f"@{ds_id}")]
        if not mine:
            flags.append(
                QAFlag("warning", "BENCHMARK_DATASET_MISSING", f"no successful runs on {ds_id}")
            )
            continue
        preds, run_info, ok = _load_preds(s, store, project_id, mine)
        all_ok &= ok
        models = _summaries(protocol, preds, run_info)
        hyps = _test_hypotheses(protocol, preds, models, holm_adjust=False)
        for h in hyps:
            h["dataset"] = ds_id
        tests += [h for h in hyps if "test" in h]
        rec = s.get(DatasetRecord, dataset_record_id(ds_id, benchmark_split(ds_id, protocol.split)))
        man = rec.manifest if rec else {}
        overlap = int(man.get("split_overlap_rows", 0) or 0)
        if overlap:
            flags.append(
                QAFlag(
                    "blocker",
                    "LEAKAGE_OVERLAP",
                    f"{ds_id}: {overlap} rows appear in both training and test",
                )
            )
        per_dataset[ds_id] = {
            "models": models,
            "hypotheses": hyps,
            "dataset": {
                k: man.get(k)
                for k in (
                    "name",
                    "license",
                    "citation",
                    "n_rows_raw",
                    "n_rows_clean",
                    "n_duplicates_removed",
                    "split_counts",
                    "split_overlap_rows",
                    "class_counts",
                )
            },
        }
    # secondary tests (the stated hypotheses repeated on every other dataset): Holm across them
    for h, p_adj in zip(tests, holm([h["test"]["p_value"] for h in tests]), strict=True):
        h["p_holm"] = p_adj
        h["result"] = "supported" if h["a_mean"] > h["b_mean"] and p_adj < 0.05 else "not supported"
    scores: dict[str, dict[str, float]] = {}
    for ds, v in per_dataset.items():
        scores[ds] = {}
        for n, m in v["models"].items():
            mean = (m["aggregate"].get(metric) or {}).get("mean")
            if mean is not None:
                scores[ds][n] = float(mean)
    return {
        "metric": metric,
        "datasets": list(per_dataset),
        "per_dataset": per_dataset,
        "scores": scores,
        "ranking": friedman_nemenyi(scores),
    }, all_ok


def _summaries(
    protocol: Protocol,
    preds: dict[tuple[str, int], pd.DataFrame],
    run_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
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
        seed_preds = [preds[(m.name, sd)] for sd in seeds]
        models[m.name] = {
            "role": m.role,
            "template": m.template,
            "params": dict(run_info.get(m.name, {}).get("params") or m.params),
            "tuned": bool(run_info.get(m.name, {}).get("tuned")),
            "tuning": run_info.get(m.name, {}).get("tuning"),
            "seeds": seeds,
            "aggregate": agg,
            "ci_seed": seeds[0],
            "ci_seeds": seeds,
            "ci_label": (
                f"mean over seeds {', '.join(str(x) for x in seeds)}"
                if len(seeds) > 1
                else f"seed {seeds[0]}"
            ),
            "macro_f1_ci": bootstrap_ci_seeds(seed_preds, "macro_f1", n_boot=500),
            "per_class": _with_exact_intervals(first["per_class"]),
            "confusion_matrix": first["confusion_matrix"],
        }
        # CIs also for every metric a hypothesis is tested on (e.g. balanced accuracy)
        for metric in {h.primary_metric for h in protocol.hypotheses} - {"macro_f1"}:
            models[m.name][f"{metric}_ci"] = bootstrap_ci_seeds(seed_preds, metric, n_boot=500)
    return models


def _test_hypotheses(
    protocol: Protocol,
    preds: dict[tuple[str, int], pd.DataFrame],
    models: dict[str, Any],
    holm_adjust: bool = True,
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for h in protocol.hypotheses:
        a, b = (x.strip() for x in h.comparison.split(">"))
        common = sorted({sd for (n, sd) in preds if n == a} & {sd for (n, sd) in preds if n == b})
        if not common:
            hypotheses.append({"id": h.id, "statement": h.statement, "result": "not tested"})
            continue
        # McNemar on every seed both models share; the largest (least favourable) p is used
        by_seed = {sd: mcnemar(preds[(a, sd)], preds[(b, sd)]) for sd in common}
        worst = max(by_seed, key=lambda sd: by_seed[sd]["p_value"])
        hypotheses.append(
            {
                "id": h.id,
                "statement": h.statement,
                "metric": h.primary_metric,
                "a": a,
                "b": b,
                "a_mean": models[a]["aggregate"][h.primary_metric]["mean"],
                "b_mean": models[b]["aggregate"][h.primary_metric]["mean"],
                "mcnemar_seed": worst,
                "mcnemar_seeds": common,
                "mcnemar_p_by_seed": {str(sd): t["p_value"] for sd, t in by_seed.items()},
                "mcnemar": by_seed[worst],
                # primary test on the hypothesis's own metric (McNemar only checks paired
                # per-row correctness, which is not the same as a balanced-accuracy difference)
                "test": paired_permutation_test(
                    [preds[(a, sd)] for sd in common],
                    [preds[(b, sd)] for sd in common],
                    h.primary_metric,
                    n_perm=1000,
                ),
                "effect": bootstrap_diff_seeds(
                    [preds[(a, sd)] for sd in common],
                    [preds[(b, sd)] for sd in common],
                    h.primary_metric,
                    n_boot=300,
                ),
            }
        )
    # Holm-Bonferroni correction across all tested hypotheses of the protocol
    tested = [hr for hr in hypotheses if "mcnemar" in hr]
    if holm_adjust:
        for hr, p_adj in zip(tested, holm([hr["test"]["p_value"] for hr in tested]), strict=True):
            hr["p_holm"] = p_adj
            hr["result"] = (
                "supported" if hr["a_mean"] > hr["b_mean"] and p_adj < 0.05 else "not supported"
            )
    return hypotheses


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


def _with_exact_intervals(per_class: dict[str, Any]) -> dict[str, Any]:
    """Add a Clopper-Pearson 95% interval to each class's recall (exact for tiny classes)."""
    out: dict[str, Any] = {}
    for c, v in per_class.items():
        v = dict(v)
        n = int(v.get("support", 0))
        k = round(float(v.get("recall", 0.0)) * n)
        v["recall_ci"] = list(clopper_pearson(k, n))
        out[c] = v
    return out


def _followup_comparison(
    s: Session, store: ArtifactStore, project_id: str, protocol: Protocol
) -> dict[str, Any] | None:
    """For a follow-up protocol: each model's metric means under the previous version, so the
    paper can show what the stricter experiment changed (feedback-controller cycle)."""
    f = protocol.followup_of
    if not f:
        return None
    prev = int(f["version"])
    runs = [
        r
        for r in s.scalars(
            select(RunRecord).where(
                RunRecord.project_id == project_id, RunRecord.status == "succeeded"
            )
        )
        if r.experiment_id.startswith(f"p{prev}-")
    ]
    by_model: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        path = store.path_for(s, project_id, r.receipt["artifacts"]["predictions.csv"])
        by_model.setdefault(r.model_name, []).append(compute_metrics(pd.read_csv(path)))
    previous = {
        name: {
            k: statistics.fmean(m[k] for m in ms)
            for k in ("macro_f1", "balanced_accuracy", "accuracy")
            if all(k in m for m in ms)
        }
        for name, ms in by_model.items()
    }
    return {**f, "previous_version": prev, "previous_means": previous}
