"""Scientific validation before any manuscript is written (professor's plan, Phase 1).

Independent, deterministic checks of the evidence; no LLM is asked whether results "look
right". Every check returns a structured result with one of four statuses:

    PASS           the check ran and found no problem
    WARNING        a limitation that the paper must state, or that needs attention
    BLOCK          the result is invalid or insufficiently supported: no manuscript
    NOT_EVALUATED  the evidence needed for the check does not exist (never counted as a pass)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cyberlab_qa.metrics import clopper_pearson

PASS, WARNING, BLOCK, NOT_EVALUATED = "PASS", "WARNING", "BLOCK", "NOT_EVALUATED"
_ORDER = {PASS: 0, NOT_EVALUATED: 1, WARNING: 2, BLOCK: 3}


def _r(check: str, status: str, detail: str, **data: Any) -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail, **data}


def worst(results: list[dict[str, Any]]) -> str:
    return max((r["status"] for r in results), key=lambda s: _ORDER[s], default=NOT_EVALUATED)


# --------------------------------------------------------------------------- data


def validate_dataset_integrity(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return _r("dataset_integrity", NOT_EVALUATED, "no dataset manifest recorded")
    if manifest.get("status") != "verified":
        return _r("dataset_integrity", BLOCK, f"dataset blocked: {manifest.get('blocked_reasons')}")
    if not manifest.get("files") or not all(len(h) == 64 for h in manifest["files"].values()):
        return _r("dataset_integrity", BLOCK, "raw file hashes are missing or incomplete")
    if not manifest.get("license"):
        return _r("dataset_integrity", BLOCK, "no licence recorded")
    return _r(
        "dataset_integrity",
        PASS,
        f"{len(manifest['files'])} raw file(s) hashed, licence recorded, dataset verified",
    )


def validate_data_preprocessing(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not audit:
        return _r(
            "data_preprocessing",
            NOT_EVALUATED,
            "no stage-by-stage preprocessing audit (prepare the dataset again with this version)",
        )
    if not audit.get("consistent"):
        return _r("data_preprocessing", BLOCK, "; ".join(audit.get("problems", [])), audit=audit)
    removed = ", ".join(
        f"{s['stage']}: {s['removed_rows']:,}" for s in audit["stages"] if s["removed_rows"]
    )
    return _r(
        "data_preprocessing",
        PASS,
        f"{audit['raw_rows']:,} raw rows -> {audit['final_rows']:,} final rows; removed: "
        f"{removed or 'none'}; every stage balances (input = removed + remaining)",
        audit=audit,
    )


def _quantised_hash(df: pd.DataFrame, features: list[str], bins: int = 20) -> pd.Series:
    """Coarse fingerprint of a row: each numeric feature replaced by its quantile bin."""
    q = pd.DataFrame(index=df.index)
    for f in features:
        col = df[f]
        if f.startswith("text:"):
            q[f] = col.astype(str).str.lower().str.replace(r"[0-9]+", "0", regex=True)
        elif col.nunique() > bins:
            q[f] = pd.qcut(col.rank(method="first"), q=bins, labels=False)
        else:
            q[f] = col
    return pd.util.hash_pandas_object(q, index=False)


def detect_data_leakage(
    manifest: dict[str, Any] | None,
    notes: dict[str, Any] | None,
    split: dict[str, Any],
    prepared: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Seven leakage checks (Step 1.3)."""
    out: list[dict[str, Any]] = []
    manifest, notes = manifest or {}, notes or {}
    overlap = int(manifest.get("split_overlap_rows", 0))
    out.append(
        _r(
            "exact_feature_overlap",
            BLOCK if overlap else PASS,
            f"{overlap} test rows have a training row with identical features",
        )
    )
    features = list(manifest.get("features") or [])
    if prepared is not None and features:
        h = _quantised_hash(prepared, features)
        train = set(h[prepared["split"] == "train"])
        test_h = h[prepared["split"] == "test"]
        share = float(test_h.isin(train).mean()) if len(test_h) else 0.0
        status = WARNING if share >= 0.5 else PASS
        out.append(
            _r(
                "near_duplicate_flows",
                status,
                f"{share:.1%} of test rows fall into the same coarse feature cell (20 quantile bins "
                "per feature) as a training row; a high share means the test set is close to the "
                "training data and scores may be optimistic",
                share=share,
            )
        )
    else:
        out.append(_r("near_duplicate_flows", NOT_EVALUATED, "prepared data not available"))
    out.append(
        _r(
            "session_identifiers",
            NOT_EVALUATED,
            "the dataset has no session identifiers, so session-level leakage could not be "
            "fully evaluated",
        )
    )
    strategy = split.get("strategy", "stratified_dedup")
    group_col = (
        "Day"
        if prepared is not None and "Day" in prepared.columns
        else "Group"
        if prepared is not None and "Group" in prepared.columns
        else "Part"
        if prepared is not None and "Part" in prepared.columns
        else None
    )
    if group_col is None or prepared is None:
        out.append(_r("capture_groups", NOT_EVALUATED, "no capture-day, scenario or host groups"))
    elif strategy in ("by_day", "by_group", "grouped", "official", "cross_dataset"):
        groups = prepared[group_col].astype(str)
        shared = set(groups[prepared["split"] == "train"]) & set(
            groups[prepared["split"] == "test"]
        )
        out.append(
            _r(
                "capture_groups",
                PASS if not shared or strategy == "official" else BLOCK,
                f"split by {group_col}: "
                + (
                    "no group is in both training and test"
                    if not shared
                    else f"{len(shared)} shared groups"
                ),
            )
        )
    else:
        alt = {"Day": "by_day", "Group": "by_group or grouped", "Part": "official"}[group_col]
        out.append(
            _r(
                "capture_groups",
                WARNING,
                f"the random split puts rows of the same {group_col.lower()} in training and "
                f"test; a {alt} split is available and gives a harder, more realistic test",
                recommended_split=alt,
            )
        )
    shortcuts = [
        s for s in notes.get("single_feature_shortcuts", []) if s.get("purity", 0) >= 0.999
    ]
    out.append(
        _r(
            "label_revealing_features",
            WARNING if shortcuts else PASS,
            (
                "features whose value alone decides the class on the training rows: "
                + ", ".join(f"{s['feature']} ({s['purity']:.3f})" for s in shortcuts)
            )
            if shortcuts
            else "no single feature decides the class on its own",
            features=shortcuts,
        )
    )
    if "one_hot_vocabulary" in notes:
        out.append(
            _r(
                "preprocessing_fitted_on_test",
                WARNING,
                "the one-hot category list was taken from all rows (no labels used); fitting it "
                "on the training rows only would be stricter",
            )
        )
    else:
        out.append(
            _r(
                "preprocessing_fitted_on_test",
                PASS,
                "scalers, encoders and text vectorisers are fitted inside each model on the "
                "training rows only; categorical vocabularies come from the training file",
            )
        )
    if group_col == "Day" and strategy != "by_day":
        out.append(
            _r(
                "temporal_leakage",
                WARNING,
                "capture days are mixed across training and test; a by-day split tests on "
                "traffic from days the models never saw",
            )
        )
    elif group_col == "Day":
        out.append(_r("temporal_leakage", PASS, "test days are disjoint from training days"))
    else:
        out.append(
            _r(
                "temporal_leakage",
                NOT_EVALUATED,
                "no timestamps or capture order are available in the prepared data",
            )
        )
    return out


# --------------------------------------------------------------------------- experiments


def validate_experiment_configuration(
    protocol: dict[str, Any], runs: list[dict[str, Any]], analysis: dict[str, Any]
) -> dict[str, Any]:
    planned = {
        (m["name"], s) for m in protocol.get("models", []) for s in protocol.get("seeds", [])
    }
    done = {(r["model"], r["seed"]) for r in runs if r["status"] == "succeeded"}
    failed = sorted(f"{m} seed {s}" for (m, s) in planned - done)
    tuning = [
        n
        for n, m in analysis.get("models", {}).items()
        if (m.get("tuning") or {}).get("split") not in (None, "validation")
    ]
    if tuning:
        return _r(
            "experiment_configuration", BLOCK, f"tuning did not use the validation split: {tuning}"
        )
    if failed:
        return _r(
            "experiment_configuration",
            WARNING,
            f"{len(failed)} planned run(s) did not succeed and are recorded, not dropped: "
            + ", ".join(failed[:10]),
            failed=failed,
        )
    return _r(
        "experiment_configuration",
        PASS,
        f"all {len(planned)} planned runs succeeded with the approved protocol; the test split "
        "was used once per run",
    )


def recalculate_metrics(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("metrics_recomputed_match"):
        return _r(
            "recalculated_metrics",
            PASS,
            "every stored metric equals its recomputation from the predictions",
        )
    return _r("recalculated_metrics", BLOCK, "stored metrics differ from recomputation")


def validate_statistical_tests(
    analysis: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    issues, notes = [], []
    for h in analysis.get("hypotheses", []):
        if "mcnemar" not in h:
            issues.append(f"{h['id']}: not tested")
            continue
        test = h.get("test") or {}
        metric = h.get("metric")
        if not test:
            if metric != "accuracy":
                issues.append(
                    f"{h['id']}: McNemar tests paired per-row correctness, not a difference in "
                    f"{metric}; recompute the statistics (Rewrite draft)"
                )
            continue
        if test.get("metric") != metric:
            issues.append(f"{h['id']}: the test used {test.get('metric')}, the hypothesis {metric}")
        if "p_holm" not in h:
            issues.append(f"{h['id']}: no multiple-comparison correction")
    n_seeds = len(protocol.get("seeds", []))
    if n_seeds < 3:
        notes.append(
            f"only {n_seeds} seed(s): model-initialisation variability is poorly estimated"
        )
    notes.append(
        "uncertainty sources reported separately: test-sample (bootstrap CI), model "
        "initialisation (sd over seeds); data-partition variability is NOT_EVALUATED (one split)"
    )
    if issues:
        return _r("statistical_tests", BLOCK, "; ".join(issues), notes=notes)
    return _r(
        "statistical_tests",
        WARNING if n_seeds < 3 else PASS,
        "each hypothesis is tested on its own metric with a paired permutation test and Holm "
        "correction; effect sizes have bootstrap intervals",
        notes=notes,
    )


def rare_class_report(analysis: dict[str, Any], min_support: int = 30) -> dict[str, Any]:
    """Per class and model: support, precision, recall, F1, false negatives, false positives,
    exact intervals, and what to do when a class is too small (Step 1.5)."""
    rows = []
    small: set[str] = set()
    for name, m in analysis.get("models", {}).items():
        for cls, pc in (m.get("per_class") or {}).items():
            n = int(pc.get("support", 0))
            tp = round(float(pc.get("recall", 0.0)) * n)
            prec = float(pc.get("precision", 0.0))
            fp = round(tp / prec - tp) if prec > 0 else None
            rows.append(
                {
                    "model": name,
                    "class": cls,
                    "support": n,
                    "precision": pc.get("precision"),
                    "recall": pc.get("recall"),
                    "f1": pc.get("f1"),
                    "true_positives": tp,
                    "false_negatives": n - tp,
                    "false_positives": fp,
                    "recall_ci_95": list(clopper_pearson(tp, n)),
                    "precision_ci_95": list(clopper_pearson(tp, tp + fp))
                    if fp is not None
                    else None,
                }
            )
            if n < min_support:
                small.add(cls)
    status = WARNING if small else PASS
    return {
        "check": "rare_class_reliability",
        "status": status,
        "min_test_support": min_support,
        "small_classes": sorted(small),
        "detail": (
            f"classes with fewer than {min_support} test rows: {', '.join(sorted(small))}. Their "
            "per-class scores rest on a handful of rows; one more or one fewer correct row moves "
            "recall by several points, and resampling the same rows does not create new evidence."
            if small
            else f"every class has at least {min_support} test rows"
        ),
        "recommendations": [
            "evaluate on additional data for the small classes",
            "repeat the evaluation with several carefully designed splits",
            "report class-specific exact intervals and interpret conservatively",
            "evaluate on a second dataset with more examples of these classes",
        ]
        if small
        else [],
        "per_class": rows,
    }


def validate_result_consistency(analysis: dict[str, Any]) -> dict[str, Any]:
    unstable = []
    for name, m in analysis.get("models", {}).items():
        for metric, v in m.get("aggregate", {}).items():
            if metric in ("macro_f1", "balanced_accuracy") and v.get("sd", 0) > 0.05:
                unstable.append(f"{name} {metric} sd {v['sd']:.3f}")
    flags = [
        f
        for f in analysis.get("qa_flags", [])
        if f.get("code") in ("SURPRISING_IMPROVEMENT", "SUSPICIOUSLY_HIGH")
    ]
    if unstable or flags:
        return _r(
            "result_consistency",
            WARNING,
            "; ".join(unstable + [f["message"] for f in flags]),
            unstable=unstable,
        )
    return _r(
        "result_consistency",
        PASS,
        "results are stable across seeds and plausible against the baselines",
    )


# --------------------------------------------------------------------------- report


def generate_validation_report(
    analysis: dict[str, Any],
    protocol: dict[str, Any],
    runs: list[dict[str, Any]],
    prepared_dir: Path | None,
) -> dict[str, Any]:
    manifest = notes = audit = None
    prepared = None
    if prepared_dir is not None and prepared_dir.exists():
        import json

        def load(name: str) -> Any:
            p = prepared_dir / name
            return json.loads(p.read_text()) if p.exists() else None

        manifest, notes, audit = (
            load("manifest.json"),
            load("preprocessing_notes.json"),
            load("data_preprocessing_audit.json"),
        )
        if (prepared_dir / "clean.parquet").exists():
            prepared = pd.read_parquet(prepared_dir / "clean.parquet")
    manifest = manifest or analysis.get("dataset")
    leakage = detect_data_leakage(manifest, notes, protocol.get("split", {}), prepared)
    stats = validate_statistical_tests(analysis, protocol)
    rare = rare_class_report(analysis, int(protocol.get("min_test_support") or 30))
    checks = [
        validate_dataset_integrity(manifest),
        validate_data_preprocessing(audit),
        *leakage,
        validate_experiment_configuration(protocol, runs, analysis),
        recalculate_metrics(analysis),
        stats,
        {k: v for k, v in rare.items() if k != "per_class"},
        validate_result_consistency(analysis),
    ]
    overall = worst(checks)
    return {
        "status": overall,
        "manuscript_allowed": overall != BLOCK,
        "checks": checks,
        "reports": {
            "data_preprocessing_audit": audit,
            "leakage_validation_report": {"status": worst(leakage), "checks": leakage},
            "statistical_validation_report": stats,
            "rare_class_reliability_report": rare,
        },
        "counts": {s: sum(1 for c in checks if c["status"] == s) for s in _ORDER},
        "_note": "NOT_EVALUATED is never treated as a pass",
        "groups": sorted(str(g) for g in (notes or {}).get("groups", [])),
    }
