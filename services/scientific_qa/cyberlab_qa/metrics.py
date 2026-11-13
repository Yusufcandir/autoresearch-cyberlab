"""Metrics and statistics computed only from stored predictions (AT-14).

Nothing here calls an LLM. Every number that reaches a manuscript comes from these functions
applied to a hashed `predictions.csv` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm


@dataclass(frozen=True)
class QAFlag:
    severity: str  # info | warning | blocker
    code: str
    message: str


def compute_metrics(preds: pd.DataFrame) -> dict[str, Any]:
    y, p = preds["y_true"].astype(str), preds["y_pred"].astype(str)
    labels = sorted(set(y) | set(p))
    prf = skm.precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    out: dict[str, Any] = {
        "n": len(preds),
        "accuracy": float(skm.accuracy_score(y, p)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, p)),
        "macro_f1": float(skm.f1_score(y, p, average="macro", zero_division=0)),
        "weighted_f1": float(skm.f1_score(y, p, average="weighted", zero_division=0)),
        "mcc": float(skm.matthews_corrcoef(y, p)),
        "per_class": {
            c: {
                "precision": float(prf[0][i]),
                "recall": float(prf[1][i]),
                "f1": float(prf[2][i]),
                "support": int(prf[3][i]),
            }
            for i, c in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": labels,
            "matrix": skm.confusion_matrix(y, p, labels=labels).tolist(),
        },
    }
    prob_cols = [f"p_{c}" for c in labels if f"p_{c}" in preds.columns]
    if len(prob_cols) == len(labels) and len(labels) > 1:
        try:
            probs = preds[prob_cols].to_numpy()
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
            if len(labels) == 2:
                out["roc_auc"] = float(skm.roc_auc_score(y == labels[1], probs[:, 1]))
            else:
                out["roc_auc_ovr_macro"] = float(
                    skm.roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=labels)
                )
        except ValueError:
            pass
    if "BENIGN" in labels:
        attack_true = y != "BENIGN"
        attack_pred = p != "BENIGN"
        benign = ~attack_true
        out["false_positive_rate"] = float((attack_pred & benign).sum() / max(benign.sum(), 1))
        out["detection_rate"] = float((attack_pred & attack_true).sum() / max(attack_true.sum(), 1))
    return out


_METRIC_FN = {
    "macro_f1": lambda y, p: skm.f1_score(y, p, average="macro", zero_division=0),
    "accuracy": skm.accuracy_score,
    "balanced_accuracy": skm.balanced_accuracy_score,
    "mcc": skm.matthews_corrcoef,
}


def bootstrap_ci(
    preds: pd.DataFrame, metric: str = "macro_f1", n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, float]:
    """Percentile bootstrap CI over test rows (stratified by true class)."""
    fn = _METRIC_FN[metric]
    y = preds["y_true"].astype(str).to_numpy()
    p = preds["y_pred"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        vals[b] = fn(y[idx], p[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"metric": metric, "point": float(fn(y, p)), "low": float(lo), "high": float(hi), "n_boot": n_boot}


def mcnemar(preds_a: pd.DataFrame, preds_b: pd.DataFrame) -> dict[str, float]:
    """Exact McNemar test on per-row correctness of two models over the same test rows."""
    a = preds_a.sort_values("row").reset_index(drop=True)
    b = preds_b.sort_values("row").reset_index(drop=True)
    if not (a["y_true"].astype(str) == b["y_true"].astype(str)).all():
        raise ValueError("predictions are not over the same test rows")
    ca = a["y_true"].astype(str) == a["y_pred"].astype(str)
    cb = b["y_true"].astype(str) == b["y_pred"].astype(str)
    n01 = int((ca & ~cb).sum())  # a right, b wrong
    n10 = int((~ca & cb).sum())
    n = n01 + n10
    pval = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": pval}


def recompute_matches(stored: dict[str, Any], preds: pd.DataFrame, tol: float = 1e-9) -> bool:
    fresh = compute_metrics(preds)
    keys = ["accuracy", "macro_f1", "balanced_accuracy", "mcc"]
    return all(abs(float(stored[k]) - fresh[k]) <= tol for k in keys if k in stored)


def leakage_audit(metrics_by_model: dict[str, dict[str, Any]], split_overlap_rows: int) -> list[QAFlag]:
    """Flags that must be resolved before results are called Verified (spec 13.2)."""
    flags: list[QAFlag] = []
    if split_overlap_rows:
        flags.append(
            QAFlag("blocker", "SPLIT_OVERLAP", f"{split_overlap_rows} test rows also appear in train")
        )
    for name, m in metrics_by_model.items():
        if m.get("macro_f1", 0) >= 0.995:
            flags.append(
                QAFlag(
                    "warning",
                    "SUSPICIOUSLY_HIGH",
                    f"{name} macro-F1 {m['macro_f1']:.4f}: audit for leakage before claiming it",
                )
            )
        for cls, pc in m.get("per_class", {}).items():
            if pc["support"] < 30:
                flags.append(
                    QAFlag("info", "SMALL_CLASS", f"{name}: class {cls} has only {pc['support']} test rows")
                )
    return flags
