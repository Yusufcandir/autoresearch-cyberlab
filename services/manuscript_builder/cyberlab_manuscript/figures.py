"""Figures generated from the stored analysis (spec FR-009, Step 21).

Every figure is drawn by tested code from the analysis artifact, never by an LLM, and is
embedded in the manuscript as a PNG data URI so the Markdown, the dashboard view and the
print/PDF view all carry the same image. `extract_figures` turns them back into files for
the LaTeX bundle.
"""

from __future__ import annotations

import base64
import io
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display on the server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PRETTY = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "macro_f1": "macro-F1",
    "balanced_accuracy": "balanced accuracy",
    "mcc": "MCC",
    "accuracy": "accuracy",
}
_DATA_URI = re.compile(r"!\[(?P<alt>[^\]]*)\]\(data:image/png;base64,(?P<b64>[A-Za-z0-9+/=]+)\)")


def _png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _primary_metric(analysis: dict[str, Any]) -> str:
    for h in analysis.get("hypotheses", []):
        if h.get("metric"):
            return str(h["metric"])
    return "macro_f1"


def ci_figure(analysis: dict[str, Any]) -> str | None:
    metric = _primary_metric(analysis)
    rows = []
    for name, m in analysis.get("models", {}).items():
        mean = (m.get("aggregate", {}).get(metric) or {}).get("mean")
        ci = m.get(f"{metric}_ci") or {}
        if mean is not None:
            rows.append((PRETTY.get(name, name), mean, ci.get("low"), ci.get("high")))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6, 0.6 * len(rows) + 1))
    for i, (_label, mean, lo, hi) in enumerate(rows):
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [i, i], color="#4b5563", lw=2)
        ax.plot(mean, i, "o", color="#4f46e5")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(f"{PRETTY.get(metric, metric)} (mean over seeds, 95% bootstrap CI)")
    ax.grid(axis="x", alpha=0.3)
    return _png(fig)


def recall_figure(analysis: dict[str, Any]) -> str | None:
    models = analysis.get("models", {})
    classes = sorted(
        {c for m in models.values() for c in (m.get("per_class") or {})},
        key=lambda c: (
            -max((m.get("per_class") or {}).get(c, {}).get("support", 0) for m in models.values())
        ),
    )
    if not classes:
        return None
    fig, ax = plt.subplots(figsize=(8, 3.5))
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(classes))
    for i, (name, m) in enumerate(models.items()):
        pc = m.get("per_class") or {}
        ax.bar(
            x + i * width,
            [pc.get(c, {}).get("recall", 0.0) for c in classes],
            width,
            label=PRETTY.get(name, name),
        )
    ax.set_xticks(x + width * (len(models) - 1) / 2, classes, rotation=30, ha="right")
    ax.set_ylabel("recall (first seed)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=min(len(models), 4))
    ax.grid(axis="y", alpha=0.3)
    return _png(fig)


def confusion_figure(analysis: dict[str, Any], model: str) -> str | None:
    m = analysis.get("models", {}).get(model) or {}
    cm = m.get("confusion_matrix")
    if not cm:
        return None
    labels = cm.get("labels") if isinstance(cm, dict) else None
    matrix = np.array(cm.get("matrix") if isinstance(cm, dict) else cm, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        return None
    labels = labels or [str(i) for i in range(matrix.shape[0])]
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            if norm[i, j] >= 0.005:
                ax.text(
                    j,
                    i,
                    f"{norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if norm[i, j] > 0.5 else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, label="share of true class")
    return _png(fig)


def figures_markdown(analysis: dict[str, Any], best: str | None) -> str:
    """The Figures sub-section of Results (empty string if nothing can be drawn)."""
    metric = PRETTY.get(_primary_metric(analysis), "macro-F1")
    parts = []
    n = 0
    for b64, caption in (
        (ci_figure(analysis), f"{metric} per model with 95% bootstrap confidence intervals."),
        (recall_figure(analysis), "Recall per class and model (first seed), largest class first."),
        (
            confusion_figure(analysis, best) if best else None,
            f"Row-normalised confusion matrix of {PRETTY.get(best or '', best)} (first seed).",
        ),
    ):
        if b64:
            n += 1
            parts.append(f"![Figure {n}](data:image/png;base64,{b64})\n\n*Figure {n}. {caption}*")
    return "\n\n".join(parts)


def extract_figures(markdown: str) -> tuple[str, dict[str, bytes]]:
    """Replace embedded PNGs by file names (for LaTeX/zip export)."""
    files: dict[str, bytes] = {}

    def sub(m: re.Match[str]) -> str:
        name = f"figure{len(files) + 1}.png"
        files[name] = base64.b64decode(m.group("b64"))
        return f"![{m.group('alt')}]({name})"

    return _DATA_URI.sub(sub, markdown), files
