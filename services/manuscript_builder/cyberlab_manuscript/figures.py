"""Figures generated from the stored analysis (spec FR-009, Step 21).

Every figure is drawn by tested code from the analysis artifact, never by an LLM, and is
embedded in the manuscript as a PNG data URI so the Markdown, the dashboard view and the
print/PDF view all carry the same image. `extract_figures` turns them back into files for
the LaTeX bundle.

House style (one look for the whole paper): serif type that matches the LaTeX body text,
sizes for a two-column journal page (3.5 in single column, 7.16 in full width), one fixed
colour per model in every figure (Okabe-Ito, colour-blind safe), models ordered by score,
legends outside the data, 300 dpi.
"""

from __future__ import annotations

import base64
import io
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    "lightgbm": "LightGBM",
    "resnet_mlp": "ResNet-MLP",
    "ft_transformer": "FT-Transformer",
    "cnn1d": "1D-CNN",
    "tabnet": "TabNet",
    "tfidf_logreg": "TF-IDF logistic regression",
    "tfidf_linear_svm": "TF-IDF linear SVM",
    "macro_f1": "macro-F1",
    "balanced_accuracy": "balanced accuracy",
    "mcc": "MCC",
    "accuracy": "accuracy",
}
_DATA_URI = re.compile(r"!\[(?P<alt>[^\]]*)\]\(data:image/png;base64,(?P<b64>[A-Za-z0-9+/=]+)\)")

# Okabe-Ito: distinguishable with every common colour-vision deficiency and in greyscale print
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#882255", "#000000"]
INK = "#222222"
GRID = "#d9d9d9"
COLUMN = 3.5  # inches: one column of a two-column journal page
FULL = 7.16  # inches: full page width

STYLE: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
}


@contextmanager
def _style() -> Iterator[None]:
    with plt.rc_context(STYLE):  # type: ignore[arg-type]
        yield


def _png(fig: Any) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _primary_metric(analysis: dict[str, Any]) -> str:
    for h in analysis.get("hypotheses", []):
        if h.get("metric"):
            return str(h["metric"])
    return "macro_f1"


def _name(model: str) -> str:
    return PRETTY.get(model, model.replace("_", " "))


def model_colors(analysis: dict[str, Any]) -> dict[str, str]:
    """One colour per model, fixed by the protocol's model order, used in every figure."""
    models = list(analysis.get("models", {}))
    bench = analysis.get("benchmark") or {}
    for v in (bench.get("per_dataset") or {}).values():
        models += [m for m in (v.get("models") or {}) if m not in models]
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _mean(m: dict[str, Any], metric: str) -> float | None:
    v = (m.get("aggregate", {}).get(metric) or {}).get("mean")
    return float(v) if v is not None else None


def _ranked(analysis: dict[str, Any], metric: str) -> list[str]:
    models = analysis.get("models", {})
    scored = [n for n in models if _mean(models[n], metric) is not None]
    return sorted(scored, key=lambda n: -(_mean(models[n], metric) or 0.0))


def _text_on(rgba: tuple[float, ...]) -> str:
    """Black or white text, whichever reads better on this cell colour."""
    r, g, b = rgba[:3]
    return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.55 else INK


def _classes_by_support(analysis: dict[str, Any]) -> tuple[list[str], dict[str, int]]:
    models = analysis.get("models", {})
    support: dict[str, int] = {}
    for m in models.values():
        for c, v in (m.get("per_class") or {}).items():
            support[c] = max(support.get(c, 0), int(v.get("support", 0) or 0))
    return sorted(support, key=lambda c: -support[c]), support


# --------------------------------------------------------------------------- figures


def ci_figure(analysis: dict[str, Any]) -> str | None:
    """Primary metric per model with its 95% bootstrap interval, best model on top."""
    metric = _primary_metric(analysis)
    models = analysis.get("models", {})
    order = _ranked(analysis, metric)
    if not order:
        return None
    colors = model_colors(analysis)
    with _style():
        fig, ax = plt.subplots(figsize=(COLUMN, 0.3 * len(order) + 0.55))
        right = 0.0
        for i, n in enumerate(order):
            mean = _mean(models[n], metric) or 0.0
            ci = models[n].get(f"{metric}_ci") or {}
            lo, hi = ci.get("low"), ci.get("high")
            if lo is not None and hi is not None:
                ax.plot([lo, hi], [i, i], color=colors[n], lw=1.6, solid_capstyle="butt")
                ax.plot([lo, lo], [i - 0.12, i + 0.12], color=colors[n], lw=0.8)
                ax.plot([hi, hi], [i - 0.12, i + 0.12], color=colors[n], lw=0.8)
                right = max(right, hi)
            ax.plot(mean, i, "o", ms=4.5, color=colors[n], mec="white", mew=0.6, zorder=3)
            ax.annotate(
                f"{mean:.3f}",
                (mean, i),
                xytext=(0, 5.5),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
            )
        ax.set_yticks(range(len(order)), [_name(n) for n in order])
        ax.set_ylim(len(order) - 0.5, -0.7)
        ax.set_xlabel(f"{_name(metric)} (mean over seeds, 95% bootstrap CI)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        return _png(fig)


def recall_figure(analysis: dict[str, Any]) -> str | None:
    """Recall per class and model as an annotated heatmap: classes by test size (largest
    first), models by the primary metric (best on top). Easier to read than grouped bars."""
    metric = _primary_metric(analysis)
    models = analysis.get("models", {})
    order = _ranked(analysis, metric) or list(models)
    classes, support = _classes_by_support(analysis)
    if not classes or not order:
        return None
    mat = np.array(
        [
            [
                ((models[n].get("per_class") or {}).get(c) or {}).get("recall", np.nan)
                for c in classes
            ]
            for n in order
        ],
        dtype=float,
    )
    with _style():
        width = min(FULL, 1.3 + 0.52 * len(classes))
        fig, ax = plt.subplots(figsize=(width, 0.3 * len(order) + 1.0))
        cmap = plt.get_cmap("Blues")
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=_text_on(cmap(mat[i, j])),
                    )
        ax.set_xticks(
            range(len(classes)),
            [f"{c} ({support[c]:,})" for c in classes],
            rotation=35,
            ha="right",
            rotation_mode="anchor",
            fontsize=6.5,
        )
        ax.set_yticks(range(len(order)), [_name(n) for n in order])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
        cb.set_label("recall", fontsize=7)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.5, width=0.4)
        return _png(fig)


def importance_figure(analysis: dict[str, Any]) -> str | None:
    """Permutation feature importance (drop in macro-F1 when a feature is shuffled) for the
    features that matter most to any model; models ordered by score."""
    metric = _primary_metric(analysis)
    models = analysis.get("models", {})
    order = [n for n in (_ranked(analysis, metric) or list(models))
             if models[n].get("permutation_importance")]  # fmt: skip
    if not order:
        return None
    imps = {n: models[n]["permutation_importance"] for n in order}
    feats: list[str] = []
    for n in order:
        feats += [f for f in list(imps[n])[:6] if f not in feats]
    feats = sorted(feats, key=lambda f: -float(np.mean([imps[n].get(f, 0.0) for n in order])))[:12]
    mat = np.array([[float(imps[n].get(f, 0.0)) for n in order] for f in feats])
    with _style():
        fig, ax = plt.subplots(figsize=(COLUMN, 0.24 * len(feats) + 1.1))
        cmap = plt.get_cmap("Blues")
        top = max(float(mat.max()), 1e-6)
        im = ax.imshow(np.clip(mat, 0, None), cmap=cmap, vmin=0, vmax=top, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=_text_on(cmap(max(mat[i, j], 0) / top)),
                )
        ax.set_xticks(range(len(order)), [_name(n) for n in order], rotation=30, ha="right",
                      rotation_mode="anchor")  # fmt: skip
        ax.set_yticks(range(len(feats)), feats, fontsize=6.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label(f"drop in {_name(metric)}", fontsize=7)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.5, width=0.4)
        return _png(fig)


def shap_figure(analysis: dict[str, Any]) -> str | None:
    """Mean |SHAP| of the best boosted-tree model, split by class (stacked bars), for its
    most important features."""
    metric = _primary_metric(analysis)
    models = analysis.get("models", {})
    lead = next((n for n in _ranked(analysis, metric) if models[n].get("shap_importance")), None)
    if lead is None:
        return None
    overall = models[lead]["shap_importance"]
    feats = list(overall)[:10][::-1]
    per = models[lead].get("shap_per_class") or {}
    with _style():
        fig, ax = plt.subplots(figsize=(COLUMN, 0.2 * len(feats) + 0.9))
        if per:
            # per-class values are kept for each class's own top features; others count as 0
            left = np.zeros(len(feats))
            for k, (cls, vals) in enumerate(per.items()):
                widths = np.array([float(vals.get(f, 0.0)) for f in feats])
                if widths.sum() == 0:
                    continue
                ax.barh(feats, widths, left=left, color=PALETTE[k % len(PALETTE)], label=cls,
                        height=0.7)  # fmt: skip
                left += widths
            rest = np.array([float(overall[f]) for f in feats]) - left
            ax.barh(feats, np.clip(rest, 0, None), left=left, color="#cccccc", height=0.7,
                    label="smaller contributions")  # fmt: skip
            ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=1)
        else:
            ax.barh(feats, [float(overall[f]) for f in feats], color=PALETTE[0], height=0.7)
        ax.set_xlabel(f"mean |SHAP value| of {_name(lead)} (summed over classes)")
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, labelsize=6.5)
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
    labels = list(labels or [str(i) for i in range(matrix.shape[0])])
    # classes by test size, largest first (the same order as the per-class figure)
    idx = sorted(range(len(labels)), key=lambda i: -matrix[i].sum())
    matrix = matrix[np.ix_(idx, idx)]
    labels = [labels[i] for i in idx]
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    with _style():
        side = min(FULL * 0.62, 1.4 + 0.33 * len(labels))
        fig, ax = plt.subplots(figsize=(side + 0.5, side))
        cmap = plt.get_cmap("Blues")
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                if norm[i, j] >= 0.01:
                    ax.text(
                        j,
                        i,
                        f"{norm[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.5 if len(labels) > 8 else 6.5,
                        color=_text_on(cmap(norm[i, j])),
                    )
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(length=0, labelsize=6.5)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("share of the true class", fontsize=7)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.5, width=0.4)
        return _png(fig)


def _dataset_name(dataset_id: str) -> str:
    from cyberlab_data import get_adapter

    return str(get_adapter(dataset_id).display)


def benchmark_heatmap(analysis: dict[str, Any]) -> str | None:
    """Score per model and dataset. The colour is the model's rank on that dataset, so the
    figure shows which model leads where (raw scores would mostly show how hard each dataset
    is); the best score per dataset is in bold."""
    b = analysis.get("benchmark") or {}
    scores = b.get("scores") or {}
    if len(scores) < 2:
        return None
    datasets = [d for d in b["datasets"] if scores.get(d)]
    avg = (b.get("ranking") or {}).get("average_rank") or {}
    models = sorted({m for d in datasets for m in scores[d]}, key=lambda m: (avg.get(m, 99), m))
    mat = np.array([[scores[d].get(m, np.nan) for d in datasets] for m in models])
    rank = np.full_like(mat, np.nan)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        ok = ~np.isnan(col)
        rank[ok, j] = (-col[ok]).argsort().argsort() + 1
    k = len(models)
    with _style():
        fig, ax = plt.subplots(figsize=(FULL * 0.8, 0.3 * k + 1.0))
        cmap = plt.get_cmap("Blues_r", k + 2)
        ax.imshow(rank, cmap=cmap, vmin=0.5, vmax=k + 2.5, aspect="auto")
        for i in range(k):
            for j in range(len(datasets)):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.4f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold" if rank[i, j] == 1 else "normal",
                    color=_text_on(cmap((rank[i, j] - 0.5) / (k + 2))),
                )
        ax.set_xticks(range(len(datasets)), [_dataset_name(d) for d in datasets])
        ax.xaxis.tick_top()
        ax.set_yticks(
            range(k),
            [f"{_name(m)} ({avg[m]:.2f})" if m in avg else _name(m) for m in models],
        )
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(datasets)), minor=True)
        ax.set_yticks(np.arange(-0.5, k), minor=True)
        ax.grid(which="minor", color="white", lw=1.2)
        ax.tick_params(which="minor", length=0)
        return _png(fig)


def transfer_figure(analysis: dict[str, Any]) -> str | None:
    """One small matrix per model: rows = training dataset, columns = test dataset, cells =
    attack-vs-BENIGN macro-F1 (mean over seeds); the diagonal is in-domain."""
    t = analysis.get("transfer") or {}
    models = [n for n, v in (t.get("models") or {}).items() if v.get("matrix")]
    datasets = list(t.get("datasets") or [])
    if not models or len(datasets) < 2:
        return None
    models.sort(key=lambda n: -(t["models"][n].get("cross_domain") or 0.0))
    short = [_dataset_name(d).replace("-v2", "").replace("NF-", "") for d in datasets]
    k = len(datasets)
    ncols = min(len(models), 3)
    nrows = -(-len(models) // ncols)
    cmap = plt.get_cmap("Blues")
    with _style():
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(FULL, (0.36 * k + 0.9) * nrows), squeeze=False
        )
        for ax in axes.flat[len(models) :]:
            ax.axis("off")
        for ax, m in zip(axes.flat, models, strict=False):
            mat = np.full((k, k), np.nan)
            for i, src in enumerate(datasets):
                for j, tgt in enumerate(datasets):
                    cell = ((t["models"][m]["matrix"].get(src) or {}).get(tgt) or {}).get(
                        "macro_f1"
                    ) or {}
                    if "mean" in cell:
                        mat[i, j] = cell["mean"]
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1)
            for i in range(k):
                for j in range(k):
                    if not np.isnan(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6.5,
                            fontweight="bold" if i == j else "normal",
                            color=_text_on(cmap(mat[i, j])),
                        )
            ax.set_title(_name(m), fontsize=8)
            ax.set_xticks(range(k), short, rotation=35, ha="right", rotation_mode="anchor",
                          fontsize=6)  # fmt: skip
            ax.set_yticks(range(k), short, fontsize=6)
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
        fig.supxlabel("tested on", fontsize=7)
        fig.supylabel("trained on", fontsize=7)
        fig.tight_layout()
        return _png(fig)


def _cliques(order: list[str], ranks: dict[str, float], cd: float) -> list[tuple[int, int]]:
    """Maximal runs of models (in rank order) whose average ranks differ by at most CD."""
    out: list[tuple[int, int]] = []
    for i in range(len(order)):
        j = i
        while j + 1 < len(order) and ranks[order[j + 1]] - ranks[order[i]] <= cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in out):
            out.append((i, j))
    return out


def rank_figure(analysis: dict[str, Any]) -> str | None:
    """Critical-difference diagram (Demšar, 2006): average ranks on an axis, models joined by
    a bar do not differ significantly (Nemenyi, alpha 0.05)."""
    r = (analysis.get("benchmark") or {}).get("ranking") or {}
    if not r.get("applicable"):
        return None
    ranks = {m: float(v) for m, v in r["average_rank"].items()}
    order = sorted(ranks, key=ranks.__getitem__)
    k = len(order)
    cd = float(r["critical_difference"])
    colors = model_colors(analysis)
    half = (k + 1) // 2
    with _style():
        rows = max(half, k - half)
        fig, ax = plt.subplots(figsize=(COLUMN, 0.95 + 0.2 * rows))
        # labels sit inside the axes, so the printed figure is exactly one column wide
        ax.set_xlim(-0.45 * k - 0.2, 1.45 * k + 0.2)
        ax.set_ylim(-(rows + 1.6) * 0.32, 1.0)
        ax.axis("off")
        # rank axis (1 = best on the left)
        ax.plot([1, k], [0, 0], color=INK, lw=0.7)
        for t in range(1, k + 1):
            ax.plot([t, t], [0, 0.08], color=INK, lw=0.7)
            ax.text(t, 0.14, str(t), ha="center", va="bottom", fontsize=7)
        # critical difference bar
        ax.plot([1, 1 + cd], [0.62, 0.62], color=INK, lw=0.9)
        for x in (1, 1 + cd):
            ax.plot([x, x], [0.57, 0.67], color=INK, lw=0.9)
        ax.text(1 + cd / 2, 0.7, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
        # models: left half labelled on the left, right half on the right
        for i, m in enumerate(order):
            row = i if i < half else k - 1 - i
            y = -(row + 1) * 0.32 - 0.25
            x = ranks[m]
            left = i < half
            edge = 0.75 if left else k + 0.25
            ax.plot([x, x, edge], [0, y, y], color=colors.get(m, INK), lw=0.8)
            ax.plot(x, 0, "o", ms=3, color=colors.get(m, INK), zorder=3)
            ax.text(
                edge - 0.05 if left else edge + 0.05,
                y,
                f"{_name(m)} ({x:.2f})" if left else f"({x:.2f}) {_name(m)}",
                ha="right" if left else "left",
                va="center",
                fontsize=7,
            )
        # groups that do not differ significantly
        for n, (a, b2) in enumerate(_cliques(order, ranks, cd)):
            y = -0.1 - 0.07 * n
            ax.plot([ranks[order[a]] - 0.04, ranks[order[b2]] + 0.04], [y, y], color=INK, lw=2.2)
        return _png(fig)


# --------------------------------------------------------------------------- assembly


def figures_markdown(analysis: dict[str, Any], best: str | None) -> str:
    """The Figures sub-section of Results (empty string if nothing can be drawn).

    The figure set follows the study: a benchmark study adds the across-dataset comparison
    (a rank-coloured score map and a critical difference diagram)."""
    metric = _name(_primary_metric(analysis))
    b = analysis.get("benchmark") or {}
    main = ""
    if b.get("datasets"):
        try:
            main = f" on {_dataset_name(b['datasets'][0])}"
        except Exception:  # noqa: BLE001 - an unknown dataset id only loses the name
            main = ""
    n_ds = len(b.get("datasets") or [])
    bench: list[tuple[Callable[[], str | None], str]] = [
        (
            lambda: rank_figure(analysis),
            f"Critical difference diagram of the average ranks on {metric} across the {n_ds} "
            "datasets (1 = best). Models joined by a horizontal bar do not differ "
            "significantly (Friedman test with Nemenyi post-hoc, alpha = 0.05).",
        ),
        (
            lambda: benchmark_heatmap(analysis),
            f"{metric[0].upper() + metric[1:]} per model and dataset (mean over seeds). Cell "
            "colour shows the model's rank on that dataset (darker = better), the best score on "
            "each dataset is in bold, and the average rank is given after each model's name.",
        ),
    ]
    single: list[tuple[Callable[[], str | None], str]] = [
        (
            lambda: ci_figure(analysis),
            f"{metric[0].upper() + metric[1:]} of each model{main} (mean over seeds) with 95% "
            "bootstrap confidence intervals over the test rows; models are ordered by score.",
        ),
        (
            lambda: recall_figure(analysis),
            f"Recall per class and model{main} (first seed); classes are ordered by test size "
            "(test rows in brackets), models by score.",
        ),
        (
            (lambda: confusion_figure(analysis, best)) if best else (lambda: None),
            f"Row-normalised confusion matrix of {_name(best or '')}{main} (first seed); cell "
            "values are the share of each true class assigned to each predicted class.",
        ),
    ]
    parts = []
    n = 0
    explain: list[tuple[Callable[[], str | None], str]] = [
        (
            lambda: importance_figure(analysis),
            f"Permutation feature importance{main}: the drop in {metric} when one feature's "
            "values are shuffled on the test rows (mean over seeds), for the features that "
            "matter most to any model.",
        ),
        (
            lambda: shap_figure(analysis),
            "SHAP explanation of the best boosted-tree model"
            + main
            + ": mean absolute SHAP value of its ten most influential features (class-balanced "
            "sample of test rows), split by the class whose prediction the feature drives.",
        ),
    ]
    # numbered in the order the Results text first cites them: model comparison, classes,
    # features, across datasets (scores, then ranks), and the confusion matrix
    xfer: list[tuple[Callable[[], str | None], str]] = [
        (
            lambda: transfer_figure(analysis),
            "Generalization of each model, trained on the row dataset and tested on the column "
            "dataset (attack versus BENIGN, macro-F1, mean over seeds); the diagonal (bold) is "
            "in-domain.",
        )
    ]
    order = single[:2] + explain + (bench[::-1] if n_ds > 1 else []) + xfer + single[2:]
    for draw, caption in order:
        b64 = draw()
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
