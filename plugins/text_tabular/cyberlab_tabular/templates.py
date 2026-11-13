"""Trusted (T1) experiment templates.

A template trains on the train split, selects nothing on test, and writes raw predictions and
class probabilities for every test row. It does NOT compute the reported metrics: scientific QA
recomputes them from the stored predictions (AT-14). This is the only code that runs on the GPU
host, and it is platform-owned and reviewed; agents choose templates and parameters, they do
not write the code that executes.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler

from cyberlab_data import load_split
from cyberlab_schemas import ExperimentManifest

_N_JOBS = int(os.environ.get("CYBERLAB_N_JOBS", "-1"))

ALLOWED_PARAMS: dict[str, set[str]] = {
    "logistic_regression": {"C", "max_iter"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample"},
    "mlp": {"hidden_layer_sizes", "alpha", "max_iter"},
}


def _check_params(template: str, params: dict[str, Any]) -> None:
    extra = set(params) - ALLOWED_PARAMS[template]
    if extra:
        raise ValueError(f"parameters not allowed for {template}: {sorted(extra)}")


def _logreg(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=p.get("C", 1.0), max_iter=p.get("max_iter", 400), random_state=seed, n_jobs=_N_JOBS
        ),
    )


def _rf(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    return RandomForestClassifier(
        n_estimators=p.get("n_estimators", 200),
        max_depth=p.get("max_depth"),
        min_samples_leaf=p.get("min_samples_leaf", 1),
        class_weight="balanced_subsample",
        n_jobs=_N_JOBS,
        random_state=seed,
    )


def _xgb(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=p.get("n_estimators", 300),
        max_depth=p.get("max_depth", 8),
        learning_rate=p.get("learning_rate", 0.1),
        subsample=p.get("subsample", 0.9),
        tree_method="hist",
        device="cuda" if gpu else "cpu",
        random_state=seed,
        n_jobs=_N_JOBS,
        eval_metric="mlogloss",
    )


def _mlp(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    sizes = p.get("hidden_layer_sizes", [128, 64])
    return make_pipeline(
        QuantileTransformer(output_distribution="normal", random_state=seed),
        MLPClassifier(
            hidden_layer_sizes=tuple(sizes),
            alpha=p.get("alpha", 1e-4),
            max_iter=p.get("max_iter", 60),
            early_stopping=True,
            random_state=seed,
        ),
    )


TEMPLATES: dict[str, Callable[[dict[str, Any], int, bool], Any]] = {
    "logistic_regression": _logreg,
    "random_forest": _rf,
    "xgboost": _xgb,
    "mlp": _mlp,
}


def run_experiment(
    manifest: ExperimentManifest, prepared_dir: Path, out_dir: Path
) -> dict[str, Any]:
    """Train one model and write predictions.csv, model_info.json. Returns run metadata."""
    tmpl = manifest.model.template
    _check_params(tmpl, manifest.model.params)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_split(prepared_dir, "train", manifest.label_scheme)
    x_test, y_test = load_split(prepared_dir, "test", manifest.label_scheme)
    enc = LabelEncoder().fit(pd.concat([y_train, y_test]))
    np.random.seed(manifest.seed)
    model = TEMPLATES[tmpl](manifest.model.params, manifest.seed, manifest.use_gpu)

    t0 = time.monotonic()
    model.fit(x_train.to_numpy(), enc.transform(y_train))
    train_s = time.monotonic() - t0
    t1 = time.monotonic()
    proba = model.predict_proba(x_test.to_numpy())
    infer_s = time.monotonic() - t1
    classes = list(enc.classes_)
    full = np.zeros((len(x_test), len(classes)))
    full[:, : proba.shape[1]] = proba
    pred = enc.inverse_transform(full.argmax(axis=1))

    preds = pd.DataFrame({"row": np.arange(len(y_test)), "y_true": y_test, "y_pred": pred})
    for i, c in enumerate(classes):
        preds[f"p_{c}"] = full[:, i].round(6)
    preds.to_csv(out_dir / "predictions.csv", index=False)
    info = {
        "experiment_id": manifest.experiment_id,
        "template": tmpl,
        "params": manifest.model.params,
        "seed": manifest.seed,
        "classes": classes,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "train_seconds": round(train_s, 3),
        "inference_seconds": round(infer_s, 3),
        "used_gpu": manifest.use_gpu,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    if hasattr(model, "feature_importances_"):
        imp = dict(
            zip(x_train.columns, np.asarray(model.feature_importances_).tolist(), strict=True)
        )
        info["feature_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    return info
