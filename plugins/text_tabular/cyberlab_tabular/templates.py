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

_TRAIN = {"lr", "weight_decay", "epochs", "batch_size", "class_weight"}

ALLOWED_PARAMS: dict[str, set[str]] = {
    "logistic_regression": {"C", "max_iter"},
    "random_forest": {"n_estimators", "max_depth", "min_samples_leaf"},
    "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample"},
    "mlp": {"hidden_layer_sizes", "alpha", "max_iter"},
    "lightgbm": {"n_estimators", "num_leaves", "learning_rate", "subsample", "colsample_bytree"},
    # deep templates (PyTorch, cyberlab_tabular.deep); shared training settings first
    "resnet_mlp": {"d", "n_blocks", "dropout", *_TRAIN},
    "ft_transformer": {"d_token", "n_blocks", "n_heads", "dropout", *_TRAIN},
    "cnn1d": {"channels", "n_layers", "kernel_size", "dropout", *_TRAIN},
    "tabnet": {"n_d", "n_steps", "gamma", *_TRAIN},
}
DEEP_TEMPLATES = {"resnet_mlp", "ft_transformer", "cnn1d", "tabnet"}


# Reviewed, documented settings used for every study (spec 7.4: no silent tuning). The protocol
# agent cannot change them; the PI can, explicitly, through a protocol revision comment.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 400},
    "random_forest": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1},
    "xgboost": {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.1, "subsample": 0.9},
    "mlp": {"hidden_layer_sizes": [128, 64], "alpha": 0.0001, "max_iter": 60},
    "lightgbm": {
        "n_estimators": 400,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    # Defaults from Gorishniy et al. (2021) and Arik and Pfister (2021), scaled for 48 GB.
    "resnet_mlp": {
        "d": 256,
        "n_blocks": 4,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 30,
        "batch_size": 1024,
        "class_weight": "none",
    },
    "ft_transformer": {
        "d_token": 96,
        "n_blocks": 3,
        "n_heads": 8,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 30,
        "batch_size": 1024,
        "class_weight": "none",
    },
    "cnn1d": {
        "channels": 32,
        "n_layers": 3,
        "kernel_size": 3,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 30,
        "batch_size": 1024,
        "class_weight": "none",
    },
    "tabnet": {
        "n_d": 32,
        "n_steps": 4,
        "gamma": 1.5,
        "lr": 2e-3,
        "weight_decay": 1e-5,
        "epochs": 30,
        "batch_size": 2048,
        "class_weight": "none",
    },
}


def _check_params(template: str, params: dict[str, Any]) -> None:
    extra = set(params) - ALLOWED_PARAMS[template]
    if extra:
        raise ValueError(f"parameters not allowed for {template}: {sorted(extra)}")


def _logreg(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 400), random_state=seed),
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


def _lgbm(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=p.get("n_estimators", 400),
        num_leaves=p.get("num_leaves", 63),
        learning_rate=p.get("learning_rate", 0.05),
        subsample=p.get("subsample", 0.9),
        subsample_freq=1,
        colsample_bytree=p.get("colsample_bytree", 0.9),
        random_state=seed,
        n_jobs=_N_JOBS,
        verbose=-1,
    )


def _deep(arch: str) -> Callable[[dict[str, Any], int, bool], Any]:
    def build(p: dict[str, Any], seed: int, gpu: bool) -> Any:
        from cyberlab_tabular.deep import TorchTabularClassifier

        full = {**DEFAULT_PARAMS[arch], **p}
        train = {
            k: full.pop(k) for k in ("lr", "weight_decay", "epochs", "batch_size", "class_weight")
        }
        return TorchTabularClassifier(arch, full, seed, gpu, **train)

    return build


TEMPLATES: dict[str, Callable[[dict[str, Any], int, bool], Any]] = {
    "logistic_regression": _logreg,
    "random_forest": _rf,
    "xgboost": _xgb,
    "mlp": _mlp,
    "lightgbm": _lgbm,
    **{arch: _deep(arch) for arch in DEEP_TEMPLATES},
}


# Reviewed search spaces (part of the platform, not chosen by an agent). Each returns the
# params for one Optuna trial; every key must be in ALLOWED_PARAMS.
def _space_logreg(t: Any) -> dict[str, Any]:
    return {"C": t.suggest_float("C", 1e-3, 1e2, log=True), "max_iter": 600}


def _space_rf(t: Any) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 500, step=50),
        "max_depth": t.suggest_categorical("max_depth", [None, 12, 20, 32]),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 8),
    }


def _space_xgb(t: Any) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "max_depth": t.suggest_int("max_depth", 3, 12),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
    }


def _space_mlp(t: Any) -> dict[str, Any]:
    width = t.suggest_categorical("width", [64, 128, 256])
    depth = t.suggest_int("depth", 1, 3)
    return {
        "hidden_layer_sizes": [max(16, width // (2**i)) for i in range(depth)],
        "alpha": t.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "max_iter": 80,
    }


def _space_lgbm(t: Any) -> dict[str, Any]:
    return {
        "n_estimators": t.suggest_int("n_estimators", 100, 800, step=50),
        "num_leaves": t.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _train_space(t: Any, lr: tuple[float, float]) -> dict[str, Any]:
    return {
        "lr": t.suggest_float("lr", *lr, log=True),
        "weight_decay": t.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "class_weight": t.suggest_categorical("class_weight", ["none", "balanced"]),
    }


def _space_resnet(t: Any) -> dict[str, Any]:
    return {
        "d": t.suggest_categorical("d", [128, 256, 512]),
        "n_blocks": t.suggest_int("n_blocks", 2, 6),
        "dropout": t.suggest_float("dropout", 0.0, 0.3),
        **_train_space(t, (1e-4, 3e-3)),
    }


def _space_ft(t: Any) -> dict[str, Any]:
    return {
        "d_token": t.suggest_categorical("d_token", [64, 96, 128]),
        "n_blocks": t.suggest_int("n_blocks", 1, 4),
        "n_heads": 8,
        "dropout": t.suggest_float("dropout", 0.0, 0.3),
        **_train_space(t, (3e-5, 1e-3)),
    }


def _space_cnn(t: Any) -> dict[str, Any]:
    return {
        "channels": t.suggest_categorical("channels", [16, 32, 64]),
        "n_layers": t.suggest_int("n_layers", 2, 4),
        "kernel_size": t.suggest_categorical("kernel_size", [3, 5]),
        "dropout": t.suggest_float("dropout", 0.0, 0.3),
        **_train_space(t, (1e-4, 3e-3)),
    }


def _space_tabnet(t: Any) -> dict[str, Any]:
    return {
        "n_d": t.suggest_categorical("n_d", [16, 32, 64]),
        "n_steps": t.suggest_int("n_steps", 3, 6),
        "gamma": t.suggest_float("gamma", 1.0, 2.0),
        **_train_space(t, (5e-4, 2e-2)),
    }


SEARCH_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "lightgbm": _space_lgbm,
    "resnet_mlp": _space_resnet,
    "ft_transformer": _space_ft,
    "cnn1d": _space_cnn,
    "tabnet": _space_tabnet,
    "logistic_regression": _space_logreg,
    "random_forest": _space_rf,
    "xgboost": _space_xgb,
    "mlp": _space_mlp,
}


def score(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
    )

    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "mcc":
        return float(matthews_corrcoef(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro"))


def _subsample(
    x: pd.DataFrame, y: np.ndarray, n: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Stratified subsample used only to speed up tuning; small classes keep their rows."""
    if len(x) <= n:
        return x, y
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(min(len(idx), 50), round(n * len(idx) / len(x)))
        keep.extend(rng.choice(idx, size=min(k, len(idx)), replace=False).tolist())
    keep.sort()
    return x.iloc[keep], y[keep]


def tune(
    manifest: ExperimentManifest,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Optuna TPE search on the validation split; the test split is never touched here.

    Cached per (protocol, model), so the other seeds of the same model reuse the same tuned
    settings instead of searching again."""
    import optuna

    spec = manifest.tuning
    assert spec is not None
    tmpl = manifest.model.template
    cache_dir = os.environ.get("CYBERLAB_TUNING_CACHE")
    key = (
        f"{manifest.project_id}-p{manifest.protocol_version}-"
        f"{manifest.protocol_sha256[:12]}-{manifest.model.name}"
    )
    cache = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache is not None and cache.exists():
        return {**json.loads(cache.read_text()), "reused": True}
    xs, ys = _subsample(x_train, y_train, spec.max_train_rows, spec.sampler_seed)
    trials: list[dict[str, Any]] = []

    def objective(t: Any) -> float:
        params = SEARCH_SPACES[tmpl](t)
        _check_params(tmpl, params)
        t0 = time.monotonic()
        m = TEMPLATES[tmpl](params, spec.sampler_seed, manifest.use_gpu)
        m.fit(xs.to_numpy(), ys)
        value = score(spec.metric, y_val, m.predict(x_val.to_numpy()))
        trials.append(
            {"params": params, "value": round(value, 6), "seconds": round(time.monotonic() - t0, 2)}
        )
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=spec.sampler_seed)
    )
    study.optimize(objective, n_trials=spec.trials, timeout=spec.timeout_minutes * 60)
    result = {
        "label": "exploratory",
        "metric": spec.metric,
        "split": "validation",
        "trials_requested": spec.trials,
        "trials_run": len(trials),
        "tuning_rows": len(xs),
        "best_params": trials[study.best_trial.number]["params"],
        "best_value": round(float(study.best_value), 6),
        "trials": trials,
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(result, indent=2, default=str))
    return {**result, "reused": False}


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
    params = dict(manifest.model.params)
    tuning: dict[str, Any] | None = None
    if manifest.tuning is not None:
        x_val, y_val = load_split(prepared_dir, "val", manifest.label_scheme)
        if len(x_val) == 0:
            raise ValueError("tuning needs a validation split (val_fraction > 0)")
        tuning = tune(manifest, x_train, enc.transform(y_train), x_val, enc.transform(y_val))
        params = tuning["best_params"]
        (out_dir / "tuning.json").write_text(json.dumps(tuning, indent=2, default=str))
    model = TEMPLATES[tmpl](params, manifest.seed, manifest.use_gpu)

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
        "params": params,
        "tuned": tuning is not None,
        "tuning_best_value": tuning["best_value"] if tuning else None,
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
    if hasattr(model, "history"):  # deep templates: learning curve and device, as evidence
        import torch

        info["training_curve"] = model.history
        info["epochs_run"] = getattr(model, "epochs_run_", None)
        info["device"] = str(model.device)
        env: dict[str, Any] = info["environment"]  # type: ignore[assignment]
        env["torch"] = str(torch.__version__)
        env["cuda"] = torch.version.cuda
    if hasattr(model, "feature_importances_"):
        imp = dict(
            zip(x_train.columns, np.asarray(model.feature_importances_).tolist(), strict=True)
        )
        info["feature_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    return info
