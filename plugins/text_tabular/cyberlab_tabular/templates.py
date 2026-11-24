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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
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
    # text templates (modality 'text': one string feature such as a URL)
    "tfidf_logreg": {"C", "ngram_max", "max_features"},
    "tfidf_linear_svm": {"C", "ngram_max", "max_features"},
}
DEEP_TEMPLATES = {"resnet_mlp", "ft_transformer", "cnn1d", "tabnet"}
TEXT_TEMPLATES = {"tfidf_logreg", "tfidf_linear_svm"}


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
    "tfidf_logreg": {"C": 4.0, "ngram_max": 5, "max_features": 200_000},
    "tfidf_linear_svm": {"C": 0.5, "ngram_max": 5, "max_features": 200_000},
}


# Allowed on every template (spec FR-008): an ablation drops feature groups (names or name
# prefixes), a robustness stress test adds Gaussian noise (in feature standard deviations) to
# the numeric TEST features only. Both are reviewed, deterministic operations.
COMMON_PARAMS = {"drop_features", "test_noise"}


def _check_params(template: str, params: dict[str, Any]) -> None:
    extra = set(params) - ALLOWED_PARAMS[template] - COMMON_PARAMS
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


def _ravel(x: Any) -> Any:
    return np.asarray(x).ravel().astype(str)


def _tfidf(p: dict[str, Any]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import FunctionTransformer

    return [
        FunctionTransformer(_ravel),
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, int(p.get("ngram_max", 5))),
            max_features=int(p.get("max_features", 200_000)),
            sublinear_tf=True,
            lowercase=True,
            dtype=np.float32,
        ),
    ]


def _tfidf_logreg(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    return make_pipeline(
        *_tfidf(p),
        LogisticRegression(C=p.get("C", 4.0), max_iter=2000, random_state=seed),
    )


def _tfidf_svm(p: dict[str, Any], seed: int, gpu: bool) -> Any:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.svm import LinearSVC

    return make_pipeline(
        *_tfidf(p),
        CalibratedClassifierCV(LinearSVC(C=p.get("C", 0.5), random_state=seed), cv=3),
    )


TEMPLATES: dict[str, Callable[[dict[str, Any], int, bool], Any]] = {
    "logistic_regression": _logreg,
    "random_forest": _rf,
    "xgboost": _xgb,
    "mlp": _mlp,
    "lightgbm": _lgbm,
    **{arch: _deep(arch) for arch in DEEP_TEMPLATES},
    "tfidf_logreg": _tfidf_logreg,
    "tfidf_linear_svm": _tfidf_svm,
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


def _space_text(t: Any) -> dict[str, Any]:
    return {
        "C": t.suggest_float("C", 1e-2, 1e2, log=True),
        "ngram_max": t.suggest_int("ngram_max", 3, 6),
        "max_features": t.suggest_categorical("max_features", [50_000, 200_000, 500_000]),
    }


SEARCH_SPACES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "lightgbm": _space_lgbm,
    "resnet_mlp": _space_resnet,
    "ft_transformer": _space_ft,
    "cnn1d": _space_cnn,
    "tabnet": _space_tabnet,
    "tfidf_logreg": _space_text,
    "tfidf_linear_svm": _space_text,
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
    spec = manifest.tuning
    assert spec is not None
    cache_dir = os.environ.get("CYBERLAB_TUNING_CACHE")
    key = (
        f"{manifest.project_id}-p{manifest.protocol_version}-"
        f"{manifest.protocol_sha256[:12]}-{manifest.model.name}"
        + (f"-{manifest.dataset_id}" if "@" in manifest.experiment_id else "")
    )
    cache = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if cache is not None and cache.exists():
        return {**json.loads(cache.read_text()), "reused": True}
    if cache is not None:
        # The seeds of one model start in parallel. Only the first one searches; the others
        # wait for its result instead of each running the same search (3x the CPU, slower).
        with _tuning_lock(cache.with_suffix(".lock"), spec.timeout_minutes * 60 + 600):
            if cache.exists():
                return {**json.loads(cache.read_text()), "reused": True}
            return _search(manifest, x_train, y_train, x_val, y_val, cache)
    return _search(manifest, x_train, y_train, x_val, y_val, None)


@contextmanager
def _tuning_lock(path: Path, max_wait: float) -> Iterator[None]:
    """Exclusive lock by O_EXCL file creation. A lock older than max_wait is treated as left
    behind by a crashed job and taken over."""
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > max_wait:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(2)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _search(
    manifest: ExperimentManifest,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    cache: Path | None,
) -> dict[str, Any]:
    import optuna

    spec = manifest.tuning
    assert spec is not None
    tmpl = manifest.model.template
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


def permutation_importance(
    model: Any, x: pd.DataFrame, y: np.ndarray, seed: int, per_class: int = 300, repeats: int = 3
) -> dict[str, float]:
    """Model-agnostic explanation (works for trees and deep models alike): how much the
    macro-F1 drops when one feature's values are shuffled. Computed on a class-balanced
    sample of the test rows (up to `per_class` rows per class, so small classes count)."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed + 101)
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
            for c in np.unique(y)
        ]
    )
    xs, ys = x.to_numpy()[idx], y[idx]

    def f1(arr: np.ndarray) -> float:
        return float(f1_score(ys, model.predict_proba(arr).argmax(axis=1), average="macro"))

    base = f1(xs)
    out: dict[str, float] = {}
    for j, col in enumerate(x.columns):
        drops = []
        for _ in range(repeats):
            xp = xs.copy()
            xp[:, j] = rng.permutation(xp[:, j])
            drops.append(base - f1(xp))
        out[str(col)] = round(float(np.mean(drops)), 6)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _balanced_rows(y: np.ndarray, seed: int, per_class: int = 300) -> np.ndarray:
    rng = np.random.default_rng(seed + 101)
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(y == c), min(per_class, int((y == c).sum())), replace=False)
            for c in np.unique(y)
        ]
    )


def shap_importance(
    model: Any, x: pd.DataFrame, y: np.ndarray, classes: list[str], seed: int
) -> dict[str, Any] | None:
    """Exact TreeSHAP values from the boosting libraries themselves (XGBoost pred_contribs,
    LightGBM pred_contrib; no extra package): mean |SHAP| per feature over a class-balanced
    sample of test rows, overall (summed over classes) and for each class."""
    idx = _balanced_rows(y, seed)
    xs = x.to_numpy()[idx]
    nf = xs.shape[1]
    try:
        if type(model).__name__ == "XGBClassifier":
            from xgboost import DMatrix

            contrib = np.asarray(model.get_booster().predict(DMatrix(xs), pred_contribs=True))
        elif type(model).__name__ == "LGBMClassifier":
            contrib = np.asarray(model.predict(xs, pred_contrib=True))
        else:
            return None
    except Exception:  # noqa: BLE001 - explanations are optional evidence
        return None
    if contrib.ndim == 2 and contrib.shape[1] == nf + 1:  # binary: one set of values
        per = contrib[:, None, :nf]
    elif contrib.ndim == 3:  # XGBoost multiclass: rows x classes x (features + bias)
        per = contrib[:, :, :nf]
    else:  # LightGBM multiclass: rows x classes * (features + bias)
        per = contrib.reshape(len(xs), -1, nf + 1)[:, :, :nf]
    mean_abs = np.abs(per).mean(axis=0)  # classes x features
    cols = [str(c) for c in x.columns]
    overall = dict(zip(cols, mean_abs.sum(axis=0).round(6).tolist(), strict=True))
    out: dict[str, Any] = {
        "overall": dict(sorted(overall.items(), key=lambda kv: -kv[1])[:15]),
    }
    if per.shape[1] == len(classes):
        out["per_class"] = {
            classes[k]: dict(
                sorted(
                    zip(cols, mean_abs[k].round(6).tolist(), strict=True), key=lambda kv: -kv[1]
                )[:5]
            )
            for k in range(len(classes))
        }
    return out


def run_experiment(
    manifest: ExperimentManifest,
    prepared_dir: Path,
    out_dir: Path,
    targets: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Train one model and write predictions.csv, model_info.json. Returns run metadata.
    With `targets` (generalization runs) the model is trained on the features every dataset
    shares and also predicts each target dataset's test split (transfer_<dataset>.csv)."""
    tmpl = manifest.model.template
    _check_params(tmpl, manifest.model.params)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, y_train = load_split(prepared_dir, "train", manifest.label_scheme)
    x_test, y_test = load_split(prepared_dir, "test", manifest.label_scheme)
    shared: list[str] | None = None
    if targets:
        cols = set(x_train.columns)
        for tdir in targets.values():
            cols &= set(json.loads((tdir / "manifest.json").read_text())["features"])
        shared = [c for c in x_train.columns if c in cols]
        if not shared:
            raise ValueError("the datasets share no features: generalization is not possible")
        x_train, x_test = x_train[shared], x_test[shared]
    enc = LabelEncoder().fit(pd.concat([y_train, y_test]))
    np.random.seed(manifest.seed)
    params = dict(manifest.model.params)
    drop = [str(d) for d in params.pop("drop_features", None) or []]
    noise = float(params.pop("test_noise", 0.0) or 0.0)
    dropped: list[str] = []
    if drop:
        dropped = [c for c in x_train.columns if any(c == d or c.startswith(d) for d in drop)]
        if len(dropped) == len(x_train.columns):
            raise ValueError("ablation would remove every feature")
        x_train, x_test = x_train.drop(columns=dropped), x_test.drop(columns=dropped)
    if noise > 0:
        num = [c for c in x_test.columns if not c.startswith("text:")]
        rng = np.random.default_rng(manifest.seed + 7919)
        sd = x_train[num].std().replace(0, 1.0).to_numpy()
        x_test = x_test.copy()
        x_test[num] = (
            x_test[num].to_numpy() + rng.normal(0, noise, size=(len(x_test), len(num))) * sd
        )
    tuning: dict[str, Any] | None = None
    if manifest.tuning is not None:
        x_val, y_val = load_split(prepared_dir, "val", manifest.label_scheme)
        if len(x_val) == 0:
            raise ValueError("tuning needs a validation split (val_fraction > 0)")
        if dropped:
            x_val = x_val.drop(columns=dropped)
        if shared is not None:
            x_val = x_val[shared]
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
        "ablation_dropped_features": dropped,
        "robustness_test_noise_sd": noise,
        "run_label": "confirmatory",
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
    for tname, tdir in (targets or {}).items():
        x_t, y_t = load_split(tdir, "test", manifest.label_scheme)
        p_t = model.predict_proba(x_t[shared].to_numpy())
        full_t = np.zeros((len(x_t), len(classes)))
        full_t[:, : p_t.shape[1]] = p_t
        df_t = pd.DataFrame(
            {
                "row": np.arange(len(y_t)),
                "y_true": y_t,
                "y_pred": enc.inverse_transform(full_t.argmax(axis=1)),
            }
        )
        for i, c in enumerate(classes):
            df_t[f"p_{c}"] = full_t[:, i].round(6)
        df_t.to_csv(out_dir / f"transfer_{tname}.csv", index=False)
    if targets:
        info["transfer_targets"] = sorted(targets)
        info["shared_features"] = len(shared or [])
    if (
        not targets
        and len(x_test.columns) <= 300
        and not any(str(c).startswith("text:") for c in x_test.columns)
    ):
        t2 = time.monotonic()
        info["permutation_importance"] = permutation_importance(
            model, x_test, enc.transform(y_test), manifest.seed
        )
        shap = shap_importance(model, x_test, enc.transform(y_test), classes, manifest.seed)
        if shap:
            info["shap_importance"] = shap
        info["explain_seconds"] = round(time.monotonic() - t2, 3)
    if hasattr(model, "feature_importances_"):
        imp = dict(
            zip(x_train.columns, np.asarray(model.feature_importances_).tolist(), strict=True)
        )
        info["feature_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:20])
    if "@" not in manifest.experiment_id:  # the main dataset's trained models are kept
        try:
            import joblib  # type: ignore[import-untyped]

            joblib.dump(model, out_dir / "model.joblib", compress=3)
            info["model_artifact"] = "model.joblib"
        except Exception:  # noqa: BLE001 - a model that cannot be pickled is still a result
            info["model_artifact"] = None
    (out_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    return info
