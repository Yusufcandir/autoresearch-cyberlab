from pathlib import Path

import numpy as np
import pandas as pd

from cyberlab_protocol.intent import min_support_from_text, seeds_from_text, tuning_from_text
from cyberlab_qa.metrics import (
    bootstrap_diff_seeds,
    clopper_pearson,
    friedman_nemenyi,
    mcnemar,
)
from cyberlab_schemas import ExperimentManifest, ModelSpec, Protocol, SplitSpec, TuningSpec


def test_intent_parsing() -> None:
    assert tuning_from_text("Tune every model with 50 trials on validation data") == 50
    assert tuning_from_text("hyper-parameter tuning please") == 30
    assert tuning_from_text("Use fixed settings, no tuning.") is None
    assert seeds_from_text("Use five seeds") == 5 and seeds_from_text("10 random seeds") == 10
    assert min_support_from_text("merge classes with fewer than 50 test rows") == 50


def test_protocol_hash_unchanged_without_new_fields() -> None:
    from cyberlab_protocol import default_protocol

    p = default_protocol("p1", "cicids2017-mlcsv-v1")
    body = p.model_dump(mode="json")
    old = {k: v for k, v in body.items() if k not in {"tuning", "min_test_support"}}
    assert Protocol.model_validate(old).sha256 == p.sha256
    tuned = p.model_copy(update={"tuning": TuningSpec(trials=5)})
    assert tuned.sha256 != p.sha256


def _preds(n: int, acc: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"] * (n // 3))
    wrong = rng.random(len(y)) > acc
    pred = np.where(wrong, np.roll(y, 1), y)
    return pd.DataFrame({"row": np.arange(len(y)), "y_true": y, "y_pred": pred})


def test_effect_sizes() -> None:
    a = [_preds(300, 0.95, s) for s in (1, 2)]
    b = [_preds(300, 0.70, s + 10) for s in (1, 2)]
    d = bootstrap_diff_seeds(a, b, "macro_f1", n_boot=100)
    assert 0.1 < d["point"] < 0.4 and d["low"] > 0
    assert mcnemar(a[0], b[0])["odds_ratio"] > 1


def test_exact_interval_and_friedman() -> None:
    lo, hi = clopper_pearson(9, 9)
    assert hi == 1.0 and 0.6 < lo < 0.75
    assert clopper_pearson(0, 9)[0] == 0.0
    scores = {
        "d1": {"xgb": 0.9, "rf": 0.85, "lr": 0.6},
        "d2": {"xgb": 0.8, "rf": 0.82, "lr": 0.5},
        "d3": {"xgb": 0.7, "rf": 0.65, "lr": 0.4},
    }
    r = friedman_nemenyi(scores)
    assert r["applicable"] and r["average_rank"]["lr"] == 3.0 and r["critical_difference"] > 0
    assert friedman_nemenyi({"d1": scores["d1"]})["applicable"] is False


def test_tuning_uses_validation_and_is_cached(tmp_path: Path, monkeypatch) -> None:
    from cyberlab_data import prepare, write_fixture
    from cyberlab_tabular.templates import run_experiment

    write_fixture(tmp_path / "raw", rows_per_class=60)
    prep = prepare(tmp_path, SplitSpec())
    monkeypatch.setenv("CYBERLAB_TUNING_CACHE", str(tmp_path / "tuning"))
    m = ExperimentManifest(
        experiment_id="p1-lr-s42",
        project_id="proj",
        protocol_version=1,
        protocol_sha256="a" * 64,
        dataset_id="cicids2017-mlcsv-v1",
        model=ModelSpec(name="lr", template="logistic_regression", role="baseline"),
        seed=42,
        label_scheme="multiclass",
        split=SplitSpec(),
        risk_tier="T1",
        max_minutes=5,
        tuning=TuningSpec(trials=4),
    )
    info = run_experiment(m, prep.root, tmp_path / "out1")
    assert info["tuned"] and (tmp_path / "out1" / "tuning.json").exists()
    m2 = m.model_copy(update={"seed": 43, "experiment_id": "p1-lr-s43"})
    run_experiment(m2, prep.root, tmp_path / "out2")
    import json

    t2 = json.loads((tmp_path / "out2" / "tuning.json").read_text())
    assert t2["reused"] is True and t2["split"] == "validation"


def test_ablation_and_robustness_params(tmp_path: Path) -> None:
    from cyberlab_data import prepare, write_fixture
    from cyberlab_tabular.templates import run_experiment

    write_fixture(tmp_path / "raw", rows_per_class=60)
    prep = prepare(tmp_path, SplitSpec())
    base = dict(
        project_id="proj",
        protocol_version=1,
        protocol_sha256="b" * 64,
        dataset_id="cicids2017-mlcsv-v1",
        seed=42,
        label_scheme="multiclass",
        split=SplitSpec(),
        risk_tier="T1",
        max_minutes=5,
    )
    m = ExperimentManifest(
        experiment_id="p1-rf_ab-s42",
        model=ModelSpec(
            name="rf_ab",
            template="random_forest",
            role="ablation",
            params={"n_estimators": 20, "drop_features": ["Flow"], "test_noise": 0.5},
        ),
        **base,
    )
    info = run_experiment(m, prep.root, tmp_path / "o")
    assert info["ablation_dropped_features"] and all(
        f.startswith("Flow") for f in info["ablation_dropped_features"]
    )
    assert info["robustness_test_noise_sd"] == 0.5 and info["run_label"] == "confirmatory"


def test_parallel_seeds_share_one_tuning_search(tmp_path: Path, monkeypatch) -> None:
    """Seeds of one model start together; only one may run the search, the rest reuse it."""
    import threading

    from cyberlab_data import prepare, write_fixture
    from cyberlab_tabular import templates
    from cyberlab_tabular.templates import run_experiment

    write_fixture(tmp_path / "raw", rows_per_class=60)
    prep = prepare(tmp_path, SplitSpec())
    monkeypatch.setenv("CYBERLAB_TUNING_CACHE", str(tmp_path / "tuning"))
    searches: list[int] = []
    real = templates._search

    def counted(*a, **kw):  # type: ignore[no-untyped-def]
        searches.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(templates, "_search", counted)
    base = ExperimentManifest(
        experiment_id="p1-lr-s42",
        project_id="proj",
        protocol_version=1,
        protocol_sha256="c" * 64,
        dataset_id="cicids2017-mlcsv-v1",
        model=ModelSpec(name="lr", template="logistic_regression", role="baseline"),
        seed=42,
        label_scheme="multiclass",
        split=SplitSpec(),
        risk_tier="T1",
        max_minutes=5,
        tuning=TuningSpec(trials=3),
    )
    threads = [
        threading.Thread(
            target=run_experiment,
            args=(
                base.model_copy(update={"seed": s, "experiment_id": f"p1-lr-s{s}"}),
                prep.root,
                tmp_path / f"out{s}",
            ),
        )
        for s in (42, 43, 44)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(searches) == 1
    assert not list((tmp_path / "tuning").glob("*.lock"))
