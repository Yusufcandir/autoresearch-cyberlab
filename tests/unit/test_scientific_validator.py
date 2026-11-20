from pathlib import Path

import numpy as np
import pandas as pd

from cyberlab_qa.metrics import paired_permutation_test
from cyberlab_qa.scientific_validator import (
    BLOCK,
    NOT_EVALUATED,
    PASS,
    WARNING,
    generate_validation_report,
    rare_class_report,
    validate_data_preprocessing,
    validate_statistical_tests,
)


def _preds(acc: float, seed: int, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array(["A", "B", "C"] * (n // 3))
    pred = np.where(rng.random(len(y)) < acc, y, np.roll(y, 1))
    return pd.DataFrame({"row": np.arange(len(y)), "y_true": y, "y_pred": pred})


def test_permutation_test_on_the_hypothesis_metric() -> None:
    good, bad = [_preds(0.95, 1), _preds(0.95, 2)], [_preds(0.6, 3), _preds(0.6, 4)]
    r = paired_permutation_test(good, bad, "balanced_accuracy", n_perm=300)
    assert r["observed_difference"] > 0.2 and r["p_value"] < 0.01
    same = paired_permutation_test(good, good, "balanced_accuracy", n_perm=300)
    assert same["p_value"] > 0.5


def test_mcnemar_only_balanced_accuracy_hypothesis_is_blocked() -> None:
    analysis = {
        "hypotheses": [
            {
                "id": "H1",
                "metric": "balanced_accuracy",
                "mcnemar": {"p_value": 0.001},
                "p_holm": 0.001,
            }
        ]
    }
    r = validate_statistical_tests(analysis, {"seeds": [42, 43, 44]})
    assert r["status"] == BLOCK and "McNemar" in r["detail"]


def test_missing_audit_is_not_a_pass() -> None:
    assert validate_data_preprocessing(None)["status"] == NOT_EVALUATED
    bad = {"consistent": False, "problems": ["stage 'x': input != removed + remaining"]}
    assert validate_data_preprocessing(bad)["status"] == BLOCK


def test_rare_class_report_counts() -> None:
    analysis = {
        "models": {
            "xgb": {
                "per_class": {
                    "Rare": {"support": 9, "recall": 8 / 9, "precision": 0.8, "f1": 0.84},
                    "BENIGN": {"support": 1000, "recall": 0.99, "precision": 0.99, "f1": 0.99},
                }
            }
        }
    }
    r = rare_class_report(analysis, 30)
    rare = next(x for x in r["per_class"] if x["class"] == "Rare")
    assert r["status"] == WARNING and r["small_classes"] == ["Rare"]
    assert (
        rare["true_positives"] == 8
        and rare["false_negatives"] == 1
        and rare["false_positives"] == 2
    )
    assert rare["recall_ci_95"][0] < 0.6 and r["recommendations"]


def test_full_report_on_prepared_fixture(tmp_path: Path) -> None:
    from cyberlab_data import prepare, write_fixture

    write_fixture(tmp_path / "raw", rows_per_class=60)
    p = prepare(tmp_path)
    analysis = {
        "models": {},
        "hypotheses": [],
        "metrics_recomputed_match": True,
        "qa_flags": [],
    }
    protocol = {"models": [], "seeds": [42, 43, 44], "split": {"strategy": "stratified_dedup"}}
    rep = generate_validation_report(analysis, protocol, [], p.root)
    by = {c["check"]: c["status"] for c in rep["checks"]}
    assert by["dataset_integrity"] == PASS and by["data_preprocessing"] == PASS
    assert by["exact_feature_overlap"] == PASS
    assert by["capture_groups"] == WARNING  # random split mixes capture days (CIC-IDS2017)
    assert by["session_identifiers"] == NOT_EVALUATED
    assert rep["manuscript_allowed"] is True
