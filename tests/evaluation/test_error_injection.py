"""Professor's plan, Step 9.4: a test set of deliberately introduced scientific errors.
Each must be detected before (or while) the manuscript is generated."""

import numpy as np
import pandas as pd

from cyberlab_manuscript.builder import EvidenceReport, render_text
from cyberlab_manuscript.consistency import consistency_issues
from cyberlab_orchestrator.agents.citation_validator import build_report
from cyberlab_qa.metrics import compute_metrics, recompute_matches
from cyberlab_qa.scientific_validator import (
    rare_class_report,
    validate_data_preprocessing,
    validate_statistical_tests,
)


def _analysis() -> dict:
    agg = lambda f1: {  # noqa: E731
        "macro_f1": {"mean": f1, "sd": 0.01, "n_seeds": 3},
        "accuracy": {"mean": 0.99, "sd": 0.0, "n_seeds": 3},
        "balanced_accuracy": {"mean": f1, "sd": 0.0, "n_seeds": 3},
    }
    return {
        "models": {
            "xgboost": {"role": "candidate", "aggregate": agg(0.9), "per_class": {}},
            "random_forest": {"role": "baseline", "aggregate": agg(0.85), "per_class": {}},
            "logistic_regression": {"role": "baseline", "aggregate": agg(0.6), "per_class": {}},
        },
        "hypotheses": [
            {
                "id": "H1",
                "a": "xgboost",
                "b": "logistic_regression",
                "metric": "macro_f1",
                "statement": "x",
                "result": "supported",
                "mcnemar": {"p_value": 0.001},
                "test": {"p_value": 0.001, "metric": "macro_f1"},
                "p_holm": 0.001,
            }
        ],
        "metrics_recomputed_match": True,
    }


def test_incorrect_row_counts_block() -> None:
    audit = {
        "consistent": False,
        "problems": ["stage 'exact duplicates': input != removed + remaining"],
    }
    assert validate_data_preprocessing(audit)["status"] == "BLOCK"


def test_invalid_statistical_test_blocks() -> None:
    a = {
        "hypotheses": [
            {
                "id": "H1",
                "metric": "balanced_accuracy",
                "mcnemar": {"p_value": 0.01},
                "p_holm": 0.01,
            }
        ]
    }
    assert validate_statistical_tests(a, {"seeds": [1, 2, 3]})["status"] == "BLOCK"


def test_mismatched_metrics_detected() -> None:
    y = np.array(["A", "B"] * 50)
    preds = pd.DataFrame({"row": range(100), "y_true": y, "y_pred": y})
    stored = compute_metrics(preds)
    stored["macro_f1"] = 0.5  # tampered value
    assert not recompute_matches(stored, preds)


def test_fabricated_reference_detected() -> None:
    md = "## Related Work\n\nX was shown [@Fake2024].\n\n## References\n\n"
    r = build_report(md, {"Fake2024": {"status": "Unverified", "doi": "10.9999/x"}}, [])
    assert r["status"] == "BLOCK" and r["reference_list"]["cited_but_unverified"] == ["Fake2024"]
    out = render_text(
        "X {{cite:Fake2024}}.",
        _analysis(),
        {"Fake2024": {"status": "Unverified"}},
        rep := EvidenceReport(),
        "related_work",
    )
    assert "[citation needed]" in out and any(f["code"] == "UNVERIFIED_CITATION" for f in rep.flags)


def test_unsupported_conclusion_detected() -> None:
    md = (
        "# T\n\n## Discussion\n\nXGBoost outperforms random forest on every dataset.\n\n"
        "## Conclusion\n\nThe results do not support H1 (supported).\n"
    )
    codes = {i["code"] for i in consistency_issues(md, _analysis())}
    assert "UNTESTED_COMPARISON" in codes  # xgboost vs random forest was never tested
    assert "HYP_CONTRADICTION" in codes


def test_typed_number_inconsistent_with_tables_detected() -> None:
    rep = EvidenceReport()
    render_text("XGBoost reached a macro-F1 of 0.97.", _analysis(), {}, rep, "discussion")
    assert any(f["code"] == "UNSUPPORTED_NUMBER" for f in rep.flags)


def test_insufficient_sample_size_warned() -> None:
    a = {
        "models": {
            "m": {
                "per_class": {"Rare": {"support": 9, "recall": 1.0, "precision": 0.9, "f1": 0.95}}
            }
        }
    }
    r = rare_class_report(a, 30)
    assert r["status"] == "WARNING" and r["small_classes"] == ["Rare"]


def test_draft_measures_and_reliability_report(tmp_path) -> None:
    from cyberlab_common.db import Database
    from cyberlab_common.evaluation import draft_measures, reliability_report

    md = (
        "## Introduction\n\nThis paper aims to study IDS. Network attacks plays a crucial role.\n\n"
        "## Related Work\n\nA [@A] and B [@B].\n\n## Discussion\n\nWe find X.\n"
    )
    m = draft_measures(md)
    assert m["generic_phrases"] == 2 and m["citations"] == 2
    db = Database(f"sqlite:///{(tmp_path / 'e.db').as_posix()}")
    db.create_all()
    r = reliability_report(db)
    assert r["runs"] == 0 and r["experiment_execution_success_rate"] is None
