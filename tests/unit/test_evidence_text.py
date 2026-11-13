"""Evidence-locked manuscript text: results come from stored evidence, never from the LLM."""

from __future__ import annotations

from cyberlab_manuscript.consistency import consistency_issues
from cyberlab_manuscript.evidence_text import forbidden, lock_evidence


def _m(f1: float, acc: float, bal: float, lo: float, hi: float, role: str) -> dict:
    return {
        "role": role,
        "ci_seed": 42,
        "macro_f1_ci": {"low": lo, "high": hi},
        "aggregate": {
            "macro_f1": {"mean": f1, "sd": 0.001},
            "accuracy": {"mean": acc},
            "balanced_accuracy": {"mean": bal},
        },
    }


ANALYSIS = {
    "models": {
        "logistic_regression": _m(0.68, 0.97, 0.69, 0.64, 0.71, "baseline"),
        "random_forest": _m(0.96, 0.999, 0.95, 0.92, 0.98, "baseline"),
        "xgboost": {
            **_m(0.99, 0.999, 0.989, 0.96, 0.99, "candidate"),
            "per_class": {"BENIGN": {"support": 800}, "Rare": {"support": 9}},
        },
    },
    "hypotheses": [
        {
            "id": "H1",
            "a": "xgboost",
            "b": "logistic_regression",
            "metric": "macro_f1",
            "result": "supported",
            "mcnemar": {"p_value": 1e-9},
        },
        {
            "id": "H2",
            "a": "random_forest",
            "b": "logistic_regression",
            "metric": "macro_f1",
            "result": "supported",
            "mcnemar": {"p_value": 1e-9},
        },
    ],
    "dataset": {"split_overlap_rows": 0},
}
FACTS = {"benign_cap": "400,000 rows", "split_seed": 42}
PROSE = [
    "BENIGN rows were randomly down-sampled to at most facts.benign_cap.",
    "One stratified split was made with seed facts.split_seed.",
    "Model 'xgboost' uses the 'xgboost' template (candidate).",
]

DRAFT = """# T

## Abstract

We study intrusion detection on CIC-IDS2017. The results support H1 but not H2. Random forest shows comparable performance to logistic regression.

## Introduction

Intrusion detection matters, and errors can be significantly costly. We compare logistic regression, random forest and XGBoost.

## Methodology

The study employs 0.9698 models and uses the 'xgboost' template.

### Metric definitions

Macro-F1 is defined.

## Results

- **LR**: 95% CI [low, high], seed N.

### Main results

| table |

## Discussion

The statistician notes say tree-based models are more effective. Class imbalance remains a challenge.

## Limitations

CIC-IDS2017 is synthetic.

## Conclusion

This study evaluated three models. The results support H1 only. Future work should use real traffic.

## Evidence

Runs: 3
"""


def test_lock_evidence_leaves_no_inconsistency_and_keeps_framing() -> None:
    md, dropped = lock_evidence(DRAFT, ANALYSIS, FACTS, PROSE)
    assert consistency_issues(md, ANALYSIS) == []
    # framing sentences survive
    for keep in (
        "We study intrusion detection on CIC-IDS2017.",
        "errors can be significantly costly",
        "We compare logistic regression, random forest and XGBoost.",
        "Class imbalance remains a challenge.",
        "CIC-IDS2017 is synthetic.",
        "This study evaluated three models.",
    ):
        assert keep in md, keep
    # result statements written by the model are gone
    for gone in (
        "but not H2",
        "comparable performance",
        "0.9698 models",
        "[low, high]",
        "statistician",
        "supports H1 only",
        "template",
    ):
        assert gone not in md, gone
    assert len(dropped) >= 4
    # evidence text is present where it belongs
    assert "BENIGN rows were randomly down-sampled to at most 400,000 rows." in md
    assert "Three models were compared" in md
    assert (
        "XGBoost reached a macro-F1 of 0.9900 ± 0.0010 (95% bootstrap CI [0.9600, 0.9900], seed 42)"
        in md
    )
    assert (
        md.count("H2 (random forest scores higher than logistic regression") == 3
    )  # abstract, results, conclusion
    conclusion = md.split("## Conclusion", 1)[1]
    assert conclusion.index("H1 (") < conclusion.index("Future work")
    assert "### Main results" in md and "| table |" in md and "### Metric definitions" in md


def test_lock_evidence_is_idempotent() -> None:
    once, _ = lock_evidence(DRAFT, ANALYSIS, FACTS, PROSE)
    twice, dropped = lock_evidence(once, ANALYSIS, FACTS, PROSE)
    assert twice == once and dropped == []


def test_framing_filter() -> None:
    known = list(ANALYSIS["models"])
    assert forbidden("XGBoost performs best on this dataset.", "", known)
    assert forbidden("Tree-based models are more effective.", "", known)
    assert forbidden("The model reached 0.95.", "", known)
    assert forbidden("These results support the hypothesis.", "", known)
    assert not forbidden("We compare logistic regression and random forest.", "", known)
    assert not forbidden("Confidence intervals are reported for every model.", "", known)
    assert not forbidden("Class imbalance can significantly affect macro-F1.", "", known)
