"""Deterministic manuscript consistency checks (found in a draft written by the local model)."""

from __future__ import annotations

from cyberlab_manuscript.builder import _closest_fact, tidy_prose
from cyberlab_manuscript.consistency import compared_pairs, consistency_issues

ANALYSIS = {
    "models": {k: {} for k in ("logistic_regression", "random_forest", "xgboost", "mlp")},
    "hypotheses": [
        {"id": "H1", "a": "xgboost", "b": "logistic_regression"},
        {"id": "H2", "a": "random_forest", "b": "logistic_regression"},
    ],
    "dataset": {"split_overlap_rows": 0},
}

DRAFT = """# T

## Abstract

We test whether gradient boosting achieves higher macro-F1 than logistic regression (H1, supported) and random forest achieves higher macro-F1 than logistic regression (H2, supported). The results support H1 but not H2 (supported), indicating the superiority of gradient boosting.

## Methodology

The study employs 0.9698 ± 0.0000 and 0.9990 ± 0.0000 models on the dataset. BENIGN rows were capped at [not reported]. No data leakage occurred because the split was checked.

## Results

The MLP model performed comparably to Random Forest. Random forest also outperforms logistic regression.

## Limitations

The high accuracy suggests potential issues related to data leakage.

## Conclusion

Gradient boosting outperforms logistic regression and random forest. Future work is needed.

## Evidence

Runs: 12
"""


def codes() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for i in consistency_issues(DRAFT, ANALYSIS):
        out.setdefault(i["code"], []).append(i["sentence"])
    return out


def test_flags_each_kind_of_inconsistency() -> None:
    c = codes()
    assert any("but not H2 (supported)" in s for s in c["HYP_CONTRADICTION"])
    assert any("0.9698" in s for s in c["RESULT_IN_METHOD"])
    assert any("[not reported]" in s for s in c["UNRESOLVED_PLACEHOLDER"])
    assert any("potential issues" in s for s in c["CONTRADICTS_LEAKAGE_CHECK"])
    untested = c["UNTESTED_COMPARISON"]
    assert any("comparably to Random Forest" in s for s in untested)
    assert any(s.startswith("Gradient boosting outperforms") for s in untested)


def test_does_not_flag_supported_statements() -> None:
    flagged = {s for v in codes().values() for s in v}
    assert not any(s.startswith("We test whether") for s in flagged)  # two tested comparisons
    assert "Random forest also outperforms logistic regression." not in flagged
    assert not any("No data leakage occurred" in s for s in flagged)


def test_comparison_pairs_follow_the_sentence_structure() -> None:
    known = list(ANALYSIS["models"])
    assert set(
        compared_pairs("XGBoost outperforms logistic regression and random forest.", known)
    ) == {
        ("xgboost", "logistic_regression"),
        ("xgboost", "random_forest"),
    }
    two = "XGBoost scores higher than logistic regression and random forest scores higher than logistic regression."
    assert set(compared_pairs(two, known)) == {
        ("xgboost", "logistic_regression"),
        ("random_forest", "logistic_regression"),
    }


def test_misspelled_fact_names_are_mapped() -> None:
    facts = {"benign_cap": "400,000", "rows_dropped_inf_nan": "2,867"}
    assert _closest_fact("BENIGN_cap", facts) == "benign_cap"
    assert _closest_fact("rows_dropped_inf_or_nan", facts) == "rows_dropped_inf_nan"
    assert _closest_fact("learning_rate", facts) is None


def test_nested_hypothesis_parentheses_are_tidied() -> None:
    assert tidy_prose("than LR (H1 (supported)).") == "than LR (H1, supported)."
