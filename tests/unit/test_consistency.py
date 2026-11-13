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


def test_et_al_does_not_split_a_sentence() -> None:
    from cyberlab_manuscript.consistency import split_sentences

    para = "For instance, Alikhanov et al. [@A] studied sampling. However, it is costly."
    assert split_sentences(para) == [
        "For instance, Alikhanov et al. [@A] studied sampling.",
        "However, it is costly.",
    ]


def test_score_without_model_is_flagged() -> None:
    md = "# T\n\n## Limitations\n\nExtremely high accuracy and 0.9964 ± 0.0001 values raise concerns.\n"
    assert [i["code"] for i in consistency_issues(md, ANALYSIS)] == ["SCORE_WITHOUT_MODEL"]


def test_removal_takes_the_orphaned_follow_up_sentence() -> None:
    from cyberlab_orchestrator.agents.reviewer import remove_sentences

    md = (
        "## Related Work\n\nIntro sentence about the field [@B]. Alikhanov et al. [@A] proposed "
        "sampling methods for detection. These methods aim to improve robustness. Last one [@C].\n"
    )
    out, n = remove_sentences(
        md, ["Alikhanov et al. [@A] proposed sampling methods for detection."]
    )
    assert n == 1 and "These methods" not in out and "Alikhanov" not in out and "Last one" in out


def test_related_work_backfill_lists_uncited_papers() -> None:
    from cyberlab_orchestrator.agents.reviewer import backfill_related_work

    papers = {
        k: {"title": f"Title {k}.", "authors": ["Ann Smith", "Bo Lee", "Cy Wu"]} for k in "ABCD"
    }
    md = "## Related Work\n\nOne sentence [@A].\n\n## Methodology\n\nM.\n"
    out, n = backfill_related_work(md, papers)
    assert n == 3 and "Smith et al., “Title B” [@B]" in out
    assert out.index("[@D]") < out.index("## Methodology")


def test_hypothesis_and_citation_wording_is_tidied() -> None:
    assert (
        tidy_prose("H1 (supported) is supported by the results.")
        == "H1 is supported by the results."
    )
    assert tidy_prose("H2 (supported) also holds true.") == "H2 also holds true."
    assert (
        tidy_prose("tree models ([@Apruzzese2022cross]).") == "tree models [@Apruzzese2022cross]."
    )
    assert tidy_prose("H1 (supported) is not supported.") == "H1 (supported) is not supported."


def _agg(f1: float, acc: float, bal: float) -> dict:
    return {
        "aggregate": {
            "macro_f1": {"mean": f1, "sd": 0.001},
            "accuracy": {"mean": acc},
            "balanced_accuracy": {"mean": bal},
        }
    }


GAP_ANALYSIS = {
    **ANALYSIS,
    "models": {
        "logistic_regression": _agg(0.68, 0.97, 0.69),
        "random_forest": _agg(0.96, 0.999, 0.952),
        "xgboost": _agg(0.99, 0.999, 0.989),
        "mlp": _agg(0.92, 0.998, 0.907),
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
    ],
}


def test_accuracy_gap_caveat_must_name_a_model_that_has_the_gap() -> None:
    md = (
        "# T\n\n## Discussion\n\n"
        "Complex models like XGBoost raise concerns about overly simplistic decision boundaries. "
        "Logistic regression shows high accuracy alongside lower balanced accuracy. "
        "High overall accuracy alongside lower balanced metrics indicate potential issues.\n"
    )
    flagged = [i["sentence"][:12] for i in consistency_issues(md, GAP_ANALYSIS)]
    assert flagged == ["Complex mode", "High overall"]


def test_ci_described_as_seed_variation_is_flagged() -> None:
    md = "# T\n\n## Results\n\nXGBoost macro-F1 has 95% CI [0.9647, 0.9916], seed 42, reflecting robustness across different seeds.\n"
    assert [i["code"] for i in consistency_issues(md, GAP_ANALYSIS)] == ["CI_MISREAD"]


def test_findings_are_added_when_abstract_and_conclusion_lack_them() -> None:
    from cyberlab_orchestrator.agents.reviewer import ensure_findings, strip_paragraph_connectives

    md = "## Abstract\n\nWe compare models.\n\n## Conclusion\n\nHowever, more work is needed.\n\n## Evidence\n"
    out, added = ensure_findings(strip_paragraph_connectives(md), GAP_ANALYSIS)
    assert added == ["Abstract", "Conclusion"]
    assert "XGBoost had the highest mean macro-F1 (0.9900" in out
    assert (
        "H1 (XGBoost above logistic regression on macro-F1) was supported, exact McNemar p < 0.001."
        in out
    )
    assert "However" not in out and "More work is needed." in out


def _codes(md: str) -> list[str]:
    return [i["code"] for i in consistency_issues(md, ANALYSIS_RESULTS)]


ANALYSIS_RESULTS = {
    **ANALYSIS,
    "hypotheses": [
        {"id": "H1", "a": "xgboost", "b": "logistic_regression", "result": "supported"},
        {"id": "H2", "a": "random_forest", "b": "logistic_regression", "result": "supported"},
    ],
}


def test_bare_hypothesis_verdicts_are_checked() -> None:
    assert _codes("## Abstract\n\nThe results support H1 but not H2.\n") == ["HYP_CONTRADICTION"]
    assert _codes("## Abstract\n\nThe results support H1 and H2.\n") == []
    assert _codes("## Abstract\n\nH2 was not supported by the data.\n") == ["HYP_CONTRADICTION"]
    assert _codes("## Methodology\n\nHypotheses H1 and H2 were tested with McNemar.\n") == []


def test_comparison_that_contradicts_a_test_is_flagged() -> None:
    bad = "## Results\n\nRandom forest shows comparable performance to logistic regression.\n"
    assert _codes(bad) == ["HYP_CONTRADICTION"]
    worse = "## Results\n\nLogistic regression outperforms random forest on macro-F1.\n"
    assert _codes(worse) == ["HYP_CONTRADICTION"]
    ok = "## Results\n\nRandom forest outperforms logistic regression on macro-F1.\n"
    assert _codes(ok) == []


def test_reader_facing_checks() -> None:
    assert _codes("## Discussion\n\nThe statistician notes highlight caveats.\n") == [
        "INTERNAL_REFERENCE"
    ]
    assert _codes("## Discussion\n\nLogistic regression achieves 0.9698 ± 0.0000.\n") == [
        "SCORE_WITHOUT_METRIC"
    ]
    assert _codes("## Discussion\n\nThe Rare class has very few test instances (331,332).\n") == [
        "FACT_MISUSED"
    ]
    assert _codes("## Limitations\n\nOnly a few studies since 2017 used this dataset.\n") == []


def test_list_items_are_read_with_their_model_label() -> None:
    md = "## Results\n\n**Logistic Regression:**\n- Macro-F1: 0.6815 ± 0.0000\n- MCC: 0.9554\n"
    assert _codes(md) == []


def test_negated_comparison_that_contradicts_a_test_is_flagged() -> None:
    md = "## Discussion\n\nRandom forest does not provide a significant advantage over logistic regression.\n"
    assert _codes(md) == ["HYP_CONTRADICTION"]
    ok = "## Discussion\n\nRandom forest is not comparable to logistic regression.\n"
    assert _codes(ok) == []


def test_empty_results_prose_gets_an_evidence_summary() -> None:
    from cyberlab_orchestrator.agents.reviewer import ensure_findings, remove_sentences

    md = (
        "## Results\n\nSummary:\n\n**Logistic Regression:**\n- bad claim about it here.\n\n"
        "### Main results\n\n| t |\n\n## Discussion\n\nText.\n"
    )
    md, n = remove_sentences(md, ["bad claim about it here."])
    assert n == 1 and "**Logistic Regression:**" not in md and "\n- " not in md
    out, added = ensure_findings(md, GAP_ANALYSIS)
    assert "Results" in added and "logistic regression reached a macro-F1 of 0.6800" in out
    assert out.index("reached a macro-F1") < out.index("### Main results")


def test_verdict_contradicted_after_the_rendered_result() -> None:
    assert _codes("## Discussion\n\nHowever, H2 (supported) is not supported.\n") == [
        "HYP_CONTRADICTION"
    ]
    assert _codes(
        "## Conclusion\n\nThe test of random forest (H2, supported) is not supported.\n"
    ) == ["HYP_CONTRADICTION"]
    md = "## Discussion\n\nThere is no significant difference between random forest and logistic regression.\n"
    assert _codes(md) == ["HYP_CONTRADICTION"]
    assert _codes("## Discussion\n\nH2 (supported) is supported by the data.\n") == []
