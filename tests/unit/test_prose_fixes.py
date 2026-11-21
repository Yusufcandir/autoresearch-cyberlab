"""Prose problems found in a real draft (UNSW-NB15, v6) and the rules that now prevent them."""

from __future__ import annotations

from typing import Any

from cyberlab_manuscript.builder import evidence_integers, strip_internal_ids
from cyberlab_manuscript.consistency import class_claim_problem
from cyberlab_manuscript.evidence_text import _rebuild, _renumber


def _analysis() -> dict[str, Any]:
    def pc(worms: float, backdoor: float) -> dict[str, Any]:
        return {
            "Worms": {"recall": worms, "support": 43},
            "Backdoor": {"recall": backdoor, "support": 57},
        }

    return {
        "models": {
            "random_forest": {"seeds": [42, 43, 44], "per_class": pc(0.4186, 0.5789)},
            "lightgbm": {"seeds": [42, 43, 44], "per_class": pc(0.4884, 0.5614)},
        }
    }


def test_per_class_recall_claim_is_checked() -> None:
    a, known = _analysis(), ["random_forest", "lightgbm"]
    bad = "LightGBM exhibits better recall for smaller classes such as Worms and Backdoor."
    assert "Backdoor" in (class_claim_problem(bad, a, known) or "")
    assert class_claim_problem("LightGBM has better recall on Worms.", a, known) is None
    assert class_claim_problem("LightGBM is fast.", a, known) is None


def test_recorded_integers_are_not_unsupported_numbers() -> None:
    nums = evidence_integers(_analysis())
    assert {"42", "43", "44", "57", "500", "1000"} <= nums
    assert "0.5" not in nums


def test_internal_finding_ids_are_removed() -> None:
    s = "Accuracy can mislead (Finding F-acc). Ranking (F1) holds [C012]."
    assert strip_internal_ids(s) == "Accuracy can mislead. Ranking holds."
    assert strip_internal_ids("macro-F1 of 0.60 [@Gupta2025improving]") == (
        "macro-F1 of 0.60 [@Gupta2025improving]"
    )


def test_dangling_pronoun_after_a_removed_paragraph_end() -> None:
    body = "LightGBM won by a lot.\n\nIt contrasts with deep models. Tree ensembles are cheap."
    out = _rebuild(body, {"LightGBM won by a lot."})
    assert "It contrasts" not in out and "Tree ensembles are cheap." in out


def test_first_and_lastly_with_two_points_become_first_and_second() -> None:
    out = _renumber(["First, one model.", "Lastly, one metric."])
    assert out == ["First, one model.", "Second, one metric."]
    three = _renumber(["First, a.", "Second, b.", "Finally, c."])
    assert three[-1] == "Finally, c."


def test_institution_is_not_an_author_and_empty_dataset_is_skipped() -> None:
    from cyberlab_orchestrator.agents.reviewer import EMPTY, INSTITUTION, _matrix_sentence

    assert INSTITUTION.search("Zarqa Technical Intermediate College")
    assert not INSTITUTION.search("Dena Abu Laila")
    assert EMPTY.fullmatch("Not specified explicitly in abstract")
    s = _matrix_sentence(
        "Laila and Aljawarneh",
        "College2025optimizing",
        {"method": "SMOTE with ensembles", "dataset": "Not specified explicitly in abstract"},
    )
    assert s is not None and "Not specified" not in s and "Main finding reported" not in s


def test_hypothesis_label_is_not_doubled() -> None:
    assert strip_internal_ids("better (H1 H1 (supported)).") == "better (H1 supported)."
    assert strip_internal_ids("H1 H1 (supported)") == "H1 (supported)"


def test_author_year_mentions_become_citations() -> None:
    from cyberlab_manuscript.builder import link_author_year

    cites = {
        "Abbas2023optimization": {"status": "Verified", "authors": ["Qaiser Abbas"], "year": 2023},
        "Gupta2025improving": {"status": "Verified", "authors": ["Harsh Gupta"], "year": 2025},
    }
    md = "## Related Work\n\nAbbas et al. (2023) proposed X. Gupta (2025) compared Y. Lee (2020) Z.\n"
    out = link_author_year(md, cites)
    assert "Abbas et al. [@Abbas2023optimization] proposed" in out
    assert "Gupta [@Gupta2025improving] compared" in out and "Lee (2020)" in out


def test_false_positive_rate_claims_are_checked() -> None:
    from cyberlab_manuscript.consistency import fpr_claim_problem

    def m(fpr: float) -> dict[str, Any]:
        return {
            "aggregate": {"false_positive_rate": {"mean": fpr}},
            "per_class": {"Backdoor": {"recall": 0.5}, "BENIGN": {"recall": 0.9}},
        }

    a = {"models": {"lightgbm": m(0.22), "random_forest": m(0.39)}}
    known = ["lightgbm", "random_forest"]
    assert fpr_claim_problem("LightGBM exhibits high false positive rates.", a, known)
    assert fpr_claim_problem("Random forest has a higher false-positive rate.", a, known) is None
    assert fpr_claim_problem("It has high false positive rates for Backdoor.", a, known)


def test_backfilled_finding_gets_a_subject_and_names_stay_capitalised() -> None:
    from cyberlab_orchestrator.agents.reviewer import _lower_first, _matrix_sentence

    assert _lower_first("Random Forest achieved high accuracy").startswith("Random Forest")
    s = _matrix_sentence(
        "Abbas et al.",
        "Abbas2023optimization",
        {"method": "hybrid ensemble", "finding": "Yielded high overall accuracy"},
    )
    assert s is not None and "that the approach yielded" in s


def test_budget_exhaustion_is_not_reported_as_no_sections() -> None:
    import pytest

    from cyberlab_llm import BudgetExceeded
    from cyberlab_orchestrator.agents.writing import write_sections

    class Ctx:
        class llm:  # noqa: N801
            @staticmethod
            def complete_json(*a, **kw):  # type: ignore[no-untyped-def]
                raise BudgetExceeded("project p used 1001258/1000000 LLM tokens")

    with pytest.raises(BudgetExceeded):
        write_sections(Ctx(), "p", {})  # type: ignore[arg-type]


def test_v10_leftovers() -> None:
    from cyberlab_orchestrator.agents.reviewer import _lower_first

    assert strip_internal_ids("leakage (Validation Warning: session_identifiers). X") == (
        "leakage. X"
    )
    assert strip_internal_ids("confirms it (H1: H1 (supported)).") == "confirms it (H1 supported)."
    assert strip_internal_ids("**Deep Learning:**\nDeep models were used.") == (
        "Deep models were used."
    )
    assert _lower_first("Recasting UNSW-NB15 data").startswith("recasting")
    assert _lower_first("Outperformed Random Forest").startswith("outperformed")
    assert _lower_first("Grid Search tuning").startswith("Grid Search")
    assert "0.05" in evidence_integers({})


def test_general_rare_class_claim_is_checked_on_the_smallest_classes() -> None:
    a = _analysis()
    for m in a["models"].values():
        m["per_class"]["Fuzzers"] = {"recall": 0.7, "support": 4000}
    claim = "LightGBM exhibits better recall for rare attack classes."
    assert class_claim_problem(claim, a, ["random_forest", "lightgbm"])  # Backdoor: RF higher


def test_v12_claims_are_checked() -> None:
    from cyberlab_manuscript.consistency import (
        NOVELTY,
        OVERFIT,
        tested_pair_called_untested,
    )
    from cyberlab_orchestrator.agents.reviewer import _lower_first

    known = ["lightgbm", "random_forest", "resnet_mlp"]
    tested = {frozenset(("lightgbm", "random_forest"))}
    wrong = (
        "LightGBM had a higher mean macro-F1 than Random Forest, but the difference was not "
        "tested for significance."
    )
    assert tested_pair_called_untested(wrong, known, tested)
    right = "LightGBM versus the ResNet-MLP was not tested for significance."
    assert tested_pair_called_untested(right, known, tested) is None
    assert NOVELTY.search("this study uniquely evaluates LightGBM")
    assert OVERFIT.search("indicating overfitting to the majority class")
    assert _lower_first("Signal-processing feature construction").startswith("signal-processing")


def test_v14_false_positive_direction_and_fragments() -> None:
    from cyberlab_manuscript.consistency import fpr_claim_problem

    def m(fpr: float) -> dict[str, Any]:
        return {"aggregate": {"false_positive_rate": {"mean": fpr}}, "per_class": {}}

    a = {"models": {"lightgbm": m(0.2247), "random_forest": m(0.3897), "resnet_mlp": m(0.3712)}}
    known = ["lightgbm", "random_forest", "resnet_mlp"]
    wrong = (
        "The false positive rate is lower for Random Forest compared to LightGBM and ResNet-MLP."
    )
    assert fpr_claim_problem(wrong, a, known)
    assert fpr_claim_problem("Random forest minimizes false alarms best.", a, known)
    assert fpr_claim_problem("LightGBM had the lowest false-positive rate.", a, known) is None
    assert fpr_claim_problem("Random forest had a higher false-positive rate.", a, known) is None
    assert strip_internal_ids("leakage (Validation Warnings). Next.") == "leakage. Next."
    assert strip_internal_ids("Metrics matter. This aligns with\n\nNext para.") == (
        "Metrics matter.\n\nNext para."
    )


def test_v20_claims_are_checked() -> None:
    from cyberlab_manuscript.consistency import COMPARE, fpr_claim_problem

    def m(fpr: float) -> dict[str, Any]:
        return {"aggregate": {"false_positive_rate": {"mean": fpr}}, "per_class": {}}

    a = {"models": {"lightgbm": m(0.2247), "random_forest": m(0.3897)}}
    known = ["lightgbm", "random_forest"]
    s = (
        "The higher macro-F1 scores suggest better performance, but random forest has a lower "
        "false positive rate compared to LightGBM."
    )
    assert fpr_claim_problem(s, a, known)
    assert COMPARE.search("LightGBM achieves different performance from random forest")
    low = "LightGBM has lower recall for Worms and Backdoor."
    assert "Worms" in (class_claim_problem(low, _analysis(), known) or "")
    assert strip_internal_ids("# Title\n\n## Related Work\n\n# Deep Learning\nText.") == (
        "# Title\n\n## Related Work\n\nText."
    )
