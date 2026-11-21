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
