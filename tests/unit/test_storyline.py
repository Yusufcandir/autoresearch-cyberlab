from cyberlab_manuscript.builder import EvidenceReport, render_text
from cyberlab_manuscript.evidence_text import lock_evidence
from cyberlab_orchestrator.agents.storyline import (
    evidence_brief,
    related_themes,
    sameness_flags,
)


def _analysis() -> dict:
    def model(f1: float, fpr: float, rec: float) -> dict:
        return {
            "role": "baseline",
            "aggregate": {
                "macro_f1": {"mean": f1, "sd": 0.01, "n_seeds": 3},
                "accuracy": {"mean": 0.99, "sd": 0.0, "n_seeds": 3},
                "balanced_accuracy": {"mean": f1, "sd": 0.0, "n_seeds": 3},
                "false_positive_rate": {"mean": fpr, "sd": 0.0, "n_seeds": 3},
            },
            "macro_f1_ci": {"low": f1 - 0.02, "high": f1 + 0.02},
            "per_class": {
                "BENIGN": {"recall": 0.99, "support": 1000},
                "Worms": {"recall": rec, "support": 9, "recall_ci": [0.6, 1.0]},
            },
        }

    return {
        "models": {"xgboost": model(0.9, 0.2, 1.0), "logistic_regression": model(0.6, 0.3, 0.2)},
        "hypotheses": [
            {
                "id": "H1",
                "a": "xgboost",
                "b": "logistic_regression",
                "metric": "macro_f1",
                "statement": "x",
                "mcnemar": {"p_value": 0.0001},
                "p_holm": 0.0001,
                "result": "supported",
                "a_mean": 0.9,
                "b_mean": 0.6,
            }
        ],
        "metrics_recomputed_match": True,
    }


def test_brief_uses_placeholders_only() -> None:
    brief = evidence_brief(_analysis(), {})
    text = " ".join(f["text"] for f in brief["findings"])
    assert "{{hyp:H1}}" in text and "{{recall:xgboost:Worms}}" in text
    assert "0.9" not in text  # no raw numbers reach the model


def test_recall_placeholder_renders_exact_interval() -> None:
    out = render_text(
        "Recall was {{recall:xgboost:Worms}}.", _analysis(), {}, EvidenceReport(), "d"
    )
    assert "1.0000 (9 test rows, 95% exact interval [0.6000, 1.0000])" in out


def test_themes_group_papers() -> None:
    cites = {
        "A": {"status": "Verified", "title": "Deep LSTM for intrusion detection"},
        "B": {"status": "Verified", "title": "SMOTE oversampling for rare attacks"},
        "C": {"status": "Unverified", "title": "x"},
    }
    themes = {t["theme"]: t["papers"] for t in related_themes(cites, {})}
    assert themes["deep learning architectures"] == ["A"]
    assert themes["class imbalance and rare attacks"] == ["B"] and "C" not in str(themes)


def test_sameness_flags_repeated_wording() -> None:
    old = ["Network intrusion detection remains an active research area with many open problems."]
    flags = sameness_flags(
        {
            "introduction": "Network intrusion detection remains an active research area with many "
            "open problems. Our study differs."
        },
        old,
    )
    assert len(flags) == 1 and flags[0]["section"] == "introduction"


def test_rich_lock_keeps_results_in_discussion_but_not_mechanisms() -> None:
    md = (
        "# T\n\n## Abstract\n\nWe report H1 (supported) for XGBoost against logistic regression.\n\n"
        "## Discussion\n\nXGBoost reached a macro-F1 of 0.9000 ± 0.0100, so it suits rare "
        "attacks in this setting. Its superior performance can be attributed to its ability to "
        "capture complex interactions.\n\n## Limitations\n\nL.\n\n## Conclusion\n\nC.\n"
    )
    out, dropped = lock_evidence(md, _analysis(), rich=True)
    disc = out.split("## Discussion")[1].split("## Limitations")[0]
    assert "0.9000" in disc  # result statement kept (numbers come from placeholders)
    assert "attributed to" not in disc  # untested mechanism removed
    assert any("mechanism" in d["reason"] for d in dropped)
    abstract = out.split("## Abstract")[1].split("## Discussion")[0]
    assert abstract.count("H1 (supported)") == 1  # findings not appended twice
