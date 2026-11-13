"""Offline responder used when LLM_PROVIDER=fake (tests, demos without network or GPU).

It returns clearly-labelled template text so the whole pipeline can run end to end. It never
produces numbers: those always come from the evidence store.
"""

from __future__ import annotations

import json
import re

_SECTIONS = {
    "abstract": (
        "[Offline draft] We compare classical and gradient-boosted classifiers for multiclass "
        "intrusion detection on a de-duplicated version of CIC-IDS2017 {{cite:Sharafaldin2018dataset}}. "
        "The best model reaches a macro-F1 of {{metric:BEST:macro_f1}}."
    ),
    "introduction": (
        "Network intrusion detection remains an active research area. Public benchmarks such as "
        "CIC-IDS2017 {{cite:Sharafaldin2018dataset}} enable comparison, but duplicated flows can "
        "inflate reported scores."
    ),
    "related_work": "Prior work on CIC-IDS2017 is summarised in the evidence matrix; see the verified references.",
    "methodology": (
        "We removed rows with invalid values, exact duplicates and conflicting labels, then used a "
        "stratified split with no train/test overlap. Each model was trained with several seeds."
    ),
    "results": (
        "Hypothesis H1 was {{hyp:H1}}. The best model obtained {{metric:BEST:macro_f1}} "
        "({{ci:BEST:macro_f1}})."
    ),
    "discussion": "Per-class results show where minority attack classes remain difficult.",
    "limitations": (
        "CIC-IDS2017 is a lab-generated dataset with known labelling issues; results may not "
        "transfer to production networks."
    ),
    "conclusion": "Evidence-linked reporting makes the comparison reproducible.",
}


def offline_responder(task: str, system: str, prompt: str) -> str:
    if task == "orchestrator.plan":
        return json.dumps(
            {
                "objective": "Compare intrusion detection models on de-duplicated CIC-IDS2017.",
                "search_queries": [
                    "CIC-IDS2017 intrusion detection machine learning",
                    "network intrusion detection gradient boosting",
                    "duplicate records intrusion detection dataset evaluation",
                ],
                "tasks": [{"agent": "literature", "task": "survey CIC-IDS2017 studies"}],
                "risks": ["dataset duplicates inflate scores", "class imbalance"],
            }
        )
    if task == "literature.matrix":
        keys = re.findall(r"UNTRUSTED PAPER (\S+):", prompt)
        return json.dumps(
            {
                "matrix": [
                    {
                        "key": k,
                        "method": "see abstract",
                        "dataset": "",
                        "finding": "",
                        "limitation": "",
                    }
                    for k in keys
                ],
                "summary": f"[Offline] {len(keys)} verified papers collected.",
            }
        )
    if task == "gap.analysis":
        return json.dumps(
            {
                "gaps": [
                    {
                        "gap": "Few studies report results after removing duplicate flows.",
                        "why_it_matters": "Duplicates leak between splits and inflate metrics.",
                        "testable_with": "de-duplicated stratified split",
                    }
                ],
                "novelty_caveats": ["CIC-IDS2017 is widely studied."],
            }
        )
    if task == "methodology.protocol":
        return json.dumps({"rationale": "[Offline] baseline protocol kept."})
    if task == "statistics.interpret":
        return json.dumps(
            {
                "interpretation": "[Offline] See the analysis artifact for numbers.",
                "caveats": ["single dataset"],
                "recommended_followups": ["evaluate on a by-day split"],
            }
        )
    if task == "writing.draft":
        ctx = json.loads(prompt) if prompt.strip().startswith("{") else {}
        models = ctx.get("models") or ["xgboost"]
        best = "xgboost" if "xgboost" in models else models[0]
        return json.dumps({k: v.replace("BEST", best) for k, v in _SECTIONS.items()})
    if task == "reviewer.review":
        return json.dumps(
            {
                "scores": {
                    "scientific_contribution": 2,
                    "methodological_rigor": 4,
                    "reproducibility": 5,
                    "statistical_validity": 4,
                    "clarity": 3,
                    "threats_to_validity": 3,
                },
                "major_issues": ["Novelty is limited for a widely studied dataset."],
                "minor_issues": [],
                "recommendation": "major revision",
            }
        )
    return "{}"
