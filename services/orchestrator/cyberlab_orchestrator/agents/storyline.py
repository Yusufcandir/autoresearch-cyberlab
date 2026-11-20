"""Writing pipeline v2: evidence brief -> story outline -> full prose -> referee -> revision.

The system decides WHAT is true (numbers, test outcomes, which claims are allowed); the model
decides HOW to say it. Facts reach the model as an evidence brief whose numbers are placeholders,
so every number in the paper is still rendered from stored results and checked afterwards.

Why: the professor found the earlier papers flat and alike. They were assembled from fixed
template sentences with the model only allowed short framing text. Here the model writes a
connected argument for each study, a referee pass criticises it like a journal reviewer, and a
sameness check stops one paper from reusing another's sentences.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cyberlab_llm import LLMError
from cyberlab_manuscript.consistency import accuracy_gap_models
from cyberlab_manuscript.evidence_text import pretty, primary_metric
from cyberlab_orchestrator.context import LabContext

# --------------------------------------------------------------------------- evidence brief

THEMES: list[tuple[str, re.Pattern[str]]] = [
    (
        "deep learning architectures",
        re.compile(r"deep|neural|cnn|lstm|transformer|attention|gan|autoencoder", re.I),
    ),
    (
        "tree ensembles and boosting",
        re.compile(r"forest|boost|xgboost|lightgbm|ensemble|tree", re.I),
    ),
    (
        "class imbalance and rare attacks",
        re.compile(r"imbalanc|smote|oversampl|minority|rare|cost[- ]sensitive|focal", re.I),
    ),
    (
        "feature selection and representation",
        re.compile(
            r"feature (?:selection|analysis|set)|pca|chi-square|information gain|netflow", re.I
        ),
    ),
    (
        "datasets, benchmarks and evaluation",
        re.compile(r"dataset|benchmark|evaluation|survey|review|duplicate|generali[sz]", re.I),
    ),
    ("federated, IoT and other settings", re.compile(r"federated|iot|edge|quantum|transfer", re.I)),
]


def related_themes(
    cites: dict[str, dict[str, Any]], matrix: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Group the verified papers by theme (method text of the evidence matrix, else the title),
    so Related Work compares approaches instead of listing papers one by one."""
    groups: dict[str, list[str]] = {}
    for key, c in cites.items():
        if c.get("status") != "Verified":
            continue
        text = " ".join(
            str(x) for x in ((matrix.get(key) or {}).get("method"), c.get("title")) if x
        )
        theme = next((name for name, pat in THEMES if pat.search(text)), "other related work")
        groups.setdefault(theme, []).append(key)
    return [{"theme": t, "papers": keys} for t, keys in groups.items()]


def evidence_brief(analysis: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    """What the study found, ranked, with every number as a placeholder."""
    metric = primary_metric(analysis)
    models = analysis.get("models", {})
    findings: list[dict[str, str]] = []
    ranked = sorted(
        models,
        key=lambda n: -((models[n].get("aggregate", {}).get(metric) or {}).get("mean") or 0.0),
    )
    if ranked:
        findings.append(
            {
                "id": "F1",
                "kind": "ranking by measured mean (not a test)",
                "text": "; ".join(
                    f"{pretty(n)}: {{{{metric:{n}:{metric}}}}} ({{{{ci:{n}:{metric}}}}})"
                    for n in ranked
                ),
            }
        )
    for h in analysis.get("hypotheses", []):
        if "mcnemar" in h:
            a, b, hm = h["a"], h["b"], h.get("metric", metric)
            findings.append(
                {
                    "id": h["id"],
                    "kind": "tested hypothesis (paired permutation test on its metric, Holm)",
                    "text": f"{pretty(a)} versus {pretty(b)} on {hm}: {{{{hyp:{h['id']}}}}}; "
                    f"means {{{{metric:{a}:{hm}}}}} and {{{{metric:{b}:{hm}}}}}; the difference "
                    "with its 95% CI and the odds ratio are in the hypothesis table",
                }
            )
    gap = accuracy_gap_models(analysis)
    if gap:
        findings.append(
            {
                "id": "F-acc",
                "kind": "accuracy hides minority-class weakness",
                "text": "; ".join(
                    f"{pretty(n)}: accuracy {{{{metric:{n}:accuracy}}}} versus balanced "
                    f"accuracy {{{{metric:{n}:balanced_accuracy}}}}"
                    for n in gap
                ),
            }
        )
    fpr = {
        n: (m.get("aggregate", {}).get("false_positive_rate") or {}).get("mean")
        for n, m in models.items()
    }
    if any(isinstance(v, float) and v >= 0.05 for v in fpr.values()):
        findings.append(
            {
                "id": "F-fpr",
                "kind": "operational: false alarms",
                "text": "; ".join(
                    f"{pretty(n)}: {{{{metric:{n}:false_positive_rate}}}}" for n in fpr
                ),
            }
        )
    if ranked:
        best = ranked[0]
        pc = models[best].get("per_class") or {}
        small = sorted(pc, key=lambda c: pc[c].get("support", 0))[:3]
        findings.append(
            {
                "id": "F-small",
                "kind": "smallest classes (first seed)",
                "text": "; ".join(
                    f"{c} ({pc[c].get('support', 0):,} test rows): recall of {pretty(best)} "
                    f"{{{{recall:{best}:{c}}}}}"
                    for c in small
                ),
            }
        )
        missed = [
            c
            for c in pc
            if c != "BENIGN"
            and all(
                ((m.get("per_class") or {}).get(c) or {}).get("recall", 1) == 0
                for m in models.values()
            )
        ]
        if missed:
            findings.append(
                {"id": "F-missed", "kind": "classes no model detected", "text": ", ".join(missed)}
            )
    tuned = [n for n, m in models.items() if m.get("tuned")]
    return {
        "primary_metric": metric,
        "findings": findings,
        "tested_pairs": [
            f"{h['a']} vs {h['b']}" for h in analysis.get("hypotheses", []) if h.get("a")
        ],
        "tuned_models": tuned,
        "facts": facts,
    }


# --------------------------------------------------------------------------- model calls

OUTLINE_SYSTEM = (
    "You plan the argument of a research paper in network security from an evidence brief. "
    "Find what is actually interesting in THIS study: the main finding, what is surprising, "
    "what a practitioner should take away, and how it relates to the prior work themes. Do not "
    "invent results: refer to findings by their id. Output JSON: {'main_claim': one sentence "
    "without numbers, 'contribution': one sentence, 'introduction': [3-4 points in order], "
    "'related_work': [{'theme', 'papers': [keys], 'point': how these works relate to this "
    "study}], 'discussion': [3-5 points, each naming finding ids], 'limitations': [3-4 points "
    "specific to this study], 'conclusion': one sentence}."
)

# Professor's plan, Step 6.1: twelve review dimensions
RUBRIC = (
    "scientific_correctness",
    "research_motivation",
    "research_gap_clarity",
    "methodological_completeness",
    "statistical_validity",
    "results_interpretation",
    "critical_discussion",
    "citation_integrity",
    "logical_coherence",
    "academic_writing_quality",
    "reproducibility",
    "consistency_between_sections",
)

REFEREE_SYSTEM = (
    "You are a demanding referee for a peer-reviewed security venue. Read the draft sections "
    "and report concrete problems: claims the evidence brief does not support or that overstate "
    "it; vague or generic sentences ('plays a crucial role', 'in today's world'); repetition "
    "across sections; Related Work that lists papers instead of comparing them; a Discussion "
    "that only restates numbers instead of interpreting them; missing connection between the "
    "findings and prior work; inconsistent terminology; weak flow between paragraphs. Quote the "
    "exact words. Score the draft 1-5 on each rubric dimension: " + ", ".join(RUBRIC) + ". "
    "Mark an issue 'major' when it affects scientific correctness, statistical validity, "
    "citation integrity or consistency with the evidence, else 'minor'. Never invent "
    "replacement evidence. Output JSON: {'rubric': {dimension: score}, 'issues': [{'section', "
    "'quote', 'problem', 'fix', 'severity'}], 'verdict': 'accept' | 'revise'} with at most 10 "
    "issues, most important first."
)


def _call(ctx: LabContext, pid: str, task: str, system: str, payload: dict[str, Any]) -> Any:
    try:
        return ctx.llm.complete_json(task, system, json.dumps(payload, default=str), project_id=pid)
    except LLMError:
        return {}


def outline(
    ctx: LabContext, pid: str, brief: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    got = _call(
        ctx,
        pid,
        "writing.outline",
        OUTLINE_SYSTEM,
        {
            "title": context.get("title"),
            "goal": context.get("goal"),
            "evidence_brief": brief,
            "related_work_themes": context.get("related_work_themes"),
            "papers": {k: v.get("title") for k, v in (context.get("papers") or {}).items()},
        },
    )
    return got if isinstance(got, dict) else {}


def critique(
    ctx: LabContext, pid: str, sections: dict[str, str], brief: dict[str, Any]
) -> dict[str, Any]:
    """Manuscript critic (professor's plan, Step 6.1): rubric scores and concrete issues."""
    got = _call(
        ctx,
        pid,
        "writing.referee",
        REFEREE_SYSTEM,
        {"evidence_brief": brief, "sections": sections},
    )
    got = got if isinstance(got, dict) else {}
    issues = got.get("issues")
    issues = [i for i in issues if isinstance(i, dict)][:10] if isinstance(issues, list) else []
    raw = got.get("rubric")
    rubric: dict[str, Any] = raw if isinstance(raw, dict) else {}
    scores = {d: rubric.get(d) for d in RUBRIC if isinstance(rubric.get(d), (int, float))}
    return {"issues": issues, "rubric": scores}


def referee(
    ctx: LabContext, pid: str, sections: dict[str, str], brief: dict[str, Any]
) -> list[dict[str, Any]]:
    return list(critique(ctx, pid, sections, brief)["issues"])


def revise(
    ctx: LabContext,
    pid: str,
    system: str,
    sections: dict[str, str],
    issues: list[dict[str, Any]],
    context: dict[str, Any],
) -> dict[str, str]:
    """Rewrite only what the issues point at; sections the model does not return are kept."""
    keys = sorted(
        {str(i.get("section", "")).lower().replace(" ", "_") for i in issues} & set(sections)
    )
    if not keys:
        return sections
    got = _call(
        ctx,
        pid,
        "writing.revise",
        system
        + "\nRevise the given sections to fix every listed issue. Keep what is not criticised. "
        "Output JSON with exactly these keys: " + ", ".join(keys) + ".",
        {"issues": issues, "sections": {k: sections[k] for k in keys}, "context": context},
    )
    out = dict(sections)
    if isinstance(got, dict):
        for k in keys:
            v = got.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v
    return out


# --------------------------------------------------------------------------- sameness

FRAMING = ("abstract", "introduction", "related_work", "discussion", "limitations", "conclusion")


def _shingles(text: str, n: int = 6) -> set[str]:
    words = re.findall(r"[a-z]+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def sameness_flags(
    sections: dict[str, str], corpus: list[str], threshold: float = 0.5
) -> list[dict[str, str]]:
    """Sentences that largely repeat earlier papers (6-word shingle overlap)."""
    seen: set[str] = set()
    for text in corpus:
        seen |= _shingles(text)
    flags = []
    for key in FRAMING:
        for sent in re.split(r"(?<=[.!?])\s+", sections.get(key, "")):
            sh = _shingles(re.sub(r"\{\{[^}]+\}\}", " ", sent))
            if len(sh) >= 4 and len(sh & seen) / len(sh) >= threshold:
                flags.append(
                    {
                        "section": key,
                        "quote": sent.strip()[:300],
                        "problem": "this sentence repeats wording from an earlier paper",
                        "fix": "say it in new words that fit this study's findings",
                    }
                )
    return flags[:10]


def evidence_package(
    title: str,
    goal: str,
    analysis: dict[str, Any],
    protocol: dict[str, Any],
    cites: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """verified_evidence_package.json (professor's plan, Step 5.3): the only facts the writer
    receives. Every numerical claim has an id, its value, its source artifact and the
    validation status of the analysis it comes from."""
    validation = analysis.get("validation") or {}
    status = (
        "PASS"
        if validation.get("status") in ("PASS", "WARNING")
        else validation.get("status", "NOT_EVALUATED")
    )
    claims: list[dict[str, Any]] = []
    n = 0
    for name, m in analysis.get("models", {}).items():
        for metric, v in (m.get("aggregate") or {}).items():
            n += 1
            claims.append(
                {
                    "claim_id": f"C{n:03d}",
                    "statement": f"{name} {metric} mean over {v.get('n_seeds', 1)} seeds",
                    "evidence_type": "experiment_result",
                    "source_artifact": analysis.get("artifact_sha256"),
                    "model": name,
                    "metric": metric,
                    "value": v.get("mean"),
                    "sd": v.get("sd"),
                    "placeholder": f"{{{{metric:{name}:{metric}}}}}",
                    "validation_status": status,
                }
            )
    for h in analysis.get("hypotheses", []):
        n += 1
        claims.append(
            {
                "claim_id": f"C{n:03d}",
                "statement": f"{h.get('id')}: {h.get('statement')}",
                "evidence_type": "hypothesis_test",
                "source_artifact": analysis.get("artifact_sha256"),
                "result": h.get("result"),
                "p_holm": h.get("p_holm"),
                "test": (h.get("test") or {}).get("method"),
                "placeholder": f"{{{{hyp:{h.get('id')}}}}}",
                "validation_status": status,
            }
        )
    return {
        "research_title": title,
        "research_objectives": goal,
        "research_questions": protocol.get("research_questions", []),
        "hypotheses": protocol.get("hypotheses", []),
        "dataset": {
            k: (analysis.get("dataset") or {}).get(k)
            for k in (
                "name",
                "license",
                "n_rows_raw",
                "n_rows_clean",
                "class_counts",
                "split_counts",
            )
        },
        "experimental_configuration": {
            "models": {n_: m.get("params") for n_, m in analysis.get("models", {}).items()},
            "seeds": protocol.get("seeds"),
            "split": protocol.get("split"),
            "tuning": protocol.get("tuning"),
        },
        "claims": claims,
        "verified_references": [k for k, c in cites.items() if c.get("status") == "Verified"],
        "limitations_and_warnings": [
            {"check": c["check"], "status": c["status"], "detail": c["detail"]}
            for c in validation.get("checks", [])
            if c.get("status") in ("WARNING", "NOT_EVALUATED")
        ],
        "validation_status": validation.get("status", "NOT_EVALUATED"),
    }
