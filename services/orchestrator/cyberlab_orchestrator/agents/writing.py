"""Agent 9: Scientific Writing. Writes prose with evidence placeholders; numbers come from evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy import func, select

from cyberlab_common.db import CitationRecord, ManuscriptRecord, Project, ProtocolVersion, audit
from cyberlab_literature import Paper, bibtex_entry
from cyberlab_llm import LLMError
from cyberlab_manuscript import build_manuscript, manuscript_sha
from cyberlab_manuscript.consistency import accuracy_gap_models
from cyberlab_manuscript.evidence_text import lock_evidence
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ResearchSpec

MAX_REVISIONS = 1

SYSTEM = (
    "You are a scientific writing agent drafting an intrusion-detection paper. Rules, strictly:\n"
    "1. NEVER type a number yourself (no percentages, counts, scores or sizes). Use only "
    "placeholders: {{metric:MODEL:METRIC}} (METRIC in macro_f1, accuracy, balanced_accuracy, mcc, "
    "false_positive_rate, detection_rate), {{ci:MODEL:macro_f1}} (the system replaces it with the full interval text including '95% CI' and the seed, so never write '95%', 'CI', brackets or a seed around it, and never write an interval yourself), {{fact:NAME}} for any method "
    "detail listed under 'facts', and {{hyp:H1}} which renders as 'H1 (supported)' or 'H1 (not supported)', so write e.g. 'the results support {{hyp:H1}}'. MODEL must "
    "be a real model name from 'models'.\n"
    "2. Describe the method ONLY with what is in 'method_facts' and 'facts'. Do not invent "
    "procedures (cross-validation, per-seed splits, tuning) that are not listed.\n"
    "3. Do not claim comparisons that were not run. There was NO experiment on non-deduplicated "
    "data, so do not say that de-duplication changed or lowered the scores.\n"
    "4. Cite with {{cite:KEY}} or {{cite:KEY1, KEY2}} using only keys in 'papers'. Say about a "
    "paper only what its title and abstract support; name authors exactly as in 'authors'. In related_work, cite every paper at least once.\n"
    "5. Be precise and modest; mention CIC-IDS2017 limitations.\n"
    "6. Compare two models (outperforms, better than, comparable to, superior) ONLY for the "
    "pairs listed in 'tested_comparisons'. Never call one model the best overall unless every "
    "pair was tested. Describe other models by their own scores, without comparing.\n"
    "7. To state a hypothesis result write only {{hyp:H1}}; never add your own 'not', "
    "'rejected' or 'supported' next to it, because the placeholder already says the result.\n"
    "8. Never put {{metric:...}} or {{ci:...}} in introduction, related_work or methodology; "
    "in methodology name models in words (logistic regression, random forest, ...).\n"
    "9. Use {{fact:NAME}} only with a NAME exactly as listed in 'facts'. The split was checked "
    "and has zero train/test overlap rows, so do not suggest data leakage.\n"
    "10. 'statistician_notes' are Proposed readings, not findings. Say 'high accuracy hides weak "
    "minority-class performance' (or 'simplistic decision boundaries') only about models listed "
    "in 'models_where_accuracy_exceeds_balanced_accuracy', and name them.\n"
    "11. The 95% CI is a bootstrap over test rows for one seed; variation across seeds is the "
    "± sd. Do not describe the CI as variation across seeds.\n"
    "12. The abstract and conclusion must state the main findings with {{hyp:...}} and "
    "{{metric:...}} placeholders; never write a bare H1/H2 with your own verdict.\n"
    "13. Write for readers of the paper: never mention 'statistician notes', 'reviewer "
    "feedback', placeholders or other internal tools. Every number must say which metric it "
    "is (e.g. 'a macro-F1 of {{metric:xgboost:macro_f1}}').\n"
    "Output JSON with keys abstract, introduction, related_work, methodology, results, "
    "discussion, limitations, conclusion (markdown strings)."
)


def method_facts(
    ctx: LabContext, pid: str, analysis: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Deterministic, true statements about how the study was run (no LLM involved)."""
    with ctx.db.session() as s:
        pv = s.scalar(
            select(ProtocolVersion).where(
                ProtocolVersion.project_id == pid,
                ProtocolVersion.version == analysis["protocol_version"],
            )
        )
        body: dict[str, Any] = pv.body if pv else {}
    notes_path = ctx.settings.dataset_root / "cicids2017" / "prepared" / "preprocessing_notes.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    ds = analysis.get("dataset") or {}
    split = body.get("split", {})
    seeds = body.get("seeds", [])
    first: dict[str, Any] = next(iter(analysis.get("models", {}).values()), {})
    ci = first.get("macro_f1_ci", {})
    support = {c: v.get("support", 0) for c, v in (first.get("per_class") or {}).items()}
    smallest = (
        f"{min(support, key=support.__getitem__)} ({min(support.values()):,} test rows)"
        if support
        else ""
    )
    facts: dict[str, Any] = {
        "n_seeds": len(seeds),
        "seeds": ", ".join(str(x) for x in seeds),
        "split_seed": split.get("seed"),
        "test_percent": f"{round(100 * float(split.get('test_fraction', 0)))}%",
        "val_percent": f"{round(100 * float(split.get('val_fraction', 0)))}%",
        "rows_raw": f"{ds.get('n_rows_raw', 0):,}",
        "rows_clean": f"{ds.get('n_rows_clean', 0):,}",
        "rows_removed_duplicates_or_conflicts": f"{ds.get('n_duplicates_removed', 0):,} rows",
        "rows_dropped_inf_nan": f"{notes.get('rows_with_inf_or_nan_dropped', 0):,} rows",
        "benign_cap": f"{notes.get('benign_cap') or 0:,} rows",
        # separate facts, so a writer cannot put the class name where a number belongs
        "smallest_class": smallest.split(" (")[0] if smallest else "",
        "smallest_class_test_rows": smallest.split("(")[1].split(" ")[0] if "(" in smallest else "",
        "n_features": len(ds.get("features", [])),
        "n_classes": len(ds.get("class_counts", {})),
        "test_rows": f"{ds.get('split_counts', {}).get('test', 0):,}",
        "train_rows": f"{ds.get('split_counts', {}).get('train', 0):,}",
        "bootstrap_resamples": ci.get("n_boot", 500),
        "ci_seed": first.get("ci_seed"),
        "ci_level": "95%",
    }
    prose = [
        "Rows with infinite or missing feature values were removed (facts.rows_dropped_inf_nan).",
        "Exact duplicate rows (same features and label) were removed, and feature vectors that "
        "appeared with conflicting labels were removed entirely "
        "(facts.rows_removed_duplicates_or_conflicts in total).",
        "Constant features and the duplicated Fwd Header Length column were dropped.",
        "Fine-grained labels were grouped into classes: BENIGN, DoS, DDoS, PortScan, BruteForce "
        "(FTP/SSH-Patator), WebAttack, Bot, and Rare (Infiltration and Heartbleed).",
        "BENIGN rows were randomly down-sampled to at most facts.benign_cap.",
        "One stratified train/validation/test split was made with a single fixed seed "
        "(facts.split_seed) after de-duplication; the same split is used for every model and seed. "
        "No test row also appears in training.",
        "The model seeds (facts.seeds) change only the randomness of model training, not the split.",
        "Every model was trained on the training split and evaluated once on the test split; no "
        "hyper-parameter tuning was performed (fixed, documented defaults).",
        "Metrics are reported as mean and standard deviation over the model seeds.",
        "Confidence intervals are percentile bootstrap intervals (facts.bootstrap_resamples "
        "resamples) over test rows, stratified by class, computed on the first seed's "
        "predictions (seed facts.ci_seed).",
        "Hypotheses were tested with an exact McNemar test on per-row correctness of two models "
        "over the same test rows (first seed).",
        "All metrics were recomputed from stored per-row predictions; numbers in the paper are "
        "inserted from these stored results.",
    ]
    for name, m in analysis.get("models", {}).items():
        prose.append(f"Model '{name}' uses the '{m.get('template')}' template ({m.get('role')}).")
    return facts, prose


SECTION_GROUPS = [
    ["introduction", "related_work"],
    ["methodology", "results"],
    ["abstract", "discussion", "limitations", "conclusion"],
]


def write_sections(ctx: LabContext, pid: str, context: dict[str, Any]) -> dict[str, str]:
    """Write the paper in three parallel calls (about 3x faster than one long call).

    Every call sees the same facts, papers and results, so the parts stay consistent.
    """
    payload = json.dumps(context, default=str)

    def one(keys: list[str]) -> dict[str, str]:
        system = (
            SYSTEM + "\nFor THIS call write ONLY these sections and output JSON with exactly "
            "these keys: " + ", ".join(keys) + ". Other sections are written separately."
        )
        out: dict[str, str] = {}
        for _attempt in range(2):
            try:
                got = ctx.llm.complete_json("writing.draft", system, payload, project_id=pid)
            except LLMError:
                continue
            out = {k: str(got[k]) for k in keys if isinstance(got.get(k), str | int | float)}
            if out:
                break
        return out

    with ThreadPoolExecutor(max_workers=len(SECTION_GROUPS)) as pool:
        parts = list(pool.map(one, SECTION_GROUPS))
    sections: dict[str, str] = {}
    for part in parts:
        sections.update(part)
    wanted = [k for group in SECTION_GROUPS for k in group]
    missing = [k for k in wanted if not sections.get(k, "").strip()]
    if missing:  # the model sometimes drops a key: ask once more for just those sections
        sections.update(one(missing))
    if not sections:
        raise LLMError("the writing model returned no sections")
    return sections


def writing_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    ctx.authorize("writing", "draft_manuscript", "T0", pid)
    analysis = _load_analysis(ctx, pid, state["analysis_sha256"])
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        assert project is not None
        spec = ResearchSpec.model_validate(project.spec)
        cites: dict[str, dict[str, Any]] = {
            c.key: {
                "status": c.status,
                "doi": c.doi,
                "title": c.title,
                "authors": c.authors,
                "year": c.year,
                "venue": c.venue,
                "abstract": c.abstract,
            }
            for c in s.scalars(select(CitationRecord).where(CitationRecord.project_id == pid))
        }
    verified_keys = [k for k, c in cites.items() if c["status"] == "Verified"]
    analysis["artifact_sha256"] = state["analysis_sha256"]
    facts, prose = method_facts(ctx, pid, analysis)
    analysis["facts"] = facts
    context = {
        "title": spec.title,
        "goal": spec.goal,
        "models": list(analysis["models"]),
        "hypotheses": [
            {"id": h["id"], "statement": h["statement"]} for h in analysis["hypotheses"]
        ],
        "papers": {
            k: {
                "title": cites[k]["title"],
                "year": cites[k]["year"],
                "authors": (cites[k].get("authors") or [])[:6],
                "abstract": (cites[k].get("abstract") or "")[:2500],
            }
            for k in verified_keys
        },
        "tested_comparisons": [
            f"{h['a']} vs {h['b']} ({h['id']})"
            for h in analysis["hypotheses"]
            if h.get("a") and h.get("b")
        ],
        "models_where_accuracy_exceeds_balanced_accuracy": accuracy_gap_models(analysis),
        "facts": facts,
        "method_facts": prose,
        "dataset": {k: analysis["dataset"].get(k) for k in ("name", "license")}
        if analysis.get("dataset")
        else {},
        "statistician_notes": state.get("interpretation", {}),
        "reviewer_feedback": state.get("review_feedback", []),
    }
    sections = write_sections(ctx, pid, context)
    # Final revision: the deterministic guardrail removes anything still unsupported.
    final = state.get("revisions", 0) >= MAX_REVISIONS
    md, report = build_manuscript(spec.title, sections, analysis, cites, "", guardrail=final)
    # Evidence statements are generated, never written by the model (see evidence_text).
    md, dropped = lock_evidence(md, analysis, facts, prose)
    if dropped:
        report.flag(
            "warning",
            "FRAMING_SENTENCES_DROPPED",
            f"{len(dropped)} model-written sentence(s) stated results and were replaced by "
            "evidence-generated text: "
            + " | ".join(f'{d["section"]}: "{d["sentence"][:100]}"' for d in dropped[:12]),
        )
    bib = "\n\n".join(
        bibtex_entry(
            Paper(
                title=c["title"],
                doi=c["doi"],
                year=c["year"],
                authors=c["authors"] or [],
                venue=c["venue"],
            ),
            key=k,
        )
        for k, c in cites.items()
        if c["status"] == "Verified"
    )
    with ctx.db.session() as s:
        version = (
            s.scalar(
                select(func.max(ManuscriptRecord.version)).where(ManuscriptRecord.project_id == pid)
            )
            or 0
        ) + 1
        s.add(
            ManuscriptRecord(
                project_id=pid,
                version=version,
                markdown=md,
                bibtex=bib,
                sha256=manuscript_sha(md),
                evidence_report=report.as_dict(),
                status="draft",
            )
        )
        audit(
            s,
            "writing_agent",
            "manuscript.drafted",
            pid,
            {"version": version, "blockers": len(report.blockers)},
        )
    ctx.set_stage(pid, "drafted", "Unverified" if report.blockers else "Measured")
    return {"manuscript_version": version, "evidence_blockers": report.blockers}


def _load_analysis(ctx: LabContext, pid: str, sha: str) -> dict[str, Any]:
    with ctx.db.session() as s:
        return dict(json.loads(ctx.store.path_for(s, pid, sha).read_text()))
