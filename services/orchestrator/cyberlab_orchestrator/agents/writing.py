"""Agent 9: Scientific Writing. Writes prose with evidence placeholders; numbers come from evidence."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sqlalchemy import func, select

from cyberlab_common.db import CitationRecord, ManuscriptRecord, Project, ProtocolVersion, audit
from cyberlab_data import get_adapter, prepared_dir_for
from cyberlab_literature import Paper, bibtex_entry
from cyberlab_llm import BudgetExceeded, LLMError
from cyberlab_manuscript import build_manuscript, manuscript_sha
from cyberlab_manuscript.builder import link_author_year, strip_internal_ids
from cyberlab_manuscript.consistency import accuracy_gap_models
from cyberlab_manuscript.evidence_text import lock_evidence
from cyberlab_orchestrator.agents.storyline import (
    critique,
    evidence_brief,
    evidence_package,
    outline,
    related_themes,
    revise,
    sameness_flags,
)
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ResearchSpec, SplitSpec

MAX_REVISIONS = 1
REFEREE_ROUNDS = 3

SYSTEM = (
    "You are an experienced researcher writing a paper on network intrusion detection for a "
    "peer-reviewed venue. Write connected, precise prose: each paragraph makes one point and "
    "leads to the next; vary sentence structure; prefer concrete statements about THIS study "
    "over generic ones. Never use filler such as 'in today's digital landscape', 'plays a "
    "crucial role', 'this paper aims to', 'it is worth noting', 'in conclusion'. Follow the "
    "story outline you are given.\n"
    "Facts, strictly:\n"
    "1. NEVER type a number (no percentages, counts, scores, sizes or p-values). Numbers appear "
    "only through placeholders: {{metric:MODEL:METRIC}} (METRIC in macro_f1, accuracy, "
    "balanced_accuracy, mcc, false_positive_rate, detection_rate; renders as mean ± sd over "
    "seeds), {{ci:MODEL:METRIC}} (renders the full 95% bootstrap interval text; do not add "
    "'95%', 'CI' or brackets yourself), {{recall:MODEL:CLASS}} (per-class recall with its exact "
    "interval and the number of test rows), {{fact:NAME}} for a name listed under 'facts', and "
    "{{hyp:H1}} (renders as 'H1 (supported)' or 'H1 (not supported)'; never add your own verdict "
    "next to it). MODEL must be a name from 'models'.\n"
    "2. Describe the method only with 'method_facts' and 'facts'; never invent procedures.\n"
    "3. Compare two models (better, outperforms, comparable) only for the pairs in "
    "'tested_comparisons'; describe other models by their own scores. Never call a model the "
    "best overall unless every pair was tested.\n"
    "4. Do not explain WHY a model did better (its architecture, decision boundaries, ability to "
    "capture interactions): the experiments measured scores, not causes. You may relate a "
    "result to what cited papers report.\n"
    "5. Cite with {{cite:KEY}} or {{cite:KEY1, KEY2}} using only keys in 'papers'; say about a "
    "paper only what its title and abstract support. In related_work, organise the papers by "
    "the themes in 'related_work_themes', compare their approaches with each other and with "
    "this study, and cite every paper.\n"
    "6. The split has zero train/test overlap rows: do not suggest data leakage.\n"
    "7. The abstract and conclusion report the tested hypotheses with {{hyp:...}} and the "
    "key numbers with placeholders. The discussion interprets the findings in 'evidence_brief' "
    "(what they mean for practice, which result is surprising, how it relates to prior work) "
    "instead of repeating the results list. Limitations are specific to this study.\n"
    "8. Write for the paper's readers: never mention placeholders, briefs, outlines, reviewers "
    "or internal tools.\n"
    "9. Refer to tables and figures only with these placeholders, which become numbered "
    "references: {{ref:table_main}}, {{ref:table_hypotheses}}, {{ref:table_per_class}}, "
    "{{ref:table_classes}}, {{ref:figure_ci}}, {{ref:figure_recall}}, {{ref:figure_confusion}}.\n"
    "Output JSON with keys abstract, introduction, related_work, methodology, results, "
    "discussion, limitations, conclusion (markdown strings, no headings)."
)


def split_prose(strategy: str, notes: dict[str, Any]) -> list[str]:
    """True sentences about how the test set was formed, one variant per split strategy."""
    if strategy == "by_day":
        return [
            "Test rows are all flows from the test days (facts.test_days); training and "
            "validation rows come only from the other days, so the models never see traffic "
            "from the test days. Duplicates were removed before splitting and no test row "
            "also appears in training."
        ]
    if strategy == "official":
        return [
            "The dataset's official partition was used: test rows come only from the "
            "official testing file and training and validation rows only from the official "
            "training file. Rows of the testing file that duplicate a training row were "
            "removed before evaluation, so no test row also appears in training."
        ]
    if strategy == "by_group":
        return [
            "Test rows are all flows from the held-out capture scenarios (facts.test_days); "
            "training and validation rows come only from the other scenarios, so the test "
            "botnets were never seen in training. No test row also appears in training."
        ]
    if strategy == "grouped":
        return [
            "The split was grouped by website: all URLs of one host are in the same split "
            "(stratified seed facts.split_seed), so no website in the test set was seen in "
            "training. No test row also appears in training."
        ]
    if strategy == "cross_dataset":
        return [
            f"Cross-dataset evaluation: models were trained on {notes.get('train_dataset')} and "
            f"tested on {notes.get('test_dataset')}, using only the "
            f"{notes.get('shared_features')} features both datasets share. Labels were reduced "
            "to attack versus BENIGN because the attack families differ between the datasets. "
            "Test rows identical to a training row were removed, so no test row also appears "
            "in training."
        ]
    return [
        "One stratified train/validation/test split was made with a single fixed seed "
        "(facts.split_seed) after de-duplication; the same split is used for every model and "
        "seed. No test row also appears in training."
    ]


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
    split_spec = SplitSpec.model_validate(body.get("split", {}))
    adapter = get_adapter(body.get("dataset_id"))
    notes_path = (
        prepared_dir_for(adapter.root(ctx.settings.dataset_root), split_spec)
        / "preprocessing_notes.json"
    )
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
        "test_days": ", ".join(split_spec.test_days),
    }
    prose = [
        "Rows with infinite or missing feature values were removed (facts.rows_dropped_inf_nan).",
        "Exact duplicate rows (same features and label) were removed, and feature vectors that "
        "appeared with conflicting labels were removed entirely "
        "(facts.rows_removed_duplicates_or_conflicts in total).",
        *adapter.preprocessing,
        *(
            ["BENIGN rows were randomly down-sampled to at most facts.benign_cap."]
            if notes.get("benign_cap")
            and (ds.get("class_counts") or {}).get("BENIGN", 0) >= notes["benign_cap"]
            else []
        ),
        (
            "The traffic was captured on a real network (externally observed data)."
            if adapter.observed
            else "The traffic was generated in a laboratory testbed (synthetic data), so results "
            "may not transfer directly to production networks."
        ),
        *split_prose(split_spec.strategy, notes),
        "The model seeds (facts.seeds) change only the randomness of model training, not the split.",
        (
            "Every model was trained on the training split with the settings selected on the "
            "validation split, and evaluated once on the test split."
            if body.get("tuning")
            else "Every model was trained on the training split and evaluated once on the test "
            "split; no hyper-parameter tuning was performed (fixed, documented defaults)."
        ),
        "Metrics are reported as mean and standard deviation over the model seeds.",
        "Confidence intervals are percentile bootstrap intervals (facts.bootstrap_resamples "
        "resamples) over test rows, stratified by class; each resample is scored on every "
        "model seed's predictions and averaged, so the interval belongs to the reported mean.",
        "Each hypothesis was tested on its own metric with a one-sided paired permutation test "
        "(1,000 permutations): the predictions of the two models were swapped on random test "
        "rows, the metric difference was averaged over the model seeds, and the p-value is the "
        "share of permuted differences at least as large as the observed one. P-values were "
        "corrected across hypotheses with the Holm-Bonferroni method; a hypothesis is "
        "supported only if the adjusted p is below 0.05 and the observed difference favours "
        "the stated model.",
        "Three sources of uncertainty are reported separately: test-sample uncertainty (the "
        "bootstrap intervals), model-initialisation variability (the standard deviation over "
        "seeds), and data-partition variability, which a single fixed split cannot measure.",
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
    errors: list[str] = []

    def one(keys: list[str]) -> dict[str, str]:
        system = (
            SYSTEM + "\nFor THIS call write ONLY these sections and output JSON with exactly "
            "these keys: " + ", ".join(keys) + ". Other sections are written separately."
        )
        out: dict[str, str] = {}
        for _attempt in range(2):
            try:
                got = ctx.llm.complete_json("writing.draft", system, payload, project_id=pid)
            except BudgetExceeded:
                raise  # the PI must raise the budget; retrying cannot help
            except LLMError as exc:
                errors.append(str(exc)[:300])
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
        why = f" (last error: {errors[-1]})" if errors else ""
        raise LLMError(f"the writing model returned no sections{why}")
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
    analysis["validation"] = _validation_summary(ctx, pid)
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
    # Writing pipeline v2: evidence brief -> outline -> sections -> referee -> revision
    brief = evidence_brief(analysis, facts)
    brief["validation_warnings"] = [
        {"check": c["check"], "status": c["status"], "detail": c["detail"]}
        for c in (analysis.get("validation") or {}).get("checks", [])
        if c["status"] in ("WARNING", "NOT_EVALUATED")
    ]
    context["evidence_brief"] = brief
    package = evidence_package(
        spec.title, spec.goal, analysis, _protocol_body(ctx, pid, analysis), cites
    )
    context["evidence_package_claims"] = [
        {k: c[k] for k in ("claim_id", "statement", "placeholder")} for c in package["claims"]
    ]
    _store_json(ctx, pid, "verified_evidence_package.json", package)
    context["related_work_themes"] = related_themes(cites, _evidence_matrix(ctx, pid))
    context["outline"] = outline(ctx, pid, brief, context)
    sections = write_sections(ctx, pid, context)
    corpus = _earlier_papers(ctx, pid)
    # Draft -> Review -> Identify problems -> Revise -> Verify, at most REFEREE_ROUNDS times;
    # the build and evidence lock below re-verify numbers and citations after the last round.
    unresolved: list[dict[str, Any]] = []
    for _round in range(REFEREE_ROUNDS + 1):
        review = critique(ctx, pid, sections, brief)
        issues = review["issues"] + sameness_flags(sections, corpus)
        with ctx.db.session() as s:
            ctx.note(
                s,
                pid,
                "writing",
                "manuscript_critique",
                {"round": _round + 1, "rubric": review["rubric"], "issues": issues},
            )
        unresolved = [
            i
            for i in review["issues"]
            if str(i.get("severity", "")).lower() == "major"
            # Methodology and Results are rebuilt from the evidence afterwards
            and str(i.get("section", "")).lower() not in ("results", "methodology")
        ]
        if not issues or _round == REFEREE_ROUNDS:
            break
        sections = revise(ctx, pid, SYSTEM, sections, issues, context)
    # Final revision: the deterministic guardrail removes anything still unsupported.
    final = state.get("revisions", 0) >= MAX_REVISIONS
    models = sorted(
        {
            str(e["model"])
            for e in ctx.llm.log
            if e.get("project_id") == pid and e.get("provider") not in ("fake", None)
        }
    )
    md, report = build_manuscript(
        spec.title, sections, analysis, cites, "", guardrail=final, llm_models=models
    )
    # Evidence statements are generated, never written by the model (see evidence_text).
    md, dropped = lock_evidence(md, analysis, facts, prose, rich=True)
    md = link_author_year(strip_internal_ids(md), cites)
    if unresolved:  # the critic still finds major problems: stop for the authors
        report.flag(
            "warning",
            "HUMAN_REVIEW_REQUIRED",
            f"after {REFEREE_ROUNDS} revision rounds the manuscript critic still reports "
            f"{len(unresolved)} major issue(s): "
            + " | ".join(f"{i.get('section')}: {i.get('problem')}" for i in unresolved[:5]),
        )
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


def _validation_summary(ctx: LabContext, pid: str) -> dict[str, Any]:
    from cyberlab_common.db import AgentNote

    with ctx.db.session() as s:
        note = s.scalar(
            select(AgentNote)
            .where(AgentNote.project_id == pid, AgentNote.kind == "scientific_validation")
            .order_by(AgentNote.id.desc())
        )
        return dict(note.content) if note else {}


def _evidence_matrix(ctx: LabContext, pid: str) -> dict[str, dict[str, Any]]:
    from cyberlab_common.db import AgentNote

    with ctx.db.session() as s:
        note = s.scalar(
            select(AgentNote)
            .where(AgentNote.project_id == pid, AgentNote.kind == "evidence_matrix")
            .order_by(AgentNote.id.desc())
        )
        rows = (note.content.get("matrix") if note else None) or []
    return {r["key"]: r for r in rows if isinstance(r, dict) and r.get("key")}


def _earlier_papers(ctx: LabContext, pid: str, limit: int = 20) -> list[str]:
    """Latest draft of other studies, for the sameness check."""
    with ctx.db.session() as s:
        rows = s.scalars(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id != pid)
            .order_by(ManuscriptRecord.created_at.desc())
            .limit(200)
        )
        latest: dict[str, str] = {}
        for r in rows:
            latest.setdefault(r.project_id, r.markdown)
    return list(latest.values())[:limit]


def _protocol_body(ctx: LabContext, pid: str, analysis: dict[str, Any]) -> dict[str, Any]:
    with ctx.db.session() as s:
        pv = s.scalar(
            select(ProtocolVersion).where(
                ProtocolVersion.project_id == pid,
                ProtocolVersion.version == analysis.get("protocol_version"),
            )
        )
        return dict(pv.body) if pv else {}


def _store_json(ctx: LabContext, pid: str, name: str, content: dict[str, Any]) -> str:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp, ctx.db.session() as s:
        path = Path(tmp) / name
        path.write_text(json.dumps(content, indent=2, default=str))
        return str(ctx.store.put(s, pid, path, name=name).sha256)
