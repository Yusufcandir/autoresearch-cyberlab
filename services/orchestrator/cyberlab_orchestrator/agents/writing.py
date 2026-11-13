"""Agent 9: Scientific Writing. Writes prose with evidence placeholders; numbers come from evidence."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import func, select

from cyberlab_common.db import CitationRecord, ManuscriptRecord, Project, audit
from cyberlab_literature import Paper, bibtex_entry
from cyberlab_manuscript import build_manuscript, manuscript_sha
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ResearchSpec

SYSTEM = (
    "You are a scientific writing agent drafting an intrusion-detection paper. NEVER write a "
    "numeric result yourself. Use placeholders only: {{metric:MODEL:METRIC}} (METRIC in macro_f1, "
    "accuracy, balanced_accuracy, mcc, false_positive_rate, detection_rate), {{ci:MODEL:macro_f1}}, "
    "{{hyp:H1}}, and {{cite:KEY}} with KEY from the allowed citation keys. Results tables are added "
    "automatically. Be precise and modest; mention CIC-IDS2017 limitations. Output JSON with keys "
    "abstract, introduction, related_work, methodology, results, discussion, limitations, "
    "conclusion (markdown strings)."
)


def writing_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    ctx.authorize("writing", "draft_manuscript", "T0", pid)
    analysis = _load_analysis(ctx, pid, state["analysis_sha256"])
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        assert project is not None
        spec = ResearchSpec.model_validate(project.spec)
        cites = {
            c.key: {"status": c.status, "doi": c.doi, "title": c.title, "authors": c.authors,
                    "year": c.year, "venue": c.venue}
            for c in s.scalars(select(CitationRecord).where(CitationRecord.project_id == pid))
        }
    verified_keys = [k for k, c in cites.items() if c["status"] == "Verified"]
    context = {
        "title": spec.title,
        "goal": spec.goal,
        "models": list(analysis["models"]),
        "hypotheses": [{"id": h["id"], "statement": h["statement"]} for h in analysis["hypotheses"]],
        "allowed_citation_keys": {k: cites[k]["title"] for k in verified_keys},
        "dataset": {k: analysis["dataset"].get(k) for k in ("name", "license")} if analysis.get("dataset") else {},
        "statistician_notes": state.get("interpretation", {}),
        "reviewer_feedback": state.get("review_feedback", []),
    }
    sections = ctx.llm.complete_json("writing.draft", SYSTEM, json.dumps(context, default=str), project_id=pid)
    sections = {k: str(v) for k, v in sections.items() if isinstance(v, str | int | float)}
    md, report = build_manuscript(spec.title, sections, analysis, cites, "")
    bib = "\n\n".join(
        bibtex_entry(Paper(title=c["title"], doi=c["doi"], year=c["year"], authors=c["authors"] or [],
                           venue=c["venue"]), key=k)
        for k, c in cites.items() if c["status"] == "Verified"
    )
    with ctx.db.session() as s:
        version = (s.scalar(select(func.max(ManuscriptRecord.version)).where(ManuscriptRecord.project_id == pid)) or 0) + 1
        s.add(ManuscriptRecord(project_id=pid, version=version, markdown=md, bibtex=bib,
                               sha256=manuscript_sha(md), evidence_report=report.as_dict(), status="draft"))
        audit(s, "writing_agent", "manuscript.drafted", pid,
              {"version": version, "blockers": len(report.blockers)})
    ctx.set_stage(pid, "drafted", "Unverified" if report.blockers else "Measured")
    return {"manuscript_version": version, "evidence_blockers": report.blockers}


def _load_analysis(ctx: LabContext, pid: str, sha: str) -> dict[str, Any]:
    with ctx.db.session() as s:
        return dict(json.loads(ctx.store.path_for(s, pid, sha).read_text()))
