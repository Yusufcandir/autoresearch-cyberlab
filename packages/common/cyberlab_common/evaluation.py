"""System-quality evaluation (professor's plan, Phase 9).

`reliability_report` measures the platform from its own records (Step 9.4); `compare_drafts`
compares two manuscript versions written from the same evidence (Step 9.2: e.g. the frozen v4
baseline against a v5 rewritten by the new writing pipeline). Human evaluation stays separate.
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select

from cyberlab_common.db import (
    AgentNote,
    CostRecord,
    Database,
    JobRecord,
    ManuscriptRecord,
    Project,
    RunRecord,
)


def reliability_report(db: Database, project_id: str | None = None) -> dict[str, Any]:
    with db.session() as s:

        def where(model: Any) -> Any:
            q = select(model)
            return q.where(model.project_id == project_id) if project_id else q

        jobs = list(s.scalars(where(JobRecord)))
        runs = list(s.scalars(where(RunRecord)))
        mss = list(s.scalars(where(ManuscriptRecord)))
        notes = list(s.scalars(where(AgentNote)))
        costs = list(s.scalars(where(CostRecord)))
        projects = list(
            s.scalars(select(Project).where(Project.id == project_id))
            if project_id
            else s.scalars(select(Project))
        )
    done_jobs = [j for j in jobs if j.state in ("succeeded", "failed", "canceled")]
    flags = [f for m in mss for f in (m.evidence_report or {}).get("flags", [])]
    validations = [n.content for n in notes if n.kind == "scientific_validation"]
    citations = [n.content for n in notes if n.kind == "citation_verification"]
    claims_total = sum(len(c.get("claims", [])) for c in citations)
    claims_bad = sum(
        (c.get("claim_support_counts") or {}).get(k, 0)
        for c in citations
        for k in ("contradicted", "insufficient_evidence")
    )
    dropped = sum(
        int(m.group(1))
        for f in flags
        if f.get("code") == "FRAMING_SENTENCES_DROPPED"
        and (m := re.match(r"(\d+)", str(f.get("message", ""))))
    )
    return {
        "scope": project_id or "all studies",
        "experiment_execution_success_rate": (
            sum(j.state == "succeeded" for j in done_jobs) / len(done_jobs) if done_jobs else None
        ),
        "runs": len(runs),
        "runs_reused": sum(1 for r in runs if (r.receipt or {}).get("reused_from_run")),
        "workflow_completion_rate": (
            sum(p.stage in ("awaiting_release", "released") for p in projects) / len(projects)
            if projects
            else None
        ),
        "validation_reports": len(validations),
        "validation_status_counts": {
            st: sum(v.get("status") == st for v in validations)
            for st in ("PASS", "WARNING", "BLOCK", "NOT_EVALUATED")
        },
        "citation_claims_checked": claims_total,
        "unsupported_citation_rate": claims_bad / claims_total if claims_total else None,
        "model_sentences_removed_by_checks": dropped,
        "evidence_flags_by_code": _count(f.get("code", "?") for f in flags),
        "manuscript_versions": len(mss),
        "gpu_minutes": round(sum(float(p.gpu_minutes_used or 0) for p in projects), 1),
        "llm_tokens": sum(int(p.llm_tokens_used or 0) for p in projects),
        "llm_cost_usd": round(sum(c.usd for c in costs), 4),
    }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


_FRAMING = ("Abstract", "Introduction", "Related Work", "Discussion", "Limitations", "Conclusion")


def _sections(md: str) -> dict[str, str]:
    parts = re.split(r"^## (.+)$", md, flags=re.M)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _shingles(text: str, n: int = 6) -> set[str]:
    w = re.findall(r"[a-z]+", text.lower())
    return {" ".join(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


def draft_measures(md: str) -> dict[str, Any]:
    secs = _sections(md)
    framing = " ".join(secs.get(k, "") for k in _FRAMING)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", framing) if s.strip()]
    words = re.findall(r"[A-Za-z]+", framing)
    sh = [_shingles(s) for s in sentences]
    repeated = sum(1 for i, a in enumerate(sh) if a and any(len(a & b) >= 3 for b in sh[:i]))
    return {
        "framing_words": len(words),
        "framing_sentences": len(sentences),
        "mean_sentence_length": round(len(words) / len(sentences), 1) if sentences else 0,
        "distinct_word_ratio": round(len({w.lower() for w in words}) / len(words), 3)
        if words
        else 0,
        "repeated_sentences_within_paper": repeated,
        "citations": len(re.findall(r"@[A-Za-z0-9_]+", secs.get("Related Work", ""))),
        "related_work_paragraphs": len(
            [p for p in secs.get("Related Work", "").split("\n\n") if p.strip()]
        ),
        "discussion_words": len(re.findall(r"[A-Za-z]+", secs.get("Discussion", ""))),
        "generic_phrases": len(
            re.findall(
                r"plays? a (?:crucial|critical|vital) role|in today's|increasingly important|"
                r"this paper aims to|it is worth noting",
                framing,
                re.I,
            )
        ),
    }


def compare_drafts(db: Database, project_id: str, a: int, b: int) -> dict[str, Any]:
    with db.session() as s:
        rows = {
            m.version: m
            for m in s.scalars(
                select(ManuscriptRecord).where(ManuscriptRecord.project_id == project_id)
            )
        }
        if a not in rows or b not in rows:
            raise ValueError(f"versions available: {sorted(rows)}")
        ma, mb = rows[a], rows[b]
        da, db_ = draft_measures(ma.markdown), draft_measures(mb.markdown)
        out = {
            "project": project_id,
            "versions": [a, b],
            "same_evidence": _evidence_sha(ma.markdown) == _evidence_sha(mb.markdown),
            "measures": {k: {"a": da[k], "b": db_[k]} for k in da},
            "evidence_flags": {
                "a": len((ma.evidence_report or {}).get("flags", [])),
                "b": len((mb.evidence_report or {}).get("flags", [])),
            },
            "release_ready": {
                "a": (ma.evidence_report or {}).get("release_ready"),
                "b": (mb.evidence_report or {}).get("release_ready"),
            },
            "note": "automated measures only; scientific accuracy, clarity and critical discussion "
            "need human rating alongside these numbers",
        }
    return out


def _evidence_sha(md: str) -> str | None:
    m = re.search(r"Analysis artifact SHA-256: `([0-9a-f]+)`", md)
    return m.group(1) if m else None
