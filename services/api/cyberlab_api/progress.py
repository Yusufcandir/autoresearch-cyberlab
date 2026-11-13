"""Turn workflow events into a human-readable progress tracker, activity feed and trust summary."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from markdown_it import MarkdownIt

from cyberlab_common.db import AuditEvent, CitationRecord, JobRecord, ManuscriptRecord

STEPS: list[tuple[str, str, str]] = [
    (
        "orchestrator",
        "Plan",
        "The chief orchestrator is turning the research goal into a plan and search queries.",
    ),
    (
        "literature",
        "Literature",
        "Searching OpenAlex and checking every DOI against Crossref before a paper is accepted.",
    ),
    (
        "research_gap",
        "Gaps",
        "Reading the verified papers to propose research gaps (proposals only, not facts).",
    ),
    (
        "methodology",
        "Protocol",
        "Designing hypotheses, models and seeds, then validating them against the allowed templates.",
    ),
    (
        "dataset",
        "Dataset",
        "Cleaning CIC-IDS2017, removing duplicates and checking that no test row leaks into training. This takes a few minutes.",
    ),
    (
        "gate_a",
        "PI approval",
        "Waiting for the PI to approve the protocol and data use. Nothing runs on the GPU before this.",
    ),
    ("coding", "Compile", "Turning the approved protocol into reviewed experiment templates."),
    ("experiments", "Experiments", "Training models on the GPU, one job at a time."),
    (
        "statistics",
        "Statistics",
        "Recomputing every metric from the saved predictions, with confidence intervals and significance tests.",
    ),
    (
        "writing",
        "Writing",
        "Drafting the paper. Numbers are inserted from the evidence store, never typed by the AI.",
    ),
    (
        "reviewer",
        "Review",
        "An independent reviewer agent scores the draft and checks every claim against the evidence.",
    ),
    ("gate_d", "PI release", "Waiting for the PI to read the draft and release it."),
]
LABEL = {k: v for k, v, _ in STEPS}
LABEL.update({"experiment": "Experiments", "wait_jobs": "Experiments", "check_runs": "Experiments"})
_ALIAS = {"experiment": "experiments", "wait_jobs": "experiments", "check_runs": "experiments"}


@dataclass
class Step:
    key: str
    label: str
    state: str  # done | running | waiting | error | todo
    detail: str = ""


def _ago(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    s = int((datetime.now(UTC) - ts).total_seconds())
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def build_progress(
    events: list[AuditEvent], stage: str, pending: dict[str, Any] | None, jobs: list[JobRecord]
) -> tuple[list[Step], str]:
    """Return the step list and a one-line 'what is happening now' message."""
    finished: set[str] = set()
    open_steps: dict[str, datetime] = {}
    failed: dict[str, str] = {}
    for e in sorted(events, key=lambda e: e.id):
        step = _ALIAS.get(str(e.detail.get("step", "")), str(e.detail.get("step", "")))
        if e.action == "agent.started":
            open_steps[step] = e.ts
            failed.pop(step, None)
        elif e.action == "agent.finished":
            if step != "experiments":  # experiments are done when all jobs are, not when submitted
                finished.add(step)
            open_steps.pop(step, None)
        elif e.action == "agent.failed":
            failed[step] = str(e.detail.get("error", ""))
            open_steps.pop(step, None)
    actions = {e.action for e in events}
    n_jobs = len(jobs)
    states = Counter(j.state for j in jobs)
    terminal = sum(states[k] for k in ("succeeded", "failed", "canceled", "blocked"))
    if "protocol.approved" in actions:
        finished.add("gate_a")
    if n_jobs and terminal == n_jobs:
        finished.add("experiments")
    if stage == "released":
        finished.add("gate_d")

    steps: list[Step] = []
    now = ""
    for key, label, desc in STEPS:
        state = "done" if key in finished else "todo"
        detail = ""
        if key == "experiments" and n_jobs:
            detail = f"{states['succeeded']}/{n_jobs} done" + (
                f", {states['failed']} failed" if states["failed"] else ""
            )
            if key not in finished:
                state = "running"
                running = [j for j in jobs if j.state == "running"]
                current = f" Now training {running[0].experiment_id}." if running else ""
                now = f"Experiments: {states['succeeded']} of {n_jobs} finished.{current} {desc}"
        if pending and (
            (key == "gate_a" and pending.get("gate") == "A")
            or (key == "gate_d" and pending.get("gate") == "D")
        ):
            state, now = "waiting", desc
        if key in open_steps and stage != "error" and key != "experiments":
            state = "running"
            since = (
                open_steps[key] if open_steps[key].tzinfo else open_steps[key].replace(tzinfo=UTC)
            )
            msg = (
                f'{label} (<span class="elapsed" data-since="{since.isoformat()}">'
                f"{_ago(open_steps[key])}</span> so far): {desc}"
            )
            now = f"{now}  |  {msg}" if now else msg
        if key in failed and stage == "error":
            state, detail = "error", failed[key][:120]
        steps.append(Step(key, label, state, detail))
    # Studies started before step events existed: everything before a waiting gate is done.
    gate_idx = max((i for i, st in enumerate(steps) if st.state == "waiting"), default=-1)
    # Anything after PI approval being done means everything before approval is done too.
    approval = next(i for i, st in enumerate(steps) if st.key == "gate_a")
    last_done = max((i for i, st in enumerate(steps) if st.state == "done"), default=-1)
    if last_done >= approval:
        gate_idx = max(gate_idx, last_done)
    if stage == "released":
        gate_idx = len(steps)
    for st in steps[: max(gate_idx, 0)]:
        if st.state == "todo":
            st.state = "done"
    if stage == "error" and not now:
        now = "Stopped on an error. Read the reason below and use Retry; the study continues from its last checkpoint."
    if stage == "released":
        now = "Finished: the PI released the manuscript."
    if not now and stage in {"created", "retrying"}:
        now = "Starting up..."
    return steps, now


_FRIENDLY = {
    "project.created": "Study created",
    "dataset.verified": "Dataset passed the gate",
    "dataset.blocked": "Dataset blocked",
    "protocol.drafted": "Protocol drafted",
    "protocol.approved": "Protocol approved",
    "protocol.rejected": "Protocol rejected; methodology agent will revise",
    "job.submitted": "Experiment queued",
    "job.started": "Experiment started",
    "job.succeeded": "Experiment finished",
    "job.failed": "Experiment failed",
    "job.canceled": "Experiment canceled",
    "job.contention": "Waiting: another user is on the GPU",
    "evidence.published": "Results analysed and evidence saved",
    "manuscript.drafted": "Draft written",
    "review.requested": "Reviewer finished",
    "manuscript.released": "Manuscript released",
    "security.policy_denied": "Blocked by the security policy",
    "budget.threshold_reached": "Compute budget reached",
    "project.paused": "Study paused",
    "project.resumed": "Study resumed",
}


def friendly_activity(events: list[AuditEvent], limit: int = 60) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for e in sorted(events, key=lambda e: e.id, reverse=True):
        d = e.detail or {}
        step = LABEL.get(str(d.get("step", "")), str(d.get("step", "")))
        if e.action == "agent.started":
            text, kind = f"{step} agent started", "info"
        elif e.action == "agent.finished":
            text, kind = f"{step} agent finished in {d.get('seconds', '?')} s", "ok"
        elif e.action == "agent.failed":
            text, kind = f"{step} agent stopped: {str(d.get('error', ''))[:160]}", "bad"
        elif e.action in _FRIENDLY:
            text = _FRIENDLY[e.action]
            extra = d.get("experiment_id") or d.get("version") or d.get("error") or ""
            if extra and e.action.startswith(("job.", "protocol.", "manuscript.")):
                text += f" ({extra})"
            kind = "bad" if e.action.endswith(("failed", "blocked", "denied")) else "ok"
        else:
            continue
        out.append({"ts": e.ts.strftime("%H:%M:%S"), "actor": e.actor, "text": text, "kind": kind})
        if len(out) >= limit:
            break
    return out


_md = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table")


def render_manuscript(markdown: str, keys_in_order: list[str]) -> tuple[str, dict[str, list[str]]]:
    """Safe HTML (raw HTML disabled) with numbered, linked citations; also returns key -> sections."""
    cited_in: dict[str, list[str]] = {}
    section = "Front matter"
    for line in markdown.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        for key in re.findall(r"\[@([A-Za-z0-9_]+)\]", line):
            if section not in cited_in.setdefault(key, []):
                cited_in[key].append(section)
    numbers = {k: i + 1 for i, k in enumerate(keys_in_order)}
    html = _md.render(markdown)

    def cite(m: re.Match[str]) -> str:
        key = m.group(1)
        n = numbers.get(key, "?")
        return f'<a class="cite" href="#ref-{key}" title="{key}">[{n}]</a>'

    html = re.sub(r"\[@([A-Za-z0-9_]+)\]", cite, html)
    for token, cls in (
        ("[number removed]", "flag"),
        ("[citation needed]", "flag"),
        ("[missing evidence]", "flag"),
        ("[not reported]", "flag"),
    ):
        html = html.replace(
            token,
            f'<mark class="{cls}" title="Removed or missing: not supported by stored evidence">{token}</mark>',
        )
    return html, cited_in


def trust_summary(
    cites: list[CitationRecord],
    matrix_note: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
    manuscript: ManuscriptRecord | None,
    artifact_check: dict[str, str],
) -> list[dict[str, str]]:
    """Each row: check, status (ok|warn|bad|pending), and what it means."""
    rows: list[dict[str, str]] = []
    if cites:
        v = sum(c.status == "Verified" for c in cites)
        u = len(cites) - v
        rows.append(
            {
                "check": "References are real",
                "status": "ok",
                "detail": f"{v} papers verified by DOI and title against Crossref/OpenAlex. "
                + (
                    f"{u} could not be verified and are excluded from the paper."
                    if u
                    else "None were rejected."
                ),
            }
        )
    else:
        rows.append(
            {
                "check": "References are real",
                "status": "pending",
                "detail": "Literature not searched yet.",
            }
        )
    if matrix_note is not None:
        dropped = int(matrix_note.get("dropped_unverified_rows", 0) or 0)
        rows.append(
            {
                "check": "Literature summary only uses verified papers",
                "status": "warn" if dropped else "ok",
                "detail": f"The AI mentioned {dropped} paper(s) that were not in the verified list; those rows were removed."
                if dropped
                else "Every row of the evidence matrix refers to a verified paper.",
            }
        )
    if analysis:
        ok = bool(analysis.get("metrics_recomputed_match"))
        rows.append(
            {
                "check": "Numbers come from real experiments",
                "status": "ok" if ok else "bad",
                "detail": "Every metric was recomputed from the saved predictions and matched the stored value."
                if ok
                else "Recomputed metrics did NOT match stored values; results are not verified.",
            }
        )
        blockers = [f for f in analysis.get("qa_flags", []) if f.get("severity") == "blocker"]
        warns = [f for f in analysis.get("qa_flags", []) if f.get("severity") == "warning"]
        rows.append(
            {
                "check": "Leakage and sanity checks",
                "status": "bad" if blockers else ("warn" if warns else "ok"),
                "detail": "; ".join(f["message"] for f in blockers + warns)[:400]
                or "No train/test overlap and no suspiciously perfect scores.",
            }
        )
    else:
        rows.append(
            {
                "check": "Numbers come from real experiments",
                "status": "pending",
                "detail": "No results yet.",
            }
        )
    if artifact_check:
        bad = [k for k, v in artifact_check.items() if v != "ok"]
        rows.append(
            {
                "check": "Evidence files untouched",
                "status": "bad" if bad else "ok",
                "detail": f"{len(artifact_check)} stored files re-hashed with SHA-256; "
                + (f"{len(bad)} missing or modified." if bad else "all match."),
            }
        )
    if manuscript is not None:
        rep = manuscript.evidence_report or {}
        flags = rep.get("flags", [])
        removed = sum(1 for f in flags if f.get("code") == "UNSUPPORTED_NUMBER")
        unverified = sum(1 for f in flags if f.get("code") == "UNVERIFIED_CITATION")
        n_blockers = sum(1 for f in flags if f.get("severity") == "blocker")
        rows.append(
            {
                "check": "Paper text matches the evidence",
                "status": "bad" if n_blockers else ("warn" if removed or unverified else "ok"),
                "detail": f"{len(rep.get('resolved', []))} numbers and references were filled in from evidence. "
                + (
                    f"{removed} number(s) the AI wrote itself were flagged{' and removed' if not n_blockers else ''}. "
                    if removed
                    else ""
                )
                + (f"{unverified} unverified citation(s) were blocked. " if unverified else "")
                + ("Release is blocked until these are fixed." if n_blockers else ""),
            }
        )
    rows.append(
        {
            "check": "AI opinions are labelled",
            "status": "ok",
            "detail": "Plans, gaps, interpretations and reviews are marked Proposed. Only Measured/Verified items come from data.",
        }
    )
    return rows
