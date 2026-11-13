"""Evidence-bound manuscript builder (spec Steps 21 and 22).

Agents write prose with placeholders; this module renders the numbers and references:

    {{metric:xgboost:macro_f1}}        mean over seeds, from analysis.json (Verified/Measured)
    {{ci:xgboost:macro_f1}}            bootstrap 95% CI
    {{hyp:H1}}                         'supported' / 'not supported'
    {{cite:Sharafaldin2018toward}}     only Verified citations; others become [citation needed]

Results tables are generated directly from the analysis artifact. Any digit-bearing number in
agent prose that did not come from a placeholder is flagged as an unsupported numeric claim;
blocker flags fail the release gate (AT-15).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

PLACEHOLDER = re.compile(r"\{\{(metric|ci|hyp|cite):([^}]+)\}\}")
NUMBER = re.compile(r"(?<![\w.\[{:])(\d+\.\d+|\d{2,}(?:\.\d+)?%?)(?![\w}])")
ALLOWED_PLAIN_NUMBERS = {"2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"}


@dataclass
class EvidenceReport:
    resolved: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, str]] = field(default_factory=list)

    def flag(self, severity: str, code: str, message: str) -> None:
        self.flags.append({"severity": severity, "code": code, "message": message})

    @property
    def blockers(self) -> list[dict[str, str]]:
        return [f for f in self.flags if f["severity"] == "blocker"]

    def as_dict(self) -> dict[str, Any]:
        return {"resolved": self.resolved, "flags": self.flags, "release_ready": not self.blockers}


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def render_text(
    text: str,
    analysis: dict[str, Any],
    citations: dict[str, dict[str, Any]],
    report: EvidenceReport,
    section: str,
) -> str:
    models = analysis.get("models", {})
    hyps = {h["id"]: h for h in analysis.get("hypotheses", [])}
    status = "Verified" if analysis.get("metrics_recomputed_match") else "Measured"

    def sub(m: re.Match[str]) -> str:
        kind, arg = m.group(1), m.group(2).strip()
        if kind == "cite":
            c = citations.get(arg)
            if c and c.get("status") == "Verified":
                report.resolved.append({"placeholder": m.group(0), "status": "Verified", "doi": c.get("doi")})
                return f"[@{arg}]"
            report.flag("blocker", "UNVERIFIED_CITATION", f"{section}: citation {arg!r} is not verified")
            return "[citation needed]"
        if kind == "hyp":
            h = hyps.get(arg)
            if not h:
                report.flag("blocker", "UNKNOWN_HYPOTHESIS", f"{section}: {arg} has no evidence")
                return "[missing evidence]"
            report.resolved.append({"placeholder": m.group(0), "value": h["result"], "status": status})
            return str(h["result"])
        try:
            model, metric = (x.strip() for x in arg.split(":"))
        except ValueError:
            report.flag("blocker", "BAD_PLACEHOLDER", f"{section}: {m.group(0)}")
            return "[missing evidence]"
        mdl = models.get(model)
        if kind == "metric":
            v = (mdl or {}).get("aggregate", {}).get(metric)
            if v is None:
                report.flag("blocker", "MISSING_EVIDENCE", f"{section}: no result for {model}/{metric}")
                return "[missing evidence]"
            report.resolved.append(
                {"placeholder": m.group(0), "value": v["mean"], "status": status,
                 "artifact": analysis.get("artifact_sha256")}
            )
            sd = f" ± {_fmt(v['sd'])}" if v.get("n_seeds", 1) > 1 else ""
            return _fmt(v["mean"]) + sd
        ci = (mdl or {}).get(f"{metric}_ci")
        if ci is None:
            report.flag("blocker", "MISSING_EVIDENCE", f"{section}: no CI for {model}/{metric}")
            return "[missing evidence]"
        report.resolved.append({"placeholder": m.group(0), "value": [ci["low"], ci["high"]], "status": status})
        return f"95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]"

    rendered = PLACEHOLDER.sub(sub, text)
    stripped = PLACEHOLDER.sub("", text)
    for num in NUMBER.findall(stripped):
        if num not in ALLOWED_PLAIN_NUMBERS:
            report.flag(
                "blocker",
                "UNSUPPORTED_NUMBER",
                f"{section}: number {num!r} is not linked to evidence; use a placeholder",
            )
    return rendered


def results_table(analysis: dict[str, Any]) -> str:
    rows = [
        "| Model | Role | Macro-F1 (mean ± sd) | 95% CI (seed {seed}) | Accuracy | MCC | FPR | Detection rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    seed = "?"
    for name, m in analysis["models"].items():
        a = m["aggregate"]
        ci = m["macro_f1_ci"]
        seed = str(m["ci_seed"])

        def g(k: str) -> str:
            return _fmt(a[k]["mean"]) if k in a else "n/a"

        rows.append(
            f"| {name} | {m['role']} | {_fmt(a['macro_f1']['mean'])} ± {_fmt(a['macro_f1']['sd'])} | "
            f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}] | {g('accuracy')} | {g('mcc')} | "
            f"{g('false_positive_rate')} | {g('detection_rate')} |"
        )
    rows[0] = rows[0].replace("{seed}", seed)
    return "\n".join(rows)


def per_class_table(analysis: dict[str, Any], model: str) -> str:
    pc = analysis["models"][model]["per_class"]
    rows = ["| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls, v in pc.items():
        rows.append(f"| {cls} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']} |")
    return "\n".join(rows)


def hypotheses_table(analysis: dict[str, Any]) -> str:
    rows = ["| Hypothesis | Comparison | Means | McNemar p | Result |", "|---|---|---|---|---|"]
    for h in analysis["hypotheses"]:
        if "mcnemar" not in h:
            rows.append(f"| {h['id']} | - | - | - | {h['result']} |")
            continue
        rows.append(
            f"| {h['id']} | {h['a']} > {h['b']} ({h['metric']}) | {_fmt(h['a_mean'])} vs {_fmt(h['b_mean'])} | "
            f"{h['mcnemar']['p_value']:.3g} | {h['result']} |"
        )
    return "\n".join(rows)


def dataset_section(analysis: dict[str, Any]) -> str:
    ds = analysis.get("dataset") or {}
    if not ds:
        return "_Dataset manifest missing._"
    counts = "\n".join(f"| {k} | {v} |" for k, v in ds.get("class_counts", {}).items())
    splits = ", ".join(f"{k}: {v}" for k, v in ds.get("split_counts", {}).items())
    return (
        f"Dataset: {ds.get('name')} (`{ds.get('dataset_id')}`), licence: {ds.get('license')}\n\n"
        f"Rows after cleaning: {ds.get('n_rows_clean')} of {ds.get('n_rows_raw')} raw; "
        f"duplicates or conflicting rows removed: {ds.get('n_duplicates_removed')}; "
        f"train/test overlap rows: {ds.get('split_overlap_rows')}. Splits: {splits}.\n\n"
        f"| Class | Rows |\n|---|---|\n{counts}"
    )


SECTION_ORDER = ["abstract", "introduction", "related_work", "methodology", "results", "discussion", "limitations", "conclusion"]


def build_manuscript(
    title: str,
    sections: dict[str, str],
    analysis: dict[str, Any],
    citations: dict[str, dict[str, Any]],
    bibtex: str,
) -> tuple[str, EvidenceReport]:
    report = EvidenceReport()
    out = [f"# {title}", "", "> DRAFT generated by AutoResearch-CyberLab. Numbers and tables are rendered "
           "from stored evidence; prose is agent-written and must be reviewed by the authors.", ""]
    for key in SECTION_ORDER:
        heading = key.replace("_", " ").title()
        body = sections.get(key, "")
        out += [f"## {heading}", ""]
        if body:
            out += [render_text(body, analysis, citations, report, key), ""]
        if key == "methodology":
            out += ["### Data", "", dataset_section(analysis), ""]
        if key == "results":
            out += ["### Main results", "", results_table(analysis), "",
                    "### Hypothesis tests", "", hypotheses_table(analysis), ""]
            best = max(analysis["models"], key=lambda n: analysis["models"][n]["aggregate"]["macro_f1"]["mean"])
            out += [f"### Per-class results ({best})", "", per_class_table(analysis, best), ""]
    for f in analysis.get("qa_flags", []):
        report.flag("blocker" if f["severity"] == "blocker" else "warning", f["code"], f["message"])
    out += ["## Evidence", "", f"Analysis artifact SHA-256: `{analysis.get('artifact_sha256')}`  ",
            f"Protocol SHA-256: `{analysis.get('protocol_sha256')}`  ",
            f"Runs: {len(analysis.get('run_ids', []))}", "", "## References", ""]
    for key, c in citations.items():
        if c.get("status") == "Verified":
            out.append(f"- [{key}] {', '.join(c.get('authors', [])[:3])} ({c.get('year')}). {c.get('title')}. "
                       f"{c.get('venue') or ''}. doi:{c.get('doi')}")
    md = "\n".join(out) + "\n"
    return md, report


def manuscript_sha(md: str) -> str:
    return hashlib.sha256(md.encode()).hexdigest()
