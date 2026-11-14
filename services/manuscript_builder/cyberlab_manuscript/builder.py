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

PLACEHOLDER = re.compile(r"\{\{(metric|ci|hyp|cite|fact):([^}]+)\}\}")
NUMBER = re.compile(r"(?<![\w.\[{:])(\d+\.\d+|\d{2,}(?:\.\d+)?%?)(?![\w}])")
ALLOWED_PLAIN_NUMBERS = {
    "95%",  # confidence level of every interval reported by the platform
    "95",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
}


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


def _closest_fact(name: str, facts: dict[str, Any]) -> str | None:
    """Writers (small local models especially) misspell fact names: 'benign_row_cap',
    'BENIGN_cap'. Map an unknown name to a clearly matching recorded fact, else None."""
    import difflib

    lower = {k.lower(): k for k in facts}
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if key in lower:
        return lower[key]
    hit = difflib.get_close_matches(key, list(lower), n=1, cutoff=0.75)
    return lower[hit[0]] if hit else None


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def render_text(
    text: str,
    analysis: dict[str, Any],
    citations: dict[str, dict[str, Any]],
    report: EvidenceReport,
    section: str,
    guardrail: bool = False,
) -> str:
    """Resolve placeholders. With guardrail=True (final revision) unsupported content is removed
    and recorded as a warning instead of blocking release."""
    sev = "warning" if guardrail else "blocker"
    missing = "[not reported]" if guardrail else "[missing evidence]"
    models = analysis.get("models", {})
    hyps = {h["id"]: h for h in analysis.get("hypotheses", [])}
    status = "Verified" if analysis.get("metrics_recomputed_match") else "Measured"

    def sub(m: re.Match[str]) -> str:
        kind, arg = m.group(1), m.group(2).strip()
        if kind == "cite":
            # Writers often group citations: {{cite:A, B}} or {{cite:A; B}}.
            keys = [k.strip() for k in re.split(r"[,;\s]+", arg) if k.strip()]
            good: list[str] = []
            for key in keys:
                c = citations.get(key)
                if c and c.get("status") == "Verified":
                    good.append(key)
                    report.resolved.append(
                        {
                            "placeholder": f"{{{{cite:{key}}}}}",
                            "status": "Verified",
                            "doi": c.get("doi"),
                        }
                    )
                else:
                    report.flag(
                        sev, "UNVERIFIED_CITATION", f"{section}: citation {key!r} is not verified"
                    )
            if good:
                return "[" + "; ".join(f"@{k}" for k in good) + "]"
            return "" if guardrail else "[citation needed]"
        if kind == "fact":
            facts = analysis.get("facts", {})
            if arg not in facts:
                arg = _closest_fact(arg, facts) or arg
            if arg not in facts:
                report.flag(sev, "UNKNOWN_FACT", f"{section}: no recorded fact named {arg!r}")
                return missing
            report.resolved.append(
                {"placeholder": m.group(0), "value": facts[arg], "status": "Verified"}
            )
            return str(facts[arg])
        if kind == "hyp":
            h = hyps.get(arg)
            if not h:
                report.flag(sev, "UNKNOWN_HYPOTHESIS", f"{section}: {arg} has no evidence")
                return missing
            report.resolved.append(
                {"placeholder": m.group(0), "value": h["result"], "status": status}
            )
            return f"{h['id']} ({h['result']})"
        try:
            model, metric = (x.strip() for x in arg.split(":"))
        except ValueError:
            report.flag(sev, "BAD_PLACEHOLDER", f"{section}: {m.group(0)}")
            return missing
        mdl = models.get(model)
        if kind == "metric":
            v = (mdl or {}).get("aggregate", {}).get(metric)
            if v is None:
                report.flag(sev, "MISSING_EVIDENCE", f"{section}: no result for {model}/{metric}")
                return missing
            report.resolved.append(
                {
                    "placeholder": m.group(0),
                    "value": v["mean"],
                    "status": status,
                    "artifact": analysis.get("artifact_sha256"),
                }
            )
            sd = f" ± {_fmt(v['sd'])}" if v.get("n_seeds", 1) > 1 else ""
            return _fmt(v["mean"]) + sd
        ci = (mdl or {}).get(f"{metric}_ci")
        if ci is None:
            report.flag(sev, "MISSING_EVIDENCE", f"{section}: no CI for {model}/{metric}")
            return missing
        report.resolved.append(
            {"placeholder": m.group(0), "value": [ci["low"], ci["high"]], "status": status}
        )
        seed = (mdl or {}).get("ci_seed")
        label = f", seed {seed}" if seed is not None else ""
        return f"95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}]{label}"

    def check_numbers(segment: str) -> str:
        def num(m: re.Match[str]) -> str:
            if m.group(1) in ALLOWED_PLAIN_NUMBERS:
                return m.group(0)
            report.flag(
                sev,
                "UNSUPPORTED_NUMBER",
                f"{section}: number {m.group(1)!r} is not linked to evidence"
                + (" and was removed" if guardrail else "; use a placeholder"),
            )
            return "[number removed]" if guardrail else m.group(0)

        return NUMBER.sub(num, segment)

    out, pos = [], 0
    for m in PLACEHOLDER.finditer(text):
        out.append(check_numbers(text[pos : m.start()]))
        out.append(sub(m))
        pos = m.end()
    out.append(check_numbers(text[pos:]))
    return "".join(out)


_NAME = r"[A-Z][A-Za-zÀ-ÿ'’\-]+"
AUTHOR_CITE = re.compile(
    rf"(?P<first>{_NAME})(?P<rest>\s+et al\.|\s+and\s+(?P<second>{_NAME}))?(?P<gap>\s*)"
    rf"\[@(?P<key>[A-Za-z0-9_]+)"
)


def _surname(full: str) -> str:
    return full.strip().split()[-1] if full.strip() else ""


def _norm(s: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fix_author_names(
    text: str, citations: dict[str, dict[str, Any]], report: EvidenceReport, section: str
) -> str:
    """'Tufan and Kaya [@Key]' must name the real authors of Key; wrong names are corrected."""

    def sub(m: re.Match[str]) -> str:
        c = citations.get(m.group("key")) or {}
        authors = [a for a in (c.get("authors") or []) if a]
        if not authors:
            return m.group(0)
        s1 = _surname(authors[0])
        s2 = _surname(authors[1]) if len(authors) > 1 else ""
        first_ok = _norm(m.group("first")) == _norm(s1)
        second = m.group("second")
        second_ok = second is None or (len(authors) == 2 and _norm(second) == _norm(s2))
        if first_ok and second_ok:
            return m.group(0)
        if not first_ok and _norm(m.group("first")) not in {_norm(_surname(a)) for a in authors}:
            if m.group("rest") is None:
                return m.group(0)  # a normal word before the citation, not an author name
        correct = (
            s1 if len(authors) == 1 else (f"{s1} and {s2}" if len(authors) == 2 else f"{s1} et al.")
        )
        written = m.group("first") + (m.group("rest") or "")
        report.flag(
            "warning",
            "AUTHOR_CORRECTED",
            f"{section}: '{written}' corrected to '{correct}' for {m.group('key')}",
        )
        return f"{correct}{m.group('gap')}[@{m.group('key')}"

    return AUTHOR_CITE.sub(sub, text)


_CI_PREFIX = re.compile(
    r"(?:CI:\s*|95\s*%\s*(?:percentile\s+)?(?:bootstrap\s+)?(?:confidence\s+interval|CI)"
    r"(?:\s+(?:of|is|was))?\s*[:=]?\s*)(?=95% CI \[)",
    re.I,
)
_HYP_VERB = re.compile(
    r"\b(confirm(?:s|ed|ing)?|support(?:s|ed|ing)?|validat(?:e|es|ed|ing)|verif(?:y|ies|ied)|"
    r"corroborat(?:e|es|ed|ing)|substantiat(?:e|es|ed|ing)|reinforc(?:e|es|ed|ing)|"
    r"consistent with|in line with)\s+"
    r"((?:H\d+ \((?:supported|not supported)\)(?:,\s*|\s+and\s+)?)+)",
    re.I,
)


def tidy_prose(text: str) -> str:
    """Fix wording problems that come from combining agent prose with rendered evidence."""
    text = _CI_PREFIX.sub("", text)
    text = re.sub(r"\(\s*\(95% CI ([^)]*)\)\s*\)", r"(95% CI \1)", text)

    def hyp(m: re.Match[str]) -> str:
        items = re.findall(r"(H\d+) \((supported|not supported)\)", m.group(2))
        ok = [h for h, r in items if r == "supported"]
        bad = [h for h, r in items if r != "supported"]
        tail = (
            m.group(2)[len(m.group(2).rstrip(", ")) :] if m.group(2).endswith((", ", " ")) else ""
        )
        parts = []
        if ok:
            parts.append(f"{m.group(1)} {' and '.join(ok)}")
        if bad:
            parts.append(f"does not support {' and '.join(bad)}")
        return ", but ".join(parts) + tail

    text = _HYP_VERB.sub(hyp, text)
    # "(H1 (supported))" -> "(H1, supported)"
    text = re.sub(r"\((H\d+) \((supported|not supported)\)\)", r"(\1, \2)", text)
    # "H1 (supported) is supported" -> "H1 is supported" (only when the wording agrees)
    text = re.sub(
        r"\b(H\d+) \(supported\)(?=\s+(?:(?:is|was|also|has been|clearly)\s+)*"
        r"(?:supported|holds|held|confirmed))",
        r"\1",
        text,
    )
    text = re.sub(
        r"\b(H\d+) \(not supported\)(?=\s+(?:(?:is|was|also|has been)\s+)*"
        r"(?:not supported|rejected))",
        r"\1",
        text,
    )
    # "only Rare (9 test rows) test rows" -> "only 9 test rows" (a fact used as a number)
    text = re.sub(r"\b[A-Z][\w-]* \(([\d,]+) test rows\) test rows\b", r"\1 test rows", text)
    # "([@Key])" -> "[@Key]"
    text = re.sub(r"\((\[@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*\])\)", r"\1", text)
    # "[[@Key]]" (writer wrapped the placeholder in brackets) -> "[@Key]"
    text = re.sub(r"\[(\[@[A-Za-z0-9_]+(?:;\s*@[A-Za-z0-9_]+)*\])\]", r"\1", text)
    # "we test H1 (supported)" should not reveal the result: drop it after test verbs.
    return re.sub(
        r"\b(test(?:s|ed|ing)?|evaluat(?:e|es|ed|ing)|examin(?:e|es|ed|ing))(\s+(?:whether\s+)?)"
        r"((?:H\d+ \((?:supported|not supported)\)(?:,\s*|\s+and\s+)?)+)",
        lambda m: (
            m.group(1) + m.group(2) + re.sub(r" \((?:supported|not supported)\)", "", m.group(3))
        ),
        text,
        flags=re.I,
    )


def metric_definitions() -> str:
    return (
        "### Metric definitions\n\n"
        "Macro-F1 is the unweighted mean of per-class F1 scores, so every class counts equally. "
        "MCC is the multiclass Matthews correlation coefficient. Balanced accuracy is the mean "
        "per-class recall. For the operational metrics the multiclass output is read as attack "
        "versus BENIGN: the detection rate is the share of attack rows predicted as any attack "
        "class, and the false-positive rate is the share of BENIGN rows predicted as an attack."
    )


def small_class_note(analysis: dict[str, Any]) -> str:
    small: dict[str, int] = {}
    for m in analysis.get("models", {}).values():
        for cls, pc in (m.get("per_class") or {}).items():
            if pc.get("support", 0) < 30:
                small[cls] = int(pc["support"])
    if not small:
        return ""
    listed = ", ".join(f"{c} ({n:,} test rows)" for c, n in sorted(small.items()))
    return (
        f"Some classes have very few test rows: {listed}. Their per-class scores, and the "
        "macro-averaged metrics they feed into, are therefore highly uncertain."
    )


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

        def g(k: str, a: dict[str, Any] = a) -> str:
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
        rows.append(
            f"| {cls} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | {v['support']:,} |"
        )
    return "\n".join(rows)


def hypotheses_table(analysis: dict[str, Any]) -> str:
    rows = ["| Hypothesis | Comparison | Means | McNemar p | Result |", "|---|---|---|---|---|"]
    for h in analysis["hypotheses"]:
        if "mcnemar" not in h:
            rows.append(f"| {h['id']} | - | - | - | {h['result']} |")
            continue
        rows.append(
            f"| {h['id']} | {h['a']} > {h['b']} ({h['metric']}) | {_fmt(h['a_mean'])} vs {_fmt(h['b_mean'])} | "
            f"{_fmt_p(h['mcnemar']['p_value'])} | {h['result']} |"
        )
    return "\n".join(rows)


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def _n(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def dataset_section(analysis: dict[str, Any]) -> str:
    ds = analysis.get("dataset") or {}
    if not ds:
        return "_Dataset manifest missing._"
    counts = "\n".join(f"| {k} | {_n(v)} |" for k, v in ds.get("class_counts", {}).items())
    splits = ", ".join(f"{k}: {_n(v)}" for k, v in ds.get("split_counts", {}).items())
    return (
        f"Dataset: {ds.get('name')} (`{ds.get('dataset_id')}`), licence: {ds.get('license')}\n\n"
        f"Rows after cleaning: {_n(ds.get('n_rows_clean'))} of {_n(ds.get('n_rows_raw'))} raw; "
        f"duplicates or conflicting rows removed: {_n(ds.get('n_duplicates_removed'))}; "
        f"train/test overlap rows: {ds.get('split_overlap_rows')}. Splits: {splits}.\n\n"
        f"| Class | Rows |\n|---|---|\n{counts}"
    )


SECTION_ORDER = [
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "results",
    "discussion",
    "limitations",
    "conclusion",
]


def build_manuscript(
    title: str,
    sections: dict[str, str],
    analysis: dict[str, Any],
    citations: dict[str, dict[str, Any]],
    bibtex: str,
    guardrail: bool = False,
) -> tuple[str, EvidenceReport]:
    report = EvidenceReport()
    out = [
        f"# {title}",
        "",
        "> DRAFT generated by AutoResearch-CyberLab. Numbers and tables are rendered "
        "from stored evidence; prose is agent-written and must be reviewed by the authors.",
        "",
    ]
    for key in SECTION_ORDER:
        heading = key.replace("_", " ").title()
        body = sections.get(key, "")
        out += [f"## {heading}", ""]
        if body:
            rendered = render_text(body, analysis, citations, report, key, guardrail)
            out += [tidy_prose(fix_author_names(rendered, citations, report, key)), ""]
        if key == "methodology":
            out += [metric_definitions(), "", "### Data", "", dataset_section(analysis), ""]
        if key == "limitations" and (note := small_class_note(analysis)):
            out += [note, ""]
        if key == "results":
            out += [
                "### Main results",
                "",
                results_table(analysis),
                "",
                "### Hypothesis tests",
                "",
                hypotheses_table(analysis),
                "",
            ]
            best = max(
                analysis["models"],
                key=lambda n: analysis["models"][n]["aggregate"]["macro_f1"]["mean"],
            )
            out += [f"### Per-class results ({best})", "", per_class_table(analysis, best), ""]
    for required in ("abstract", "introduction", "methodology", "results", "conclusion"):
        if not str(sections.get(required, "")).strip():
            report.flag("blocker", "EMPTY_SECTION", f"the {required} section is empty")
    for f in analysis.get("qa_flags", []):
        report.flag("blocker" if f["severity"] == "blocker" else "warning", f["code"], f["message"])
    out += [
        "## Evidence",
        "",
        f"Analysis artifact SHA-256: `{analysis.get('artifact_sha256')}`  ",
        f"Protocol SHA-256: `{analysis.get('protocol_sha256')}`  ",
        f"Runs: {len(analysis.get('run_ids', []))}",
        "",
        "## References",
        "",
    ]
    for key, c in citations.items():
        if c.get("status") == "Verified":
            out.append(
                f"- [{key}] {', '.join(c.get('authors', [])[:3])} ({c.get('year')}). {c.get('title')}. "
                f"{c.get('venue') or ''}. doi:{c.get('doi')}"
            )
    md = "\n".join(out) + "\n"
    return md, report


def manuscript_sha(md: str) -> str:
    return hashlib.sha256(md.encode()).hexdigest()
