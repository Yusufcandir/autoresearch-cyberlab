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

PLACEHOLDER = re.compile(r"\{\{(metric|ci|hyp|cite|fact|recall):([^}]+)\}\}")
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


# counts the platform itself uses and reports in the Methodology (bootstrap resamples,
# permutations)
METHOD_COUNTS = {"500", "1000", "0.05"}  # 0.05: the significance level


# '(Finding F-acc)', '(F1)', '[C012]', 'claim C003': labels from the evidence brief
INTERNAL_ID = re.compile(
    r"\s*[\(\[](?:(?:finding|claim)\s+)?(?:F-?[A-Za-z0-9]+|C\d{3}|H\d+-?claim)[\)\]]"
    r"|\s*\((?:see\s+)?(?:finding|claim)\s+[A-Za-z0-9-]+\)",
    re.I,
)


_FRAMING_SECTIONS = ("Introduction", "Related Work", "Discussion", "Limitations", "Conclusion")


def drop_repeated_sentences(text: str) -> str:
    """A writer sometimes says the same thing twice in a section ('These works explore deep
    models ... but do not compare them with tree ensembles'). Keep the first of any two
    number-free sentences that share most of their words."""
    from cyberlab_manuscript.consistency import split_sentences

    parts = re.split(r"(?m)^(## .+)$", text)
    for i in range(1, len(parts) - 1, 2):
        if parts[i][3:].strip() not in _FRAMING_SECTIONS:
            continue
        seen: list[set[str]] = []
        lines = []
        for line in parts[i + 1].split("\n"):
            if not line.strip() or line.lstrip().startswith(("|", "#", ">", "!", "*")):
                lines.append(line)
                continue
            kept = []
            for s in split_sentences(line):
                words = set(re.findall(r"[a-z]{3,}", s.lower()))
                # share of the shorter sentence's words found in an earlier one
                if (
                    len(words) >= 8
                    and not re.search(r"\d", s)
                    and any(len(words & w) / min(len(words), len(w)) >= 0.75 for w in seen)
                ):
                    continue
                seen.append(words)
                kept.append(s)
            if kept:
                lines.append(" ".join(kept))
        parts[i + 1] = "\n".join(lines)
    return "".join(parts)


MAX_KEYS_PER_CITATION = 4


def trim_citation_dumps(text: str) -> str:
    def trim(m: re.Match[str]) -> str:
        keys = [k.strip() for k in m.group(1).split(";") if k.strip()]
        if len(keys) <= MAX_KEYS_PER_CITATION + 1:
            return m.group(0)
        return "[" + "; ".join(keys[:MAX_KEYS_PER_CITATION]) + "]"

    return re.sub(r"\[(@[^\]\n]+)\]", trim, text)


def strip_internal_ids(text: str) -> str:
    text = INTERNAL_ID.sub("", text)
    # the writer often names the hypothesis before its placeholder: 'H1 {{hyp:H1}}'
    text = re.sub(r"\s*\((?:validation )?warnings?(?::\s*[a-z_ ,]+)?\)", "", text, flags=re.I)
    # a sentence cut off where an unsupported citation was removed: 'This aligns with'
    text = re.sub(
        r"(?m)(?:(?<=[.!?])[ \t]+|^)[A-Z][^.!?\n]{0,120}\b(?:with|to|by|of|in|and|as|from|than|"
        r"such as|including|like)[ \t]*$",
        "",
        text,
    )
    text = re.sub(r"\b(H\d+):?\s+(?=\1\b)", "", text)
    head, nl, rest = text.partition("\n")
    if rest:
        text = head + nl + re.sub(r"(?m)^#[ \t]+[^\n]*\n", "", rest)
    # '**Tree Ensembles and Boosting:**' on its own line reads as a slide, not a paper
    text = re.sub(r"(?m)^\*\*[^*\n]{2,80}:?\*\*:?[ \t]*\n", "", text)
    text = drop_repeated_sentences(text)
    text = trim_citation_dumps(text)
    text = re.sub(
        r"(?m)^((?:The|This|Our) (?:study|analysis|paper|work|evaluation)) also\b ", r"\1 ", text
    )
    return re.sub(r"\((H\d+) \(([a-z ]+)\)\)", r"(\1 \2)", text)


def link_author_year(markdown: str, citations: dict[str, dict[str, Any]]) -> str:
    """'Abbas et al. (2023)' written as plain text becomes 'Abbas et al. [@Abbas2023...]' when
    exactly one verified reference has that first author and year, so it counts as cited."""
    index: dict[tuple[str, int], list[str]] = {}
    for key, c in citations.items():
        if c.get("status") != "Verified" or not c.get("authors") or not c.get("year"):
            continue
        index.setdefault((_surname(c["authors"][0]).lower(), int(c["year"])), []).append(key)

    def repl(m: re.Match[str]) -> str:
        keys = index.get((m.group(1).lower(), int(m.group(3))), [])
        if len(keys) != 1:
            return m.group(0)
        return f"{m.group(1)}{m.group(2) or ''} [@{keys[0]}]"

    body, sep, refs = markdown.partition("\n## References")
    verified = [k for k, c in citations.items() if c.get("status") == "Verified"]
    if verified:
        body = re.sub(
            r"(?<![@\w\[])("
            + "|".join(re.escape(k) for k in sorted(verified, key=len, reverse=True))
            + r")\b(?!\])",
            r"[@\1]",
            body,
        )

    def bracket(m: re.Match[str]) -> str:
        keys = index.get((m.group(1).lower(), int(m.group(2))), [])
        return f"[@{keys[0]}]" if len(keys) == 1 else m.group(0)

    body = re.sub(
        r"\[([A-Z][A-Za-z'\-]+)(?: et al\.| and [A-Z][A-Za-z'\-]+)?,? (\d{4})\]", bracket, body
    )
    body = re.sub(
        r"\b([A-Z][A-Za-z'\-]+)((?: et al\.| and [A-Z][A-Za-z'\-]+)?) \((\d{4})\)(?! ?\[@)",
        repl,
        body,
    )
    return body + sep + refs


def evidence_integers(analysis: dict[str, Any]) -> set[str]:
    """Whole numbers recorded in the analysis (seeds, per-class test rows, split sizes, trials):
    a writer quoting one of them is quoting evidence, not inventing a number."""
    found: set[str] = set(METHOD_COUNTS)

    def walk(v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            if 10 <= v < 10_000_000:
                found.add(str(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list | tuple):
            for x in v:
                walk(x)

    walk(analysis)
    return found


@dataclass
class EvidenceReport:
    resolved: list[dict[str, Any]] = field(default_factory=list)
    flags: list[dict[str, str]] = field(default_factory=list)
    # AI-use statement kept beside the paper (for the journal's submission form), not in it
    ai_disclosure: str = ""

    def flag(self, severity: str, code: str, message: str) -> None:
        self.flags.append({"severity": severity, "code": code, "message": message})

    @property
    def blockers(self) -> list[dict[str, str]]:
        return [f for f in self.flags if f["severity"] == "blocker"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "resolved": self.resolved,
            "flags": self.flags,
            "release_ready": not self.blockers,
            "ai_disclosure": self.ai_disclosure,
        }


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
        if kind == "recall":  # {{recall:MODEL:CLASS}} -> recall with its exact interval
            pc = ((mdl or {}).get("per_class") or {}).get(metric)
            if not pc:
                report.flag(sev, "MISSING_EVIDENCE", f"{section}: no per-class result {arg}")
                return missing
            report.resolved.append(
                {"placeholder": m.group(0), "value": pc["recall"], "status": status}
            )
            ci = pc.get("recall_ci")
            exact = f", 95% exact interval [{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci else ""
            return f"{_fmt(pc['recall'])} ({pc.get('support', 0):,} test rows{exact})"
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
        return f"95% CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}], {_ci_label(mdl or {})}"

    recorded = evidence_integers(analysis)

    def check_numbers(segment: str) -> str:
        shelved: list[str] = []

        def shelve(m: re.Match[str]) -> str:
            if m.group(0).replace(",", "") in recorded:
                shelved.append(m.group(0))
                # letters, not digits, so the number check below cannot see the placeholder
                tag = "".join(chr(97 + int(d)) for d in str(len(shelved) - 1))
                return f"\x00{tag}\x00"
            return m.group(0)

        segment = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", shelve, segment)
        out = _check_plain(segment)
        return re.sub(
            r"\x00([a-j]+)\x00",
            lambda m: shelved[int("".join(str(ord(c) - 97) for c in m.group(1)))],
            out,
        )

    def _check_plain(segment: str) -> str:
        def num(m: re.Match[str]) -> str:
            if m.group(1) in ALLOWED_PLAIN_NUMBERS or m.group(1) in recorded:
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
    """Small-class policy: classes under the protocol's minimum test support (default 30) are
    named with an exact (Clopper-Pearson) interval for the best model's recall."""
    threshold = int(analysis.get("min_test_support") or 30)
    small: dict[str, int] = {}
    for m in analysis.get("models", {}).values():
        for cls, pc in (m.get("per_class") or {}).items():
            if pc.get("support", 0) < threshold:
                small[cls] = int(pc["support"])
    if not small:
        return ""
    best = max(
        analysis.get("models", {}).items(),
        key=lambda kv: (kv[1].get("aggregate", {}).get("macro_f1") or {}).get("mean", 0.0),
    )
    exact = []
    for c in sorted(small):
        pc = (best[1].get("per_class") or {}).get(c) or {}
        ci = pc.get("recall_ci")
        if ci:
            exact.append(
                f"{c}: recall {pc.get('recall', 0):.4f}, 95% exact interval [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
    listed = ", ".join(f"{c} ({n:,} test rows)" for c, n in sorted(small.items()))
    return (
        f"Some classes have very few test rows: {listed}. Their per-class scores, and the "
        "macro-averaged metrics they feed into, are therefore highly uncertain. A bootstrap "
        "interval understates this: when a model detects all (or none) of a tiny class, every "
        "resample does too, so the interval can look narrow although a few more rows could "
        "change the score a lot."
        + (
            f" Exact binomial intervals for {_pretty_model(best[0])} show the real uncertainty ("
            + "; ".join(exact)
            + ")."
            if exact
            else ""
        )
    )


def _pretty_model(name: str) -> str:
    return {
        "logistic_regression": "logistic regression",
        "random_forest": "random forest",
        "xgboost": "XGBoost",
        "mlp": "the MLP",
        "lightgbm": "LightGBM",
        "resnet_mlp": "the ResNet-MLP",
        "ft_transformer": "the FT-Transformer",
        "cnn1d": "the 1D-CNN",
        "tabnet": "TabNet",
        "tfidf_logreg": "TF-IDF logistic regression",
        "tfidf_linear_svm": "TF-IDF linear SVM",
    }.get(name, name)


def _ci_label(m: dict[str, Any]) -> str:
    if m.get("ci_label"):
        return str(m["ci_label"])
    return f"seed {m['ci_seed']}" if m.get("ci_seed") is not None else "bootstrap"


def results_table(analysis: dict[str, Any]) -> str:
    rows = [
        "| Model | Role | Macro-F1 (mean ± sd) | 95% CI ({seed}) | Accuracy | MCC | FPR | Detection rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    seed = "?"
    for name, m in analysis["models"].items():
        a = m["aggregate"]
        ci = m["macro_f1_ci"]
        seed = _ci_label(m)

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
    rows = [
        "| Hypothesis | Comparison | Means | Difference (95% CI) | Odds ratio | "
        "Paired test p (one-sided) | Holm-adjusted p | Result |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for h in analysis["hypotheses"]:
        if "mcnemar" not in h:
            rows.append(f"| {h['id']} | - | - | - | - | - | - | {h['result']} |")
            continue
        holm_p = _fmt_p(h["p_holm"]) if "p_holm" in h else "-"
        eff = h.get("effect") or {}
        diff = (
            f"{eff['point']:+.4f} [{eff['low']:+.4f}, {eff['high']:+.4f}]"
            if {"point", "low", "high"} <= set(eff)
            else "-"
        )
        odds = h["mcnemar"].get("odds_ratio")
        odds_txt = f"{odds:.2f}" if odds is not None else "-"
        rows.append(
            f"| {h['id']} | {h['a']} > {h['b']} ({h['metric']}) | {_fmt(h['a_mean'])} vs {_fmt(h['b_mean'])} | "
            f"{diff} | {odds_txt} | {_fmt_p((h.get('test') or h['mcnemar'])['p_value'])} | {holm_p} | {h['result']} |"
        )
    return "\n".join(rows)


def _fmt_p(p: float) -> str:
    return "< 0.001" if p < 1e-3 else f"{p:.3f}"


def _n(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def benchmark_datasets_table(analysis: dict[str, Any]) -> str:
    b = analysis.get("benchmark") or {}
    rows = []
    for ds_id, v in (b.get("per_dataset") or {}).items():
        d = v.get("dataset") or (analysis.get("dataset") if ds_id == b["datasets"][0] else {})
        d = d or {}
        sc = d.get("split_counts") or {}
        rows.append(
            f"| {d.get('name') or ds_id} | {_n(d.get('n_rows_clean'))} | "
            f"{_n(d.get('n_duplicates_removed'))} | {_n(sc.get('train'))} | {_n(sc.get('test'))} | "
            f"{d.get('split_overlap_rows', 0)} |"
        )
    if not rows:
        return ""
    return (
        "| Dataset | Rows after cleaning | Removed (duplicates/conflicts) | Train | Test | "
        "Train/test overlap |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


def benchmark_table(analysis: dict[str, Any]) -> str:
    """Models x datasets on the primary metric (mean ± sd over seeds) and the average rank."""
    b = analysis.get("benchmark") or {}
    if not b.get("datasets"):
        return ""
    metric = b.get("metric", "macro_f1")
    datasets = b["datasets"]
    from cyberlab_data import get_adapter

    heads = [get_adapter(d).display for d in datasets]
    ranks = (b.get("ranking") or {}).get("average_rank") or {}
    models = list((b["per_dataset"][datasets[0]].get("models") or {}).keys())
    lines = [
        "| Model | " + " | ".join(heads) + (" | Average rank |" if ranks else " |"),
        "|---|" + "---|" * len(heads) + ("---|" if ranks else ""),
    ]
    for m in models:
        cells = []
        for d in datasets:
            agg = ((b["per_dataset"][d].get("models") or {}).get(m) or {}).get("aggregate") or {}
            v = agg.get(metric) or {}
            cells.append(f"{v['mean']:.4f} ± {v.get('sd', 0):.4f}" if "mean" in v else "–")
        rank = f" | {ranks[m]:.2f}" if m in ranks else (" | –" if ranks else "")
        lines.append(f"| {m} | " + " | ".join(cells) + rank + " |")
    return "\n".join(lines)


def benchmark_tests_table(analysis: dict[str, Any]) -> str:
    b = analysis.get("benchmark") or {}
    from cyberlab_data import get_adapter

    rows = []
    for d in b.get("datasets", [])[1:]:
        for h in (b["per_dataset"].get(d) or {}).get("hypotheses", []):
            if "p_holm" not in h:
                continue
            p = h["test"]["p_value"]
            rows.append(
                f"| {get_adapter(d).display} | {h['id']} | {h['a']} > {h['b']} ({h.get('metric')}) "
                f"| {_fmt(h['a_mean'])} vs {_fmt(h['b_mean'])} | "
                f"{'< 0.001' if p < 0.001 else f'{p:.4f}'} | "
                f"{'< 0.001' if h['p_holm'] < 0.001 else format(h['p_holm'], '.4f')} | "
                f"{h['result']} |"
            )
    if not rows:
        return ""
    return (
        "| Dataset | Hypothesis | Comparison | Means | Paired test p (one-sided) | "
        "Holm-adjusted p | Result |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )


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


# Plain-language limitation sentences for the scientific validator's warnings (Principle 4:
# the system states where its evidence is insufficient).
_VALIDATION_SENTENCES = {
    "near_duplicate_flows": "Many test rows lie close to training rows in feature space, so the "
    "test set may be easier than traffic from a new network.",
    "capture_groups": "The random split places traffic from the same capture period or source in "
    "both training and test; a group-aware split would give a stricter test.",
    "temporal_leakage": "Training and test traffic come from the same capture days, so the "
    "results do not show how the models handle traffic from later days.",
    "label_revealing_features": "Some single features almost determine the class on their own, "
    "which suggests the dataset contains shortcuts that may not exist in other networks.",
    "preprocessing_fitted_on_test": "The list of categorical values was taken from all rows "
    "(without labels) rather than from the training rows only.",
    "session_identifiers": "Overlap between training and test at the level of network sessions "
    "could not be evaluated, because the dataset has no session identifiers.",
    "result_consistency": "Some results vary noticeably between seeds or exceed the baselines by "
    "an unusually large margin, so they should be confirmed on further data.",
}


def validation_note(analysis: dict[str, Any]) -> str:
    checks = (analysis.get("validation") or {}).get("checks") or []
    out = [
        _VALIDATION_SENTENCES[c["check"]]
        for c in checks
        if c.get("status") in ("WARNING", "NOT_EVALUATED")
        and c.get("check") in _VALIDATION_SENTENCES
    ]
    if not out:
        return ""
    return (
        "The automatic scientific validation also found: "
        + " ".join(out)
        + (
            " Variation from re-partitioning the data was not measured, because one fixed split was "
            "used."
        )
    )


def followup_section(analysis: dict[str, Any]) -> str:
    """Results of a follow-up experiment next to the version it revisits."""
    f = analysis.get("followup")
    if not f:
        return ""
    metric = next(
        (h.get("metric") for h in analysis.get("hypotheses", []) if h.get("metric")), "macro_f1"
    )
    change = f.get("change") or {}
    what = (
        f"a {change['split']['strategy'].replace('_', ' ')} split"
        if "split" in change
        else f"{len(change['seeds'])} seeds"
        if "seeds" in change
        else "a revised protocol"
    )
    rows = [
        f"### Follow-up to protocol version {f['previous_version']}",
        "",
        f"The scientific validation of version {f['previous_version']} found that "
        f"{str(f.get('reason', '')).rstrip('.')}. This version repeats the experiments with {what} "
        f"(decision {f.get('decision_id')}, rule {f.get('rule')}, approved by the PI).",
        "",
        f"| Model | {metric} (version {f['previous_version']}) | {metric} (this version) | Change |",
        "|---|---|---|---|",
    ]
    for name, m in analysis.get("models", {}).items():
        now = (m.get("aggregate", {}).get(metric) or {}).get("mean")
        before = (f.get("previous_means") or {}).get(name, {}).get(metric)
        if now is None or before is None:
            continue
        rows.append(f"| {name} | {_fmt(before)} | {_fmt(now)} | {now - before:+.4f} |")
    return "\n".join(rows)


def disclosure(models: list[str]) -> str:
    """Disclosure statement (spec Step 21) describing exactly what the AI did."""
    who = ", ".join(sorted(set(models))) if models else "a large language model"
    return (
        "This manuscript was drafted with AutoResearch-CyberLab, an agentic research platform. "
        f"The framing text (introduction, related work, discussion and limitations) was written "
        f"by {who} and checked by deterministic rules. All numbers, tables, figures, the "
        "Methodology and the Results were generated by code from stored, hashed experiment "
        "predictions; every reference was verified against its DOI. Experiments ran only after "
        "the principal investigator approved the protocol. The authors reviewed and edited the "
        "text and take full responsibility for its content."
    )


def build_manuscript(
    title: str,
    sections: dict[str, str],
    analysis: dict[str, Any],
    citations: dict[str, dict[str, Any]],
    bibtex: str,
    guardrail: bool = False,
    llm_models: list[str] | None = None,
) -> tuple[str, EvidenceReport]:
    report = EvidenceReport()
    tables: list[str] = []

    def numbered(key: str, caption: str, text: str) -> str:
        """Put 'Table n. caption' above the first table in text (numbered in reading order)."""
        lines = text.split("\n")
        at = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")), None)
        if at is None:
            return text
        tables.append(key)
        lines[at:at] = [f"*Table {len(tables)}. {caption}*", ""]
        return "\n".join(lines)

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
            out += [
                metric_definitions(),
                "",
                "### Data",
                "",
                numbered(
                    "table_classes", "Rows per class after cleaning.", dataset_section(analysis)
                ),
                "",
            ]
            if dt := benchmark_datasets_table(analysis):
                out += [
                    numbered(
                        "table_datasets",
                        "All datasets of the benchmark after cleaning and splitting.",
                        dt,
                    ),
                    "",
                ]
        if key == "limitations" and (note := small_class_note(analysis)):
            out += [note, ""]
        if key == "limitations" and (vnote := validation_note(analysis)):
            out += [vnote, ""]
        if key == "results":
            if fu := followup_section(analysis):
                out += [
                    numbered(
                        "table_followup", "Results of this version next to the previous one.", fu
                    ),
                    "",
                ]
            out += [
                "### Main results",
                "",
                numbered(
                    "table_main",
                    "Main results per model (means and standard deviations over seeds).",
                    results_table(analysis),
                ),
                "",
                "### Hypothesis tests",
                "",
                numbered(
                    "table_hypotheses",
                    "Hypothesis tests with effect sizes and Holm-adjusted p-values.",
                    hypotheses_table(analysis),
                ),
                "",
            ]
            if bt := benchmark_table(analysis):
                out += [
                    "### Results across datasets",
                    "",
                    numbered(
                        "table_benchmark",
                        "Primary metric per model and dataset (means and standard deviations "
                        "over seeds) and the average rank across datasets (1 = best).",
                        bt,
                    ),
                    "",
                ]
                if tt := benchmark_tests_table(analysis):
                    out += [
                        numbered(
                            "table_benchmark_tests",
                            "The stated hypotheses repeated on the other datasets (secondary "
                            "tests, Holm-adjusted across them).",
                            tt,
                        ),
                        "",
                    ]
            from cyberlab_manuscript.figures import _primary_metric, figures_markdown

            metric = _primary_metric(analysis)
            best = max(
                analysis["models"],
                key=lambda n: (analysis["models"][n]["aggregate"].get(metric) or {}).get(
                    "mean", float("-inf")
                ),
            )
            out += [
                f"### Per-class results ({best})",
                "",
                numbered(
                    "table_per_class",
                    f"Per-class results of {best} (first seed).",
                    per_class_table(analysis, best),
                ),
                "",
            ]
            try:
                figs = figures_markdown(analysis, best)
            except Exception as exc:  # noqa: BLE001 - a figure failure must not block the draft
                report.flag("warning", "FIGURES_FAILED", f"figures could not be drawn: {exc}")
                figs = ""
            if figs:
                out += ["### Figures", "", figs, ""]
    for required in ("abstract", "introduction", "methodology", "results", "conclusion"):
        if not str(sections.get(required, "")).strip():
            report.flag("blocker", "EMPTY_SECTION", f"the {required} section is empty")
    for f in analysis.get("qa_flags", []):
        report.flag("blocker" if f["severity"] == "blocker" else "warning", f["code"], f["message"])
    report.ai_disclosure = disclosure(llm_models or [])
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
    md = resolve_refs("\n".join(out) + "\n", tables)
    return md, report


_FIGURE_KEYS = (
    ("figure_ci", "confidence intervals"),
    ("figure_recall", "recall per class"),
    ("figure_confusion", "confusion matrix"),
)


def resolve_refs(md: str, tables: list[str]) -> str:
    """Cross-references (professor's plan, Phase 8): {{ref:table_main}} -> 'Table 2',
    {{ref:figure_ci}} -> 'Figure 1', numbered in reading order."""
    refs = {k: f"Table {i}" for i, k in enumerate(tables, 1)}
    for m in re.finditer(r"^\*Figure (\d+)\. (.+?)\*$", md, re.M):
        for key, needle in _FIGURE_KEYS:
            if needle in m.group(2).lower():
                refs.setdefault(key, f"Figure {m.group(1)}")
    return re.sub(r"\{\{ref:([a-z_]+)\}\}", lambda m: refs.get(m.group(1), "the table below"), md)


def manuscript_sha(md: str) -> str:
    return hashlib.sha256(md.encode()).hexdigest()
