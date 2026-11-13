"""Evidence-locked manuscript text.

The writing model (often a small local model) kept inventing new ways to misstate results.
Instead of chasing each one, the paper is split into two kinds of text:

* Evidence text, generated here from the stored analysis and method facts only (no LLM):
  the Methodology and Results prose, the findings in the Abstract and Conclusion, and the
  interpretation paragraph that opens the Discussion.
* Framing text, written by the LLM: motivation, related work, discussion of context,
  limitations and future work. Every LLM sentence passes a whitelist filter: a sentence that
  states a score, a hypothesis, a confidence interval, a test or an evaluative claim about a
  model is dropped, because those statements belong to the evidence text.

`lock_evidence` applies both. It is idempotent and needs no LLM, so it runs on every draft.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

from cyberlab_manuscript.consistency import (
    COMPARE,
    SIG_DIFF,
    SUPERIORITY,
    accuracy_gap_models,
    consistency_issues,
    mentions,
    sections,
    split_sentences,
)

PRETTY = {
    "logistic_regression": "logistic regression",
    "random_forest": "random forest",
    "xgboost": "XGBoost",
    "mlp": "the MLP",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _and(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _num(n: int) -> str:
    words = "zero one two three four five six seven eight nine ten".split()
    return words[n] if 0 <= n < len(words) else str(n)


def _metric(name: str) -> str:
    return {"macro_f1": "macro-F1", "balanced_accuracy": "balanced accuracy"}.get(
        name, name.replace("_", " ")
    )


def _p(p: float | None) -> str:
    if p is None:
        return ""
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def primary_metric(analysis: dict[str, Any]) -> str:
    """The study's primary metric: the one its hypotheses are tested on (macro-F1 otherwise)."""
    for h in analysis.get("hypotheses", []):
        if h.get("metric"):
            return str(h["metric"])
    return "macro_f1"


def _best(analysis: dict[str, Any], metric: str | None = None) -> str | None:
    metric = metric or primary_metric(analysis)
    models = analysis.get("models", {})
    scored = [n for n in models if "mean" in (models[n].get("aggregate", {}).get(metric) or {})]
    return max(scored, key=lambda n: models[n]["aggregate"][metric]["mean"]) if scored else None


def _tested_hyps(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return [h for h in analysis.get("hypotheses", []) if h.get("a") and h.get("b")]


def hypothesis_sentences(analysis: dict[str, Any]) -> list[str]:
    out = []
    models = analysis.get("models", {})
    for h in _tested_hyps(analysis):
        metric = h.get("metric", "macro_f1")
        a_mean = h.get(
            "a_mean", (models.get(h["a"], {}).get("aggregate", {}).get(metric) or {}).get("mean")
        )
        b_mean = h.get(
            "b_mean", (models.get(h["b"], {}).get("aggregate", {}).get(metric) or {}).get("mean")
        )
        means = (
            f"; {a_mean:.4f} vs {b_mean:.4f}" if a_mean is not None and b_mean is not None else ""
        )
        ptxt = _p((h.get("mcnemar") or {}).get("p_value"))
        test = f" (exact McNemar test, {ptxt})" if ptxt else ""
        out.append(
            f"{h['id']} ({pretty(h['a'])} scores higher than {pretty(h['b'])} on "
            f"{_metric(metric)}{means}) was {h['result']}{test}."
        )
    return out


def untested_pairs_sentence(analysis: dict[str, Any]) -> str:
    tested = {frozenset((h["a"], h["b"])) for h in _tested_hyps(analysis)}
    untested = [
        f"{pretty(a)} vs {pretty(b)}"
        for a, b in itertools.combinations(analysis.get("models", {}), 2)
        if frozenset((a, b)) not in tested
    ]
    if not untested:
        return ""
    return (
        "No significance test was run for the other pairs ("
        + "; ".join(untested)
        + "), so the paper makes no claim about their relative order beyond the measured means."
    )


def findings_text(analysis: dict[str, Any]) -> str:
    """Main findings for the Abstract and Conclusion."""
    out = []
    metric = primary_metric(analysis)
    best = _best(analysis, metric)
    if best:
        v = analysis["models"][best]["aggregate"][metric]
        out.append(
            f"On the de-duplicated test split, {pretty(best)} had the highest mean "
            f"{_metric(metric)} ({v['mean']:.4f} ± {v.get('sd', 0):.4f} over the model seeds)."
        )
    out += hypothesis_sentences(analysis)
    return " ".join(out)


def results_text(analysis: dict[str, Any]) -> str:
    """The complete Results prose, from the stored aggregates and tests."""
    paras = []
    lines = []
    metric = primary_metric(analysis)
    order = [metric] + [
        k for k in ("macro_f1", "balanced_accuracy", "accuracy", "mcc") if k != metric
    ]
    for name, m in analysis.get("models", {}).items():
        agg = m.get("aggregate", {})
        lead = agg.get(metric) or {}
        if "mean" not in lead:
            continue
        txt = (
            f"{_cap(pretty(name))} reached a {_metric(metric) if metric != 'mcc' else 'MCC'} of "
            f"{lead['mean']:.4f} ± {lead.get('sd', 0):.4f}"
        )
        ci = m.get(f"{metric}_ci") or {}
        if "low" in ci and "high" in ci:
            seed = m.get("ci_seed")
            txt += f" (95% bootstrap CI [{ci['low']:.4f}, {ci['high']:.4f}]" + (
                f", seed {seed})" if seed is not None else ")"
            )
        extra = []
        for key in order[1:]:
            v = agg.get(key) or {}
            if "mean" in v:
                label = "MCC" if key == "mcc" else _metric(key)
                ci2 = m.get(f"{key}_ci") or {}
                ci_txt = (
                    f" (95% CI [{ci2['low']:.4f}, {ci2['high']:.4f}])"
                    if "low" in ci2 and "high" in ci2
                    else ""
                )
                extra.append(f"{label} {v['mean']:.4f}{ci_txt}")
        if extra:
            txt += ", with " + ", ".join(extra)
        lines.append(txt + ".")
    if lines:
        paras.append(
            "Scores are means ± standard deviations over the model seeds on the test split.\n\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )
    minority = minority_recall_text(analysis)
    if minority:
        paras.append(minority)
    hyp = hypothesis_sentences(analysis)
    scope = untested_pairs_sentence(analysis)
    if hyp or scope:
        paras.append(" ".join(hyp + ([scope] if scope else [])))
    best = _best(analysis)
    if best:
        support = {
            c: v.get("support", 0)
            for c, v in (analysis["models"][best].get("per_class") or {}).items()
        }
        if support:
            small = min(support, key=support.__getitem__)
            paras.append(
                f"Per-class results for {pretty(best)} are listed below; the smallest class, "
                f"{small}, has only {support[small]:,} test rows."
            )
    return "\n\n".join(paras)


def minority_recall_text(analysis: dict[str, Any], n: int = 3) -> str:
    """Recall of every model on the smallest classes (first seed's predictions)."""
    models = analysis.get("models", {})
    support: dict[str, int] = {}
    for m in models.values():
        for c, v in (m.get("per_class") or {}).items():
            support[c] = max(support.get(c, 0), int(v.get("support", 0)))
    if len(support) < 3:
        return ""
    small = sorted(support, key=support.__getitem__)[:n]
    parts = []
    for name, m in models.items():
        pc = m.get("per_class") or {}
        vals = [f"{c} {pc[c]['recall']:.4f}" for c in small if c in pc and "recall" in pc[c]]
        if vals:
            parts.append(f"{pretty(name)}: " + ", ".join(vals))
    if not parts:
        return ""
    return (
        "Recall on the "
        + _num(len(small))
        + " smallest classes ("
        + ", ".join(f"{c}, {support[c]:,} test rows" for c in small)
        + "), first seed: "
        + "; ".join(parts)
        + "."
    )


def gap_text(analysis: dict[str, Any]) -> str:
    names = accuracy_gap_models(analysis)
    if not names:
        return ""
    parts = []
    for n in names:
        agg = analysis["models"][n]["aggregate"]
        parts.append(
            f"{pretty(n)} (accuracy {agg['accuracy']['mean']:.4f} vs balanced accuracy "
            f"{agg['balanced_accuracy']['mean']:.4f})"
        )
    return (
        "Accuracy is much higher than balanced accuracy for "
        + " and ".join(parts)
        + ", so for these models the high accuracy mostly reflects the large BENIGN class and "
        "hides weaker performance on the minority attack classes."
    )


def discussion_lead(analysis: dict[str, Any]) -> str:
    sents = []
    supported = [h for h in _tested_hyps(analysis) if h.get("result") == "supported"]
    rejected = [h for h in _tested_hyps(analysis) if h.get("result") != "supported"]
    if supported:
        sents.append(
            "The tested comparisons favoured "
            + _and([f"{pretty(h['a'])} over {pretty(h['b'])} ({h['id']})" for h in supported])
            + "."
        )
    if rejected:
        sents.append(
            "No significant advantage was found for "
            + _and([f"{pretty(h['a'])} over {pretty(h['b'])} ({h['id']})" for h in rejected])
            + "."
        )
    scope = untested_pairs_sentence(analysis)
    if scope:
        sents.append(scope)
    return " ".join(sents)


def methodology_text(facts: dict[str, Any], prose: list[str], analysis: dict[str, Any]) -> str:
    """Methodology from the deterministic method facts ('facts.x' references filled in)."""

    def fill(s: str) -> str:
        return re.sub(r"facts\.([a-z_]+)", lambda m: str(facts.get(m.group(1), m.group(0))), s)

    statements = [fill(s) for s in prose if not s.startswith("Model '")]
    models = analysis.get("models", {})
    if models:
        statements.append(
            f"{_cap(_num(len(models)))} models were compared: "
            + _and([f"{pretty(n)} ({m.get('role', 'model')})" for n, m in models.items()])
            + "."
        )
    # paragraphs: data preparation / split and training / statistics
    cut1 = next((i for i, s in enumerate(statements) if "split" in s.lower()), len(statements))
    cut2 = next(
        (i for i, s in enumerate(statements) if "confidence interval" in s.lower()), len(statements)
    )
    groups = [statements[:cut1], statements[cut1:cut2], statements[cut2:]]
    return "\n\n".join(" ".join(g) for g in groups if g)


# ---------------------------------------------------------------- framing-text whitelist

FILTERED_SECTIONS = {
    "Abstract",
    "Introduction",
    "Related Work",
    "Discussion",
    "Limitations",
    "Conclusion",
}
_NUMBER = re.compile(r"\b\d+\.\d+\b")
# verdicts and test outcomes (mentioning that intervals or tests are reported is fine)
_STATS = re.compile(
    r"\bH\d+\b|hypothes\w*[^.]{0,80}\b(?:support|confirm|reject|refut|hold|true|false|valid)\w*|"
    r"\b(?:support|confirm|reject|refut|validat|contradict)\w*[^.]{0,40}\bhypothes|"
    r"p[- ]?values?\b|McNemar|statistically|standard deviations?|\bvariance\b|variabilit|"
    r"\bsignificant(?:ly)? (?:\w+ )?(?:better|worse|higher|lower|differen|improv|advantage|"
    r"outperform|gain)\w*",
    re.I,
)
_EVAL = re.compile(
    r"\b(?:more|less|most|least|highly) (?:effective|accurate|robust|reliable|consistent)(?:ly)?\b|"
    r"\b(?:shown|show|shows|showed) promise\b|"
    r"\b(?:best|worst|strongest|weakest|top|leading)\b|"
    r"\bperform(?:s|ed|ing)? (?:better|worse|best|well|poorly|strongly|comparabl\w*|similarly)\b|"
    r"\b(?:achiev|obtain|reach|attain|deliver|yield)\w*\b|\b(?:benefit|excel)\w*\b|"
    r"\b(?:high|higher|low|lower|strong|weak|good|poor|excellent|superior|inferior)\b[^.]{0,30}"
    r"\b(?:score|accuracy|performance|F1|macro-F1|MCC|recall|precision|detection)\w*",
    re.I,
)
_RESULT_CLAIM = re.compile(
    r"\b(?:results?|findings?|evaluation|experiments?|analysis)\b[^.]{0,60}"
    r"\b(?:show|indicat|reveal|demonstrat|suggest|confirm|prove)\w*",
    re.I,
)
_GROUP = re.compile(
    r"\b(?:tree[- ]based|complex|simpler|deep|ensemble|linear) (?:models?|methods?)\b", re.I
)


def forbidden(sentence: str, context: str, known: list[str]) -> str | None:
    """Why an LLM sentence may not stay in a framing section (None = allowed)."""
    if _NUMBER.search(sentence):
        return "states a measured value"
    if _STATS.search(sentence):
        return "states a hypothesis, test or interval result"
    named = mentions(f"{context} {sentence}", known) or _GROUP.search(sentence)
    if named and (
        COMPARE.search(sentence)
        or SIG_DIFF.search(sentence)
        or SUPERIORITY.search(sentence)
        or _EVAL.search(sentence)
    ):
        return "makes an evaluative claim about a model"
    if _RESULT_CLAIM.search(sentence) and (COMPARE.search(sentence) or _EVAL.search(sentence)):
        return "states a result"
    return None


_CONNECTIVE = re.compile(
    r"^(?:However|Similarly|Moreover|Additionally|Furthermore|In addition|Likewise|Nevertheless|"
    r"Consequently|Therefore|Thus|Also|Specifically|In particular|In contrast|Conversely),\s+"
)
_PRONOUN = re.compile(r"^(?:It|Its|They|Their|Them|This|These|Such)\b")


_ORDINALS = ["First", "Second", "Third", "Fourth", "Fifth", "Sixth"]
_ORDINAL = re.compile(
    r"^(?:First(?:ly)?|Second(?:ly)?|Third(?:ly)?|Fourth(?:ly)?|Fifth(?:ly)?|Sixth(?:ly)?|"
    r"Finally|Lastly),\s+"
)


def _renumber(sentences: list[str]) -> list[str]:
    """'First, ... Third, ...' after a removal becomes 'First, ... Second, ...'."""
    ordinal_idx = [i for i, s in enumerate(sentences) if _ORDINAL.match(s)]
    if len(ordinal_idx) < 2:
        if len(ordinal_idx) == 1 and not sentences[ordinal_idx[0]].startswith(
            ("First", "Finally", "Lastly")
        ):
            i = ordinal_idx[0]  # a lone 'Third,' reads as a gap: drop the ordinal
            sentences[i] = _cap(_ORDINAL.sub("", sentences[i], count=1))
        return sentences
    for n, i in enumerate(ordinal_idx):
        s = sentences[i]
        if s.startswith(("Finally", "Lastly")) and n == len(ordinal_idx) - 1:
            continue
        sentences[i] = _ORDINAL.sub(f"{_ORDINALS[min(n, len(_ORDINALS) - 1)]}, ", s, count=1)
    return sentences


def _rebuild(body: str, drop: set[str]) -> str:
    """Rebuild a section without the dropped sentences, cleaning connectives and orphans."""
    out_lines = []
    for line in body.split("\n"):
        if not line.strip() or line.lstrip().startswith(("|", "#", ">")):
            out_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)", line)
        prefix, text = (m.group(1), line[m.end() :]) if m else ("", line)
        kept: list[str] = []
        prev_dropped = False
        for s in split_sentences(text):
            if s in drop or (prev_dropped and _PRONOUN.match(s)):
                prev_dropped = True
                continue
            if (not kept or prev_dropped) and _CONNECTIVE.match(s):
                s = _cap(_CONNECTIVE.sub("", s, count=1))
            kept.append(s)
            prev_dropped = False
        if kept:
            out_lines.append(prefix + " ".join(_renumber(kept)))
    text = "\n".join(out_lines)
    text = re.sub(r"(?m)^\*\*[^*\n]{1,80}:?\*\*:?[ \t]*\n(?=\s*(?:\n|\*\*|$))", "", text)
    return re.sub(r"\n{3,}", "\n\n", text)


def _replace_section(markdown: str, name: str, new_body: str) -> str:
    pattern = re.compile(rf"(?ms)^(## {re.escape(name)}\n)(.*?)(?=^## |\Z)")
    return pattern.sub(lambda m: m.group(1) + "\n" + new_body.strip() + "\n\n", markdown, count=1)


def _prose_and_rest(body: str) -> tuple[str, str]:
    """Split a section body into its prose and its generated sub-sections ('### ...')."""
    idx = body.find("\n### ")
    return (body, "") if idx < 0 else (body[:idx], body[idx:])


def lock_evidence(
    markdown: str,
    analysis: dict[str, Any],
    facts: dict[str, Any] | None = None,
    prose: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Make every evidence statement generated and every LLM sentence framing-only.

    Returns the new markdown and a list of {section, sentence, reason} that were dropped."""
    known = list(analysis.get("models", {}))
    dropped: list[dict[str, str]] = []
    bodies = dict(sections(markdown))

    # 1. Methodology and Results prose come only from stored facts and results.
    if "Methodology" in bodies and facts is not None and prose is not None:
        _, rest = _prose_and_rest(bodies["Methodology"])
        markdown = _replace_section(
            markdown, "Methodology", methodology_text(facts, prose, analysis) + "\n" + rest
        )
    if "Results" in bodies:
        _, rest = _prose_and_rest(bodies["Results"])
        markdown = _replace_section(markdown, "Results", results_text(analysis) + "\n" + rest)

    # 2. Framing sections keep only sentences that state no results.
    issues = {i["sentence"]: i["reason"] for i in consistency_issues(markdown, analysis)}
    # sentences this module generates are removed silently and re-added below (idempotent)
    generated = set(
        split_sentences(
            " ".join([findings_text(analysis), discussion_lead(analysis), gap_text(analysis)])
        )
    )
    for name, body in sections(markdown):
        if name not in FILTERED_SECTIONS:
            continue
        prose_part, rest = _prose_and_rest(body)
        drop: set[str] = set()
        intro = ""
        for line in prose_part.split("\n"):
            if not line.strip() or line.lstrip().startswith(("|", "#", ">")):
                continue
            is_item = bool(re.match(r"\s*(?:[-*+]|\d+\.)\s", line))
            text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", line) if is_item else line
            ctx = intro if is_item else ""
            if not is_item:
                intro = line
            for s in split_sentences(text):
                if s in generated:
                    drop.add(s)
                    continue
                if name == "Related Work" and "[@" in s:
                    continue  # statements about cited papers are checked against their abstracts
                why = issues.get(s) or forbidden(s, ctx, known)
                if why:
                    drop.add(s)
                    dropped.append({"section": name, "sentence": s, "reason": why})
        kept = _rebuild(prose_part, drop).strip()

        # 3. Evidence statements are added where the paper needs them.
        if name == "Abstract":
            kept = (kept + " " + findings_text(analysis)).strip()
        elif name == "Conclusion":
            sents = split_sentences(" ".join(kept.split()))
            future = re.compile(r"(?i)(?:additionally,\s*|moreover,\s*)?(?:future|further)\b")
            at = next((i for i, s in enumerate(sents) if future.match(s)), len(sents))
            kept = " ".join(sents[:at] + [findings_text(analysis)] + sents[at:])
        elif name == "Discussion":
            lead = [p for p in (discussion_lead(analysis), gap_text(analysis)) if p]
            kept = "\n\n".join(lead + ([kept] if kept else []))
        if not kept.strip() and name in ("Introduction", "Limitations"):
            # every model-written sentence was a result claim: state the scope factually
            models = _and([pretty(n) for n in analysis.get("models", {})])
            kept = (
                f"This study compares {models} under one fixed, de-duplicated evaluation protocol."
                if name == "Introduction"
                else "All results come from a single dataset and a single fixed train/test split, "
                "so they may not transfer to other networks or splits."
            )
        markdown = _replace_section(markdown, name, kept + ("\n" + rest if rest else ""))
    return markdown, dropped
