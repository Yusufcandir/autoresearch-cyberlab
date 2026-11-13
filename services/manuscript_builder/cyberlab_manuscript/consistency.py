"""Deterministic consistency checks on the rendered manuscript.

These run without an LLM, so they catch the same mistakes every time, whichever model wrote the
draft (a small local model makes them far more often than Gemini):

- HYP_CONTRADICTION: prose negates a hypothesis the evidence supports, or vice versa
  ("supports H1 but not H2 (supported)").
- UNTESTED_COMPARISON: two models are compared ("outperforms", "comparable to", ...) but no
  hypothesis test covered that pair; or one model is called superior to all others when not
  every pair was tested.
- CONTRADICTS_LEAKAGE_CHECK: prose suggests data leakage although the stored split has zero
  train/test overlap rows.
- RESULT_IN_METHOD: a measured score appears in Introduction, Related Work or Methodology,
  usually because a metric placeholder was used where a model name was meant.
- UNRESOLVED_PLACEHOLDER: a sentence still contains '[not reported]', '[missing evidence]' or
  '[number removed]', i.e. the writer referred to a fact that does not exist.
"""

from __future__ import annotations

import itertools
import re
from typing import Any

MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "logistic_regression": ("logistic regression", "logistic_regression", r"\bLR\b"),
    "random_forest": ("random forest", "random_forest", r"\bRF\b"),
    "xgboost": ("xgboost", "gradient boosting", "gradient-boosting", r"\bGBM\b", r"\bGBDT\b"),
    "mlp": (r"\bmlp\b", "neural network", "multi-layer perceptron", "multilayer perceptron"),
}

COMPARE = re.compile(
    r"\b(outperform\w*|better than|worse than|superior|inferior|surpass\w*|exceed\w*|"
    r"beat\w*|comparabl\w*|on par|similar(?:ly)? to|(?:higher|lower|greater) (?:\S+ ){0,2}than|"
    r"advantage over|edge over)\b",
    re.I,
)
SUPERIORITY = re.compile(
    r"\b(superiority|superior to (?:all|the other|every)|outperform\w* (?:all|the other|every)|"
    r"best[- ]performing|the best model)\b",
    re.I,
)
NEGATION = re.compile(
    r"\b(not|no|fail\w*|reject\w*|refut\w*|contradict\w*|unsupported|lack\w*|neither|nor)\b", re.I
)
LEAK = re.compile(r"\bleak\w*", re.I)
NO_LEAK = re.compile(
    r"\b(no|without|zero|prevent\w*|avoid\w*|free of|ensur\w*)\b[^.]{0,40}\bleak\w*|"
    r"\bleak\w*[^.]{0,30}\b(was|were) (not|ruled out|excluded)",
    re.I,
)
SCORE = re.compile(r"\b\d\.\d{4}\b")
UNRESOLVED = ("[not reported]", "[missing evidence]", "[number removed]", "[citation needed]")
NO_RESULTS_SECTIONS = {"Introduction", "Related Work", "Methodology"}
SKIP_SECTIONS = {"Evidence", "References"}


def sections(markdown: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^## (.+)$", markdown)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def prose_sentences(body: str) -> list[str]:
    out = []
    for para in body.split("\n"):
        if not para.strip() or para.startswith(("|", "#", ">")):
            continue
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", para) if s.strip()]
    return out


def mentions(sentence: str, known: list[str]) -> list[tuple[int, str]]:
    """(position, model) for every model named in the sentence, in reading order."""
    found: list[tuple[int, str]] = []
    for name in known:
        pats = MODEL_ALIASES.get(name, (re.escape(name), re.escape(name.replace("_", " "))))
        for pat in pats:
            found += [(m.start(), name) for m in re.finditer(pat, sentence, re.I)]
    return sorted(set(found))


def compared_pairs(sentence: str, known: list[str]) -> list[tuple[str, str]]:
    """Model pairs that a comparison word puts against each other.

    'A outperforms B and C' -> (A,B), (A,C). In 'A beats B, and C beats B' the C before the
    second comparison is its subject, not an object of the first."""
    ms = mentions(sentence, known)
    comps = list(COMPARE.finditer(sentence))
    pairs: list[tuple[str, str]] = []
    for i, c in enumerate(comps):
        lo = comps[i - 1].end() if i else 0
        hi = comps[i + 1].start() if i + 1 < len(comps) else len(sentence)
        left = [n for pos, n in ms if lo <= pos < c.start()]
        right = [n for pos, n in ms if c.end() <= pos < hi]
        if i + 1 < len(comps) and right:
            right = right[:-1]  # subject of the next comparison
        if not left:
            continue
        subject = left[-1]
        pairs += [(subject, o) for o in dict.fromkeys(right) if o != subject]
    return pairs


def _hyp_contradiction(sentence: str) -> bool:
    for m in re.finditer(r"(H\d+) \((supported|not supported)\)", sentence):
        before = sentence[max(0, m.start() - 40) : m.start()]
        # look only at the clause right before the hypothesis
        clause = re.split(r"[,;]|\b(?:but|and|while|whereas)\b", before)[-1]
        negated = bool(NEGATION.search(clause))
        if m.group(2) == "supported" and negated:
            return True
        if (
            m.group(2) == "not supported"
            and re.search(r"\bsupport\w*\b", clause, re.I)
            and not negated
        ):
            return True
    return False


def consistency_issues(markdown: str, analysis: dict[str, Any]) -> list[dict[str, str]]:
    known = list(analysis.get("models", {}))
    tested = {
        frozenset((h["a"], h["b"]))
        for h in analysis.get("hypotheses", [])
        if h.get("a") and h.get("b")
    }
    all_pairs_tested = all(frozenset(p) in tested for p in itertools.combinations(known, 2))
    overlap = (analysis.get("dataset") or {}).get("split_overlap_rows")
    issues: list[dict[str, str]] = []

    def add(section: str, sentence: str, code: str, reason: str) -> None:
        issues.append({"section": section, "sentence": sentence, "code": code, "reason": reason})

    for name, body in sections(markdown.split("\n## Evidence", 1)[0]):
        if name in SKIP_SECTIONS:
            continue
        for sent in prose_sentences(body):
            if any(tag in sent for tag in UNRESOLVED):
                add(
                    name,
                    sent,
                    "UNRESOLVED_PLACEHOLDER",
                    "refers to a fact or number that does not exist in the stored evidence",
                )
                continue
            if name in NO_RESULTS_SECTIONS and SCORE.search(sent):
                add(
                    name,
                    sent,
                    "RESULT_IN_METHOD",
                    f"a measured score appears in {name}; scores belong in Results",
                )
                continue
            if _hyp_contradiction(sent):
                add(
                    name,
                    sent,
                    "HYP_CONTRADICTION",
                    "the wording contradicts the stored hypothesis result",
                )
                continue
            if overlap == 0 and LEAK.search(sent) and not NO_LEAK.search(sent):
                add(
                    name,
                    sent,
                    "CONTRADICTS_LEAKAGE_CHECK",
                    "suggests data leakage, but the stored split has 0 train/test overlap rows",
                )
                continue
            if name == "Related Work" or "[@" in sent:
                continue  # statements about other papers are checked against their abstracts
            mentioned = [n for _, n in mentions(sent, known)]
            if COMPARE.search(sent):
                untested = [
                    f"{a} vs {b}"
                    for a, b in dict.fromkeys(compared_pairs(sent, known))
                    if frozenset((a, b)) not in tested
                ]
                if untested:
                    add(
                        name,
                        sent,
                        "UNTESTED_COMPARISON",
                        "compares models that no hypothesis test compared: " + ", ".join(untested),
                    )
                    continue
            if SUPERIORITY.search(sent) and mentioned and not all_pairs_tested:
                add(
                    name,
                    sent,
                    "UNTESTED_COMPARISON",
                    "claims one model is best overall, but not every pair of models was tested",
                )
    return issues
