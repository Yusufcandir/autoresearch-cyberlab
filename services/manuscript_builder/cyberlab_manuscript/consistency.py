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
# Group words that stand for several models at once ('tree-based models' = RF and XGBoost).
MODEL_GROUPS: dict[str, tuple[str, ...]] = {
    r"\btree[- ]based (?:models?|methods?|ensembles?|classifiers?)\b": ("random_forest", "xgboost"),
}

COMPARE = re.compile(
    r"\b(outperform\w*|better than|worse than|superior|inferior|surpass\w*|exceed\w*|"
    r"beat\w*|comparabl\w*|on par|similar(?:ly)? to|(?:higher|lower|greater) (?:\S+ ){0,2}than|"
    r"advantage over|edge over|improvement over|gain over|"
    r"(?:better|worse|higher|lower)\b[^.,;]{0,40}?\bcompared (?:to|with))\b",
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
    r"\b(no|without|zero|prevent\w*|avoid\w*|free of|ensur\w*|consider\w*|check\w*|audit\w*|"
    r"guard\w* against|detect\w*|importance of)\b[^.]{0,60}\bleak\w*|"
    r"\bleak\w*[^.]{0,30}\b(was|were) (not|ruled out|excluded)",
    re.I,
)
SCORE = re.compile(r"\b\d\.\d{4}\b")
UNRESOLVED = ("[not reported]", "[missing evidence]", "[number removed]", "[citation needed]")
# Prompt or template text copied literally into the paper ('95% CI [low, high], seed N').
TEMPLATE_LEAK = re.compile(
    r"\[low, ?high\]|\bseed N\b|\{\{|\}\}|\bMODEL:METRIC\b|\bfacts?\.[a-z_]+\b|\bKEY\d?\b",
)
SAME_CI = re.compile(
    r"\bsame (?:95% )?(?:CI|confidence interval)s?\b|\bidentical (?:CI|confidence interval)", re.I
)
NO_RESULTS_SECTIONS = {"Introduction", "Related Work", "Methodology"}
SKIP_SECTIONS = {"Evidence", "References"}


def sections(markdown: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?m)^## (.+)$", markdown)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


_ABBREV = re.compile(r"\b(et al|e\.g|i\.e|vs|cf|Fig|Eq|approx|resp)\.", re.I)
_DOT = "․"  # one-dot leader: a stand-in for an abbreviation's full stop while splitting


def split_sentences(para: str) -> list[str]:
    """Split a paragraph into sentences without breaking 'Alikhanov et al. [@k]' or 'e.g.'."""
    protected = _ABBREV.sub(lambda m: m.group(0).replace(".", _DOT), para)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[*])", protected)
    return [p.replace(_DOT, ".").strip() for p in parts if p.strip()]


def prose_items(body: str) -> list[tuple[str, str]]:
    """(sentence, context). A list item's context is the line that introduces the list, e.g.
    '**Logistic Regression:**' above '- Macro-F1: 0.6815', so the item is read with its model."""
    out: list[tuple[str, str]] = []
    intro = ""
    for para in body.split("\n"):
        if not para.strip() or para.startswith(("|", "#", ">", "![", "*Figure")):
            continue
        is_item = bool(re.match(r"\s*(?:[-*+]|\d+\.)\s", para))
        ctx = intro if is_item else ""
        if not is_item:
            intro = para
        text = re.sub(r"^\s*(?:[-*+]|\d+\.)\s+", "", para) if is_item else para
        out += [(s, ctx) for s in split_sentences(text)]
    return out


def prose_sentences(body: str) -> list[str]:
    return [s for s, _ in prose_items(body)]


def mentions(sentence: str, known: list[str]) -> list[tuple[int, str]]:
    """(position, model) for every model named in the sentence, in reading order."""
    found: list[tuple[int, str]] = []
    for name in known:
        # 'resnet_mlp' is written 'ResNet-MLP', 'resnet mlp' or 'ResNetMLP' in prose
        loose = r"\b" + r"[\s_-]?".join(re.escape(w) for w in name.split("_")) + r"\b"
        pats = (*MODEL_ALIASES.get(name, ()), loose)
        for pat in pats:
            found += [(m.start(), name) for m in re.finditer(pat, sentence, re.I)]
    for pat, members in MODEL_GROUPS.items():
        for m in re.finditer(pat, sentence, re.I):
            found += [(m.start(), n) for n in members if n in known]
    return sorted(set(found))


def comparisons(sentence: str, known: list[str]) -> list[tuple[str, str, str]]:
    """(subject, object, comparison word) for every model comparison in the sentence.

    'A outperforms B and C' -> (A,B), (A,C). In 'A beats B, and C beats B' the C before the
    second comparison is its subject, not an object of the first."""
    ms = mentions(sentence, known)
    comps = list(COMPARE.finditer(sentence))
    out: list[tuple[str, str, str]] = []
    for i, c in enumerate(comps):
        lo = comps[i - 1].end() if i else 0
        hi = comps[i + 1].start() if i + 1 < len(comps) else len(sentence)
        left = [(pos, n) for pos, n in ms if lo <= pos < c.start()]
        right = [n for pos, n in ms if c.end() <= pos < hi]
        if i + 1 < len(comps) and right:
            right = right[:-1]  # subject of the next comparison
        if not left:
            continue
        subj_pos = left[-1][0]
        subjects = [n for p, n in left if p == subj_pos]  # a group word names several models
        # 'random forest does not provide an advantage over ...' negates the comparison
        negated = bool(CMP_NEG.search(sentence[subj_pos : c.start()]))
        word = ("not " if negated else "") + c.group(0)
        out += [(s, o, word) for s in subjects for o in dict.fromkeys(right) if o not in subjects]
    return out


def compared_pairs(sentence: str, known: list[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b, _ in comparisons(sentence, known)]


CMP_NEG = re.compile(r"\b(?:not|never|neither)\b|n't\b|\bfail\w* to\b", re.I)
EQUAL_WORDS = re.compile(r"comparabl|on par|similar", re.I)
WORSE_WORDS = re.compile(r"worse|inferior|lower", re.I)


def comparison_contradiction(
    sentence: str, known: list[str], hyps: list[dict[str, Any]]
) -> str | None:
    """A comparison of a TESTED pair that disagrees with the test: 'random forest is comparable
    to logistic regression' when H2 (random forest > logistic regression) was supported."""
    for subj, obj, word in comparisons(sentence, known):
        for h in hyps:
            if {subj, obj} != {h.get("a"), h.get("b")} or h.get("result") not in (
                "supported",
                "not supported",
            ):
                continue
            supported = h.get("result") == "supported"
            negated = word.startswith("not ")
            if negated:
                if EQUAL_WORDS.search(word):
                    continue  # 'not comparable' = different: agrees with a supported test
                claimed = obj if WORSE_WORDS.search(word) else subj  # said NOT to be better
                if supported and claimed == h["a"]:
                    return (
                        f"says {claimed} is not better than the other model, but {h['id']} "
                        f"found {h['a']} > {h['b']} (supported)"
                    )
                continue
            if EQUAL_WORDS.search(word):
                if supported:
                    return (
                        f"calls {subj} and {obj} comparable, but {h['id']} found "
                        f"{h['a']} > {h['b']} (supported)"
                    )
                continue
            # who does the sentence say is better?
            better = obj if WORSE_WORDS.search(word) else subj
            if supported and better != h["a"]:
                return f"says {better} is better, but {h['id']} found {h['a']} > {h['b']}"
            if not supported and better == h["a"]:
                return f"says {h['a']} is better than {h['b']}, but {h['id']} was not supported"
    return None


_VERDICT_NEG = re.compile(
    r"\b(not|no|fail\w*|reject\w*|refut\w*|unsupported|neither|nor|contradict\w*|disprov\w*|"
    r"undermin\w*)\b",
    re.I,
)
_VERDICT_POS = re.compile(r"\b(support\w*|confirm\w*|validat\w*|corroborat\w*|hold\w*)\b", re.I)


def claimed_verdicts(sentence: str) -> list[tuple[str, bool]]:
    """(hypothesis id, claimed supported?) for bare mentions such as 'support H1 but not H2' or
    'H2 was not supported'. Rendered 'H1 (supported)' mentions are handled separately."""
    out = []
    for m in re.finditer(r"\b(H\d+)\b(?! \((?:supported|not supported)\))", sentence):
        before = sentence[max(0, m.start() - 50) : m.start()]
        clause = re.split(r"[,;]|\b(?:but|and|while|whereas)\b", before)[-1]
        after = re.split(r"[,;.]|\b(?:but|while|whereas)\b", sentence[m.end() : m.end() + 50])[0]
        if _VERDICT_NEG.search(clause):
            out.append((m.group(1), False))
        elif _VERDICT_POS.search(clause):
            out.append((m.group(1), True))
        elif re.match(r"\s+(?:was|is|were|has been)\s+(?:not|rejected|unsupported)", after, re.I):
            out.append((m.group(1), False))
        elif re.match(
            r"\s+(?:was|is|were|has been)\s+(?:also\s+)?(?:supported|confirmed)", after, re.I
        ) or re.match(r"\s+(?:also\s+)?holds", after, re.I):
            out.append((m.group(1), True))
    return out


RENDERED_HYP = re.compile(
    r"\b(?P<id>H\d+) \((?P<r>supported|not supported)\)|\((?P<id2>H\d+), (?P<r2>supported|not supported)\)"
)
_AFTER_NEG = re.compile(
    r"\s*\)?\s*(?:is|was|were|has been|remains?)\s+(?:also\s+)?"
    r"(?:not\b|rejected|unsupported|refuted|not confirmed)",
    re.I,
)
_AFTER_POS = re.compile(
    r"\s*\)?\s*(?:is|was|were|has been)\s+(?:also\s+)?(?:supported|confirmed)", re.I
)


def _hyp_contradiction(sentence: str) -> bool:
    """A rendered verdict ('H2 (supported)' or '(H2, supported)') that the surrounding words
    contradict: 'but not H2 (supported)', 'H2 (supported) is not supported'."""
    for m in RENDERED_HYP.finditer(sentence):
        result = m.group("r") or m.group("r2")
        before = sentence[max(0, m.start() - 40) : m.start()]
        after = sentence[m.end() : m.end() + 40]
        # look only at the clause right before the hypothesis
        clause = re.split(r"[,;]|\b(?:but|and|while|whereas)\b", before)[-1]
        negated = bool(NEGATION.search(clause))
        if result == "supported" and (negated or _AFTER_NEG.match(after)):
            return True
        if result == "not supported" and (
            (re.search(r"\bsupport\w*\b", clause, re.I) and not negated) or _AFTER_POS.match(after)
        ):
            return True
    return False


NO_DIFF = re.compile(
    r"\bno (?:statistically )?(?:significant |meaningful |clear )?"
    r"(?:difference|improvement|advantage|gain)s?\b",
    re.I,
)


def no_difference_contradiction(
    sentence: str, known: list[str], hyps: list[dict[str, Any]]
) -> str | None:
    """'no significant difference between random forest and logistic regression' when the test
    of that pair was supported."""
    m = NO_DIFF.search(sentence)
    if not m:
        return None
    named = {n for _, n in mentions(sentence, known)}
    for h in hyps:
        if h.get("result") == "supported" and {h.get("a"), h.get("b")} <= named:
            return f"claims no difference, but {h['id']} found {h['a']} > {h['b']} (supported)"
    return None


SIG_DIFF = re.compile(r"\bsignificant(?:ly)? (?:\w+ )?differen\w*\b", re.I)


def untested_significance(
    sentence: str, known: list[str], tested: set[frozenset[str]]
) -> str | None:
    """'significant differences between logistic regression and random forest, XGBoost and the
    MLP' claims a significance test for every pair named, but only some pairs were tested."""
    if not SIG_DIFF.search(sentence) or NO_DIFF.search(sentence):
        return None
    named = list(dict.fromkeys(n for _, n in mentions(sentence, known)))
    untested = [
        f"{a} vs {b}"
        for i, a in enumerate(named)
        for b in named[i + 1 :]
        if frozenset((a, b)) not in tested
    ]
    # only pairs that involve the first-named model are being contrasted in such sentences
    first = named[0] if named else None
    untested = [p for p in untested if first and p.startswith(first)]
    return ", ".join(untested) or None


CLASS_CLAIM = re.compile(
    r"\b(better|higher|stronger|superior|improved|outperform\w*|best)\b[^.]{0,60}\brecall\b"
    r"|\brecall\b[^.]{0,60}\b(better|higher|stronger|superior|improved|best)\b",
    re.I,
)


def class_claim_problem(sentence: str, analysis: dict[str, Any], known: list[str]) -> str | None:
    """'X has better recall for Worms and Backdoor': X must have the highest stored recall
    (first seed) on every class the sentence names."""
    if not CLASS_CLAIM.search(sentence):
        return None
    models = analysis.get("models", {})
    named = [n for _, n in mentions(sentence, known)]
    if not named:
        return None
    subject = named[0]
    per_class = {n: (models.get(n) or {}).get("per_class") or {} for n in models}
    classes = {c for pc in per_class.values() for c in pc}
    cited = [c for c in classes if re.search(rf"\b{re.escape(c)}\b", sentence, re.I)]
    if not cited and re.search(
        r"\b(rare|minority|small(?:er|est)?|infrequent|under-?represented)\b[^.]{0,30}"
        r"\b(class|classes|attacks?)\b",
        sentence,
        re.I,
    ):
        support = {
            c: (per_class.get(subject, {}).get(c) or {}).get("support") or 0
            for c in classes
            if c != "BENIGN"
        }
        cited = sorted(support, key=lambda c: support[c])[:3]  # the three smallest classes
    wrong = []
    for c in cited:
        mine = (per_class.get(subject, {}).get(c) or {}).get("recall")
        others = [
            (per_class[n].get(c) or {}).get("recall")
            for n in per_class
            if n != subject and per_class[n].get(c)
        ]
        if mine is not None and any(o is not None and o > mine for o in others):
            wrong.append(c)
    if not wrong:
        return None
    return (
        f"says {subject.replace('_', ' ')} has better recall on {', '.join(sorted(wrong))}, but another "
        "model's stored recall on that class is higher"
    )


FPR_WORDS = re.compile(r"\bfalse[- ]positive", re.I)
HIGH = re.compile(r"\b(high|higher|highest|elevated|excessive|many|more)\b", re.I)


def fpr_claim_problem(sentence: str, analysis: dict[str, Any], known: list[str]) -> str | None:
    """'LightGBM exhibits high false-positive rates' when LightGBM has the lowest stored rate,
    or a false-positive rate 'for Backdoor' (it is defined on BENIGN rows only)."""
    if not FPR_WORDS.search(sentence):
        return None
    models = analysis.get("models", {})
    classes = {c for m in models.values() for c in (m.get("per_class") or {}) if c != "BENIGN"}
    after = sentence[FPR_WORDS.search(sentence).end() :]  # type: ignore[union-attr]
    if any(re.search(rf"\b(?:for|on|of)\b[^.]{{0,40}}\b{re.escape(c)}\b", after) for c in classes):
        return "gives a false-positive rate per attack class, but it is defined on BENIGN rows"
    if not HIGH.search(sentence):
        return None
    named = [n for _, n in mentions(sentence, known)]
    rates: dict[str, float] = {}
    for n in models:
        r = ((models[n].get("aggregate") or {}).get("false_positive_rate") or {}).get("mean")
        if r is not None:
            rates[n] = float(r)
    if named and len(rates) > 1 and named[0] in rates and rates[named[0]] == min(rates.values()):
        return (
            f"says {named[0].replace('_', ' ')} has a high false-positive rate, but it has the "
            "lowest stored rate of all models"
        )
    return None


NOT_TESTED = re.compile(
    r"\b(?:not|never)\s+(?:been\s+|statistically\s+)?tested\b|\buntested\b|"
    r"\bno (?:significance|statistical) test\b|\bnot tested for significance\b",
    re.I,
)
# 'this study uniquely evaluates', 'the first study to': a novelty claim the platform cannot check
NOVELTY = re.compile(
    r"\b(uniquely|for the first time|the first (?:study|work|paper|to)|novel(?:ty)?|"
    r"no (?:prior|previous) (?:study|work)|unlike (?:all|any) (?:prior|previous))\b",
    re.I,
)
# overfitting needs a train/test comparison, which a test-set accuracy gap does not show
OVERFIT = re.compile(r"\boverfit\w*|\bover-fit\w*", re.I)


def tested_pair_called_untested(
    sentence: str, known: list[str], tested: set[frozenset[str]]
) -> str | None:
    if not NOT_TESTED.search(sentence):
        return None
    named = list(dict.fromkeys(n for _, n in mentions(sentence, known)))
    hit = [
        f"{a} vs {b}"
        for i, a in enumerate(named)
        for b in named[i + 1 :]
        if frozenset((a, b)) in tested
    ]
    if not hit or len(named) > 2 and "not tested" not in sentence.lower():
        return None
    return "says a difference was not tested, but this pair was tested: " + ", ".join(hit).replace(
        "_", " "
    )


def consistency_issues(markdown: str, analysis: dict[str, Any]) -> list[dict[str, str]]:
    known = list(analysis.get("models", {}))
    tested = {
        frozenset((h["a"], h["b"]))
        for h in analysis.get("hypotheses", [])
        if h.get("a") and h.get("b")
    }
    all_pairs_tested = all(frozenset(p) in tested for p in itertools.combinations(known, 2))
    overlap = (analysis.get("dataset") or {}).get("split_overlap_rows")
    gap_models = accuracy_gap_models(analysis)
    hyps = [h for h in analysis.get("hypotheses", []) if h.get("a") and h.get("b")]
    results = {h["id"]: h.get("result") for h in analysis.get("hypotheses", []) if h.get("id")}
    issues: list[dict[str, str]] = []

    def add(section: str, sentence: str, code: str, reason: str) -> None:
        issues.append({"section": section, "sentence": sentence, "code": code, "reason": reason})

    for name, body in sections(markdown.split("\n## Evidence", 1)[0]):
        if name in SKIP_SECTIONS:
            continue
        for sent, ctx in prose_items(body):
            if any(tag in sent for tag in UNRESOLVED):
                add(
                    name,
                    sent,
                    "UNRESOLVED_PLACEHOLDER",
                    "refers to a fact or number that does not exist in the stored evidence",
                )
                continue
            if TEMPLATE_LEAK.search(sent):
                add(
                    name,
                    sent,
                    "TEMPLATE_TEXT",
                    "contains instruction or template text copied literally (e.g. "
                    "'[low, high], seed N') instead of a real value",
                )
                continue
            if SAME_CI.search(sent):
                named = list(dict.fromkeys(n for _, n in mentions(f"{ctx} {sent}", known)))
                cis = {
                    n: (analysis["models"][n].get("macro_f1_ci") or {})
                    for n in named
                    if n in analysis.get("models", {})
                }
                spans = {(c.get("low"), c.get("high")) for c in cis.values() if c}
                if len(named) >= 2 and len(spans) > 1:
                    add(
                        name,
                        sent,
                        "CI_MISREAD",
                        "says two models have the same confidence interval, but the stored "
                        "intervals differ",
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
            called_untested = tested_pair_called_untested(sent, known, tested)
            if called_untested and name not in ("Results", "Methodology"):
                add(name, sent, "TESTED_CALLED_UNTESTED", called_untested)
                continue
            if name not in ("Related Work", "Results", "Methodology") and NOVELTY.search(sent):
                add(
                    name,
                    sent,
                    "NOVELTY_CLAIM",
                    "claims novelty ('uniquely', 'the first') that the literature search does "
                    "not establish",
                )
                continue
            if name not in ("Related Work", "Results", "Methodology") and OVERFIT.search(sent):
                add(
                    name,
                    sent,
                    "OVERFIT_CLAIM",
                    "infers overfitting, which needs training-versus-test scores; the stored "
                    "evidence has test scores only",
                )
                continue
            wrong_fpr = fpr_claim_problem(sent, analysis, known)
            if wrong_fpr:
                add(name, sent, "FPR_CLAIM_WRONG", wrong_fpr)
                continue
            wrong_class = class_claim_problem(sent, analysis, known)
            if wrong_class:
                add(name, sent, "CLASS_CLAIM_WRONG", wrong_class)
                continue
            if SCORE.search(sent) and not mentions(f"{ctx} {sent}", known):
                add(
                    name,
                    sent,
                    "SCORE_WITHOUT_MODEL",
                    "a measured score appears without the model it belongs to (a metric "
                    "placeholder was probably used in place of a metric name)",
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
            wrong_verdict = [
                hid
                for hid, claimed in claimed_verdicts(sent)
                if hid in results and claimed != (results[hid] == "supported")
            ]
            if wrong_verdict:
                add(
                    name,
                    sent,
                    "HYP_CONTRADICTION",
                    "states the wrong result for "
                    + ", ".join(f"{h} (stored result: {results[h]})" for h in wrong_verdict),
                )
                continue
            if INTERNAL.search(sent):
                add(
                    name,
                    sent,
                    "INTERNAL_REFERENCE",
                    "refers to the lab's internal notes or tooling, which readers cannot see",
                )
                continue
            if SCORE.search(sent) and not METRIC_WORD.search(sent):
                add(
                    name,
                    sent,
                    "SCORE_WITHOUT_METRIC",
                    "gives a measured value without saying which metric it is",
                )
                continue
            if FEW.search(sent) and any(_is_count(n) for n in BIG_COUNT.findall(sent)):
                add(
                    name,
                    sent,
                    "FACT_MISUSED",
                    "calls something 'few' but gives a count in the thousands (a recorded fact "
                    "was used for the wrong quantity)",
                )
                continue
            claimed = _claimed_primary(sent)
            if claimed and claimed != _study_primary(analysis):
                add(
                    name,
                    sent,
                    "WRONG_PRIMARY_METRIC",
                    f"calls {claimed.replace('_', ' ')} the primary metric, but the study's "
                    f"primary metric is {_study_primary(analysis).replace('_', ' ')}",
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
            mentioned = [n for _, n in mentions(f"{ctx} {sent}", known)]
            contradiction = comparison_contradiction(
                sent, known, hyps
            ) or no_difference_contradiction(sent, known, hyps)
            if contradiction:
                add(name, sent, "HYP_CONTRADICTION", contradiction)
                continue
            sig = untested_significance(sent, known, tested)
            if sig:
                add(
                    name,
                    sent,
                    "UNTESTED_COMPARISON",
                    "claims a significant difference for pairs that were never tested: " + sig,
                )
                continue
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
                continue
            if CI_ACROSS_SEEDS.search(sent):
                add(
                    name,
                    sent,
                    "CI_MISREAD",
                    "the 95% CI is a bootstrap over test rows for one seed, not variation "
                    "across seeds (that is the ± sd)",
                )
                continue
            if GAP_CLAIM.search(sent):
                wrong = [m for m in mentioned if m not in gap_models]
                if wrong or not mentioned:
                    who = ", ".join(gap_models) or "none of the models"
                    add(
                        name,
                        sent,
                        "GAP_MISATTRIBUTED",
                        "claims high accuracy hides weak balanced performance"
                        + (f" for {', '.join(wrong)}" if wrong else " without naming the model")
                        + f"; the stored results show this gap only for: {who}",
                    )
    return issues


GAP_CLAIM = re.compile(
    r"simplistic decision boundar\w*|"
    r"high (?:overall )?accuracy[^.]{0,80}\b(?:lower|low|poor|weak)\w* balanced|"
    r"accuracy[^.]{0,60}\bmislead\w*|"
    r"high (?:overall )?accuracy[^.]{0,80}\b(?:raise|rais\w+|indicat\w*) (?:concern|issue|problem)\w*",
    re.I,
)
# A CI read as variability/robustness across seeds, classes or runs ("mean over seeds" is fine)
CI_ACROSS_SEEDS = re.compile(
    r"\b(?:CI|confidence interval)s?\b.{0,160}?"
    r"\b(?:variab\w*|robust\w*|consisten\w*|stabl\w*|stabilit\w*|spread|fluctuat\w*)\b"
    r".{0,60}?\b(?:across|over|between) (?:\w+ ){0,2}(?:seeds|classes|runs|experiments)\b",
    re.I,
)
GAP_THRESHOLD = 0.05
INTERNAL = re.compile(
    r"\bstatistician(?:'s)? notes?\b|\breviewer feedback\b|\bmethod_facts\b|\bplaceholder\w*\b|"
    r"\bthe (?:writing|statistic\w*|reviewer) agent\b",
    re.I,
)
METRIC_WORD = re.compile(
    r"macro[- ]?F1|\bF1\b|accuracy|\bMCC\b|Matthews|detection rate|false[- ]positive|\bFPR\b|"
    r"precision|recall|\bAUC\b",
    re.I,
)
FEW = re.compile(r"\b(?:very |only )?(?:few|handful|scarce|rare instances)\b", re.I)
BIG_COUNT = re.compile(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b")


def _is_count(n: str) -> bool:
    v = int(n.replace(",", ""))
    return v >= 1000 and not (1900 <= v <= 2100 and "," not in n)  # not a year


def accuracy_gap_models(analysis: dict[str, Any]) -> list[str]:
    """Models whose accuracy exceeds balanced accuracy by more than GAP_THRESHOLD, i.e. where a
    'high accuracy hides weak minority-class performance' caveat actually applies."""
    out = []
    for name, m in analysis.get("models", {}).items():
        agg = m.get("aggregate", {})
        acc = (agg.get("accuracy") or {}).get("mean")
        bal = (agg.get("balanced_accuracy") or {}).get("mean")
        if acc is not None and bal is not None and acc - bal > GAP_THRESHOLD:
            out.append(name)
    return out


_METRIC_WORDS = {
    "balanced_accuracy": r"balanced accuracy",
    "macro_f1": r"macro[- ]?f1|macro[- ]averaged f1",
    "mcc": r"mcc|matthews",
    "accuracy": r"(?<!balanced )accuracy",
}
_PRIMARY = re.compile(
    r"(?i)\b(?P<m>balanced accuracy|macro[- ]?f1|macro[- ]averaged f1|mcc|matthews[\w ]*?|accuracy)"
    r"\s+(?:as|is|was|being)\s+(?:the|a|our)\s+(?:sole\s+|main\s+)?primary\s+(?:evaluation\s+)?metric"
)


def _claimed_primary(sentence: str) -> str | None:
    m = _PRIMARY.search(sentence)
    if not m:
        return None
    word = m.group("m").lower()
    for key, pat in _METRIC_WORDS.items():
        if re.fullmatch(pat, word) or (key == "mcc" and word.startswith("matthews")):
            return key
    return None


def _study_primary(analysis: dict[str, Any]) -> str:
    for h in analysis.get("hypotheses", []):
        if h.get("metric"):
            return str(h["metric"])
    return "macro_f1"
