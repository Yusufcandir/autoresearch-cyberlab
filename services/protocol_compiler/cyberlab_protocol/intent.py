"""What the researcher actually asked for, read deterministically from the study goal.

The protocol agent (an LLM) proposes a protocol; this module makes sure the protocol tests the
researcher's own hypotheses with the researcher's own primary metric, and drops research
questions the platform cannot answer with its experiments (runtime, scalability, ...).
"""

from __future__ import annotations

import re
from typing import Any

METRICS: list[tuple[str, str]] = [
    (r"balanced[- ]accuracy", "balanced_accuracy"),
    (r"macro[- ]?f1|macro[- ]averaged f1", "macro_f1"),
    (r"\bmcc\b|matthews", "mcc"),
    (r"(?<!balanced )\baccuracy\b", "accuracy"),
]
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "logistic_regression": ("logistic regression", "logistic_regression"),
    "random_forest": ("random forest", "random_forest"),
    "xgboost": ("xgboost", "gradient boosting", "gradient-boosting", "gbdt"),
    "mlp": ("mlp", "neural network", "multi-layer perceptron", "multilayer perceptron"),
}
# Things no experiment on this platform measures.
UNTESTABLE = re.compile(
    r"real[- ]time|scalab|latency|runtime|run time|inference time|training time|deploy|"
    r"efficien|energy|throughput|memory use|other datasets?|cross[- ]dataset|explainab|"
    r"interpretab|adversarial|robustness to attack",
    re.I,
)


def _metric(text: str) -> str | None:
    for pat, name in METRICS:
        if re.search(pat, text, re.I):
            return name
    return None


def primary_metric(goal: str, constraints: list[str]) -> str | None:
    text = " ".join([goal, *constraints])
    m = re.search(
        r"primary metric\s*(?:is|:|=)?\s*([a-z\- ]{3,30})|"
        r"(?:use|using)\s+([a-z\- ]{3,30}?)\s+as (?:the )?primary metric|"
        r"([a-z\- ]{3,30}?)\s+as (?:the )?primary metric",
        text,
        re.I,
    )
    if not m:
        return None
    return _metric(next(g for g in m.groups() if g))


def _models_in(text: str) -> list[tuple[int, str]]:
    found = []
    for name, aliases in MODEL_ALIASES.items():
        for a in aliases:
            found += [(m.start(), name) for m in re.finditer(re.escape(a), text, re.I)]
    return sorted(set(found))


def stated_hypotheses(goal: str, default_metric: str) -> list[tuple[str, str, str]]:
    """'whether random forest achieves higher balanced accuracy than logistic regression'
    -> [('random_forest', 'logistic_regression', 'balanced_accuracy')]."""
    out: list[tuple[str, str, str]] = []
    for clause in re.split(r"\bwhether\b|;|\. ", goal, flags=re.I):
        m = re.search(r"\b(?:higher|better|greater|more)\b(.{0,40}?)\bthan\b", clause, re.I)
        if not m:
            continue
        left = [n for pos, n in _models_in(clause) if pos < m.start()]
        right = [n for pos, n in _models_in(clause) if pos > m.end() - 5]
        if not left or not right or left[-1] == right[0]:
            continue
        metric = _metric(m.group(1)) or default_metric
        pair = (left[-1], right[0], metric)
        if pair not in out:
            out.append(pair)
    return out


def research_intent(goal: str, constraints: list[str]) -> dict[str, Any]:
    pm = primary_metric(goal, constraints)
    return {
        "primary_metric": pm,
        "hypotheses": stated_hypotheses(goal, pm or "macro_f1"),
    }


def testable_questions(questions: list[str]) -> tuple[list[str], list[str]]:
    keep = [q for q in questions if not UNTESTABLE.search(q)]
    dropped = [q for q in questions if UNTESTABLE.search(q)]
    return keep, dropped


PRETTY = {
    "logistic_regression": "logistic regression",
    "random_forest": "random forest",
    "xgboost": "XGBoost",
    "mlp": "the MLP",
    "balanced_accuracy": "balanced accuracy",
    "macro_f1": "macro-F1",
    "mcc": "MCC",
    "accuracy": "accuracy",
}


def hypothesis_statement(a: str, b: str, metric: str) -> str:
    text = (
        f"{PRETTY.get(a, a)} achieves higher {PRETTY.get(metric, metric)} than {PRETTY.get(b, b)}."
    )
    return text[:1].upper() + text[1:]


DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_TEMPORAL = re.compile(
    r"by[- ]day|day[- ]based|temporal(?:ly)? split|unseen days?|days? (?:it|they|the models?) "
    r"never saw|train(?:ed|ing)? on (?:earlier|the first)",
    re.I,
)


_OFFICIAL = re.compile(
    r"official (?:train(?:ing)?[/ -]?(?:and )?test(?:ing)? )?(?:split|partition|sets?)|"
    r"(?:provided|predefined|standard) train(?:ing)?[/ -]?(?:and )?test(?:ing)? (?:split|sets?)",
    re.I,
)


def split_from_goal(goal: str) -> dict[str, Any] | None:
    """The official partition when the goal asks for it (UNSW-NB15), or a by-day split with
    the test days it names ('train on Monday to Wednesday and test on Thursday and Friday')."""
    if _OFFICIAL.search(goal):
        return {"strategy": "official"}
    if not _TEMPORAL.search(goal) and not re.search(r"\btest(?:ed|ing)? on (?:\w+day)", goal, re.I):
        return None
    m = re.search(r"\btest(?:ed|ing)?\b(.{0,80})", goal, re.I | re.S)
    days = [d for d in DAYS if m and re.search(d, m.group(1), re.I)]
    if not days:
        return None
    return {"strategy": "by_day", "test_days": days}
