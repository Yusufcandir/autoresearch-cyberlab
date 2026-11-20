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
# Regular expressions (case-insensitive) naming each template in plain words.
MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "logistic_regression": (r"logistic[ _]regression",),
    "random_forest": (r"random[ _]forests?",),
    "xgboost": (r"xgboost", r"gradient[- ]boost(?:ing|ed)(?! machine)", r"\bgbdt\b"),
    "lightgbm": (r"light ?gbm",),
    "mlp": (
        r"(?<!resnet[- ])\bmlp\b",
        r"multi-?layer perceptron",
        r"(?<!graph )(?<!deep )\bneural network\b",
    ),
    "resnet_mlp": (r"resnet[- _]?mlp", r"\bresnet\b"),
    "ft_transformer": (r"ft[- _]?transformer", r"feature[- ]tokeni[sz]er transformer"),
    "cnn1d": (r"\b1d[- ]?cnn\b", r"\bcnn1d\b", r"one-dimensional (?:cnn|convolutional)"),
    "tabnet": (r"\btabnet\b",),
}
BASELINES = {"logistic_regression", "random_forest"}
# Things no experiment on this platform measures.
UNTESTABLE = re.compile(
    r"real[- ]time|scalab|latency|runtime|run time|inference time|training time|deploy|"
    r"efficien|energy|throughput|memory use|explainab|"
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
            found += [(m.start(), name) for m in re.finditer(a, text, re.I)]
    return sorted(set(found))


def models_named(goal: str) -> list[str]:
    """Templates the goal names, in order of first mention."""
    out: list[str] = []
    for _pos, name in _models_in(goal):
        if name not in out:
            out.append(name)
    return out


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
    "lightgbm": "LightGBM",
    "resnet_mlp": "the ResNet-MLP",
    "ft_transformer": "the FT-Transformer",
    "cnn1d": "the 1D-CNN",
    "tabnet": "TabNet",
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


_NF = r"NF[- ](?:UNSW[- ]NB15|CSE[- ]CIC[- ]IDS[- ]?2018|ToN[- ]IoT|BoT[- ]IoT)[- ]v2"
_CROSS = re.compile(rf"\btest(?:ed|ing)?\s+on\s+(?:the\s+)?(?P<test>{_NF})", re.I)
_GROUPS = re.compile(
    r"\btest(?:ed|ing)?\s+on\s+(?:the\s+)?(?:capture\s+)?scenarios?\s+"
    r"(?P<g>\d{1,2}(?:\s*(?:,|and|&)\s*\d{1,2})*)",
    re.I,
)


def split_from_goal(goal: str) -> dict[str, Any] | None:
    """The split the goal asks for: cross-dataset ('train on NF-UNSW-NB15-v2 and test on
    NF-ToN-IoT-v2'), held-out capture scenarios ('test on scenarios 10 and 11', CTU-13), the
    official partition (UNSW-NB15), or a by-day split with the test days it names."""
    m = _CROSS.search(goal)
    if m:
        return {"strategy": "cross_dataset", "test_days": [m.group("test")]}
    m = _GROUPS.search(goal)
    if m:
        return {"strategy": "by_group", "test_days": re.findall(r"\d{1,2}", m.group("g"))}
    if _OFFICIAL.search(goal):
        return {"strategy": "official"}
    if not _TEMPORAL.search(goal) and not re.search(r"\btest(?:ed|ing)? on (?:\w+day)", goal, re.I):
        return None
    m = re.search(r"\btest(?:ed|ing)?\b(.{0,80})", goal, re.I | re.S)
    days = [d for d in DAYS if m and re.search(d, m.group(1), re.I)]
    if not days:
        return None
    return {"strategy": "by_day", "test_days": days}


_TUNE = re.compile(
    r"\b(?:tun(?:e|ed|ing)|hyper-?parameter (?:search|optimi[sz]ation|tuning)|optuna)\b", re.I
)
_TRIALS = re.compile(r"\b(\d{1,3})\s+(?:tuning\s+)?trials?\b", re.I)
_SEEDS = re.compile(r"\b(\d{1,2}|three|five|ten)\s+(?:random\s+)?seeds?\b", re.I)
_SUPPORT = re.compile(
    r"(?:fewer than|less than|under|minimum of|at least)\s+(\d{1,4})\s+test\s+rows", re.I
)
_WORDS = {"three": 3, "five": 5, "ten": 10}


def tuning_from_text(text: str) -> int | None:
    """Number of tuning trials asked for ('tune with 50 trials'), 30 if tuning is asked for
    without a number, None if the text does not ask for tuning ('no tuning' -> None)."""
    if not _TUNE.search(text) or re.search(
        r"\b(?:no|without)\s+(?:hyper-?parameter\s+)?tuning\b", text, re.I
    ):
        return None
    m = _TRIALS.search(text)
    return max(1, min(200, int(m.group(1)))) if m else 30


def seeds_from_text(text: str) -> int | None:
    m = _SEEDS.search(text)
    if not m:
        return None
    v = m.group(1).lower()
    return _WORDS.get(v) or int(v)


def min_support_from_text(text: str) -> int | None:
    m = _SUPPORT.search(text)
    return int(m.group(1)) if m else None
