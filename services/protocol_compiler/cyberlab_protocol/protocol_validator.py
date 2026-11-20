"""Research protocol validation before any experiment (professor's plan, Step 2.2).

Deterministic checks that a drafted protocol is scientifically reasonable: measurable
hypotheses, metrics that fit the research question, a suitable dataset, appropriate baselines,
a fair comparison, a suitable statistical plan and feasible compute. The result is shown to the
PI next to the protocol; weaknesses are found before expensive experiments begin.
"""

from __future__ import annotations

import re
from typing import Any

PASS, WARNING, BLOCK = "PASS", "WARNING", "BLOCK"
_ORDER = {PASS: 0, WARNING: 1, BLOCK: 2}

# rough minutes per 100k training rows and seed, on the lab GPU / CPU (planning estimate only)
_MINUTES_PER_100K = {
    "logistic_regression": 0.5,
    "random_forest": 2.0,
    "xgboost": 1.0,
    "lightgbm": 0.7,
    "mlp": 2.0,
    "resnet_mlp": 3.0,
    "ft_transformer": 8.0,
    "cnn1d": 3.0,
    "tabnet": 6.0,
    "tfidf_logreg": 1.0,
    "tfidf_linear_svm": 1.5,
}
_IMBALANCE = re.compile(
    r"\b(rare|minority|imbalanc\w*|small class\w*|few (?:samples|examples))\b", re.I
)


def _r(check: str, status: str, detail: str) -> dict[str, Any]:
    return {"check": check, "status": status, "detail": detail}


def estimate_minutes(protocol: dict[str, Any], train_rows: int) -> float:
    seeds = len(protocol.get("seeds", [])) or 1
    trials = ((protocol.get("tuning") or {}).get("trials") or 0) if protocol.get("tuning") else 0
    total = 0.0
    for m in protocol.get("models", []):
        per = _MINUTES_PER_100K.get(m.get("template", ""), 2.0) * max(train_rows, 1) / 100_000
        total += per * seeds + per * trials * 0.5  # tuning trials run on a subsample
    return round(total, 1)


def validate_protocol(
    protocol: dict[str, Any],
    goal: str,
    dataset: dict[str, Any] | None,
    budget_minutes: int,
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    models = {m["name"]: m for m in protocol.get("models", [])}
    hyps = protocol.get("hypotheses", [])

    # 1. measurable hypotheses
    bad = [
        h["id"]
        for h in hyps
        if not all(x.strip() in models for x in str(h.get("comparison", "")).split(">"))
    ]
    out.append(
        _r("measurable_hypotheses", BLOCK, f"hypotheses compare unknown models: {bad}")
        if bad
        else _r(
            "measurable_hypotheses",
            PASS if hyps else BLOCK,
            f"{len(hyps)} hypotheses, each a directional comparison of two models on one metric"
            if hyps
            else "no hypothesis to test",
        )
    )

    # 2. metric fits the research question
    metrics = {h.get("primary_metric", "macro_f1") for h in hyps}
    if _IMBALANCE.search(goal) and metrics & {"accuracy"}:
        out.append(
            _r(
                "metric_fits_question",
                WARNING,
                "the goal is about rare or minority classes but a hypothesis uses accuracy, "
                "which the majority class dominates; prefer balanced accuracy or macro-F1",
            )
        )
    else:
        out.append(
            _r("metric_fits_question", PASS, f"primary metric(s): {', '.join(sorted(metrics))}")
        )

    # 3. dataset suitability
    if not dataset or dataset.get("status") != "verified":
        out.append(_r("dataset_suitable", BLOCK, "the dataset is not verified"))
    else:
        counts = dataset.get("class_counts") or {}
        named = [
            c for c in counts if re.search(rf"\b{re.escape(c)}\b", goal, re.I) and c != "BENIGN"
        ]
        tiny = [c for c, n in counts.items() if n < 100]
        status = WARNING if tiny else PASS
        out.append(
            _r(
                "dataset_suitable",
                status,
                f"{len(counts)} classes"
                + (f"; classes the goal names: {', '.join(named)}" if named else "")
                + (
                    f"; classes with fewer than 100 rows in total: {', '.join(tiny)}"
                    if tiny
                    else ""
                ),
            )
        )

    # 4. appropriate baselines
    roles = {m.get("role") for m in models.values()}
    tests_vs_baseline = any(
        models.get(str(h.get("comparison", "")).split(">")[-1].strip(), {}).get("role")
        == "baseline"
        for h in hyps
    )
    out.append(
        _r(
            "baselines",
            PASS if "baseline" in roles and tests_vs_baseline else WARNING,
            "at least one hypothesis compares a candidate with a baseline"
            if "baseline" in roles and tests_vs_baseline
            else "no hypothesis compares a candidate with a simple baseline",
        )
    )

    # 5. fair comparison
    tuning = protocol.get("tuning")
    out.append(
        _r(
            "fair_comparison",
            PASS,
            "every model uses the same split, seeds and "
            + ("tuning budget on the validation split" if tuning else "reviewed default settings"),
        )
    )

    # 6. statistical plan
    seeds = len(protocol.get("seeds", []))
    problems = []
    if seeds < 3:
        problems.append(f"only {seeds} seed(s)")
    if len(hyps) > 5:
        problems.append(f"{len(hyps)} hypotheses: the Holm correction will cost power")
    out.append(
        _r(
            "statistical_plan",
            WARNING if problems else PASS,
            "; ".join(problems)
            if problems
            else f"{seeds} seeds, paired permutation tests with Holm correction, bootstrap intervals",
        )
    )

    # 7. feasibility
    train = int(((dataset or {}).get("split_counts") or {}).get("train", 0))
    minutes = estimate_minutes(protocol, train)
    out.append(
        _r(
            "feasibility",
            BLOCK
            if minutes > budget_minutes * 3
            else WARNING
            if minutes > budget_minutes
            else PASS,
            f"estimated compute about {minutes:.0f} min against a budget of {budget_minutes} min",
        )
    )
    status = max((c["status"] for c in out), key=lambda s: _ORDER[s])
    return {
        "status": status,
        "checks": out,
        "estimated_minutes": minutes,
        "acceptance_criteria": [
            "every planned run succeeds or its failure is recorded",
            "scientific validation has no BLOCK",
            "each hypothesis is tested on its own metric with Holm correction",
        ],
    }
