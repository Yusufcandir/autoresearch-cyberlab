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
    "lightgbm": "LightGBM",
    "resnet_mlp": "the ResNet-MLP",
    "ft_transformer": "the FT-Transformer",
    "cnn1d": "the 1D-CNN",
    "tabnet": "TabNet",
    "tfidf_logreg": "TF-IDF logistic regression",
    "tfidf_linear_svm": "TF-IDF linear SVM",
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


def hypothesis_sentences(analysis: dict[str, Any], compact: bool = False) -> list[str]:
    """One sentence per tested hypothesis. compact=True (Abstract, Conclusion) keeps the
    verdict and the Holm-adjusted p; the means and the raw test are in Results."""
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
        tst = h.get("test") or {}
        ptxt = _p(tst.get("p_value") if tst else (h.get("mcnemar") or {}).get("p_value"))
        if tst and ptxt and "p_holm" in h:
            test = (
                f" (one-sided paired permutation test on {_metric(metric)}, {ptxt}; "
                f"Holm-adjusted {_p(h['p_holm'])})"
            )
        elif ptxt and "p_holm" in h:
            test = (
                f" (exact McNemar test, largest over seeds {ptxt}; Holm-adjusted {_p(h['p_holm'])})"
            )
        else:
            test = f" (exact McNemar test, {ptxt})" if ptxt else ""
        if compact:
            tail = (
                f" (Holm-adjusted {_p(h['p_holm'])})"
                if "p_holm" in h
                else (f" ({ptxt})" if ptxt else "")
            )
            out.append(
                f"{h['id']} ({pretty(h['a'])} scores higher than {pretty(h['b'])} on "
                f"{_metric(metric)}) was {h['result']}{tail}."
            )
            continue
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
    out += hypothesis_sentences(analysis, compact=True)
    return " ".join(out)


def hypothesis_by_dataset(analysis: dict[str, Any]) -> dict[str, tuple[list[str], list[str]]]:
    """For each hypothesis: (datasets where it was supported, datasets where it was not),
    the main dataset included."""
    b = analysis.get("benchmark") or {}
    if not b.get("datasets"):
        return {}
    main = b["datasets"][0]
    out: dict[str, tuple[list[str], list[str]]] = {}
    for h in _tested_hyps(analysis):
        out[h["id"]] = ([main], []) if h.get("result") == "supported" else ([], [main])
    for ds in b["datasets"][1:]:
        for h in (b["per_dataset"].get(ds) or {}).get("hypotheses", []):
            if h.get("id") in out and "result" in h:
                out[h["id"]][0 if h["result"] == "supported" else 1].append(ds)
    return out


def benchmark_findings_text(analysis: dict[str, Any]) -> str:
    """One or two sentences on the cross-dataset result, without numbers (they are in Results)."""
    b = analysis.get("benchmark") or {}
    if len(b.get("datasets", [])) < 2:
        return ""
    n = _num(len(b["datasets"]))
    out = []
    rank = b.get("ranking") or {}
    if rank.get("applicable"):
        order = sorted(rank["average_rank"], key=rank["average_rank"].__getitem__)
        sig = rank["p_value"] < 0.05
        out.append(
            f"Across the {n} datasets, {pretty(order[0])} had the best average rank, and the "
            f"Friedman test {'found' if sig else 'did not find'} a significant difference "
            "between the models' ranks."
        )
    parts = []
    for hid, (yes, no) in hypothesis_by_dataset(analysis).items():
        if not no:
            parts.append(f"{hid} held on all {n} datasets")
        elif not yes:
            parts.append(f"{hid} held on none of them")
        else:
            parts.append(
                f"{hid} held on {_num(len(yes))} of the {n} datasets (not on "
                f"{_and([dataset_label(d) for d in no])})"
            )
    if parts:
        out.append("Repeated on every dataset, " + "; ".join(parts) + ".")
    return " ".join(out)


def benchmark_discussion_text(analysis: dict[str, Any]) -> str:
    """Which model led on each dataset, and where a hypothesis did not carry over (no numbers)."""
    b = analysis.get("benchmark") or {}
    scores = b.get("scores") or {}
    if len(scores) < 2:
        return ""
    lead = {ds: max(s, key=s.__getitem__) for ds, s in scores.items() if s}
    out = [
        f"The model with the highest mean {_metric(b.get('metric', 'macro_f1'))} was "
        + _and([f"{pretty(m)} on {dataset_label(ds)}" for ds, m in lead.items()])
        + "."
    ]
    for hid, (yes, no) in hypothesis_by_dataset(analysis).items():
        if yes and no:
            out.append(
                f"{hid} did not hold on {_and([dataset_label(d) for d in no])}, so the result "
                "on one NetFlow dataset does not carry over to every other."
            )
    t = (analysis.get("transfer") or {}).get("models") or {}
    gaps = [
        v["in_domain"] - v["cross_domain"]
        for v in t.values()
        if v.get("in_domain") is not None and v.get("cross_domain") is not None
    ]
    if gaps and min(gaps) > 0:
        out.append(
            "Every model scored lower on datasets it was not trained on than on its own, so "
            "results measured within one dataset overstate performance on a new network."
        )
    return " ".join(out)


def results_text(analysis: dict[str, Any]) -> str:
    """The complete Results prose, from the stored aggregates and tests.

    Written as connected paragraphs that point to the tables and figures (resolved to their
    numbers after the build): the ranking on the primary metric with the leader's interval,
    then the leader's other metrics, the smallest classes, the hypothesis tests and, for a
    benchmark, the comparison across datasets. Every model appears with its score."""
    paras = []
    metric = primary_metric(analysis)
    models = analysis.get("models", {})
    label = "MCC" if metric == "mcc" else _metric(metric)
    scored = [n for n in models if "mean" in (models[n].get("aggregate", {}).get(metric) or {})]
    order = sorted(scored, key=lambda n: -models[n]["aggregate"][metric]["mean"])
    b = analysis.get("benchmark") or {}
    where = f" on {dataset_label(b['datasets'][0])}" if b.get("datasets") else ""
    if order:

        def score(n: str, with_ci: bool = False) -> str:
            v = models[n]["aggregate"][metric]
            txt = f"{v['mean']:.4f} ± {v.get('sd', 0):.4f}"
            ci = models[n].get(f"{metric}_ci") or {}
            if with_ci and "low" in ci and "high" in ci:
                ci_label = models[n].get("ci_label") or (
                    f"seed {models[n]['ci_seed']}" if models[n].get("ci_seed") is not None else ""
                )
                txt += f", 95% bootstrap CI [{ci['low']:.4f}, {ci['high']:.4f}]" + (
                    f", {ci_label}" if ci_label else ""
                )
            return txt

        lead = order[0]
        sents = [
            "{{ref:table_main}} lists every model's scores on the test split"
            + where
            + " (means ± standard deviations over the model seeds), and {{ref:figure_ci}} shows "
            f"the {label} of each model with its 95% bootstrap confidence interval.",
            f"{_cap(pretty(lead))} had the highest mean {label} ({score(lead, True)})"
            + (
                ", followed by " + _and([f"{pretty(n)} ({score(n)})" for n in order[1:]])
                if len(order) > 1
                else ""
            )
            + ".",
        ]
        others = []
        for key in ("macro_f1", "balanced_accuracy", "accuracy", "mcc"):
            v = models[lead]["aggregate"].get(key) or {}
            if key != metric and "mean" in v:
                others.append(f"{'MCC' if key == 'mcc' else _metric(key)} {v['mean']:.4f}")
        if others:
            sents.append(f"{_cap(pretty(lead))} also reached " + _and(others) + ".")
        paras.append(" ".join(sents))
    minority = minority_recall_text(analysis)
    if minority:
        paras.append(
            "{{ref:figure_recall}} shows the recall of every model on every class. " + minority
        )
    if imp := importance_text(analysis):
        paras.append(imp)
    hyp = hypothesis_sentences(analysis)
    scope = untested_pairs_sentence(analysis)
    if hyp or scope:
        paras.append(
            " ".join(
                ["The hypothesis tests are summarised in {{ref:table_hypotheses}}."]
                + hyp
                + ([scope] if scope else [])
            )
        )
    for extra in (efficiency_text(analysis), ablation_text(analysis)):
        if extra:
            paras.append(extra)
    bench = benchmark_text(analysis)
    if bench:
        # the tables and figures are cited in their numbered order, so each one can be
        # placed right after this paragraph in reading order
        paras.append(
            "{{ref:table_benchmark}} and {{ref:figure_benchmark}} give the scores on each "
            "dataset, and {{ref:figure_rank}} the average ranks. " + bench
        )
    if xfer := transfer_text(analysis):
        paras.append(xfer)
    best = _best(analysis)
    if best:
        support = {
            c: v.get("support", 0)
            for c, v in (analysis["models"][best].get("per_class") or {}).items()
        }
        if support:
            small = min(support, key=support.__getitem__)
            paras.append(
                f"Per-class results for {pretty(best)} are given in {{{{ref:table_per_class}}}} "
                "and its confusion matrix in {{ref:figure_confusion}}; the smallest class, "
                f"{small}, has only {support[small]:,} test rows."
            )
    return "\n\n".join(paras)


def importance_text(analysis: dict[str, Any]) -> str:
    """Which features each model relies on (permutation importance), from the stored runs."""
    models = analysis.get("models", {})
    have = [n for n in models if models[n].get("permutation_importance")]
    if not have:
        return ""
    metric = _metric(primary_metric(analysis))
    best = _best(analysis)
    lead: str = best if best is not None and best in have else have[0]
    top = list(models[lead]["permutation_importance"].items())[:3]
    sents = [
        "{{ref:figure_importance}} shows how much each model relies on each feature "
        f"(permutation importance: the drop in {metric} when the feature's values are shuffled "
        "on the test rows, averaged over seeds)."
    ]
    if top:
        sents.append(
            f"For {pretty(lead)}, shuffling "
            + _and([f"{f} lowered {metric} by {d:.3f}" for f, d in top])
            + "."
        )
    if len(have) > 1:
        shared = set(list(models[have[0]]["permutation_importance"])[:5])
        for n in have[1:]:
            shared &= set(list(models[n]["permutation_importance"])[:5])
        sents.append(
            f"Among the five most important features of every model were {_and(sorted(shared))}."
            if shared
            else "No feature was among the five most important of every model."
        )
    shap_lead = next(
        (n for n in sorted(models, key=lambda n: n != best) if models[n].get("shap_importance")),
        None,
    )
    if shap_lead:
        top_s = list(models[shap_lead]["shap_importance"])[:3]
        sents.append(
            f"The SHAP values of {pretty(shap_lead)} ({{{{ref:figure_shap}}}}) agree on what "
            f"drives its decisions: its most influential features were {_and(top_s)}."
            if set(top_s) & set(list(models[shap_lead].get("permutation_importance") or {})[:5])
            else f"The SHAP values of {pretty(shap_lead)} ({{{{ref:figure_shap}}}}) rank "
            f"{_and(top_s)} highest."
        )
        per = models[shap_lead].get("shap_per_class") or {}
        if per:
            sents.append(
                "By class, "
                + "; ".join(f"{c} was driven most by {next(iter(v))}" for c, v in per.items() if v)
                + "."
            )
    return " ".join(sents)


def efficiency_text(analysis: dict[str, Any]) -> str:
    models = analysis.get("models", {})
    eff = {n: m["efficiency"] for n, m in models.items() if m.get("efficiency")}
    if len(eff) < 2:
        return ""
    fast = min(eff, key=lambda n: eff[n]["train_seconds"])
    slow = max(eff, key=lambda n: eff[n]["train_seconds"])
    inf = [e["inference_ms_per_1000"] for e in eff.values() if e.get("inference_ms_per_1000")]
    out = (
        "{{ref:table_efficiency}} gives the time of the final training run (excluding the "
        f"settings search) and the inference latency. {_cap(pretty(fast))} trained fastest "
        f"({eff[fast]['train_seconds']:.1f} s) and {pretty(slow)} slowest "
        f"({eff[slow]['train_seconds']:.1f} s)"
    )
    if inf:
        out += f"; inference took {min(inf):.2f} to {max(inf):.2f} ms per 1,000 test rows"
    return out + "."


def ablation_text(analysis: dict[str, Any]) -> str:
    abl = analysis.get("ablations") or []
    if not abl:
        return ""
    sents = [
        "{{ref:table_ablation}} compares each full model with the same model trained without selected features."
    ]
    for x in abl:
        eff = x.get("effect") or {}
        ci = (
            f", 95% CI [{eff['low']:+.4f}, {eff['high']:+.4f}]"
            if {"low", "high"} <= set(eff)
            else ""
        )
        sents.append(
            f"Removing {_and(x['dropped']) or 'the listed features'} from {pretty(x['base'])} "
            f"changed {_metric(x['metric'])} from {x['base_mean']:.4f} to "
            f"{x['ablated_mean']:.4f} (difference {x['base_mean'] - x['ablated_mean']:+.4f}{ci}; "
            f"Holm-adjusted {_p(x['p_holm'])}), "
            + ("a significant loss." if x.get("significant") else "not a significant loss.")
        )
    return " ".join(sents)


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


FPR_NOTABLE = 0.05


def operational_text(analysis: dict[str, Any]) -> str:
    """Facts a reader of an IDS paper needs that aggregate scores hide: a high false-positive
    rate, and classes that no model detected at all."""
    out = []
    models = analysis.get("models", {})
    fprs: dict[str, float] = {}
    for n, m in models.items():
        v = (m.get("aggregate", {}).get("false_positive_rate") or {}).get("mean")
        if isinstance(v, (int, float)):
            fprs[n] = float(v)
    if fprs and min(fprs.values()) >= FPR_NOTABLE:
        # every value is named with its model, so the consistency checks can trace it
        parts = _and(
            [f"{v:.4f} for {pretty(n)}" for n, v in sorted(fprs.items(), key=lambda x: x[1])]
        )
        out.append(
            "The false-positive rate (the share of BENIGN test rows flagged as an attack) was "
            f"{parts}; at these rates the models would need further work before operational use."
        )
    per_class = [m.get("per_class") or {} for m in models.values()]
    if per_class:
        classes = set.intersection(*(set(pc) for pc in per_class))
        missed = sorted(
            c
            for c in classes
            if c != "BENIGN"
            and all((pc.get(c) or {}).get("recall", 1) == 0 for pc in per_class)
            and all((pc.get(c) or {}).get("support", 0) > 0 for pc in per_class)
        )
        if missed:
            out.append(
                f"No model detected any {_and(missed)} test row (recall 0 for every model, "
                "first seed)."
            )
    return " ".join(out)


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


def _setting(v: Any) -> str:
    """Model settings in text: tiny floats in scientific notation so they never look like a
    measured score (e.g. alpha=1e-04, not 0.0001)."""
    if isinstance(v, float) and 0 < abs(v) < 0.01:
        return f"{v:.0e}"
    if isinstance(v, float):
        # three significant digits (0.0443, not 0.04429649570464297), and never four decimals,
        # which would read as a measured score
        txt = f"{v:.3g}"
        return f"{v:.2e}" if re.fullmatch(r"\d\.\d{4}", txt) else txt
    return str(v)


def dataset_label(dataset_id: str) -> str:
    from cyberlab_data import get_adapter

    return str(get_adapter(dataset_id).display)


def benchmark_method_text(analysis: dict[str, Any]) -> str:
    """How the benchmark datasets were used (Methodology)."""
    b = analysis.get("benchmark") or {}
    others = [d for d in b.get("datasets", [])[1:]]
    if not others:
        return ""
    return (
        f"To test whether the findings hold beyond one dataset, the same models, seeds, settings "
        f"search and budget were also run on {_and([dataset_label(d) for d in others])}. Each "
        "dataset was cleaned, de-duplicated and split on its own (its official partition when it "
        "has one, otherwise a stratified split), and no rows are shared between datasets. The "
        "stated hypotheses were repeated on every additional dataset as secondary tests, with "
        "Holm correction across those tests. Models were ranked on each dataset by their mean "
        f"{_metric(b.get('metric', 'macro_f1'))}, and the ranks were compared across datasets with "
        "the Friedman test and the Nemenyi critical difference (alpha 0.05)."
    )


def transfer_method_text(analysis: dict[str, Any]) -> str:
    t = analysis.get("transfer") or {}
    if len(t.get("datasets", [])) < 2:
        return ""
    return (
        "To evaluate generalization, every model was also trained on each dataset and tested "
        "on the test split of every dataset, using the features all datasets share. Because "
        "the attack families differ between the datasets, labels were reduced to attack versus "
        "BENIGN for these runs, and the reviewed default settings were used (no settings "
        "search). A model tested on its own dataset gives the in-domain score; tested on another "
        "dataset, the cross-dataset score."
    )


def transfer_text(analysis: dict[str, Any]) -> str:
    """Generalization results (Results), from the stored transfer matrix."""
    t = analysis.get("transfer") or {}
    models = {n: v for n, v in (t.get("models") or {}).items() if v.get("cross_domain") is not None}
    if not models:
        return ""
    order = sorted(models, key=lambda n: -models[n]["cross_domain"])
    sents = [
        "{{ref:table_transfer}} and {{ref:figure_transfer}} show how well each model "
        "generalizes when it is trained on one dataset and tested on another (attack versus "
        "BENIGN, macro-F1, mean over seeds)."
    ]
    sents.append(
        "Averaged over all pairs of different datasets, "
        + _and(
            [
                f"{pretty(n)} reached {models[n]['cross_domain']:.4f} (in-domain "
                f"{models[n]['in_domain']:.4f})"
                for n in order
            ]
        )
        + "."
    )
    best = order[0]
    w = models[best].get("worst")
    if w:
        sents.append(
            f"The hardest transfer for {pretty(best)} was from {dataset_label(w['train'])} to "
            f"{dataset_label(w['test'])} (macro-F1 {w['macro_f1']:.4f})."
        )
    return " ".join(sents)


def benchmark_text(analysis: dict[str, Any]) -> str:
    """Results across datasets (Results), from the stored ranking and secondary tests."""
    b = analysis.get("benchmark") or {}
    if not b.get("datasets"):
        return ""
    names = _and([dataset_label(d) for d in b["datasets"]])
    rank = b.get("ranking") or {}
    out = []
    if rank.get("applicable"):
        order = sorted(rank["average_rank"], key=rank["average_rank"].__getitem__)
        ranks = "; ".join(f"{pretty(m)} {rank['average_rank'][m]:.2f}" for m in order)
        sig = rank["p_value"] < 0.05
        out.append(
            f"Across the {_num(len(b['datasets']))} datasets ({names}), the average ranks on "
            f"{_metric(b.get('metric', 'macro_f1'))} (1 = best) were: {ranks}. The Friedman test "
            f"{'found' if sig else 'did not find'} a difference between the models' ranks "
            f"(p {'< 0.001' if rank['p_value'] < 0.001 else '= ' + _pval(rank['p_value'])}); the Nemenyi critical difference is "
            f"{rank['critical_difference']:.2f}, so only models whose average ranks differ by "
            "more than this differ significantly."
            + (
                " With few datasets the test has little power, so a non-significant result does "
                "not show that the models are equal."
                if not sig
                else ""
            )
        )
    else:
        out.append(
            f"Results on each of the {_num(len(b['datasets']))} datasets ({names}) are listed in "
            "the table below; a Friedman test across datasets needs at least two datasets and "
            "three models, so no ranking test was run."
        )
    tests = [
        h
        for ds in b.get("datasets", [])[1:]
        for h in (b["per_dataset"].get(ds) or {}).get("hypotheses", [])
        if "p_holm" in h
    ]
    if tests:
        # the full list is in its table; the text names only what was supported
        held = [h for h in tests if h.get("result") == "supported"]
        intro = (
            f"The stated hypotheses were repeated on the other datasets as {_num(len(tests))} "
            "secondary tests (Holm-adjusted across them, {{ref:table_benchmark_tests}})"
        )
        if held:
            out.append(
                intro
                + "; of these, only "
                + _and(
                    [
                        f"{h['id']} on {dataset_label(h['dataset'])} (Holm-adjusted p "
                        f"{'< 0.001' if h['p_holm'] < 0.001 else '= ' + _pval(h['p_holm'])})"
                        for h in held
                    ]
                )
                + (" was" if len(held) == 1 else " were")
                + " supported."
            )
        else:
            out.append(intro + "; none of them was supported.")
    return " ".join(out)


def _pval(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def methodology_text(facts: dict[str, Any], prose: list[str], analysis: dict[str, Any]) -> str:
    """Methodology from the deterministic method facts ('facts.x' references filled in)."""

    def fill(s: str) -> str:
        return re.sub(r"facts\.([a-z_]+)", lambda m: str(facts.get(m.group(1), m.group(0))), s)

    statements = [fill(s) for s in prose if not s.startswith("Model '")]
    if bench := benchmark_method_text(analysis):
        statements.append(bench)
    if xfer := transfer_method_text(analysis):
        statements.append(xfer)
    models = analysis.get("models", {})
    if models:
        statements.append(
            f"{_cap(_num(len(models)))} models were compared: "
            + _and([f"{pretty(n)} ({m.get('role', 'model')})" for n, m in models.items()])
            + "."
        )
        settings = [
            f"{pretty(n)}: " + ", ".join(f"{k}={_setting(v)}" for k, v in m["params"].items())
            for n, m in models.items()
            if m.get("params")
        ]
        tuned = [m for m in models.values() if m.get("tuned") and m.get("tuning")]
        if settings and tuned:
            t = tuned[0]["tuning"]
            statements.append(
                "Hyper-parameters were tuned with Optuna (TPE sampler) on the validation split, "
                f"maximising {_metric(str(t.get('metric', 'macro_f1')))}, with the same budget of "
                f"{t.get('trials_requested')} trials for every model and at most "
                f"{int(t.get('tuning_rows') or 0):,} training rows per trial; the test split was "
                "used once, after tuning. Tuning runs are exploratory; only the final test "
                "evaluation is confirmatory. The selected settings were: "
                + "; ".join(settings)
                + "."
            )
        elif settings:
            statements.append(
                "All models used fixed, reviewed settings (no tuning on the test split): "
                + "; ".join(settings)
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
    r"\bstruggl\w*|\bfail\w* to (?:detect|identify|recogni[sz]e)\b|\bpoorly\b|"
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
    r"\b(?:tree[- ]based|complex|simpler|deep|ensemble|linear|neural) (?:models?|methods?)\b|"
    r"\btree ensembles?\b|\bneural networks?\b|"
    r"\b(?:all|every|each|both|most) (?:of )?(?:the )?(?:\w+ )?(?:models?|classifiers?)\b",
    re.I,
)


# Related Work sentences about the literature itself ('These studies ...', 'Most prior work
# ...') or about what this study adds are framing, not results.
_LITERATURE = re.compile(
    r"^(?:(?:However|Yet|But|Still),\s+)?(?:These|Such|Prior|Previous|Existing|Earlier|Recent|"
    r"Most|Many|Few|Other) (?:(?:of )?(?:these |the )?(?:prior |previous |existing |recent )?"
    r"(?:studies|works?|papers|approaches|research|literature|methods))\b|"
    r"^(?:This|Our|The present) (?:study|work|paper) (?:contributes|adds|differs|extends|"
    r"complements|addresses|fills)\b",
    re.I,
)


def forbidden(sentence: str, context: str, known: list[str]) -> str | None:
    """Why an LLM sentence may not stay in a framing section (None = allowed)."""
    if _NUMBER.search(sentence):
        return "states a measured value"
    from cyberlab_manuscript.consistency import is_question

    if is_question(sentence) and not _STATS.search(sentence):
        return None  # 'This study evaluates whether A outperforms B' is the research question
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
    if named and _MECHANISM.search(sentence):
        return "explains a result with a model mechanism the study did not test"
    return None


# 'X does better because of its decision boundaries': the experiments measure scores, not why.
_MECHANISM = re.compile(
    r"\b(?:be )?attributed to\b|\battributable to\b|"
    r"\bdecision boundar(?:y|ies)\b|\bability to (?:capture|model|learn|handle)\b|"
    r"\b(?:due to|because of|owing to|thanks to) (?:their|its)\b",
    re.I,
)


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
        # 'Finally' reads well after three or more points; after one it reads as a gap
        if s.startswith(("Finally", "Lastly")) and n == len(ordinal_idx) - 1 and n >= 2:
            continue
        sentences[i] = _ORDINAL.sub(f"{_ORDINALS[min(n, len(_ORDINALS) - 1)]}, ", s, count=1)
    return sentences


def _rebuild(body: str, drop: set[str]) -> str:
    """Rebuild a section without the dropped sentences, cleaning connectives and orphans."""
    out_lines = []
    # A removal at the end of one paragraph also orphans an 'It ...' that opens the next one.
    prev_dropped = False
    for line in body.split("\n"):
        if not line.strip() or line.lstrip().startswith(("|", "#", ">")):
            out_lines.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*+]|\d+\.)\s+)", line)
        prefix, text = (m.group(1), line[m.end() :]) if m else ("", line)
        kept: list[str] = []
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


# Rich mode (writing pipeline v2): in these sections the writer may state results and compare
# models, because every number comes from a placeholder (the renderer rejects typed numbers)
# and the consistency checks reject untested comparisons and contradicted hypotheses.
RICH_SECTIONS = {"Abstract", "Discussion", "Conclusion", "Limitations"}
_RENDERED_HYP = re.compile(r"\bH\d+ \((?:not )?supported\)")
_VERDICT = re.compile(
    r"hypothes\w*[^.]{0,80}\b(?:support|confirm|reject|refut|hold|true|false|valid)\w*|"
    r"\b(?:support|confirm|reject|refut|validat|contradict)\w*[^.]{0,40}\bhypothes|"
    r"\bp[- ]?values?\s*(?:of|=|<|>|was|were)",
    re.I,
)


def forbidden_rich(sentence: str, context: str, known: list[str]) -> str | None:
    """Rich-mode rule: no hand-written test verdicts and no untested mechanism stories."""
    if _VERDICT.search(_RENDERED_HYP.sub("", sentence)) and not _RENDERED_HYP.search(sentence):
        return "states a hypothesis verdict without the evidence placeholder"
    if _MECHANISM.search(sentence):  # 'its ability to capture...', 'attributed to its ...'
        return "explains a result with a model mechanism the study did not test"
    return None


def _states_all_hypotheses(text: str, analysis: dict[str, Any]) -> bool:
    ids = [h["id"] for h in _tested_hyps(analysis)]
    return bool(ids) and all(
        re.search(
            rf"\b{re.escape(i)} \((?:not )?supported\)|\({re.escape(i)}, (?:not )?supported\)",
            text,
        )
        for i in ids
    )


def _missing_operational(text: str, analysis: dict[str, Any]) -> str:
    """The false-alarm and missed-class facts, unless the writer already discussed them."""
    keep = []
    for s in split_sentences(operational_text(analysis)):
        if s.startswith("The false-positive rate") and re.search(
            r"false[- ](?:positive|alarm)", text, re.I
        ):
            continue
        if s.startswith("No model detected"):
            missed = re.findall(r"detected any (.+?) test row", s)
            if missed and all(
                c.strip() in text for c in re.split(r",| and ", missed[0]) if c.strip()
            ):
                continue
        keep.append(s)
    return " ".join(keep)


def lock_evidence(
    markdown: str,
    analysis: dict[str, Any],
    facts: dict[str, Any] | None = None,
    prose: list[str] | None = None,
    rich: bool = False,
) -> tuple[str, list[dict[str, str]]]:
    """Make every evidence statement generated and every LLM sentence framing-only.

    rich=True (writing pipeline v2) lets Abstract, Discussion, Conclusion and Limitations state
    results through placeholders, checked by the consistency rules.
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
        from cyberlab_manuscript.builder import resolve_refs

        markdown = resolve_refs(markdown)

    # 2. Framing sections keep only sentences that state no results.
    issues = {i["sentence"]: i["reason"] for i in consistency_issues(markdown, analysis)}
    # sentences this module generates are removed silently and re-added below (idempotent)
    generated = set(
        split_sentences(
            " ".join(
                [
                    findings_text(analysis),
                    discussion_lead(analysis),
                    gap_text(analysis),
                    operational_text(analysis),
                    benchmark_discussion_text(analysis),
                ]
            )
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
                if (
                    name == "Related Work"
                    and _LITERATURE.match(s)
                    and not _NUMBER.search(s)
                    and s not in issues
                ):
                    continue
                why = issues.get(s) or (
                    forbidden_rich(s, ctx, known)
                    if rich and name in RICH_SECTIONS
                    else forbidden(s, ctx, known)
                )
                if why:
                    drop.add(s)
                    dropped.append({"section": name, "sentence": s, "reason": why})
        kept = _rebuild(prose_part, drop).strip()

        # 3. Evidence statements are added where the paper needs them.
        if rich and name in ("Abstract", "Conclusion") and _states_all_hypotheses(kept, analysis):
            pass  # the writer already reported every tested hypothesis (from placeholders)
        elif rich and name == "Discussion":
            lead = [p for p in (_missing_operational(kept, analysis),) if p]
            if accuracy_gap_models(analysis) and not re.search(r"balanced accuracy", kept, re.I):
                lead.append(gap_text(analysis))
            bench = benchmark_discussion_text(analysis)
            if bench and not any(
                dataset_label(d) in kept for d in (analysis.get("benchmark") or {})["datasets"][1:]
            ):
                lead.insert(0, bench)
            kept = "\n\n".join(([kept] if kept else []) + lead)
        elif name == "Abstract":
            kept = (kept + " " + findings_text(analysis)).strip()
        elif name == "Conclusion":
            sents = split_sentences(" ".join(kept.split()))
            future = re.compile(r"(?i)(?:additionally,\s*|moreover,\s*)?(?:future|further)\b")
            at = next((i for i, s in enumerate(sents) if future.match(s)), len(sents))
            kept = " ".join(sents[:at] + [findings_text(analysis)] + sents[at:])
        elif name == "Discussion":
            lead = [
                p
                for p in (discussion_lead(analysis), gap_text(analysis), operational_text(analysis))
                if p
            ]
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
