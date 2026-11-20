from cyberlab_protocol import default_protocol
from cyberlab_protocol.protocol_validator import validate_protocol

DS = {
    "status": "verified",
    "class_counts": {"BENIGN": 1000, "Worms": 40},
    "split_counts": {"train": 100_000},
}


def test_good_protocol_passes_or_warns_only() -> None:
    p = default_protocol("p", "cicids2017-mlcsv-v1").model_dump(mode="json")
    r = validate_protocol(p, "Compare models with macro-F1.", DS, 480)
    by = {c["check"]: c["status"] for c in r["checks"]}
    assert by["measurable_hypotheses"] == "PASS" and by["baselines"] == "PASS"
    assert by["dataset_suitable"] == "WARNING"  # Worms has fewer than 100 rows
    assert r["status"] != "BLOCK" and r["estimated_minutes"] > 0


def test_accuracy_for_rare_class_question_and_infeasible_budget() -> None:
    p = default_protocol("p", "x").model_dump(mode="json")
    for h in p["hypotheses"]:
        h["primary_metric"] = "accuracy"
    r = validate_protocol(p, "How well are rare attack classes detected?", DS, 480)
    assert {c["check"]: c["status"] for c in r["checks"]}["metric_fits_question"] == "WARNING"
    big = {**DS, "split_counts": {"train": 100_000_000}}
    assert validate_protocol(p, "x", big, 10)["status"] == "BLOCK"
    assert validate_protocol(p, "x", None, 480)["status"] == "BLOCK"
