"""Acceptance tests that had no dedicated automated check yet (spec section 13)."""

from __future__ import annotations

import sqlite3
import tarfile
from pathlib import Path

import pandas as pd

from cyberlab_common.restore import verify_backup


def test_duplicates_and_conflicting_labels_are_removed_at05(tmp_path: Path) -> None:
    from cyberlab_data.cicids2017 import prepare, write_fixture

    paths = write_fixture(tmp_path / "raw", rows_per_class=80)
    # inject a feature vector that appears with two different labels
    df = pd.read_csv(paths[1], encoding="latin-1")
    clash = df.iloc[[10]].copy()
    clash[" Label"] = "PortScan" if clash[" Label"].iloc[0] != "PortScan" else "BENIGN"
    clash.to_csv(paths[1], mode="a", header=False, index=False, encoding="latin-1")

    ds = prepare(tmp_path)
    m = ds.manifest
    notes = (ds.root / "preprocessing_notes.json").read_text()
    assert m.split_overlap_rows == 0 and m.status == "verified"
    assert m.n_duplicates_removed >= 4 * 4  # the fixture repeats 4 rows in each of 4 files
    import json

    assert json.loads(notes)["conflicting_label_rows_dropped"] == 2  # both copies removed
    clean = pd.read_parquet(ds.clean_path)
    feats = [c for c in clean.columns if c not in {"Class", "Day", "split"}]
    assert not clean.duplicated(subset=feats).any()  # no feature vector left twice


def test_stored_metrics_must_match_recomputation_at14() -> None:
    from cyberlab_qa import compute_metrics, recompute_matches

    preds = pd.DataFrame({"row": range(8), "y_true": list("AABBCCAA"), "y_pred": list("AABBCAAB")})
    stored = compute_metrics(preds)
    assert recompute_matches(stored, preds)
    tampered = {**stored, "macro_f1": stored["macro_f1"] + 0.01}
    assert not recompute_matches(tampered, preds)


def test_foreign_gpu_processes_are_detected_at09(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_exec.broker as broker

    class Done:
        stdout = "4242, 30000\n777, 1024\n"

    monkeypatch.setattr(broker.shutil, "which", lambda _n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(broker.subprocess, "run", lambda *a, **k: Done())
    procs = broker.foreign_gpu_processes()
    assert procs == [{"pid": "4242", "used_mib": "30000"}, {"pid": "777", "used_mib": "1024"}]
    monkeypatch.setattr(broker.shutil, "which", lambda _n: None)
    assert broker.foreign_gpu_processes() == []  # no GPU tools: report nothing, claim nothing


def _make_backup(tmp_path: Path, tamper: bool = False) -> Path:
    src = tmp_path / "src"
    payload = b"predictions,1,2,3\n"
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()
    art = src / "artifacts" / sha[:2] / sha
    art.mkdir(parents=True)
    (art / "predictions.csv").write_bytes(payload + (b"x" if tamper else b""))
    con = sqlite3.connect(src / "cyberlab.db")
    con.executescript(
        "CREATE TABLE artifacts (sha256 TEXT); CREATE TABLE approvals (id TEXT);"
        "CREATE TABLE projects (id TEXT);"
    )
    con.execute("INSERT INTO artifacts VALUES (?)", (sha,))
    con.execute("INSERT INTO approvals VALUES ('a1')")
    con.execute("INSERT INTO projects VALUES ('p1')")
    con.commit()
    con.close()
    tarball = tmp_path / "backup.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        for p in src.iterdir():
            tar.add(p, arcname=p.name)
    return tarball


def test_restore_drill_verifies_hashes_and_approvals_at21(tmp_path: Path) -> None:
    report = verify_backup(_make_backup(tmp_path), tmp_path / "restore")
    assert report["ok"] and report["artifacts_verified"] == 1 and report["approvals"] == 1


def test_restore_drill_detects_a_modified_artifact_at21(tmp_path: Path) -> None:
    report = verify_backup(_make_backup(tmp_path, tamper=True), tmp_path / "restore")
    assert not report["ok"] and "does not match its hash" in report["problems"][0]


def test_fast_bootstrap_metrics_match_sklearn_exactly() -> None:
    import numpy as np
    from sklearn import metrics as skm

    from cyberlab_qa.metrics import _balanced_accuracy_cm, _mcc_cm

    rng = np.random.default_rng(3)
    labels = np.array(["A", "B", "C", "D"])
    y = labels[rng.choice(4, size=5000, p=[0.7, 0.2, 0.09, 0.01])]
    p = np.where(rng.random(5000) < 0.8, y, labels[rng.integers(0, 4, size=5000)])
    p[:3] = "E"  # a label that only appears in predictions
    allv, codes = np.unique(np.concatenate([y, p]), return_inverse=True)
    k = len(allv)
    cm = np.bincount(codes[:5000] * k + codes[5000:], minlength=k * k).reshape(k, k)
    assert abs(_balanced_accuracy_cm(cm) - skm.balanced_accuracy_score(y, p)) < 1e-12
    assert abs(_mcc_cm(cm) - skm.matthews_corrcoef(y, p)) < 1e-12
