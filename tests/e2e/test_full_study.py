"""End-to-end: brief -> literature -> protocol -> PI approval -> experiments -> analysis -> draft -> release."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import select

from cyberlab_authz.rbac import Forbidden, Principal, create_user
from cyberlab_common.db import (
    CitationRecord,
    JobRecord,
    ManuscriptRecord,
    ProtocolVersion,
    RunRecord,
)
from cyberlab_orchestrator.lab import GateError, Lab
from cyberlab_schemas import ResearchSpec
from tests.fixtures.fakes import FAKE_DOI, fake_gateway, fake_scholar, make_settings

CHECKS = {"science": True, "ethics": True, "security": True}

SPEC = ResearchSpec(
    title="Duplicate-aware evaluation of IDS models on CIC-IDS2017",
    goal="Compare classical and gradient boosted classifiers for multiclass intrusion detection.",
)


@pytest.fixture
def lab(tmp_path: Path) -> Lab:
    lab = Lab(make_settings(tmp_path), llm=fake_gateway(), scholar=fake_scholar())
    yield lab
    lab.close()


def _users(lab: Lab) -> tuple[Principal, Principal]:
    with lab.db.session() as s:
        pi = create_user(s, "prof", "correct-horse-battery", "pi")
        other = create_user(s, "student", "correct-horse-battery", "researcher")
        return Principal(pi.id, pi.username, pi.role), Principal(
            other.id, other.username, other.role
        )


def test_full_study(lab: Lab) -> None:
    pi, outsider = _users(lab)
    pid = lab.create_project(pi, SPEC)
    status = lab.start(pid)
    assert status["pending"]["gate"] == "A", status

    with lab.db.session() as s:
        cites = {
            c.doi: c.status
            for c in s.scalars(select(CitationRecord).where(CitationRecord.project_id == pid))
        }
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        version, sha = pv.version, pv.sha256
    assert cites["10.5220/0006639801080116"] == "Verified"
    assert cites[FAKE_DOI] == "Unverified"  # AT-03

    with pytest.raises(GateError):  # AT-06: stale/mismatched approval does not authorize
        lab.approve_protocol(pi, pid, version, "0" * 64, "approved")
    with pytest.raises(Exception):  # non-member cannot approve  # noqa: B017
        lab.approve_protocol(outsider, pid, version, sha, "approved")

    status = lab.approve_protocol(pi, pid, version, sha, "approved")
    assert status["pending"]["gate"] == "jobs"
    with lab.db.session() as s:
        n_jobs = len(list(s.scalars(select(JobRecord).where(JobRecord.project_id == pid))))
    assert n_jobs == 12  # 4 models x 3 seeds

    for _ in range(n_jobs + 2):
        lab.worker_tick(wait=True)
    status = lab.status(pid)
    assert status["pending"] and status["pending"]["gate"] == "D", status

    with lab.db.session() as s:
        runs = list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid)))
        assert all(r.status == "succeeded" for r in runs), [r.receipt.get("error") for r in runs]
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report, mversion = ms.markdown, ms.evidence_report, ms.version
    assert "| xgboost |" in md and "[citation needed]" not in md
    assert report["release_ready"], report["flags"]
    # the AI-use statement is kept beside the paper, not inside it
    assert "AI use disclosure" not in md and "AutoResearch-CyberLab" in report["ai_disclosure"]

    with pytest.raises(Forbidden):
        lab.release(outsider_as_member(lab, pid, outsider), pid, mversion, CHECKS)
    with pytest.raises(GateError):  # spec Step 22: no release without the review checklist
        lab.release(pi, pid, mversion)
    final = lab.release(pi, pid, mversion, CHECKS)
    assert final["stage"] == "released"
    audit_lines = (lab.settings.data_dir.parent / "audit.jsonl").read_text().splitlines()
    assert any(json.loads(line)["code"] == "ALLOWED" for line in audit_lines)


def outsider_as_member(lab: Lab, pid: str, outsider: Principal) -> Principal:
    """Give the outsider a reviewer role: members below PI still cannot release (gate D)."""
    from cyberlab_authz.rbac import add_member

    with lab.db.session() as s:
        add_member(s, pid, outsider.id, "reviewer")
    return outsider


def test_redraft_rewrites_from_stored_results(lab: Lab) -> None:
    pi, _ = _users(lab)
    pid = lab.create_project(pi, SPEC)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D"
    with lab.db.session() as s:
        before = len(
            list(s.scalars(select(ManuscriptRecord).where(ManuscriptRecord.project_id == pid)))
        )
        runs = len(list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid))))
    status = lab.redraft(pi, pid)
    assert status["pending"]["gate"] == "D"
    with lab.db.session() as s:
        after = len(
            list(s.scalars(select(ManuscriptRecord).where(ManuscriptRecord.project_id == pid)))
        )
        assert len(list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid)))) == runs
    assert after == before + 1


def test_full_study_on_unsw_nb15(lab: Lab) -> None:
    """The same pipeline end to end on the second dataset, with its official partition."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Intrusion detection models on UNSW-NB15",
        goal="Compare classical and gradient boosted classifiers on UNSW-NB15 using the "
        "official train/test split, reporting macro-F1.",
        dataset_hint="UNSW-NB15",
    )
    pid = lab.create_project(pi, spec)
    status = lab.start(pid)
    assert status["pending"]["gate"] == "A", status
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        assert pv.body["dataset_id"] == "unsw-nb15-v1"
        assert pv.body["split"]["strategy"] == "official"
        pv_split = dict(pv.body["split"])
        cites = {c.doi: c.status for c in s.scalars(select(CitationRecord))}
        version, sha = pv.version, pv.sha256
    assert cites["10.1109/milcis.2015.7348942"] == "Verified"
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report = ms.markdown, ms.evidence_report
    body = md.split("## References")[0]
    assert "CIC-IDS2017" not in body and "Fwd Header" not in body
    assert "official" in md and "[citation needed]" not in md
    assert report["release_ready"], report["flags"]

    # A split prepared by an older version has no preprocessing audit: a rewrite prepares that
    # exact split again, so the audit is evaluated instead of NOT_EVALUATED.
    from cyberlab_data import get_adapter, prepared_dir_for
    from cyberlab_schemas import SplitSpec

    split = SplitSpec.model_validate(pv_split)
    audit_file = (
        prepared_dir_for(get_adapter("unsw-nb15-v1").root(lab.settings.dataset_root), split)
        / "data_preprocessing_audit.json"
    )
    audit_file.unlink()
    lab.redraft(pi, pid)
    assert audit_file.exists()


def test_full_study_with_tuning(lab: Lab) -> None:
    """Tuning requested in the goal: validation-only search, tuned settings in the paper."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Tuned intrusion detection models on CIC-IDS2017",
        goal="Compare classical and gradient boosted classifiers on CIC-IDS2017 with macro-F1. "
        "Tune every model with 3 trials on the validation split and use three seeds.",
    )
    pid = lab.create_project(pi, spec)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None and pv.body["tuning"]["trials"] == 3
        assert len(pv.body["seeds"]) == 3
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report = ms.markdown, ms.evidence_report
    assert "tuned with Optuna" in md and "Difference (95% CI)" in md
    assert "no hyper-parameter tuning was performed" not in md
    assert report["release_ready"], report["flags"]


def test_full_study_with_deep_models(lab: Lab) -> None:
    """Models named in the goal (including deep templates) become the protocol's line-up."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Deep versus tree models on CIC-IDS2017",
        goal="Compare logistic regression, random forest, the FT-Transformer and a ResNet-MLP "
        "on CIC-IDS2017. Test whether the FT-Transformer achieves higher macro-F1 than "
        "logistic regression. Use three seeds.",
    )
    pid = lab.create_project(pi, spec)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        names = [m["name"] for m in pv.body["models"]]
        assert names == ["logistic_regression", "random_forest", "ft_transformer", "resnet_mlp"]
        assert pv.body["hypotheses"][0]["comparison"] == "ft_transformer > logistic_regression"
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(16):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        runs = list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid)))
        assert all(r.status == "succeeded" for r in runs), [r.receipt.get("error") for r in runs]
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report = ms.markdown, ms.evidence_report
    assert "FT-Transformer" in md and "| ft_transformer |" in md
    assert report["release_ready"], report["flags"]


def test_cross_dataset_study(lab: Lab) -> None:
    """Train on one NetFlow-v2 dataset, test on another: binary labels, shared features."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Cross-dataset generalisation of NetFlow intrusion detectors",
        goal="Train logistic regression, random forest and XGBoost on NF-UNSW-NB15-v2 and test "
        "on NF-ToN-IoT-v2. Test whether XGBoost achieves higher macro-F1 than logistic "
        "regression.",
        dataset_hint="NF-UNSW-NB15-v2",
    )
    pid = lab.create_project(pi, spec)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        assert pv.body["split"]["strategy"] == "cross_dataset"
        assert pv.body["label_scheme"] == "binary"
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report = ms.markdown, ms.evidence_report
    assert "Cross-dataset evaluation" in md and "NF-ToN-IoT-v2" in md
    assert report["release_ready"], report["flags"]


def test_text_modality_study(lab: Lab) -> None:
    """Second modality (spec definition of done): URL text, text templates, grouped split."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Phishing URL detection from the URL text alone",
        goal="Detect phishing URLs from the URL string using character n-gram models, "
        "reporting macro-F1.",
        dataset_hint="PhiUSIIL phishing URLs",
    )
    pid = lab.create_project(pi, spec)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        assert {m["template"] for m in pv.body["models"]} == {"tfidf_logreg", "tfidf_linear_svm"}
        assert pv.body["split"]["strategy"] == "grouped"
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(10):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        runs = list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid)))
        assert all(r.status == "succeeded" for r in runs), [r.receipt.get("error") for r in runs]
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md, report = ms.markdown, ms.evidence_report
    assert "grouped by website" in md and "TF-IDF" in md
    assert report["release_ready"], report["flags"]


def test_feedback_controller_improvement_cycle(lab: Lab) -> None:
    """Professor's plan, Phase 3 deliverable: validate -> critique -> PI-approved follow-up with
    a stricter split -> rerun -> paper compares both versions."""
    pi, _ = _users(lab)
    pid = lab.create_project(pi, SPEC)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D"
    decisions = lab.research_decisions(pid)
    split_rule = next(d for d in decisions if d["rule"] == "B" and d["protocol_change"])
    assert split_rule["protocol_change"]["split"]["strategy"] == "by_day"
    status = lab.start_followup(pi, pid, split_rule["note_id"])
    assert status["pending"]["gate"] == "A"  # the follow-up still needs the PI's approval
    with lab.db.session() as s:
        v2 = s.scalar(
            select(ProtocolVersion)
            .where(ProtocolVersion.project_id == pid)
            .order_by(ProtocolVersion.version.desc())
        )
        assert v2 is not None and v2.body["followup_of"]["version"] == version
        assert v2.body["split"]["strategy"] == "by_day"
        v2_version, v2_sha = v2.version, v2.sha256
    lab.approve_protocol(pi, pid, v2_version, v2_sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    assert lab.status(pid)["pending"]["gate"] == "D", lab.status(pid)
    with lab.db.session() as s:
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
    assert f"Follow-up to protocol version {version}" in ms.markdown
    approved = [d for d in lab.research_decisions(pid) if d["approval_status"] == "approved"]
    assert approved and "created" in approved[0]["final_outcome"]


def test_identical_experiments_are_reused(lab: Lab) -> None:
    """Research memory: a follow-up that only adds seeds reruns only the new seeds."""
    from cyberlab_common.db import AgentNote

    pi, _ = _users(lab)
    pid = lab.create_project(pi, SPEC)
    lab.start(pid)
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")
    for _ in range(14):
        lab.worker_tick(wait=True)
    with lab.db.session() as s:
        note = AgentNote(
            project_id=pid,
            agent="feedback_controller",
            kind="research_decision",
            content={
                "decision_id": "D-test",
                "rule": "D",
                "identified_problem": "performance varies between seeds",
                "proposed_action": "add a fourth seed",
                "protocol_change": {"seeds": [42, 43, 44, 45]},
                "approval_status": "pending_pi",
            },
        )
        s.add(note)
        s.flush()
        note_id = note.id
    lab.start_followup(pi, pid, note_id)
    with lab.db.session() as s:
        v2 = s.scalar(
            select(ProtocolVersion)
            .where(ProtocolVersion.project_id == pid)
            .order_by(ProtocolVersion.version.desc())
        )
        assert v2 is not None
        v2v, v2sha = v2.version, v2.sha256
    lab.approve_protocol(pi, pid, v2v, v2sha, "approved")
    for _ in range(18):
        lab.worker_tick(wait=True)
    with lab.db.session() as s:
        runs = [
            r
            for r in s.scalars(select(RunRecord).where(RunRecord.project_id == pid))
            if r.experiment_id.startswith(f"p{v2v}-")
        ]
    reused = [r for r in runs if r.receipt.get("reused_from_run")]
    assert len(runs) == 16 and len(reused) == 12  # only the 4 runs with seed 45 were trained


def test_pause_stop_resume_restart(lab: Lab) -> None:
    pi, _ = _users(lab)
    pid = lab.create_project(pi, SPEC)

    # Pause before the study starts: the first step holds instead of running.
    lab.pause(pi, pid, True)
    status = lab.start(pid)
    assert status["pending"]["gate"] == "paused", status
    status = lab.resume(pi, pid)
    assert status["pending"]["gate"] == "A", status

    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        version, sha = pv.version, pv.sha256
    lab.approve_protocol(pi, pid, version, sha, "approved")

    # Stop: every job is canceled and the finished-jobs check does not advance the study.
    assert lab.stop(pi, pid) == 12
    lab.worker_tick(wait=True)
    status = lab.status(pid)
    assert status["paused"] and status["blocked_reason"].startswith("Stopped by prof")
    assert status["pending"]["gate"] == "jobs"
    with lab.db.session() as s:
        states = {j.state for j in s.scalars(select(JobRecord).where(JobRecord.project_id == pid))}
    assert states == {"canceled"}

    # Resume: the canceled experiments run again and the study reaches the release gate.
    lab.resume(pi, pid)
    for _ in range(14):
        lab.worker_tick(wait=True)
    status = lab.status(pid)
    assert status["pending"] and status["pending"]["gate"] == "D", status
    with lab.db.session() as s:
        runs = list(s.scalars(select(RunRecord).where(RunRecord.project_id == pid)))
    assert len(runs) == 12 and all(r.status == "succeeded" for r in runs)

    # Restart: back to the first step; a new protocol version waits for approval.
    status = lab.restart(pi, pid)
    assert status["pending"]["gate"] == "A", status
    with lab.db.session() as s:
        versions = list(
            s.scalars(select(ProtocolVersion.version).where(ProtocolVersion.project_id == pid))
        )
    assert max(versions) == version + 1


def test_multi_dataset_benchmark_study(lab: Lab) -> None:
    """One protocol, two datasets: experiments on both, a ranking across them, one paper."""
    pi, _ = _users(lab)
    spec = ResearchSpec(
        title="Benchmark of IDS models across CIC-IDS2017 and UNSW-NB15",
        goal="Compare logistic regression, random forest, gradient boosting and an MLP on "
        "CIC-IDS2017 and UNSW-NB15, reporting macro-F1 on each dataset and a ranking across "
        "both datasets.",
        dataset_hint="CIC-IDS2017",
    )
    pid = lab.create_project(pi, spec)
    status = lab.start(pid)
    assert status["pending"]["gate"] == "A", status
    with lab.db.session() as s:
        pv = s.scalar(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
        assert pv is not None
        assert pv.body["benchmark_datasets"] == ["unsw-nb15-v1"]
        version, sha = pv.version, pv.sha256
        keys = {
            c.key for c in s.scalars(select(CitationRecord).where(CitationRecord.project_id == pid))
        }
    assert "Moustafa2015unswnb15" in keys  # the second dataset's paper is cited too
    lab.approve_protocol(pi, pid, version, sha, "approved")
    with lab.db.session() as s:
        jobs = list(s.scalars(select(JobRecord).where(JobRecord.project_id == pid)))
    n_models = len(pv.body["models"])
    assert len(jobs) == 2 * n_models * 3
    assert sum("@unsw-nb15-v1" in j.experiment_id for j in jobs) == n_models * 3
    for _ in range(len(jobs) + 4):
        lab.worker_tick(wait=True)
    status = lab.status(pid)
    assert status["pending"] and status["pending"]["gate"] == "D", status
    with lab.db.session() as s:
        ms = s.scalar(
            select(ManuscriptRecord)
            .where(ManuscriptRecord.project_id == pid)
            .order_by(ManuscriptRecord.version.desc())
        )
        assert ms is not None
        md = ms.markdown
    assert "### Results across datasets" in md and "Average rank" in md
    assert "Friedman" in md and "UNSW-NB15" in md
    assert "All datasets of the benchmark" in md
    assert "per model and dataset (means over seeds)" in md  # heatmap figure
    assert "Average rank of each model across datasets" in md  # ranking figure
    assert "were also run on UNSW-NB15" in md  # Methodology states the benchmark
    with lab.db.session() as s:
        dups = s.scalars(select(CitationRecord.key).where(CitationRecord.project_id == pid))
        assert len(list(dups)) == len(set(keys))
