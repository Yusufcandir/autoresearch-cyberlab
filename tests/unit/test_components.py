"""Unit tests mapped to the spec's acceptance tests (AT-xx)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from cyberlab_common.db import Database, JobRecord, Project, User
from cyberlab_exec import Broker, GpuLock, LockBusy
from cyberlab_literature import Paper
from cyberlab_llm import BudgetExceeded, FakeProvider, LLMError, LLMGateway
from cyberlab_llm.gateway import parse_json, redact
from cyberlab_manuscript import EvidenceReport, render_text
from cyberlab_protocol import CompileError, compile_protocol, default_protocol
from cyberlab_qa import bootstrap_ci, compute_metrics, mcnemar
from cyberlab_schemas import Approval, ApprovalGate, DatasetManifest, ResearchSpec
from tests.fixtures.fakes import FAKE_DOI, fake_scholar, make_settings


def test_unknown_schema_version_rejected() -> None:  # AT-08 of Step 08
    with pytest.raises(ValidationError):
        ResearchSpec.model_validate({"schema_version": "9", "title": "x" * 10, "goal": "y" * 30})


def _dataset(status: str = "verified") -> DatasetManifest:
    return DatasetManifest(
        dataset_id="d",
        name="d",
        source_url="u",
        license="l",
        citation="c",
        files={},
        n_rows_raw=1,
        n_rows_clean=1,
        n_duplicates_removed=0,
        features=["f"],
        label_column="Class",
        class_counts={"BENIGN": 1},
        status=status,  # type: ignore[arg-type]
    )


def _approval(protocol_sha: str, version: int = 1) -> Approval:
    return Approval(
        project_id="p",
        gate=ApprovalGate.A_PROTOCOL,
        subject_version=version,
        subject_sha256=protocol_sha,
        approver="pi",
        decision="approved",
    )


def test_compiler_requires_matching_approval_at06() -> None:
    proto = default_protocol("p", "d")
    with pytest.raises(CompileError, match="no PI approval"):
        compile_protocol(proto, None, _dataset(), max_job_minutes=10, use_gpu=False)
    changed = proto.model_copy(update={"seeds": [1]})
    with pytest.raises(CompileError, match="different protocol version"):
        compile_protocol(
            changed, _approval(proto.sha256), _dataset(), max_job_minutes=10, use_gpu=False
        )
    out = compile_protocol(
        proto, _approval(proto.sha256), _dataset(), max_job_minutes=10, use_gpu=True
    )
    assert len(out) == 12
    assert {m.use_gpu for m in out if m.model.template == "xgboost"} == {True}


def test_compiler_blocks_unverified_dataset_at04() -> None:
    proto = default_protocol("p", "d")
    with pytest.raises(CompileError, match="dataset blocked"):
        compile_protocol(
            proto, _approval(proto.sha256), _dataset("blocked"), max_job_minutes=10, use_gpu=False
        )


def test_compiler_blocks_t3_at12() -> None:
    proto = default_protocol("p", "d").model_copy(update={"risk_tier": "T3"})
    with pytest.raises(CompileError, match="ISOLATION_REQUIRED"):
        compile_protocol(
            proto, _approval(proto.sha256), _dataset(), max_job_minutes=10, use_gpu=False
        )


def test_citation_verification_at03() -> None:
    scholar = fake_scholar()
    good = scholar.verify(
        Paper(
            title="Gradient boosting for flow-based attack classification",
            doi="10.1000/real.two",
            year=2022,
        )
    )
    assert good.status == "Verified"
    invented = scholar.verify(Paper(title="Made up", doi=FAKE_DOI, year=2024))
    assert invented.status == "Unverified"
    wrong_title = scholar.verify(
        Paper(title="Completely different title", doi="10.1000/real.one", year=2021)
    )
    assert wrong_title.status == "Unverified"
    assert scholar.verify(Paper(title="No DOI", doi=None, year=2020)).status == "Unverified"


def test_gateway_privacy_routing_and_budget() -> None:
    class Cloud(FakeProvider):
        name, local = "cloud", False

    cloud, local = Cloud(), FakeProvider()
    gw = LLMGateway(cloud, local, budget_lookup=lambda _p: (100, 100))
    with pytest.raises(BudgetExceeded):
        gw.complete("t", "s", "p", project_id="x")
    gw = LLMGateway(cloud, local)
    assert gw.complete("t", "s", "secret study data", sensitive=True).provider == "fake"
    assert gw.complete("t", "s", "public text").provider == "cloud"
    with pytest.raises(LLMError):
        LLMGateway(cloud, None).complete("t", "s", "x", sensitive=True)


def test_redaction_and_json_parsing() -> None:
    assert "AIza" not in redact("key AIzaSyA0123456789abcdefghijklmnopqrstuv here")
    assert parse_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}


def test_metrics_and_statistics() -> None:
    df = pd.DataFrame(
        {
            "row": range(8),
            "y_true": ["BENIGN"] * 4 + ["DoS"] * 4,
            "y_pred": ["BENIGN"] * 3 + ["DoS"] * 5,
        }
    )
    m = compute_metrics(df)
    assert m["accuracy"] == pytest.approx(7 / 8)
    assert m["false_positive_rate"] == pytest.approx(0.25)
    ci = bootstrap_ci(df, n_boot=50)
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert mcnemar(df, df)["p_value"] == 1.0


def test_manuscript_flags_unsupported_numbers_and_citations_at15() -> None:
    analysis = {
        "models": {"xgb": {"aggregate": {"macro_f1": {"mean": 0.91, "sd": 0.01, "n_seeds": 3}}}},
        "hypotheses": [],
        "metrics_recomputed_match": True,
        "artifact_sha256": "abc",
    }
    cites = {
        "Good2020": {"status": "Verified", "doi": "10.1/x"},
        "Bad2021": {"status": "Unverified"},
    }
    rep = EvidenceReport()
    out = render_text(
        "F1 was {{metric:xgb:macro_f1}} {{cite:Good2020}}.", analysis, cites, rep, "results"
    )
    assert "0.9100 ± 0.0100" in out and not rep.blockers
    rep = EvidenceReport()
    out = render_text(
        "We reached 99.7% accuracy {{cite:Bad2021}} and {{metric:svm:macro_f1}}.",
        analysis,
        cites,
        rep,
        "abstract",
    )
    codes = {f["code"] for f in rep.blockers}
    assert codes == {"UNSUPPORTED_NUMBER", "UNVERIFIED_CITATION", "MISSING_EVIDENCE"}
    assert "[citation needed]" in out


def test_gpu_lock_is_exclusive_at08(tmp_path: Path) -> None:
    lock = GpuLock(tmp_path / "gpu.lock")
    lock.acquire("job-1")
    with pytest.raises(LockBusy):
        lock.acquire("job-2")
    lock.release("job-2")  # not the owner: no effect
    assert lock.owner()["job_id"] == "job-1"
    lock.release("job-1")
    lock.acquire("job-2")


def test_broker_idempotent_submission_at10(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    db = Database(settings.database_url.get_secret_value())  # type: ignore[union-attr]
    db.create_all()
    proto = default_protocol("p1", "d")
    manifests = compile_protocol(
        proto, _approval(proto.sha256), _dataset(), max_job_minutes=1, use_gpu=False
    )
    broker = Broker(db, settings)
    with db.session() as s:
        s.add(User(id="u", username="u", password_hash="x", role="pi"))  # noqa: S106
        s.flush()
        s.add(Project(id="p1", title="t", owner_id="u", spec={}))
        s.flush()
        first = broker.submit(s, manifests[0], "agent").id
        second = broker.submit(s, manifests[0], "agent").id  # e.g. retried after a crash
    assert first == second
    with db.session() as s:
        assert len(list(s.query(JobRecord))) == 1
        job = s.get(JobRecord, first)
        assert job is not None
        job.state = "running"  # simulate a crash mid-run
    assert broker.reconcile() == 1
    with db.session() as s:
        job = s.get(JobRecord, first)
        assert job is not None and job.state == "queued"


def test_dashboard_list_filter_never_iterates_characters() -> None:
    from cyberlab_api.web import _aslist

    assert _aslist("CIC-IDS2017 is widely studied.") == ["CIC-IDS2017 is widely studied."]
    assert _aslist(["a", "b"]) == ["a", "b"] and _aslist(None) == []
