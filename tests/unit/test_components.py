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


def test_fast_bootstrap_macro_f1_matches_sklearn() -> None:
    import numpy as np
    from sklearn.metrics import f1_score

    from cyberlab_qa.metrics import _macro_f1_cm

    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 2000)
    p = np.where(rng.random(2000) < 0.2, rng.integers(0, 5, 2000), y)
    cm = np.bincount(y * 5 + p, minlength=25).reshape(5, 5)
    assert abs(_macro_f1_cm(cm) - f1_score(y, p, average="macro")) < 1e-12


def test_grouped_citations_facts_and_hypothesis_rendering() -> None:
    analysis = {
        "models": {},
        "hypotheses": [{"id": "H1", "result": "supported"}],
        "metrics_recomputed_match": True,
        "facts": {"n_seeds": 3, "test_percent": "20%"},
    }
    cites = {"A": {"status": "Verified"}, "B": {"status": "Verified"}}
    rep = EvidenceReport()
    out = render_text(
        "We used {{fact:n_seeds}} seeds and a {{fact:test_percent}} test split with 95% CIs "
        "{{cite:A, B}}; {{hyp:H1}}.",
        analysis,
        cites,
        rep,
        "methodology",
    )
    assert out == "We used 3 seeds and a 20% test split with 95% CIs [@A; @B]; H1 (supported)."
    assert not rep.flags


def test_reviewer_extracts_citing_sentences() -> None:
    from cyberlab_orchestrator.agents.reviewer import citing_sentences

    md = "## Intro\n\nFirst claim [@A]. No cite here. Second [@A; @B].\n\n## References\n\n- [A] x"
    got = citing_sentences(md)
    assert [g["keys"] for g in got] == [["A"], ["A", "B"]]


def test_tracker_rolls_back_when_writing_restarts() -> None:
    from datetime import UTC, datetime

    from cyberlab_api.progress import build_progress
    from cyberlab_common.db import AuditEvent

    def ev(i: int, action: str, step: str = "") -> AuditEvent:
        return AuditEvent(
            id=i,
            ts=datetime.now(UTC),
            actor="x",
            action=action,
            detail={"step": step} if step else {},
        )

    events = [ev(1, "protocol.approved")]
    for i, st in enumerate(["statistics", "writing", "reviewer"], start=2):
        events += [ev(i * 10, "agent.started", st), ev(i * 10 + 1, "agent.finished", st)]
    events += [ev(100, "manuscript.redraft_requested"), ev(101, "agent.started", "writing")]
    steps, now = build_progress(events, "drafted", None, [])
    state = {s.key: s.state for s in steps}
    assert state["statistics"] == "done"
    assert state["writing"] == "running"
    assert state["reviewer"] == "todo"
    assert "Rewriting the draft" in now


def test_prose_tidying_and_automatic_sections() -> None:
    from cyberlab_manuscript.builder import small_class_note, tidy_prose

    assert (
        tidy_prose("Testing confirms H1 (supported) and H2 (supported).")
        == "Testing confirms H1 and H2."
    )
    assert tidy_prose("we test H1 (supported).") == "we test H1."
    assert (
        tidy_prose("F1 0.9 (CI: 95% CI [0.8, 0.95], seed 42)")
        == "F1 0.9 (95% CI [0.8, 0.95], seed 42)"
    )
    analysis = {"models": {"m": {"per_class": {"Rare": {"support": 9}, "DoS": {"support": 900}}}}}
    assert "Rare (9 test rows)" in small_class_note(analysis)


def test_untested_deduplication_claims_are_caught_and_removed() -> None:
    from cyberlab_orchestrator.agents.reviewer import remove_sentences, untested_effect_sentences

    md = "## Discussion\n\nDe-duplication prevented inflated scores. Trees did best.\n\n## Evidence\n"
    found = untested_effect_sentences(md)
    assert found == ["De-duplication prevented inflated scores."]
    assert "inflated" not in remove_sentences(md, found)[0]


def test_sentence_removal_never_empties_a_required_section() -> None:
    from cyberlab_orchestrator.agents.reviewer import remove_sentences

    md = "## Abstract\n\nOnly one sentence claiming inflated scores here.\n\n## Results\n\nok\n"
    out, n = remove_sentences(md, ["Only one sentence claiming inflated scores here."])
    assert n == 0 and out == md
    para = "First sentence here is long enough. Second sentence is also long."
    md2 = f"## Discussion\n\n{para} Keep this one.\n"
    assert remove_sentences(md2, [para])[1] == 0  # multi-sentence chunks are never removed


def test_more_hypothesis_verbs_are_tidied() -> None:
    from cyberlab_manuscript.builder import tidy_prose

    assert (
        tidy_prose("corroborating H1 (supported) and H2 (supported).") == "corroborating H1 and H2."
    )
