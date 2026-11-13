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

    with pytest.raises(Forbidden):
        lab.release(outsider_as_member(lab, pid, outsider), pid, mversion)
    final = lab.release(pi, pid, mversion)
    assert final["stage"] == "released"
    audit_lines = (lab.settings.data_dir.parent / "audit.jsonl").read_text().splitlines()
    assert any(json.loads(line)["code"] == "ALLOWED" for line in audit_lines)


def outsider_as_member(lab: Lab, pid: str, outsider: Principal) -> Principal:
    """Give the outsider a reviewer role: members below PI still cannot release (gate D)."""
    from cyberlab_authz.rbac import add_member

    with lab.db.session() as s:
        add_member(s, pid, outsider.id, "reviewer")
    return outsider
