"""Deleting a study (NFR-009): PI only, typed confirmation, released studies protected,
records kept, admin restore."""

from __future__ import annotations

from pathlib import Path

import pytest

from cyberlab_authz.rbac import (
    Forbidden,
    NotFoundOrForbidden,
    Principal,
    accessible_projects,
    create_user,
    require_project_role,
)
from cyberlab_common.db import AuditEvent, Project
from cyberlab_orchestrator.lab import GateError, Lab
from cyberlab_schemas import ResearchSpec
from tests.fixtures.fakes import fake_gateway, fake_scholar, make_settings

TITLE = "Rare attack detection study"


@pytest.fixture
def lab(tmp_path: Path):  # type: ignore[no-untyped-def]
    lab = Lab(make_settings(tmp_path), llm=fake_gateway(), scholar=fake_scholar())
    yield lab
    lab.close()


def _users(lab: Lab) -> tuple[Principal, Principal, Principal]:
    with lab.db.session() as s:
        out = [
            create_user(s, n, "correct-horse-battery", r)
            for n, r in (("prof", "pi"), ("res", "researcher"), ("root", "admin"))
        ]
        return tuple(Principal(u.id, u.username, u.role) for u in out)  # type: ignore[return-value]


def test_delete_hides_the_study_and_admin_can_restore(lab: Lab) -> None:
    pi, res, admin = _users(lab)
    pid = lab.create_project(
        pi, ResearchSpec(title=TITLE, goal="Detect rare attacks in CIC-IDS2017 data.")
    )
    with pytest.raises(GateError, match="type the study title"):
        lab.delete_project(pi, pid, "wrong title")
    with pytest.raises((Forbidden, NotFoundOrForbidden)):
        lab.delete_project(res, pid, TITLE)  # researchers cannot delete
    lab.delete_project(pi, pid, TITLE)
    with lab.db.session() as s:
        assert pid not in [p.id for p in accessible_projects(s, pi)]
        assert pid not in [p.id for p in accessible_projects(s, admin)]
        with pytest.raises(NotFoundOrForbidden):
            require_project_role(s, pi, pid)
        p = s.get(Project, pid)
        assert p is not None and p.deleted_by == "prof" and p.paused  # record kept
        assert any(
            e.action == "project.deleted" for e in s.query(AuditEvent).filter_by(project_id=pid)
        )
    assert [d["id"] for d in lab.deleted_projects(admin)] == [pid]
    with pytest.raises(Forbidden):
        lab.restore_project(pi, pid)
    lab.restore_project(admin, pid)
    with lab.db.session() as s:
        assert pid in [p.id for p in accessible_projects(s, pi)]


def test_released_study_cannot_be_deleted(lab: Lab) -> None:
    pi, _, _ = _users(lab)
    pid = lab.create_project(
        pi, ResearchSpec(title=TITLE, goal="Detect rare attacks in CIC-IDS2017 data.")
    )
    with lab.db.session() as s:
        s.get(Project, pid).stage = "released"  # type: ignore[union-attr]
    with pytest.raises(GateError, match="protected record"):
        lab.delete_project(pi, pid, TITLE)
