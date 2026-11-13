"""API and dashboard: authentication, project isolation (AT-02), gates and pages."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cyberlab_api.main import app, get_lab, get_settings
from cyberlab_authz.rbac import create_user
from cyberlab_orchestrator.lab import Lab
from tests.fixtures.fakes import fake_gateway, fake_scholar, make_settings

PW = "correct-horse-battery"


@pytest.fixture
def client(tmp_path: Path) -> Iterator[tuple[TestClient, Lab]]:
    settings = make_settings(tmp_path)
    lab = Lab(settings, llm=fake_gateway(), scholar=fake_scholar())
    with lab.db.session() as s:
        create_user(s, "prof", PW, "pi")
        create_user(s, "mallory", PW, "researcher")
        create_user(s, "root", PW, "admin")
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_lab] = lambda: lab
    with TestClient(app) as c:
        yield c, lab
    app.dependency_overrides.clear()
    lab.close()


def token(c: TestClient, user: str) -> dict[str, str]:
    r = c.post("/api/v1/auth/token", data={"username": user, "password": PW})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_auth_required(client: tuple[TestClient, Lab]) -> None:
    c, _ = client
    assert c.get("/api/v1/projects").status_code == 401
    assert (
        c.post(
            "/api/v1/auth/token", data={"username": "prof", "password": "wrong-password"}
        ).status_code
        == 401
    )


def test_project_flow_and_isolation(client: tuple[TestClient, Lab]) -> None:
    c, lab = client
    prof, mallory = token(c, "prof"), token(c, "mallory")
    r = c.post(
        "/api/v1/projects",
        headers=prof,
        json={
            "title": "IDS comparison on CIC-IDS2017",
            "goal": "Compare models for intrusion detection on CIC-IDS2017.",
        },
    )
    assert r.status_code == 201, r.text
    pid = r.json()["project_id"]  # background task already ran inside TestClient
    detail = c.get(f"/api/v1/projects/{pid}", headers=prof).json()
    assert detail["pending"]["gate"] == "A"

    # AT-02: another user cannot see or infer the study
    assert c.get(f"/api/v1/projects/{pid}", headers=mallory).status_code == 404
    assert c.get(f"/api/v1/projects/{pid}/citations", headers=mallory).status_code == 404
    assert c.get("/api/v1/projects", headers=mallory).json() == []
    assert c.get(f"/api/v1/audit?project_id={pid}", headers=mallory).status_code == 404

    protos = c.get(f"/api/v1/projects/{pid}/protocols", headers=prof).json()
    p = protos[-1]
    stale = c.post(
        f"/api/v1/protocols/{p['id']}/approve",
        headers=prof,
        json={
            "expected_version": p["version"],
            "expected_sha256": "f" * 64,
            "decision": "approved",
        },
    )
    assert stale.status_code == 409
    assert (
        c.post(
            f"/api/v1/protocols/{p['id']}/approve",
            headers=mallory,
            json={
                "expected_version": p["version"],
                "expected_sha256": p["sha256"],
                "decision": "approved",
            },
        ).status_code
        == 404
    )
    ok = c.post(
        f"/api/v1/protocols/{p['id']}/approve",
        headers=prof,
        json={
            "expected_version": p["version"],
            "expected_sha256": p["sha256"],
            "decision": "approved",
        },
    )
    assert ok.status_code == 200
    jobs = c.get(f"/api/v1/projects/{pid}/jobs", headers=prof).json()
    assert len(jobs) == 12 and all(j["state"] == "queued" for j in jobs)
    cancel = c.post(f"/api/v1/jobs/{jobs[0]['id']}/cancel", headers=prof)
    assert cancel.json()["state"] == "canceled"
    assert (
        c.get("/api/v1/datasets/cicids2017-mlcsv-v1", headers=prof).json()["status"] == "verified"
    )


def test_dashboard_pages(client: tuple[TestClient, Lab]) -> None:
    c, _ = client
    assert c.get("/", follow_redirects=False).status_code == 303
    r = c.post("/login", data={"username": "root", "password": PW}, follow_redirects=False)
    assert r.status_code == 303
    assert "Studies" in c.get("/").text
    assert "New research study" in c.get("/projects/new").text
    assert "Admin console" in c.get("/admin").text
    form = c.get("/projects/new").text
    csrf = form.split('name="csrf" value="')[1].split('"')[0]
    bad = c.post("/projects/new", data={"title": "A valid title", "goal": "g" * 30, "csrf": "nope"})
    assert bad.status_code == 403
    r = c.post(
        "/projects/new",
        data={
            "title": "A valid title here",
            "goal": "Compare IDS models on CIC-IDS2017 data.",
            "csrf": csrf,
        },
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = c.get(r.headers["location"])
    assert page.status_code == 200 and "Approve protocol" in page.text
