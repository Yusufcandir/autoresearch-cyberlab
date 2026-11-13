"""REST API (spec section 10.1) and PI dashboard (section 10.3).

Owner and permissions always come from the token, never from request fields. Non-members get
404 for other studies (no existence leak). Mutating study actions run in the background so the
browser never waits on an LLM call; progress is visible through SSE and the dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import BackgroundTasks, Depends, FastAPI, Form, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from cyberlab_api.deps import get_lab, get_settings
from cyberlab_authz.rbac import (
    AuthError,
    Forbidden,
    NotFoundOrForbidden,
    Principal,
    accessible_projects,
    authenticate,
    create_user,
    issue_token,
    principal_from_token,
    require_project_role,
)
from cyberlab_common.config import Settings
from cyberlab_common.db import (
    AgentNote,
    AuditEvent,
    CitationRecord,
    DatasetRecord,
    JobRecord,
    ManuscriptRecord,
    ProtocolVersion,
    RunRecord,
)
from cyberlab_orchestrator.lab import GateError, Lab
from cyberlab_schemas import ResearchSpec

app = FastAPI(title="AutoResearch-CyberLab API", version="1.0.0")


SettingsDep = Annotated[Settings, Depends(get_settings)]
LabDep = Annotated[Lab, Depends(get_lab)]


# --------------------------------------------------------------------------- errors
def problem(status: int, title: str, detail: str) -> JSONResponse:
    return JSONResponse(
        {"type": "about:blank", "title": title, "status": status, "detail": detail},
        status_code=status,
        media_type="application/problem+json",
    )


@app.exception_handler(NotFoundOrForbidden)
def _nf(_r: Request, _e: Exception) -> JSONResponse:
    return problem(404, "Not found", "resource not found")


@app.exception_handler(Forbidden)
def _fb(_r: Request, e: Exception) -> JSONResponse:
    return problem(403, "Forbidden", str(e))


@app.exception_handler(AuthError)
def _au(_r: Request, e: Exception) -> JSONResponse:
    return problem(401, "Unauthorized", str(e))


@app.exception_handler(GateError)
def _ga(_r: Request, e: Exception) -> JSONResponse:
    return problem(409, "Conflict", str(e))


@app.exception_handler(ValidationError)
def _va(_r: Request, e: ValidationError) -> JSONResponse:
    return problem(422, "Invalid input", str(e.errors()[:5]))


# --------------------------------------------------------------------------- auth
def current_principal(request: Request, lab: LabDep, settings: SettingsDep) -> Principal:
    token = None
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    token = token or request.cookies.get("cyberlab_session")
    if not token or settings.secret_key is None:
        raise AuthError("not signed in")
    with lab.db.session() as s:
        return principal_from_token(s, token, settings.secret_key.get_secret_value())


PrincipalDep = Annotated[Principal, Depends(current_principal)]


@app.post("/api/v1/auth/token")
def login(
    lab: LabDep, settings: SettingsDep, username: str = Form(), password: str = Form()
) -> dict[str, str]:
    assert settings.secret_key is not None
    with lab.db.session() as s:
        p = authenticate(s, username, password)
    return {
        "access_token": issue_token(p.id, settings.secret_key.get_secret_value()),
        "token_type": "bearer",
    }


class NewUser(BaseModel):
    username: str = Field(pattern=r"^[a-zA-Z0-9_.-]{3,64}$")
    password: str = Field(min_length=10)
    role: str


@app.post("/api/v1/users", status_code=201)
def add_user(body: NewUser, me: PrincipalDep, lab: LabDep) -> dict[str, str]:
    if not me.is_admin:
        raise Forbidden("admin only")
    with lab.db.session() as s:
        u = create_user(s, body.username, body.password, body.role)
        return {"id": u.id, "username": u.username, "role": u.role}


# --------------------------------------------------------------------------- health
@app.get("/health/live")
def live() -> dict[str, str]:
    return {"status": "live"}


@app.get("/health/ready")
def ready(settings: SettingsDep) -> JSONResponse:
    missing = settings.missing_required()
    if missing:
        body: dict[str, Any] = {
            "type": "about:blank",
            "title": "Not ready",
            "status": 503,
            "detail": "Required configuration is missing; dependent features are blocked.",
            "missing": missing,
        }
        return JSONResponse(body, status_code=503, media_type="application/problem+json")
    return JSONResponse({"status": "ready"})


# --------------------------------------------------------------------------- projects
def _advance(lab: Lab, fn: Any, *args: Any) -> None:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - the stage and reason are stored by Lab._invoke
        logging.getLogger(__name__).exception("study step failed")


@app.post("/api/v1/projects", status_code=201)
def create_project(
    spec: ResearchSpec, me: PrincipalDep, lab: LabDep, bg: BackgroundTasks
) -> dict[str, Any]:
    pid = lab.create_project(me, spec)
    bg.add_task(_advance, lab, lab.start, pid)
    return {"project_id": pid, "status": "started"}


@app.get("/api/v1/projects")
def list_projects(me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        projects = accessible_projects(s, me)
        return [
            {
                "id": p.id,
                "title": p.title,
                "stage": p.stage,
                "status": p.status_label,
                "blocked_reason": p.blocked_reason,
            }
            for p in projects
        ]


@app.get("/api/v1/projects/{pid}")
def get_project(pid: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
    return lab.status(pid)


@app.post("/api/v1/projects/{pid}/plan", status_code=202)
def plan(pid: str, me: PrincipalDep, lab: LabDep, bg: BackgroundTasks) -> dict[str, str]:
    with lab.db.session() as s:
        require_project_role(s, me, pid, "researcher")
    bg.add_task(_advance, lab, lab.start, pid)
    return {"status": "accepted"}


@app.post("/api/v1/projects/{pid}/pause")
def pause(pid: str, me: PrincipalDep, lab: LabDep, paused: bool = True) -> dict[str, Any]:
    lab.pause(me, pid, paused)
    return {"paused": paused}


@app.get("/api/v1/projects/{pid}/protocols")
def protocols(pid: str, me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        rows = s.scalars(
            select(ProtocolVersion)
            .where(ProtocolVersion.project_id == pid)
            .order_by(ProtocolVersion.version)
        )
        return [
            {
                "id": r.id,
                "version": r.version,
                "sha256": r.sha256,
                "body": r.body,
                "rationale": r.rationale,
                "created_at": r.created_at.isoformat(),
            }
            for r in rows
        ]


class ApprovalBody(BaseModel):
    expected_version: int
    expected_sha256: str
    decision: str = Field(pattern="^(approved|rejected)$")
    comment: str = ""


@app.post("/api/v1/protocols/{protocol_id}/approve")
def approve(
    protocol_id: str, body: ApprovalBody, me: PrincipalDep, lab: LabDep, bg: BackgroundTasks
) -> dict[str, str]:
    with lab.db.session() as s:
        pv = s.get(ProtocolVersion, protocol_id)
        if pv is None:
            raise NotFoundOrForbidden(protocol_id)
        require_project_role(s, me, pv.project_id, "pi")
        pid = pv.project_id
    pend = lab.pending(pid)
    if not pend or pend.get("gate") != "A":
        raise GateError("this study is not waiting for protocol approval")
    if (
        pend.get("protocol_version") != body.expected_version
        or pend.get("protocol_sha256") != body.expected_sha256
    ):
        raise GateError("stale approval: protocol version or hash does not match")
    bg.add_task(
        _advance,
        lab,
        lab.approve_protocol,
        me,
        pid,
        body.expected_version,
        body.expected_sha256,
        body.decision,
        body.comment,
    )
    return {"status": "accepted"}


@app.get("/api/v1/projects/{pid}/citations")
def citations(pid: str, me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        return [
            {"key": c.key, "doi": c.doi, "title": c.title, "year": c.year, "status": c.status}
            for c in s.scalars(select(CitationRecord).where(CitationRecord.project_id == pid))
        ]


@app.get("/api/v1/projects/{pid}/notes")
def notes(pid: str, me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        return [
            {
                "agent": n.agent,
                "kind": n.kind,
                "label": "Proposed",
                "content": n.content,
                "created_at": n.created_at.isoformat(),
            }
            for n in s.scalars(
                select(AgentNote).where(AgentNote.project_id == pid).order_by(AgentNote.id)
            )
        ]


# --------------------------------------------------------------------------- datasets
@app.get("/api/v1/datasets/{dataset_id}")
def dataset(dataset_id: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        d = s.get(DatasetRecord, dataset_id)
        if d is None:
            raise NotFoundOrForbidden(dataset_id)
        return {"id": d.id, "status": d.status, "manifest": d.manifest}


@app.post("/api/v1/datasets/verify", status_code=202)
def verify_dataset(me: PrincipalDep, lab: LabDep, bg: BackgroundTasks) -> dict[str, str]:
    """Re-run the CIC-IDS2017 gate on files already placed by an operator (never a shell download)."""
    if me.role not in {"admin", "pi", "researcher"}:
        raise Forbidden("researcher role required")
    from cyberlab_orchestrator.agents.dataset import dataset_agent

    bg.add_task(_advance, lab, lambda: dataset_agent(lab.ctx, {"project_id": "_registry"}))
    return {"status": "accepted"}


# --------------------------------------------------------------------------- jobs and evidence
def _job_dict(j: JobRecord) -> dict[str, Any]:
    return {
        "id": j.id,
        "experiment_id": j.experiment_id,
        "state": j.state,
        "external_ref": j.external_ref,
        "failure_reason": j.failure_reason,
        "attempts": j.attempts,
        "model": j.manifest["model"]["name"],
        "seed": j.manifest["seed"],
        "max_minutes": j.manifest["max_minutes"],
        "use_gpu": j.manifest["use_gpu"],
        "updated_at": j.updated_at.isoformat(),
    }


@app.get("/api/v1/projects/{pid}/jobs")
def project_jobs(pid: str, me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        return [
            _job_dict(j)
            for j in s.scalars(
                select(JobRecord).where(JobRecord.project_id == pid).order_by(JobRecord.created_at)
            )
        ]


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        j = s.get(JobRecord, job_id)
        if j is None:
            raise NotFoundOrForbidden(job_id)
        require_project_role(s, me, j.project_id)
        return _job_dict(j)


@app.post("/api/v1/jobs/{job_id}/cancel")
def cancel_job(job_id: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        j = s.get(JobRecord, job_id)
        if j is None:
            raise NotFoundOrForbidden(job_id)
        require_project_role(s, me, j.project_id, "researcher")
        j = lab.ctx.broker.cancel(s, job_id, me.username)
        return _job_dict(j)


@app.get("/api/v1/runs/{run_id}/evidence")
def run_evidence(run_id: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        r = s.get(RunRecord, run_id)
        if r is None:
            raise NotFoundOrForbidden(run_id)
        require_project_role(s, me, r.project_id)
        hashes = lab.ctx.store.verify_all(s, r.project_id)
        return {
            "run_id": r.id,
            "experiment_id": r.experiment_id,
            "status": r.status,
            "receipt": r.receipt,
            "metrics": r.metrics,
            "artifact_check": {
                k: hashes.get(v, "unknown") for k, v in r.receipt["artifacts"].items()
            },
        }


@app.post("/api/v1/projects/{pid}/analyze")
def analyze(pid: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    from cyberlab_qa.analysis import analyze_project

    with lab.db.session() as s:
        require_project_role(s, me, pid, "researcher")
        latest = s.scalar(
            select(ProtocolVersion)
            .where(ProtocolVersion.project_id == pid)
            .order_by(ProtocolVersion.version.desc())
        )
        if latest is None:
            raise GateError("no protocol yet")
        a = analyze_project(s, lab.ctx.store, pid, latest.version)
    return {
        "artifact_sha256": a["artifact_sha256"],
        "hypotheses": a["hypotheses"],
        "qa_flags": a["qa_flags"],
    }


# --------------------------------------------------------------------------- manuscripts
@app.get("/api/v1/projects/{pid}/manuscripts")
def manuscripts(pid: str, me: PrincipalDep, lab: LabDep) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        return [
            {
                "id": m.id,
                "version": m.version,
                "status": m.status,
                "sha256": m.sha256,
                "release_ready": m.evidence_report.get("release_ready"),
                "flags": m.evidence_report.get("flags"),
                "review": m.review,
            }
            for m in s.scalars(
                select(ManuscriptRecord)
                .where(ManuscriptRecord.project_id == pid)
                .order_by(ManuscriptRecord.version)
            )
        ]


@app.get("/api/v1/manuscripts/{mid}.md", response_class=PlainTextResponse)
def manuscript_md(mid: str, me: PrincipalDep, lab: LabDep) -> str:
    with lab.db.session() as s:
        m = s.get(ManuscriptRecord, mid)
        if m is None:
            raise NotFoundOrForbidden(mid)
        require_project_role(s, me, m.project_id)
        return m.markdown


@app.get("/api/v1/manuscripts/{mid}.bib", response_class=PlainTextResponse)
def manuscript_bib(mid: str, me: PrincipalDep, lab: LabDep) -> str:
    with lab.db.session() as s:
        m = s.get(ManuscriptRecord, mid)
        if m is None:
            raise NotFoundOrForbidden(mid)
        require_project_role(s, me, m.project_id)
        return m.bibtex


@app.post("/api/v1/projects/{pid}/manuscripts", status_code=202)
def manuscripts_create(
    pid: str, me: PrincipalDep, lab: LabDep, bg: BackgroundTasks
) -> dict[str, str]:
    """Drafting happens inside the workflow; this nudges a stalled study forward."""
    with lab.db.session() as s:
        require_project_role(s, me, pid, "researcher")
    bg.add_task(_advance, lab, lab.start, pid)
    return {"status": "accepted"}


@app.post("/api/v1/manuscripts/{mid}/release")
def release(mid: str, me: PrincipalDep, lab: LabDep) -> dict[str, Any]:
    with lab.db.session() as s:
        m = s.get(ManuscriptRecord, mid)
        if m is None:
            raise NotFoundOrForbidden(mid)
        require_project_role(s, me, m.project_id, "pi")
        pid, version = m.project_id, m.version
    return lab.release(me, pid, version)


# --------------------------------------------------------------------------- audit and events
@app.get("/api/v1/audit")
def audit_search(
    me: PrincipalDep, lab: LabDep, project_id: str | None = None, limit: int = 200
) -> list[dict[str, Any]]:
    with lab.db.session() as s:
        allowed = {p.id for p in accessible_projects(s, me)}
        q = select(AuditEvent).order_by(AuditEvent.id.desc()).limit(min(limit, 1000))
        if project_id:
            if project_id not in allowed:
                raise NotFoundOrForbidden(project_id)
            q = q.where(AuditEvent.project_id == project_id)
        rows = [e for e in s.scalars(q) if me.is_admin or e.project_id in allowed]
        return [
            {
                "ts": e.ts.isoformat(),
                "actor": e.actor,
                "project_id": e.project_id,
                "action": e.action,
                "detail": e.detail,
            }
            for e in rows
        ]


@app.get("/api/v1/projects/{pid}/events")
async def events(pid: str, me: PrincipalDep, lab: LabDep) -> StreamingResponse:
    """Server-sent events with the study's audit stream (spec 10.1: SSE instead of polling)."""
    with lab.db.session() as s:
        require_project_role(s, me, pid)

    async def gen() -> AsyncIterator[str]:
        last = 0
        for _ in range(3600):
            with lab.db.session() as s:
                rows = list(
                    s.scalars(
                        select(AuditEvent)
                        .where(AuditEvent.project_id == pid, AuditEvent.id > last)
                        .order_by(AuditEvent.id)
                    )
                )
            for e in rows:
                last = e.id
                yield f"event: {e.action}\ndata: {json.dumps({'ts': e.ts.isoformat(), 'actor': e.actor, 'detail': e.detail}, default=str)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(gen(), media_type="text/event-stream")


from cyberlab_api import web  # noqa: E402

__all__ = ["app", "get_lab", "get_settings"]

app.include_router(web.router)
