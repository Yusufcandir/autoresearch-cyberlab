"""Server-rendered PI dashboard (spec 10.3). Jinja autoescaping is on: research text is never HTML."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select

from cyberlab_api.deps import get_lab
from cyberlab_api.progress import (
    build_progress,
    friendly_activity,
    render_manuscript,
    trust_summary,
)
from cyberlab_authz.rbac import (
    AuthError,
    Forbidden,
    NotFoundOrForbidden,
    Principal,
    accessible_projects,
    authenticate,
    create_user,
    issue_token,
    project_role,
    require_project_role,
)
from cyberlab_common.db import (
    AgentNote,
    ArtifactRecord,
    AuditEvent,
    CitationRecord,
    DatasetRecord,
    JobRecord,
    ManuscriptRecord,
    Project,
    ProtocolVersion,
    RunRecord,
    User,
)
from cyberlab_orchestrator.lab import GateError, Lab
from cyberlab_schemas import ResearchSpec

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _aslist(value: Any) -> list[Any]:
    """LLMs sometimes return a string or dict where a list was asked for; never iterate characters."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    return str(value)


templates.env.filters["aslist"] = _aslist
templates.env.filters["text"] = _text
COOKIE = "cyberlab_session"


def lab_dep(lab: Lab = Depends(get_lab)) -> Lab:
    return lab


def _secret(lab: Lab) -> str:
    key = lab.settings.secret_key
    if key is None:
        raise AuthError("SECRET_KEY not configured")
    return str(key.get_secret_value())


def csrf_for(lab: Lab, token: str) -> str:
    return hmac.new(_secret(lab).encode(), ("csrf:" + token).encode(), hashlib.sha256).hexdigest()[
        :32
    ]


def web_user(request: Request, lab: Lab = Depends(lab_dep)) -> Principal | None:
    from cyberlab_authz.rbac import principal_from_token

    token = request.cookies.get(COOKIE)
    if not token:
        return None
    try:
        with lab.db.session() as s:
            return principal_from_token(s, token, _secret(lab))
    except AuthError:
        return None


def check_csrf(request: Request, csrf: str, lab: Lab) -> None:
    token = request.cookies.get(COOKIE, "")
    if not token or not hmac.compare_digest(csrf_for(lab, token), csrf):
        raise Forbidden("invalid form token; reload the page")


def render(request: Request, lab: Lab, name: str, me: Principal | None, **ctx: Any) -> HTMLResponse:
    token = request.cookies.get(COOKIE, "")
    return templates.TemplateResponse(
        request, name, {"me": me, "csrf": csrf_for(lab, token) if token else "", **ctx}
    )


def _login_redirect() -> RedirectResponse:
    return RedirectResponse("/login", status_code=303)


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, lab: Lab = Depends(lab_dep)) -> HTMLResponse:
    return render(request, lab, "login.html", None, error=None)


@router.post("/login")
def login_post(
    request: Request, username: str = Form(), password: str = Form(), lab: Lab = Depends(lab_dep)
) -> Response:
    try:
        with lab.db.session() as s:
            p = authenticate(s, username, password)
    except AuthError:
        return render(request, lab, "login.html", None, error="Wrong username or password.")
    resp = RedirectResponse("/", status_code=303)
    resp.set_cookie(
        COOKIE, issue_token(p.id, _secret(lab)), httponly=True, samesite="strict", max_age=12 * 3600
    )
    return resp


@router.get("/logout")
def logout() -> RedirectResponse:
    resp = _login_redirect()
    resp.delete_cookie(COOKIE)
    return resp


@router.get("/", response_class=HTMLResponse)
def portfolio(
    request: Request, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    if me is None:
        return _login_redirect()
    rows: list[dict[str, Any]] = []
    with lab.db.session() as s:
        for p in accessible_projects(s, me):
            jobs = dict(
                s.execute(
                    select(JobRecord.state, func.count())
                    .where(JobRecord.project_id == p.id)
                    .group_by(JobRecord.state)
                ).all()
            )
            rows.append({"p": p, "jobs": jobs})
    pend = {r["p"].id: lab.pending(r["p"].id) for r in rows}
    return render(request, lab, "portfolio.html", me, rows=rows, pending=pend)


@router.get("/projects/new", response_class=HTMLResponse)
def new_project(
    request: Request, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    if me is None:
        return _login_redirect()
    return render(request, lab, "new_project.html", me, error=None)


@router.post("/projects/new")
def new_project_post(
    request: Request,
    bg: BackgroundTasks,
    title: str = Form(),
    goal: str = Form(),
    constraints: str = Form(""),
    data_may_leave_host: str = Form("no"),
    gpu_budget_minutes: int = Form(120),
    llm_token_budget: int = Form(200000),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    try:
        spec = ResearchSpec(
            title=title,
            goal=goal,
            constraints=[c.strip() for c in constraints.splitlines() if c.strip()],
            data_may_leave_host=data_may_leave_host == "yes",
            gpu_budget_minutes=gpu_budget_minutes,
            llm_token_budget=llm_token_budget,
        )
    except ValueError as exc:
        return render(request, lab, "new_project.html", me, error=str(exc))
    pid = lab.create_project(me, spec)
    bg.add_task(_safe, lab.start, pid)
    return RedirectResponse(f"/projects/{pid}", status_code=303)


def _safe(fn: Any, *args: Any) -> None:
    try:
        fn(*args)
    except Exception:  # noqa: BLE001 - stored on the project by Lab._invoke
        logging.getLogger(__name__).exception("study step failed")


@router.get("/projects/{pid}", response_class=HTMLResponse)
def project_page(
    pid: str,
    request: Request,
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    with lab.db.session() as s:
        try:
            project = require_project_role(s, me, pid)
        except NotFoundOrForbidden:
            return HTMLResponse("Not found", status_code=404)
        role = project_role(s, me, pid)
        protocol = s.scalar(
            select(ProtocolVersion)
            .where(ProtocolVersion.project_id == pid)
            .order_by(ProtocolVersion.version.desc())
        )
        cites = list(
            s.scalars(
                select(CitationRecord)
                .where(CitationRecord.project_id == pid)
                .order_by(CitationRecord.status.desc())
            )
        )
        notes = {
            n.kind: n.content
            for n in s.scalars(
                select(AgentNote).where(AgentNote.project_id == pid).order_by(AgentNote.id)
            )
        }
        jobs = list(
            s.scalars(
                select(JobRecord).where(JobRecord.project_id == pid).order_by(JobRecord.created_at)
            )
        )
        runs = {
            r.job_id: r for r in s.scalars(select(RunRecord).where(RunRecord.project_id == pid))
        }
        mss = list(
            s.scalars(
                select(ManuscriptRecord)
                .where(ManuscriptRecord.project_id == pid)
                .order_by(ManuscriptRecord.version.desc())
            )
        )
        dataset = s.get(DatasetRecord, protocol.body["dataset_id"]) if protocol else None
        analysis_rec = s.scalar(
            select(ArtifactRecord)
            .where(ArtifactRecord.project_id == pid, ArtifactRecord.name == "analysis.json")
            .order_by(ArtifactRecord.id.desc())
        )
        analysis = (
            json.loads(Path(analysis_rec.path).read_text())
            if analysis_rec and Path(analysis_rec.path).exists()
            else None
        )
        if analysis is not None and analysis_rec is not None:
            analysis["artifact_sha256"] = analysis_rec.sha256
        events = list(
            s.scalars(
                select(AuditEvent).where(AuditEvent.project_id == pid).order_by(AuditEvent.id)
            )
        )
        artifact_check = lab.ctx.store.verify_all(s, pid)
        status = lab.status(pid)
    latest_ms = mss[0] if mss else None
    verified = [c for c in cites if c.status == "Verified"]
    rendered, cited_in = (
        render_manuscript(latest_ms.markdown, [c.key for c in verified]) if latest_ms else ("", {})
    )
    steps, now = build_progress(events, status["stage"], status["pending"], jobs)
    trust = trust_summary(cites, notes.get("evidence_matrix"), analysis, latest_ms, artifact_check)
    best = (
        max(
            analysis["models"], key=lambda n: analysis["models"][n]["aggregate"]["macro_f1"]["mean"]
        )
        if analysis and analysis.get("models")
        else None
    )
    job_counts: dict[str, int] = {}
    for j in jobs:
        job_counts[j.state] = job_counts.get(j.state, 0) + 1
    return render(
        request,
        lab,
        "project.html",
        me,
        project=project,
        role=role,
        status=status,
        protocol=protocol,
        cites=cites,
        verified=verified,
        cited_in=cited_in,
        notes=notes,
        jobs=jobs,
        job_counts=job_counts,
        runs=runs,
        manuscripts=mss,
        rendered=rendered,
        dataset=dataset,
        analysis=analysis,
        best=best,
        activity=friendly_activity(events),
        steps=steps,
        now=now,
        trust=trust,
        stamp=events[-1].id if events else 0,
    )


@router.get("/projects/{pid}/live")
def project_live(
    pid: str, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> dict[str, Any]:
    """Tiny poll target: the page reloads its content only when something changed."""
    if me is None:
        raise AuthError("not signed in")
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        last = s.scalar(
            select(AuditEvent.id)
            .where(AuditEvent.project_id == pid)
            .order_by(AuditEvent.id.desc())
            .limit(1)
        )
        running = s.scalar(
            select(JobRecord.updated_at)
            .where(JobRecord.project_id == pid)
            .order_by(JobRecord.updated_at.desc())
            .limit(1)
        )
    return {"stamp": f"{last or 0}:{running.isoformat() if running else ''}"}


@router.post("/projects/{pid}/approve")
def approve_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    version: int = Form(),
    sha256: str = Form(),
    decision: str = Form(),
    comment: str = Form(""),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    if decision not in {"approved", "rejected"}:
        raise GateError("bad decision")
    with lab.db.session() as s:
        require_project_role(s, me, pid, "pi")
    pend = lab.pending(pid)
    if not pend or pend.get("gate") != "A" or pend.get("protocol_sha256") != sha256:
        raise GateError("stale approval: the protocol changed; reload the page")
    bg.add_task(_safe, lab.approve_protocol, me, pid, version, sha256, decision, comment)
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/release")
def release_post(
    pid: str,
    request: Request,
    version: int = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    lab.release(me, pid, version)
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/pause")
def pause_post(
    pid: str,
    request: Request,
    paused: str = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    lab.pause(me, pid, paused == "yes")
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/redraft")
def redraft_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    with lab.db.session() as s:
        require_project_role(s, me, pid, "researcher")
    bg.add_task(_safe, lab.redraft, me, pid)
    return RedirectResponse(f"/projects/{pid}#manuscript", status_code=303)


@router.post("/projects/{pid}/retry")
def retry_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    with lab.db.session() as s:
        require_project_role(s, me, pid, "researcher")
    bg.add_task(_safe, lab.start, pid)
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/jobs/{job_id}/cancel")
def cancel_post(
    job_id: str,
    request: Request,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    with lab.db.session() as s:
        j = s.get(JobRecord, job_id)
        if j is None:
            raise NotFoundOrForbidden(job_id)
        require_project_role(s, me, j.project_id, "researcher")
        lab.ctx.broker.cancel(s, job_id, me.username)
        pid = j.project_id
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.get("/manuscripts/{mid}/download/{kind}")
def download(
    mid: str, kind: str, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    if me is None:
        return _login_redirect()
    with lab.db.session() as s:
        m = s.get(ManuscriptRecord, mid)
        if m is None:
            raise NotFoundOrForbidden(mid)
        require_project_role(s, me, m.project_id)
        body, ext = (m.markdown, "md") if kind == "md" else (m.bibtex, "bib")
        name = f"manuscript-v{m.version}.{ext}"
    return Response(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/admin", response_class=HTMLResponse)
def admin(
    request: Request, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    if me is None:
        return _login_redirect()
    if not me.is_admin:
        return HTMLResponse("Admins only", status_code=403)
    with lab.db.session() as s:
        users = list(s.scalars(select(User).order_by(User.created_at)))
        jobs = list(s.scalars(select(JobRecord).order_by(JobRecord.updated_at.desc()).limit(50)))
        denials = list(
            s.scalars(
                select(AuditEvent)
                .where(AuditEvent.action == "security.policy_denied")
                .order_by(AuditEvent.id.desc())
                .limit(20)
            )
        )
        projects = {p.id: p.title for p in s.scalars(select(Project))}
    return render(
        request,
        lab,
        "admin.html",
        me,
        users=users,
        jobs=jobs,
        denials=denials,
        projects=projects,
        lock=lab.ctx.broker.lock.owner(),
        settings=lab.settings,
        llm_log=lab.ctx.llm.log[-20:],
    )


@router.post("/admin/users")
def admin_add_user(
    request: Request,
    username: str = Form(),
    password: str = Form(),
    role: str = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None or not me.is_admin:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    with lab.db.session() as s:
        create_user(s, username, password, role)
    return RedirectResponse("/admin", status_code=303)
