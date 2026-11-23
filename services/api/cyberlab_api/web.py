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
    overall_progress,
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
    get_setting,
    set_setting,
)
from cyberlab_data import ADAPTERS, dataset_record_id, get_adapter
from cyberlab_llm.gateway import CLAUDE_MODELS, CLAUDE_PRICES
from cyberlab_orchestrator.lab import GateError, Lab
from cyberlab_schemas import ResearchSpec, SplitSpec

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
    return render(request, lab, "portfolio.html", me, rows=rows, pending=pend, **_model_ctx(lab))


def _model_ctx(lab: Lab) -> dict[str, Any]:
    chosen = get_setting(lab.db, "claude_model")
    return {
        "claude_on": (lab.settings.llm_provider or "").lower() == "claude",
        "claude_models": CLAUDE_MODELS,
        "claude_current": chosen if chosen in CLAUDE_PRICES else lab.settings.claude_model,
    }


@router.post("/settings/model")
def model_post(
    request: Request,
    model: str = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """Dashboard model switch (admin or PI). Applies to the next Claude call of every study."""
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    if me.role not in ("admin", "pi"):
        return _message("Not allowed", "Only an admin or PI can change the model.", "/")
    if model not in dict(CLAUDE_MODELS):
        return _message("Unknown model", f"{model!r} is not one of the offered models.", "/")
    set_setting(lab.db, "claude_model", model, me.username)
    return RedirectResponse("/", status_code=303)


@router.get("/projects/new", response_class=HTMLResponse)
def new_project(
    request: Request, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    if me is None:
        return _login_redirect()
    return render(request, lab, "new_project.html", me, error=None, datasets=_dataset_choices(lab))


def _dataset_choices(lab: Lab) -> list[dict[str, Any]]:
    """Registered datasets, and whether their files are on this server yet."""
    out = []
    for a in ADAPTERS.values():
        raw = a.root(lab.settings.dataset_root) / "raw"
        out.append(
            {
                "value": a.display,
                "label": a.info.name,
                "ready": raw.exists() and any(raw.glob("*.csv")),
            }
        )
    return out


@router.post("/projects/new")
def new_project_post(
    request: Request,
    bg: BackgroundTasks,
    title: str = Form(),
    goal: str = Form(),
    constraints: str = Form(""),
    data_may_leave_host: str = Form("no"),
    dataset: str = Form("CIC-IDS2017"),
    gpu_budget_minutes: int = Form(120),
    llm_token_budget: int = Form(200000),
    llm_usd_budget: float = Form(10.0),
    deadline: str = Form(""),
    output_format: str = Form("latex"),
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
            dataset_hint=get_adapter(dataset).display,
            gpu_budget_minutes=gpu_budget_minutes,
            llm_token_budget=llm_token_budget,
            llm_usd_budget=llm_usd_budget,
            deadline=deadline or None,
            output_format=output_format,
        )
    except ValueError as exc:
        # keep what the user typed so a validation error does not wipe the form
        form = {
            "title": title,
            "goal": goal,
            "constraints": constraints,
            "data_may_leave_host": data_may_leave_host,
            "dataset": dataset,
            "gpu_budget_minutes": gpu_budget_minutes,
            "llm_token_budget": llm_token_budget,
            "llm_usd_budget": llm_usd_budget,
            "deadline": deadline,
            "output_format": output_format,
        }
        return render(
            request,
            lab,
            "new_project.html",
            me,
            error=str(exc),
            form=form,
            datasets=_dataset_choices(lab),
        )
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
        dataset = (
            s.get(
                DatasetRecord,
                dataset_record_id(
                    protocol.body["dataset_id"], SplitSpec.model_validate(protocol.body["split"])
                ),
            )
            or s.get(DatasetRecord, protocol.body["dataset_id"])
            if protocol
            else None
        )
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
        decisions=lab.research_decisions(pid),
        validation=_latest_note(lab, pid, "scientific_validation"),
        protocol_check=_latest_note(lab, pid, "protocol_validation"),
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
        progress=overall_progress(steps, jobs, runs, status["stage"]),
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
    return {"stamp": _live_stamp(lab, pid)}


def _live_stamp(lab: Lab, pid: str) -> str:
    with lab.db.session() as s:
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
    return f"{last or 0}:{running.isoformat() if running else ''}"


@router.get("/projects/{pid}/events")
def project_events(
    pid: str, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    """Server-sent events (spec 10.1): the page is told when the study changed, instead of
    polling. The stream ends after ten minutes; the browser reconnects on its own."""
    import time

    from fastapi.responses import StreamingResponse

    if me is None:
        raise AuthError("not signed in")
    with lab.db.session() as s:
        require_project_role(s, me, pid)

    def stream() -> Any:
        last = ""
        end = time.monotonic() + 600
        beat = 0.0
        while time.monotonic() < end:
            stamp = _live_stamp(lab, pid)
            if stamp != last:
                last = stamp
                yield f"event: change\ndata: {stamp}\n\n"
            elif time.monotonic() - beat > 15:
                yield ": keep-alive\n\n"
                beat = time.monotonic()
            time.sleep(2)

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
    )


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
    check_science: str = Form(""),
    check_ethics: str = Form(""),
    check_security: str = Form(""),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    checklist = {
        "science": check_science == "on",
        "ethics": check_ethics == "on",
        "security": check_security == "on",
    }
    try:
        lab.release(me, pid, version, checklist)
    except GateError as exc:
        return _message("Release not possible", str(exc), f"/projects/{pid}")
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/pause")
def pause_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    paused: str = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """Pause: nothing new starts (no experiment, no AI step); running experiments finish.
    Resume: continue from where the study was held (also after Stop)."""
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    if paused == "yes":
        lab.pause(me, pid, True)
    else:
        with lab.db.session() as s:
            require_project_role(s, me, pid, "researcher")
        bg.add_task(_safe, lab.resume, me, pid)
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/stop")
def stop_post(
    pid: str,
    request: Request,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """Stop now: running and queued experiments are canceled; Resume runs them again."""
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    try:
        lab.stop(me, pid)
    except GateError as exc:
        return _message("Cannot stop", str(exc), f"/projects/{pid}")
    return RedirectResponse(f"/projects/{pid}", status_code=303)


@router.post("/projects/{pid}/restart")
def restart_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """Run the whole study again from the first step (PI only)."""
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    with lab.db.session() as s:
        p = require_project_role(s, me, pid, "pi")
        if p.stage == "released":
            return _message(
                "Cannot restart", "A released study cannot be restarted.", f"/projects/{pid}"
            )
    bg.add_task(_safe, lab.restart, me, pid)
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


@router.post("/projects/{pid}/budget")
def budget_post(
    pid: str,
    request: Request,
    bg: BackgroundTasks,
    extra: int = Form(),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    lab.raise_budget(me, pid, extra)
    bg.add_task(_safe, lab.start, pid)  # continue from the last checkpoint right away
    return RedirectResponse(f"/projects/{pid}", status_code=303)


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
        if kind == "tex":
            # LaTeX bundle: .tex + .bib + the figure PNGs, ready to compile
            import io
            import zipfile

            from cyberlab_manuscript.figures import extract_figures
            from cyberlab_manuscript.latex import markdown_to_latex

            stem = f"manuscript-v{m.version}"
            md_files, figs = extract_figures(m.markdown)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(f"{stem}.tex", markdown_to_latex(md_files, stem))
                z.writestr(f"{stem}.bib", m.bibtex or "")
                for fname, data in figs.items():
                    z.writestr(fname, data)
            return Response(
                buf.getvalue(),
                media_type="application/zip",
                headers={"Content-Disposition": f'attachment; filename="{stem}-latex.zip"'},
            )
        if kind == "docx":
            from cyberlab_manuscript.docx_export import markdown_to_docx

            return Response(
                markdown_to_docx(m.markdown),
                media_type=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
                headers={
                    "Content-Disposition": f'attachment; filename="manuscript-v{m.version}.docx"'
                },
            )
        if kind == "bib":
            body, ext = m.bibtex, "bib"
        else:
            body, ext = m.markdown, "md"
        name = f"manuscript-v{m.version}.{ext}"
    return Response(
        body,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@router.get("/manuscripts/{mid}/print", response_class=HTMLResponse)
def print_view(
    mid: str, me: Principal | None = Depends(web_user), lab: Lab = Depends(lab_dep)
) -> Response:
    """Print-ready page: the browser's 'Save as PDF' produces the PDF rendering (FR-009)."""
    if me is None:
        return _login_redirect()
    with lab.db.session() as s:
        m = s.get(ManuscriptRecord, mid)
        if m is None:
            raise NotFoundOrForbidden(mid)
        require_project_role(s, me, m.project_id)
        keys = [
            c.key
            for c in s.scalars(
                select(CitationRecord).where(
                    CitationRecord.project_id == m.project_id, CitationRecord.status == "Verified"
                )
            )
        ]
        html, _ = render_manuscript(m.markdown, keys)
        version = m.version
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Manuscript v{version}</title>
<style>
body{{font-family:Georgia,'Times New Roman',serif;max-width:46rem;margin:2rem auto;padding:0 1rem;
line-height:1.5;color:#111;background:#fff}}
h1{{font-size:1.6rem}} h2{{font-size:1.25rem;margin-top:1.6rem}} h3{{font-size:1.05rem}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;margin:.8rem 0}}
th,td{{border-bottom:1px solid #bbb;padding:.25rem .4rem;text-align:left}}
blockquote{{color:#555;border-left:3px solid #ccc;margin:0;padding-left:.8rem}}
.bar{{font-family:system-ui,sans-serif;margin-bottom:1rem}}
@media print{{.bar{{display:none}} body{{margin:0;max-width:none}} h2{{break-after:avoid}}
table,tr{{break-inside:avoid}}}}
</style></head><body>
<div class="bar"><button onclick="window.print()">Save as PDF</button>
<a href="javascript:history.back()">Back</a></div>
{html}
</body></html>"""
    return HTMLResponse(page)


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
        deleted=lab.deleted_projects(me),
    )


def _message(title: str, text: str, back: str) -> HTMLResponse:
    import html as _html

    return HTMLResponse(
        f"<!doctype html><meta charset='utf-8'><title>{_html.escape(title)}</title>"
        "<body style='font-family:system-ui;max-width:40rem;margin:4rem auto;padding:0 1rem'>"
        f"<h1>{_html.escape(title)}</h1><p>{_html.escape(text)}</p>"
        f"<p><a href='{_html.escape(back)}'>Back</a></p></body>",
        status_code=409,
    )


@router.post("/projects/{pid}/delete")
def delete_post(
    pid: str,
    request: Request,
    confirm_title: str = Form(""),
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    try:
        lab.delete_project(me, pid, confirm_title)
    except GateError as exc:
        return _message("Study not deleted", str(exc), f"/projects/{pid}")
    return RedirectResponse("/", status_code=303)


@router.post("/admin/projects/{pid}/restore")
def restore_post(
    pid: str,
    request: Request,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    try:
        lab.restore_project(me, pid)
    except (GateError, Forbidden) as exc:
        return _message("Study not restored", str(exc), "/admin")
    return RedirectResponse("/admin", status_code=303)


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


@router.get("/projects/{pid}/diff/{kind}", response_class=HTMLResponse)
def diff_view(
    pid: str,
    kind: str,
    a: int | None = None,
    b: int | None = None,
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """Version diff before approval or release (spec Step 22: version-diff approval)."""
    import difflib
    import re as _re

    if me is None:
        return _login_redirect()
    if kind not in ("protocol", "manuscript"):
        raise NotFoundOrForbidden(kind)
    with lab.db.session() as s:
        require_project_role(s, me, pid)
        if kind == "protocol":
            rows = {
                r.version: json.dumps(r.body, indent=2, sort_keys=True, default=str)
                for r in s.scalars(select(ProtocolVersion).where(ProtocolVersion.project_id == pid))
            }
        else:
            rows = {
                r.version: _re.sub(r"\(data:image/png;base64,[^)]+\)", "(figure)", r.markdown)
                for r in s.scalars(
                    select(ManuscriptRecord).where(ManuscriptRecord.project_id == pid)
                )
            }
    versions = sorted(rows)
    if len(versions) < 2:
        return _message(
            "Nothing to compare", f"This study has only one {kind} version.", f"/projects/{pid}"
        )
    b = b if b in rows else versions[-1]
    a = (
        a
        if a in rows
        else versions[versions.index(b) - 1]
        if versions.index(b) > 0
        else versions[0]
    )
    table = difflib.HtmlDiff(wrapcolumn=90).make_table(
        rows[a].splitlines(), rows[b].splitlines(), f"version {a}", f"version {b}", context=True
    )
    return HTMLResponse(
        "<!doctype html><meta charset='utf-8'><title>Compare versions</title>"
        "<style>body{font-family:system-ui;margin:1.5rem}table.diff{font-family:monospace;"
        "font-size:12px;border-collapse:collapse}.diff_add{background:#d7f5dd}"
        ".diff_chg{background:#fff3c4}.diff_sub{background:#ffd8d8}td{padding:0 .3rem}</style>"
        f"<h1>{kind.title()}: version {a} versus {b}</h1>"
        f"<p><a href='/projects/{pid}'>Back to the study</a></p>{table}"
    )


def _latest_note(lab: Lab, pid: str, kind: str) -> dict[str, Any]:
    from cyberlab_common.db import AgentNote

    with lab.db.session() as s:
        n = s.scalar(
            select(AgentNote)
            .where(AgentNote.project_id == pid, AgentNote.kind == kind)
            .order_by(AgentNote.id.desc())
        )
        return dict(n.content) if n else {}


@router.post("/projects/{pid}/followup/{note_id}")
def followup_post(
    pid: str,
    note_id: int,
    request: Request,
    bg: BackgroundTasks,
    csrf: str = Form(),
    me: Principal | None = Depends(web_user),
    lab: Lab = Depends(lab_dep),
) -> Response:
    """PI approves a feedback-controller decision: a new protocol version is drafted and waits
    for gate A (nothing runs before that approval)."""
    if me is None:
        return _login_redirect()
    check_csrf(request, csrf, lab)
    try:
        lab.start_followup(me, pid, note_id)
    except GateError as exc:
        return _message("Follow-up not possible", str(exc), f"/projects/{pid}")
    return RedirectResponse(f"/projects/{pid}#protocol", status_code=303)
