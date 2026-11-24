"""Lab facade: the one entry point used by the API, the worker and the CLI."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from sqlalchemy import select

from cyberlab_authz.rbac import Forbidden, Principal, add_member, require_project_role
from cyberlab_common.config import Settings
from cyberlab_common.db import (
    ApprovalRecord,
    Database,
    JobRecord,
    ManuscriptRecord,
    Project,
    ProtocolVersion,
    audit,
    get_setting,
    now,
)
from cyberlab_evidence import ArtifactStore
from cyberlab_exec import Broker
from cyberlab_literature import ScholarlyClient
from cyberlab_llm import LLMGateway, build_gateway
from cyberlab_orchestrator.context import LabContext
from cyberlab_orchestrator.graph import build_graph
from cyberlab_schemas import ApprovalGate, ResearchSpec


class GateError(ValueError):
    """The requested gate action does not match the current state (stale or wrong gate)."""


# Release review checklist (spec Step 22): science, ethics and security.
REVIEW_ITEMS: dict[str, str] = {
    "science": "every claim and number checked against the Results and Evidence",
    "ethics": "dataset licences, citations and the AI-use statement checked",
    "security": "no private data, credentials or host details in the paper",
}


class Lab:
    def __init__(
        self,
        settings: Settings,
        llm: LLMGateway | None = None,
        scholar: ScholarlyClient | None = None,
        db: Database | None = None,
    ) -> None:
        if settings.database_url is None:
            raise RuntimeError("DATABASE_URL is not set")
        self.settings = settings
        self.db = db or Database(settings.database_url.get_secret_value())
        self.db.create_all()
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        store = ArtifactStore(settings.artifact_root)
        gateway = llm or build_gateway(settings)
        gateway.budget_lookup, gateway.charge = self._budget, self._charge
        gateway.charge_usd = self._charge_usd
        gateway.private_lookup = self._private
        gateway.paused_lookup = self._paused
        gateway.on_status = self._llm_status
        gateway.model_lookup = lambda: get_setting(self.db, "claude_model")
        self.ctx = LabContext(
            settings=settings,
            db=self.db,
            llm=gateway,
            scholar=scholar or ScholarlyClient(settings.literature_contact_email),
            broker=Broker(self.db, settings),
            store=store,
        )
        self._conn = sqlite3.connect(
            str(settings.data_dir / "checkpoints.sqlite"), check_same_thread=False
        )
        self.saver = SqliteSaver(self._conn)
        self.graph = build_graph(self.ctx, self.saver)
        self._locks: dict[str, threading.Lock] = {}
        self._pool = ThreadPoolExecutor(
            max_workers=settings.cpu_job_slots + 1, thread_name_prefix="job"
        )
        self._futures: dict[str, Future[None]] = {}
        self._gpu_job: dict[str, bool] = {}
        self._locks_guard = threading.Lock()

    # ----------------------------------------------------------------- budgets
    def _budget(self, project_id: str) -> tuple[int, int]:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                return 0, 0
            usd_limit = float(p.spec.get("llm_usd_budget", self.settings.llm_usd_budget) or 0)
            if usd_limit and (p.llm_usd_used or 0.0) >= usd_limit:
                return 1, 1  # dollar limit reached: the gateway refuses further paid calls
            return p.llm_tokens_used, int(p.spec.get("llm_token_budget", 0))

    def _llm_status(self, project_id: str, task: str, message: str) -> None:
        with self.db.session() as s:
            audit(
                s,
                "llm_gateway",
                "llm.status",
                project_id,
                {"task": task, "message": message},
                emit=False,
            )

    def _paused(self, project_id: str) -> bool:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            return bool(p is not None and p.paused)

    def _private(self, project_id: str) -> bool:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            return p is None or not bool(p.spec.get("data_may_leave_host", False))

    def _charge_usd(self, project_id: str, usd: float, model: str, table: str) -> None:
        """Record spend; warn once at 50, 80 and 100% of the study's dollar budget."""
        from cyberlab_common.db import CostRecord

        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                return
            before = p.llm_usd_used or 0.0
            p.llm_usd_used = before + usd
            s.add(CostRecord(project_id=project_id, model=model, usd=usd, price_table=table))
            limit = float(p.spec.get("llm_usd_budget", self.settings.llm_usd_budget) or 0)
            for share in (0.5, 0.8, 1.0):
                if limit and before < share * limit <= p.llm_usd_used:
                    audit(
                        s,
                        "llm_gateway",
                        "budget.threshold_reached",
                        project_id,
                        {"share": share, "usd_used": round(p.llm_usd_used, 4), "usd_limit": limit},
                    )

    def _charge(self, project_id: str, tokens: int) -> None:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is not None:
                p.llm_tokens_used += tokens

    # ----------------------------------------------------------------- helpers
    @contextmanager
    def _project_lock(self, project_id: str) -> Iterator[None]:
        with self._locks_guard:
            lock = self._locks.setdefault(project_id, threading.Lock())
        with lock:
            yield

    @staticmethod
    def _config(project_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": project_id}, "recursion_limit": 100}

    def _invoke(self, project_id: str, payload: Any) -> dict[str, Any]:
        with self._project_lock(project_id):
            try:
                self.graph.invoke(payload, self._config(project_id))
            except Exception as exc:
                self.ctx.set_stage(
                    project_id, "error", "Blocked", f"{type(exc).__name__}: {exc}"[:2000]
                )
                raise
        return self.status(project_id)

    # ----------------------------------------------------------------- lifecycle
    def create_project(self, principal: Principal, spec: ResearchSpec) -> str:
        if principal.role not in {"admin", "pi", "researcher"}:
            raise Forbidden("reviewers cannot create studies")
        with self.db.session() as s:
            p = Project(title=spec.title, owner_id=principal.id, spec=spec.model_dump(mode="json"))
            s.add(p)
            s.flush()
            add_member(
                s, p.id, principal.id, "pi" if principal.role in {"pi", "admin"} else "researcher"
            )
            audit(s, principal.username, "project.created", p.id, {"title": spec.title})
            return p.id

    def start(self, project_id: str) -> dict[str, Any]:
        """Start a new study, or resume one that stopped on an error from its last checkpoint."""
        state = self.graph.get_state(self._config(project_id))
        if state.next:
            if self.pending(project_id) is not None:
                return self.status(project_id)  # waiting for a human or for jobs: nothing to do
            self.ctx.set_stage(project_id, "retrying", "Running")
            return self._invoke(project_id, None)  # re-run the failed step from the checkpoint
        if state.values:
            return self.status(project_id)  # finished or stopped at an end state
        return self._invoke(project_id, {"project_id": project_id})

    def pending(self, project_id: str) -> dict[str, Any] | None:
        state = self.graph.get_state(self._config(project_id))
        for task in state.tasks:
            for intr in getattr(task, "interrupts", ()):
                return dict(intr.value)
        return None

    def status(self, project_id: str) -> dict[str, Any]:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None:
                raise KeyError(project_id)
            info: dict[str, Any] = {
                "project_id": p.id,
                "title": p.title,
                "stage": p.stage,
                "status_label": p.status_label,
                "blocked_reason": p.blocked_reason,
                "paused": p.paused,
                "llm_tokens_used": p.llm_tokens_used,
                "llm_usd_used": round(p.llm_usd_used or 0.0, 4),
                "gpu_minutes_used": round(p.gpu_minutes_used, 2),
            }
        info["pending"] = self.pending(project_id)
        return info

    def approve_protocol(
        self,
        principal: Principal,
        project_id: str,
        version: int,
        sha256: str,
        decision: str,
        comment: str = "",
    ) -> dict[str, Any]:
        pend = self.pending(project_id)
        if not pend or pend.get("gate") != "A":
            raise GateError("this study is not waiting for protocol approval")
        with self.db.session() as s:
            require_project_role(s, principal, project_id, "pi")
            latest = s.scalar(
                select(ProtocolVersion)
                .where(ProtocolVersion.project_id == project_id)
                .order_by(ProtocolVersion.version.desc())
            )
            if latest is None or latest.version != version or latest.sha256 != sha256:
                raise GateError(
                    "stale approval: protocol version or hash does not match the current protocol"
                )
            s.add(
                ApprovalRecord(
                    project_id=project_id,
                    gate=ApprovalGate.A_PROTOCOL.value,
                    subject_version=version,
                    subject_sha256=sha256,
                    approver_id=principal.id,
                    decision=decision,
                    comment=comment,
                )
            )
            audit(
                s,
                principal.username,
                f"protocol.{decision}",
                project_id,
                {"version": version, "sha256": sha256},
            )
        return self._invoke(project_id, Command(resume={"decision": decision, "comment": comment}))

    def release(
        self,
        principal: Principal,
        project_id: str,
        version: int,
        checklist: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        """PI release (gate D). Spec Step 22: the science, ethics and security review items
        must all be confirmed; a missing item blocks the RELEASED transition."""
        missing = [k for k in REVIEW_ITEMS if not (checklist or {}).get(k)]
        if missing:
            raise GateError(
                "release needs the review checklist: " + "; ".join(REVIEW_ITEMS[k] for k in missing)
            )
        pend = self.pending(project_id)
        if not pend or pend.get("gate") != "D" or pend.get("manuscript_version") != version:
            raise GateError("this manuscript version is not awaiting release")
        with self.db.session() as s:
            require_project_role(s, principal, project_id, "pi")
            ms = s.scalar(
                select(ManuscriptRecord).where(
                    ManuscriptRecord.project_id == project_id, ManuscriptRecord.version == version
                )
            )
            if ms is None:
                raise GateError("manuscript not found")
            if not ms.evidence_report.get("release_ready"):
                raise GateError("release blocked: manuscript has unresolved evidence flags")
            ms.status = "released"
            s.add(
                ApprovalRecord(
                    project_id=project_id,
                    gate=ApprovalGate.D_RELEASE.value,
                    subject_version=version,
                    subject_sha256=ms.sha256,
                    approver_id=principal.id,
                    decision="approved",
                    comment="review checklist: " + ", ".join(sorted(REVIEW_ITEMS)),
                )
            )
            audit(
                s,
                principal.username,
                "manuscript.released",
                project_id,
                {"version": version, "sha256": ms.sha256, "checklist": dict(checklist or {})},
            )
        return self._invoke(project_id, Command(resume={"released": True}))

    def research_decisions(self, project_id: str) -> list[dict[str, Any]]:
        from cyberlab_common.db import AgentNote

        with self.db.session() as s:
            notes = s.scalars(
                select(AgentNote)
                .where(AgentNote.project_id == project_id, AgentNote.kind == "research_decision")
                .order_by(AgentNote.id.desc())
            )
            return [{"note_id": n.id, **n.content} for n in notes]

    def start_followup(self, principal: Principal, project_id: str, note_id: int) -> dict[str, Any]:
        """The PI approves a feedback-controller decision: its protocol change becomes a new
        protocol version, which still needs gate A before anything runs (professor's plan,
        Step 3.3: human approval for protocol changes, capped number of cycles)."""
        import json as _json

        from cyberlab_common.db import AgentNote, ProtocolVersion
        from cyberlab_orchestrator.agents.feedback_controller import MAX_FOLLOWUPS
        from cyberlab_schemas import Protocol

        state = self.graph.get_state(self._config(project_id))
        if not state.values.get("analysis_sha256"):
            raise GateError("there are no analysed results to follow up yet")
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "pi")
            if p.stage == "released":
                raise GateError("the study was released; start a new study for further experiments")
            note = s.get(AgentNote, note_id)
            if note is None or note.project_id != project_id or note.kind != "research_decision":
                raise GateError("decision not found")
            decision = dict(note.content)
            change = decision.get("protocol_change")
            if not change or decision.get("approval_status") != "pending_pi":
                raise GateError("this decision has no pending experiment to approve")
            done = len(
                list(
                    s.scalars(
                        select(ProtocolVersion).where(
                            ProtocolVersion.project_id == project_id,
                            ProtocolVersion.created_by == "feedback_controller",
                        )
                    )
                )
            )
            if done >= MAX_FOLLOWUPS:
                raise GateError(f"the limit of {MAX_FOLLOWUPS} follow-up cycles is reached")
            latest = s.scalar(
                select(ProtocolVersion)
                .where(ProtocolVersion.project_id == project_id)
                .order_by(ProtocolVersion.version.desc())
            )
            assert latest is not None
            body = dict(latest.body)
            if "split" in change:
                body["split"] = {**(body.get("split") or {}), **change["split"]}
            if "seeds" in change:
                body["seeds"] = change["seeds"]
            body["version"] = latest.version + 1
            body["followup_of"] = {
                "version": latest.version,
                "decision_id": decision["decision_id"],
                "rule": decision["rule"],
                "reason": decision["identified_problem"],
                "change": change,
            }
            protocol = Protocol.model_validate(_json.loads(_json.dumps(body, default=str)))
            s.add(
                ProtocolVersion(
                    project_id=project_id,
                    version=protocol.version,
                    sha256=protocol.sha256,
                    body=protocol.model_dump(mode="json"),
                    rationale=f"Follow-up ({decision['rule']}): {decision['proposed_action']}",
                    created_by="feedback_controller",
                )
            )
            decision["approval_status"] = "approved"
            decision["final_outcome"] = f"protocol version {protocol.version} created"
            note.content = decision
            audit(
                s,
                principal.username,
                "followup.approved",
                project_id,
                {"decision_id": decision["decision_id"], "version": protocol.version},
            )
        self.graph.update_state(
            self._config(project_id),
            {
                "protocol_version": protocol.version,
                "protocol_sha256": protocol.sha256,
                "gate_a": None,
                "revisions": 0,
                "review_feedback": [],
                "needs_revision": False,
                "validation_blocked": False,
                "blocked_reason": None,
                "compile_error": None,
            },
            as_node="revise_protocol",
        )
        return self._invoke(project_id, None)

    def redraft(self, principal: Principal, project_id: str) -> dict[str, Any]:
        """Write the paper again (e.g. after the writer improved).

        Experiments are not repeated. The statistics are recomputed from the stored per-row
        predictions first, so a study analysed by an older version gets the current
        statistics (seed-averaged intervals, Holm correction) and the Methodology, which
        describes the current procedure, always matches the numbers.
        """
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "researcher")
            if p.stage == "released":
                raise GateError("this manuscript was already released")
            audit(s, principal.username, "manuscript.redraft_requested", project_id)
        state = self.graph.get_state(self._config(project_id))
        if not state.values.get("analysis_sha256"):
            raise GateError("there are no analysed results to write about yet")
        self.graph.update_state(
            self._config(project_id),
            {
                "revisions": 0,
                "review_feedback": [],
                "needs_revision": False,
                "blocked_reason": None,
            },
            as_node="check_runs",
        )
        return self._invoke(project_id, None)

    def raise_budget(self, principal: Principal, project_id: str, extra_tokens: int) -> int:
        """PI raises the study's LLM token budget (recorded in the audit log)."""
        if not 1_000 <= extra_tokens <= 10_000_000:
            raise GateError("choose an increase between 1,000 and 10,000,000 tokens")
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "pi")
            spec = dict(p.spec)
            old = int(spec.get("llm_token_budget", 0))
            spec["llm_token_budget"] = max(old, p.llm_tokens_used) + extra_tokens
            p.spec = spec
            audit(
                s,
                principal.username,
                "budget.raised",
                project_id,
                {"from": old, "to": spec["llm_token_budget"]},
            )
            return int(spec["llm_token_budget"])

    def set_writing(
        self, principal: Principal, project_id: str, allow_cloud: bool, usd_budget: float
    ) -> None:
        """PI sets, for an existing study, whether the paid cloud model may write (public data
        only; otherwise every call stays on the local model) and the study's dollar limit."""
        if not 0 <= usd_budget <= 10_000:
            raise GateError("the dollar limit must be between 0 and 10,000")
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "pi")
            spec = dict(p.spec)
            old = {k: spec.get(k) for k in ("data_may_leave_host", "llm_usd_budget")}
            spec["data_may_leave_host"] = allow_cloud
            spec["llm_usd_budget"] = round(float(usd_budget), 2)
            p.spec = spec
            audit(
                s,
                principal.username,
                "project.writing_settings",
                project_id,
                {"from": old, "to": {k: spec[k] for k in old}},
            )

    def pause(self, principal: Principal, project_id: str, paused: bool) -> None:
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "researcher")
            p.paused = paused
            audit(
                s, principal.username, "project.paused" if paused else "project.resumed", project_id
            )

    # ------------------------------------------------------------- study controls
    STOP_PREFIX = "Stopped by "

    def stop(self, principal: Principal, project_id: str) -> int:
        """Stop the study now: running and queued experiments are canceled and nothing new
        starts. Resume continues from the last checkpoint and runs the canceled experiments
        again. Returns how many jobs were canceled."""
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "researcher")
            if p.stage == "released":
                raise GateError("this study was already released")
            p.paused = True
            p.status_label = "Blocked"
            p.blocked_reason = (
                f"{self.STOP_PREFIX}{principal.username}. Resume continues from the last "
                "checkpoint and runs the canceled experiments again."
            )
            jobs = list(
                s.scalars(
                    select(JobRecord).where(
                        JobRecord.project_id == project_id,
                        JobRecord.state.in_(["queued", "running"]),
                    )
                )
            )
            for job in jobs:
                self.ctx.broker.cancel(s, job.id, principal.username)
            audit(s, principal.username, "project.stopped", project_id, {"jobs": len(jobs)})
            return len(jobs)

    def resume(self, principal: Principal, project_id: str) -> dict[str, Any]:
        """Continue a paused or stopped study from where it was held."""
        from cyberlab_common.db import ArtifactRecord, RunRecord

        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "researcher")
            stopped = (p.blocked_reason or "").startswith(self.STOP_PREFIX)
            p.paused = False
            requeued = 0
            if stopped:
                p.blocked_reason = None
                p.status_label = "Running"
                for job in s.scalars(
                    select(JobRecord).where(
                        JobRecord.project_id == project_id, JobRecord.state == "canceled"
                    )
                ):
                    for run in s.scalars(
                        select(RunRecord).where(
                            RunRecord.job_id == job.id, RunRecord.status != "succeeded"
                        )
                    ):
                        for a in s.scalars(
                            select(ArtifactRecord).where(ArtifactRecord.run_id == run.id)
                        ):
                            a.run_id = None
                        s.delete(run)
                    job.state, job.cancel_requested, job.failure_reason = "queued", False, None
                    requeued += 1
            audit(s, principal.username, "project.resumed", project_id, {"requeued_jobs": requeued})
        with self._locks_guard:
            lock = self._locks.get(project_id)
        if lock is not None and lock.locked():
            return self.status(project_id)  # a step is running; it simply continues now
        pend = self.pending(project_id)
        if pend and pend.get("gate") == "paused":
            return self._invoke(project_id, Command(resume={"paused": "resumed"}))
        if pend is None:
            return self.start(project_id)  # e.g. a stopped step: retry from the checkpoint
        return self.status(project_id)  # waiting for jobs or a PI decision

    def restart(self, principal: Principal, project_id: str) -> dict[str, Any]:
        """Run the whole study again from the first step (PI only). Earlier protocol and
        manuscript versions stay in the record; identical experiments are reused."""
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "pi")
            if p.stage == "released":
                raise GateError("a released study cannot be restarted; start a new study")
            for job in s.scalars(
                select(JobRecord).where(
                    JobRecord.project_id == project_id,
                    JobRecord.state.in_(["queued", "running"]),
                )
            ):
                self.ctx.broker.cancel(s, job.id, principal.username)
            p.paused = False
            p.stage, p.status_label, p.blocked_reason = "created", "Running", None
            audit(s, principal.username, "project.restarted", project_id)
        with self._project_lock(project_id):
            self.saver.delete_thread(project_id)
        return self._invoke(project_id, {"project_id": project_id})

    def delete_project(self, principal: Principal, project_id: str, confirm_title: str) -> None:
        """Soft-delete a study (NFR-009). PI only; the typed title must match; released
        studies are protected records and cannot be deleted. Queued jobs are canceled and
        running ones stopped; all records and evidence stay for audit and can be restored."""
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "pi")
            if p.stage == "released":
                raise GateError("a released study is a protected record and cannot be deleted")
            if confirm_title.strip() != p.title.strip():
                raise GateError("type the study title exactly to confirm the deletion")
            jobs = list(
                s.scalars(
                    select(JobRecord).where(
                        JobRecord.project_id == project_id,
                        JobRecord.state.in_(["queued", "running"]),
                    )
                )
            )
            for job in jobs:
                self.ctx.broker.cancel(s, job.id, principal.username)
            p.paused = True
            p.deleted_at = now()
            p.deleted_by = principal.username
            audit(
                s, principal.username, "project.deleted", project_id, {"jobs_canceled": len(jobs)}
            )

    def restore_project(self, principal: Principal, project_id: str) -> None:
        """Admin only: bring a deleted study back (it stays paused until resumed)."""
        if not principal.is_admin:
            raise Forbidden("only an administrator can restore a deleted study")
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is None or p.deleted_at is None:
                raise GateError("no deleted study with that id")
            p.deleted_at, p.deleted_by = None, None
            audit(s, principal.username, "project.restored", project_id)

    def deleted_projects(self, principal: Principal) -> list[dict[str, Any]]:
        if not principal.is_admin:
            return []
        with self.db.session() as s:
            return [
                {"id": p.id, "title": p.title, "deleted_at": p.deleted_at, "by": p.deleted_by}
                for p in s.scalars(
                    select(Project)
                    .where(Project.deleted_at.is_not(None))
                    .order_by(Project.deleted_at.desc())
                )
            ]

    # ----------------------------------------------------------------- worker
    def worker_tick(self, prepared_dir: Any = None, wait: bool = False) -> dict[str, Any]:
        """Start queued jobs up to capacity (several CPU jobs + one GPU job in parallel),
        then advance any study whose jobs are all finished.

        `wait=True` blocks until the started jobs finish (used by tests and `--once`).
        """
        broker = self.ctx.broker
        fixed = broker.reconcile()
        self._futures = {k: f for k, f in self._futures.items() if not f.done()}
        cpu_running = sum(1 for k in self._futures if not self._gpu_job.get(k))
        gpu_running = any(self._gpu_job.get(k) for k in self._futures)
        started: list[str] = []
        while True:
            cpu_free = cpu_running < self.settings.cpu_job_slots
            gpu_free = not gpu_running and not broker.gpu_busy()
            if not cpu_free and not gpu_free:
                break
            claimed = broker.claim(gpu_free=gpu_free)
            if claimed is None:
                break
            job_id, manifest = claimed
            if not manifest.use_gpu and not cpu_free:  # claimed a CPU job without a CPU slot
                with self.db.session() as s:
                    job = s.get(JobRecord, job_id)
                    if job is not None:
                        job.state, job.attempts = "queued", max(0, job.attempts - 1)
                break
            self._gpu_job[job_id] = manifest.use_gpu
            self._futures[job_id] = self._pool.submit(
                broker.execute, job_id, manifest, prepared_dir
            )
            started.append(job_id)
            if manifest.use_gpu:
                gpu_running = True
            else:
                cpu_running += 1
        if wait:
            for f in list(self._futures.values()):
                f.result()
        ran = started[0] if started else None
        advanced = []
        with self.db.session() as s:
            waiting = [
                p.id
                for p in s.scalars(
                    select(Project).where(Project.stage == "running", Project.deleted_at.is_(None))
                )
            ]
        with self.db.session() as s:
            held = {p.id for p in s.scalars(select(Project).where(Project.paused.is_(True)))}
        for pid in waiting:
            if pid in held:  # paused or stopped: the PI decides when it continues
                continue
            pend = self.pending(pid)
            if pend and pend.get("gate") == "jobs":
                with self.db.session() as s:
                    active = s.scalar(
                        select(JobRecord).where(
                            JobRecord.project_id == pid, JobRecord.state.in_(["queued", "running"])
                        )
                    )
                if active is None:
                    self._invoke(pid, Command(resume={"jobs": "done"}))
                    advanced.append(pid)
        return {
            "reconciled": fixed,
            "ran_job": ran,
            "started": started,
            "running": len([f for f in self._futures.values() if not f.done()]),
            "advanced": advanced,
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._conn.close()
