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
        gateway.private_lookup = self._private
        gateway.on_status = self._llm_status
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

    def _private(self, project_id: str) -> bool:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            return p is None or not bool(p.spec.get("data_may_leave_host", False))

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

    def release(self, principal: Principal, project_id: str, version: int) -> dict[str, Any]:
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
                )
            )
            audit(
                s,
                principal.username,
                "manuscript.released",
                project_id,
                {"version": version, "sha256": ms.sha256},
            )
        return self._invoke(project_id, Command(resume={"released": True}))

    def redraft(self, principal: Principal, project_id: str) -> dict[str, Any]:
        """Write the paper again from the stored analysis (e.g. after the writer improved).

        Experiments and statistics are not repeated; the workflow jumps back to the writer.
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
            {"revisions": 0, "review_feedback": [], "needs_revision": False},
            as_node="statistics",
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

    def pause(self, principal: Principal, project_id: str, paused: bool) -> None:
        with self.db.session() as s:
            p = require_project_role(s, principal, project_id, "researcher")
            p.paused = paused
            audit(
                s, principal.username, "project.paused" if paused else "project.resumed", project_id
            )

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
            waiting = [p.id for p in s.scalars(select(Project).where(Project.stage == "running"))]
        for pid in waiting:
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
