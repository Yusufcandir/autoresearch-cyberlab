"""Job broker for the single shared GPU host.

- Submission is idempotent: the same idempotency key never creates a second job (AT-10).
- A host-wide file lock lets at most one broker job run at a time (AT-08).
- Before a GPU job, other users' GPU processes are detected and the job waits (AT-09). This is
  detection and courtesy, not enforcement: the broker cannot stop non-broker users.
- Every job has a wall-clock cap; on timeout or cancel the process is killed and partial
  artifacts are kept (AT-11).
- After a crash, `reconcile()` finds jobs left in `running` without a live lock owner and
  requeues them under the same key (no duplicate runs).
- Every job passes the security policy first; T3 never runs here (AT-12).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess  # noqa: S404 - fixed interpreter and module only
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from cyberlab_authz.policy import ExecutionRequest, decide
from cyberlab_common.config import Settings
from cyberlab_common.db import ArtifactRecord, Database, JobRecord, Project, RunRecord, audit, now
from cyberlab_evidence import ArtifactStore
from cyberlab_qa import compute_metrics
from cyberlab_schemas import ExperimentManifest

MAX_ATTEMPTS = 2


class LockBusy(RuntimeError):
    pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":  # pragma: no cover - Windows dev PCs
        out = subprocess.run(  # noqa: S603, S607
            ["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, check=False
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GpuLock:
    """Exclusive host lock implemented with O_EXCL file creation and stale-owner detection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def owner(self) -> dict[str, Any] | None:
        try:
            return dict(json.loads(self.path.read_text()))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def acquire(self, job_id: str) -> None:
        info = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "job_id": job_id,
            "since": time.time(),
        }
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                owner = self.owner()
                same_host = owner and owner.get("host") == socket.gethostname()
                if owner and same_host and not _pid_alive(int(owner.get("pid", -1))):
                    self.path.unlink(missing_ok=True)  # stale lock from a crashed broker
                    continue
                raise LockBusy(f"GPU lock held by {owner}") from None
            with os.fdopen(fd, "w") as fh:
                json.dump(info, fh)
            return
        raise LockBusy("could not acquire GPU lock")

    def release(self, job_id: str) -> None:
        owner = self.owner()
        if owner and owner.get("job_id") == job_id:
            self.path.unlink(missing_ok=True)


def foreign_gpu_processes() -> list[dict[str, str]]:
    """GPU compute processes not started by this broker (best effort, read-only)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    out = subprocess.run(  # noqa: S603
        [exe, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).stdout
    procs = []
    for line in out.splitlines():
        if not line.strip():
            continue
        pid, mem = (x.strip() for x in line.split(","))
        procs.append({"pid": pid, "used_mib": mem})
    return procs


class Broker:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.store = ArtifactStore(settings.artifact_root)
        self.lock = GpuLock(settings.data_dir / "locks" / "gpu.lock")
        self.work_root = settings.data_dir / "work"

    # ------------------------------------------------------------------ submission
    def submit(self, s: Session, manifest: ExperimentManifest, actor: str) -> JobRecord:
        key = f"{manifest.project_id}:{manifest.protocol_sha256[:16]}:{manifest.experiment_id}"
        existing = s.scalar(select(JobRecord).where(JobRecord.idempotency_key == key))
        if existing is not None:
            return existing
        decision = decide(
            ExecutionRequest("run_experiment_template", manifest.risk_tier), self.settings
        )
        state = "queued" if decision.allowed else "blocked"
        job = JobRecord(
            project_id=manifest.project_id,
            experiment_id=manifest.experiment_id,
            idempotency_key=key,
            manifest=manifest.model_dump(mode="json"),
            state=state,
            failure_reason=None if decision.allowed else decision.code + ": " + decision.reason,
        )
        s.add(job)
        s.flush()
        audit(
            s,
            actor,
            "job.submitted" if decision.allowed else "security.policy_denied",
            manifest.project_id,
            {"job_id": job.id, "experiment_id": manifest.experiment_id, "decision": decision.code},
        )
        return job

    def cancel(self, s: Session, job_id: str, actor: str) -> JobRecord:
        job = s.get(JobRecord, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.state == "queued":
            job.state = "canceled"
            job.failure_reason = f"canceled by {actor}"
        elif job.state == "running":
            job.cancel_requested = True
        job.updated_at = now()
        audit(s, actor, "job.canceled", job.project_id, {"job_id": job_id, "state": job.state})
        return job

    # ------------------------------------------------------------------ recovery
    def reconcile(self) -> int:
        """Requeue jobs stranded in `running` by a crash. Returns how many were fixed."""
        fixed = 0
        owner = self.lock.owner()
        live_job = owner.get("job_id") if owner and _pid_alive(int(owner.get("pid", -1))) else None
        with self.db.session() as s:
            for job in s.scalars(select(JobRecord).where(JobRecord.state == "running")):
                if job.id == live_job:
                    continue
                done = s.scalar(select(RunRecord).where(RunRecord.job_id == job.id))
                if done is not None:
                    job.state = done.status
                elif job.attempts < MAX_ATTEMPTS:
                    job.state = "queued"
                    job.failure_reason = "interrupted (broker restart); requeued"
                else:
                    job.state = "failed"
                    job.failure_reason = "interrupted too many times"
                job.updated_at = now()
                audit(
                    s,
                    "broker",
                    "job.reconciled",
                    job.project_id,
                    {"job_id": job.id, "state": job.state},
                )
                fixed += 1
        return fixed

    # ------------------------------------------------------------------ execution
    def next_job_id(self, project_id: str | None = None) -> str | None:
        with self.db.session() as s:
            q = select(JobRecord).where(JobRecord.state == "queued").order_by(JobRecord.created_at)
            if project_id:
                q = q.where(JobRecord.project_id == project_id)
            for job in s.scalars(q):
                proj = s.get(Project, job.project_id)
                if proj is None or proj.paused:
                    continue
                budget = float(proj.spec.get("gpu_budget_minutes", 0) or 0)
                if budget and proj.gpu_minutes_used >= budget:  # AT-11: budget cap
                    job.state = "blocked"
                    job.failure_reason = (
                        f"compute budget exhausted ({proj.gpu_minutes_used:.1f}/{budget:.0f} min)"
                    )
                    audit(
                        s, "broker", "budget.threshold_reached", job.project_id, {"job_id": job.id}
                    )
                    continue
                return job.id
        return None

    def run_once(
        self, project_id: str | None = None, prepared_dir: Path | None = None
    ) -> str | None:
        job_id = self.next_job_id(project_id)
        if job_id is None:
            return None
        with self.db.session() as s:
            job = s.get(JobRecord, job_id)
            assert job is not None
            manifest = ExperimentManifest.model_validate(job.manifest)
            if manifest.use_gpu:
                others = foreign_gpu_processes()
                if others:
                    job.failure_reason = f"waiting: GPU in use by other processes {others}"
                    audit(
                        s,
                        "broker",
                        "job.contention",
                        job.project_id,
                        {"job_id": job.id, "others": others},
                    )
                    return None
        try:
            self.lock.acquire(job_id)
        except LockBusy:
            return None
        try:
            self._execute(job_id, manifest, prepared_dir)
        finally:
            self.lock.release(job_id)
        return job_id

    def run_all(self, project_id: str | None = None, prepared_dir: Path | None = None) -> list[str]:
        done = []
        while (jid := self.run_once(project_id, prepared_dir)) is not None:
            done.append(jid)
        return done

    def _execute(
        self, job_id: str, manifest: ExperimentManifest, prepared_dir: Path | None
    ) -> None:
        prepared = prepared_dir or (self.settings.dataset_root / "cicids2017" / "prepared")
        work = self.work_root / job_id
        work.mkdir(parents=True, exist_ok=True)
        (work / "manifest.json").write_text(manifest.model_dump_json())
        started = datetime.now(UTC)
        with self.db.session() as s:
            job = s.get(JobRecord, job_id)
            assert job is not None
            job.state = "running"
            job.attempts += 1
            job.external_ref = f"local:{socket.gethostname()}:{os.getpid()}:{job_id[:8]}"
            job.updated_at = now()
            audit(
                s,
                "broker",
                "job.started",
                job.project_id,
                {"job_id": job_id, "ref": job.external_ref},
            )

        cmd = [
            sys.executable,
            "-m",
            "cyberlab_exec.runner",
            str(work / "manifest.json"),
            str(prepared),
            str(work / "out"),
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0" if manifest.use_gpu else ""}
        proc = subprocess.Popen(
            cmd, stdout=(work / "stdout.log").open("w"), stderr=subprocess.STDOUT, env=env
        )  # noqa: S603
        deadline = time.monotonic() + manifest.max_minutes * 60
        reason: str | None = None
        while proc.poll() is None:
            time.sleep(0.5)
            if time.monotonic() > deadline:
                reason = f"timeout after {manifest.max_minutes} min"
            else:
                with self.db.session() as s:
                    j = s.get(JobRecord, job_id)
                    if j is not None and j.cancel_requested:
                        reason = "canceled"
            if reason:
                proc.kill()
                proc.wait()
                break
        finished = datetime.now(UTC)
        self._finish(job_id, manifest, work, proc.returncode, reason, started, finished)

    def _finish(
        self,
        job_id: str,
        manifest: ExperimentManifest,
        work: Path,
        returncode: int | None,
        reason: str | None,
        started: datetime,
        finished: datetime,
    ) -> None:
        out = work / "out"
        with self.db.session() as s:
            job = s.get(JobRecord, job_id)
            assert job is not None
            artifacts: dict[str, str] = {}
            run_id = None
            for name in ("predictions.csv", "model_info.json", "error.json"):
                if (out / name).exists():
                    rec = self.store.put(s, manifest.project_id, out / name, name=name)
                    artifacts[name] = rec.sha256
            if (work / "stdout.log").exists():
                artifacts["stdout.log"] = self.store.put(
                    s, manifest.project_id, work / "stdout.log"
                ).sha256
            ok = reason is None and returncode == 0 and "predictions.csv" in artifacts
            metrics: dict[str, Any] = {}
            if ok:
                metrics = compute_metrics(pd.read_csv(out / "predictions.csv"))
            error = reason or (None if ok else _error_text(out, returncode))
            info = (
                json.loads((out / "model_info.json").read_text())
                if (out / "model_info.json").exists()
                else {}
            )
            receipt = {
                "job_id": job_id,
                "experiment_id": manifest.experiment_id,
                "status": "succeeded" if ok else "failed",
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "artifacts": artifacts,
                "code_version": _code_version(),
                "environment": info.get("environment", {}),
                "external_ref": job.external_ref,
                "error": error,
            }
            run = RunRecord(
                job_id=job_id,
                project_id=manifest.project_id,
                experiment_id=manifest.experiment_id,
                model_name=manifest.model.name,
                seed=manifest.seed,
                status=receipt["status"],
                receipt=receipt,
                metrics=metrics,
            )
            s.add(run)
            s.flush()
            run_id = run.id
            for a in s.scalars(
                select(ArtifactRecord).where(
                    ArtifactRecord.sha256.in_(list(artifacts.values())),
                    ArtifactRecord.project_id == manifest.project_id,
                    ArtifactRecord.run_id.is_(None),
                )
            ):
                a.run_id = run_id
            job.state = "succeeded" if ok else ("canceled" if reason == "canceled" else "failed")
            job.failure_reason = error
            job.updated_at = now()
            proj = s.get(Project, manifest.project_id)
            if proj is not None:
                proj.gpu_minutes_used += (finished - started).total_seconds() / 60
            audit(
                s,
                "broker",
                f"job.{job.state}",
                manifest.project_id,
                {"job_id": job_id, "run_id": run_id, "artifacts": artifacts, "error": error},
            )
        shutil.rmtree(out, ignore_errors=True)


def _error_text(out: Path, returncode: int | None) -> str:
    err = out / "error.json"
    if err.exists():
        return str(json.loads(err.read_text()).get("error", "error"))[:2000]
    return f"runner exited with code {returncode}"


def _code_version() -> str:
    git = shutil.which("git")
    if git is None:
        return "unknown"
    res = subprocess.run(
        [git, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )  # noqa: S603
    return res.stdout.strip() or "unknown"
