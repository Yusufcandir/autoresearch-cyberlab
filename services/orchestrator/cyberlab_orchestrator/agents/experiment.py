"""Agent 7: Experiment Execution. Submits jobs to the broker; long jobs run in the worker, not here."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from cyberlab_common.db import JobRecord
from cyberlab_orchestrator.context import LabContext
from cyberlab_schemas import ExperimentManifest

TERMINAL = {"succeeded", "failed", "canceled", "blocked"}


def experiment_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    ctx.authorize("experiment", "run_experiment_template", "T1", pid)
    job_ids = []
    with ctx.db.session() as s:
        for raw in state.get("experiments", []):
            job = ctx.broker.submit(s, ExperimentManifest.model_validate(raw), actor="experiment_agent")
            job_ids.append(job.id)
    ctx.set_stage(pid, "running", "Running")
    return {"job_ids": job_ids}


def jobs_status(ctx: LabContext, project_id: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    with ctx.db.session() as s:
        for job in s.scalars(select(JobRecord).where(JobRecord.project_id == project_id)):
            counts[job.state] = counts.get(job.state, 0) + 1
    return counts


def all_jobs_done(ctx: LabContext, project_id: str) -> bool:
    counts = jobs_status(ctx, project_id)
    return bool(counts) and all(k in TERMINAL for k in counts)
