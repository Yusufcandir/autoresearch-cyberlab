"""Agent 6: Coding. Turns the approved protocol into allowlisted experiment manifests.

The coding agent does not write code that runs on the GPU host: arbitrary generated code is T2
and needs human review (security policy v1). It selects reviewed templates and validated
parameters through the protocol compiler, which also enforces the gate-A approval hash.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from cyberlab_common.db import ApprovalRecord, DatasetRecord, ProtocolVersion
from cyberlab_orchestrator.context import LabContext
from cyberlab_protocol import CompileError, compile_protocol
from cyberlab_schemas import Approval, ApprovalGate, DatasetManifest, Protocol


def coding_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid, version = state["project_id"], state["protocol_version"]
    with ctx.db.session() as s:
        pv = s.scalar(
            select(ProtocolVersion).where(
                ProtocolVersion.project_id == pid, ProtocolVersion.version == version
            )
        )
        assert pv is not None
        protocol = Protocol.model_validate(pv.body)
        rec = s.scalar(
            select(ApprovalRecord)
            .where(
                ApprovalRecord.project_id == pid,
                ApprovalRecord.gate == ApprovalGate.A_PROTOCOL.value,
            )
            .order_by(ApprovalRecord.created_at.desc())
        )
        approval = None
        if rec is not None:
            approval = Approval(
                project_id=pid,
                gate=ApprovalGate(rec.gate),
                subject_version=rec.subject_version,
                subject_sha256=rec.subject_sha256,
                approver=rec.approver_id,
                decision=rec.decision,
            )
        ds = s.get(DatasetRecord, protocol.dataset_id)
        dataset = DatasetManifest.model_validate(ds.manifest) if ds else None
    try:
        manifests = compile_protocol(
            protocol,
            approval,
            dataset,
            max_job_minutes=ctx.settings.gpu_job_max_minutes,
            use_gpu=ctx.settings.use_gpu,
        )
    except CompileError as exc:
        ctx.set_stage(pid, "compile_blocked", "Blocked", "; ".join(exc.reasons))
        return {"compile_error": exc.reasons}
    with ctx.db.session() as s:
        ctx.note(
            s,
            pid,
            "coding",
            "experiment_manifests",
            {"count": len(manifests), "experiments": [m.experiment_id for m in manifests]},
        )
    return {"experiments": [m.model_dump(mode="json") for m in manifests]}
