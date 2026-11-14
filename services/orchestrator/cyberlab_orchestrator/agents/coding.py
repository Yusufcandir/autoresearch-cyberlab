"""Agent 6: Coding. Turns the approved protocol into allowlisted experiment manifests.

The coding agent does not write code that runs on the GPU host: arbitrary generated code is T2
and needs human review (security policy v1). It selects reviewed templates and validated
parameters through the protocol compiler, which also enforces the gate-A approval hash.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from cyberlab_common.db import ApprovalRecord, DatasetRecord, ProtocolVersion, audit
from cyberlab_data import DatasetBlocked, dataset_record_id, is_default_split, prepare
from cyberlab_orchestrator.context import LabContext
from cyberlab_protocol import CompileError, compile_protocol
from cyberlab_schemas import Approval, ApprovalGate, DatasetManifest, Protocol


def ensure_split_prepared(ctx: LabContext, pid: str, protocol: Protocol) -> str | None:
    """Prepare and audit the data for a non-default split once; return a blocking reason."""
    if is_default_split(protocol.split):
        return None
    rec_id = dataset_record_id(protocol.dataset_id, protocol.split)
    with ctx.db.session() as s:
        rec = s.get(DatasetRecord, rec_id)
        if rec is not None and rec.status == "verified":
            return None
    try:
        prepared = prepare(
            ctx.settings.dataset_root / "cicids2017",
            protocol.split,
            benign_cap=ctx.extra.get("benign_cap", 400_000),
        )
    except DatasetBlocked as exc:
        return "dataset for this split blocked: " + "; ".join(exc.reasons)
    m = prepared.manifest
    with ctx.db.session() as s:
        rec = s.get(DatasetRecord, rec_id)
        if rec is None:
            s.add(
                DatasetRecord(
                    id=rec_id,
                    name=f"{m.name} [{protocol.split.strategy} split]",
                    manifest=m.model_dump(mode="json"),
                    status=m.status,
                )
            )
        else:
            rec.manifest, rec.status = m.model_dump(mode="json"), m.status
        audit(s, "coding_agent", f"dataset.{m.status}", pid, {"split": protocol.split.model_dump()})
    if m.status != "verified":
        return "dataset for this split blocked: " + "; ".join(m.blocked_reasons)
    return None


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
    # The protocol's own split (e.g. by day) gets its own prepared, audited copy of the data.
    split_error = ensure_split_prepared(ctx, pid, protocol)
    if split_error:
        ctx.set_stage(pid, "dataset_blocked", "Blocked", split_error)
        return {"compile_error": [split_error]}
    with ctx.db.session() as s:
        ds = s.get(DatasetRecord, dataset_record_id(protocol.dataset_id, protocol.split))
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
