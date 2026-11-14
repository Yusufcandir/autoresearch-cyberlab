"""Agent 5: Dataset. Deterministic: provenance, licence, hashes, cleaning and split integrity."""

from __future__ import annotations

from typing import Any

from cyberlab_common.db import DatasetRecord, audit
from cyberlab_data import DatasetBlocked
from cyberlab_orchestrator.context import LabContext
from cyberlab_orchestrator.datasets import project_adapter
from cyberlab_schemas import DatasetManifest, SplitSpec


def dataset_agent(ctx: LabContext, state: dict[str, Any]) -> dict[str, Any]:
    pid = state["project_id"]
    ctx.authorize("dataset", "dataset_register", "T1", pid)
    adapter = project_adapter(ctx, pid)
    root = adapter.root(ctx.settings.dataset_root)
    with ctx.db.session() as s:
        existing = s.get(DatasetRecord, adapter.dataset_id)
        if (
            existing is not None
            and existing.status == "verified"
            and (root / "prepared" / "clean.parquet").exists()
        ):
            m = DatasetManifest.model_validate(existing.manifest)
            return {"dataset_status": "verified", "dataset_summary": _summary(m)}
    try:
        prepared = adapter.prepare(
            root, SplitSpec(), benign_cap=ctx.extra.get("benign_cap", 400_000)
        )
        manifest = prepared.manifest
    except DatasetBlocked as exc:
        ctx.set_stage(pid, "dataset_blocked", "Blocked", "; ".join(exc.reasons))
        with ctx.db.session() as s:
            audit(s, "dataset_agent", "dataset.blocked", pid, {"reasons": exc.reasons})
        return {"dataset_status": "blocked", "blocked_reason": "; ".join(exc.reasons)}
    with ctx.db.session() as s:
        rec = s.get(DatasetRecord, adapter.dataset_id)
        if rec is None:
            s.add(
                DatasetRecord(
                    id=adapter.dataset_id,
                    name=manifest.name,
                    manifest=manifest.model_dump(mode="json"),
                    status=manifest.status,
                )
            )
        else:
            rec.manifest, rec.status = manifest.model_dump(mode="json"), manifest.status
        audit(s, "dataset_agent", f"dataset.{manifest.status}", pid, _summary(manifest))
    if manifest.status != "verified":
        ctx.set_stage(pid, "dataset_blocked", "Blocked", "; ".join(manifest.blocked_reasons))
        return {"dataset_status": "blocked", "blocked_reason": "; ".join(manifest.blocked_reasons)}
    return {"dataset_status": "verified", "dataset_summary": _summary(manifest)}


def _summary(m: DatasetManifest) -> dict[str, Any]:
    return {
        "dataset_id": m.dataset_id,
        "rows": m.n_rows_clean,
        "raw_rows": m.n_rows_raw,
        "duplicates_removed": m.n_duplicates_removed,
        "classes": m.class_counts,
        "splits": m.split_counts,
        "overlap": m.split_overlap_rows,
        "license": m.license,
    }
