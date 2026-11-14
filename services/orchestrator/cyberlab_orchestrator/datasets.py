"""Which registered dataset a study uses (chosen on the new-study form, stored in the spec)."""

from __future__ import annotations

from cyberlab_common.db import Project
from cyberlab_data import DatasetAdapter, get_adapter
from cyberlab_orchestrator.context import LabContext


def project_adapter(ctx: LabContext, pid: str) -> DatasetAdapter:
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        hint = (project.spec or {}).get("dataset_hint") if project else None
    return get_adapter(hint)
