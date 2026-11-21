"""Which registered dataset a study uses (chosen on the new-study form, stored in the spec)."""

from __future__ import annotations

from cyberlab_common.db import Project
from cyberlab_data import DatasetAdapter, get_adapter
from cyberlab_orchestrator.context import LabContext


def datasets_named(text: str) -> list[str]:
    """Dataset ids named in free text, in reading order. Longer names are matched first, so
    'NF-UNSW-NB15-v2' is not also read as 'UNSW-NB15'."""
    import re

    from cyberlab_data import ADAPTERS

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    flat = norm(text)
    found: list[tuple[int, str]] = []
    for a in sorted(ADAPTERS.values(), key=lambda a: -len(norm(a.display))):
        for name in {norm(a.display), norm(a.key)}:
            if len(name) < 5:
                continue
            at = flat.find(name)
            if at >= 0:
                found.append((at, a.dataset_id))
                flat = flat[:at] + "#" * len(name) + flat[at + len(name) :]
                break
    return [d for _, d in sorted(found)]


def study_adapters(ctx: LabContext, pid: str) -> list[DatasetAdapter]:
    """The study's dataset and any other dataset of the same kind its goal names (benchmark)."""
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        spec = (project.spec or {}) if project else {}
    main = get_adapter(spec.get("dataset_hint"))
    out = [main]
    for d in datasets_named(" ".join([spec.get("goal", ""), *spec.get("constraints", [])])):
        a = get_adapter(d)
        if a.dataset_id not in {x.dataset_id for x in out} and a.modality == main.modality:
            out.append(a)
    return out


def project_adapter(ctx: LabContext, pid: str) -> DatasetAdapter:
    with ctx.db.session() as s:
        project = s.get(Project, pid)
        hint = (project.spec or {}).get("dataset_hint") if project else None
    return get_adapter(hint)
