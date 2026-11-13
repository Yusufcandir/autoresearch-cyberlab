"""Content-addressed artifact store.

Artifacts are copied to `<root>/<sha[:2]>/<sha>/<name>` and recorded in the database with
their hash. Reading an artifact re-checks the hash, so a modified file is detected instead
of silently feeding a manuscript.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cyberlab_common.db import ArtifactRecord


class EvidenceError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(
        self, s: Session, project_id: str, src: Path, name: str | None = None, run_id: str | None = None
    ) -> ArtifactRecord:
        name = name or src.name
        if "/" in name or "\\" in name or name.startswith("."):
            raise EvidenceError(f"invalid artifact name {name!r}")
        sha = sha256_path(src)
        dest = self.root / sha[:2] / sha / name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            shutil.copyfile(src, tmp)
            tmp.replace(dest)  # atomic publish
        rec = ArtifactRecord(
            sha256=sha,
            project_id=project_id,
            run_id=run_id,
            name=name,
            path=str(dest),
            size_bytes=dest.stat().st_size,
        )
        s.add(rec)
        s.flush()
        return rec

    def path_for(self, s: Session, project_id: str, sha256: str) -> Path:
        rec = s.scalar(
            select(ArtifactRecord).where(
                ArtifactRecord.sha256 == sha256, ArtifactRecord.project_id == project_id
            )
        )
        if rec is None:
            raise EvidenceError(f"artifact {sha256[:12]} not found for this project")
        path = Path(rec.path)
        if not path.exists():
            raise EvidenceError(f"artifact {sha256[:12]} is missing from the store")
        if sha256_path(path) != sha256:
            raise EvidenceError(f"artifact {sha256[:12]} failed its hash check (modified?)")
        return path

    def verify_all(self, s: Session, project_id: str) -> dict[str, str]:
        """Return {sha: 'ok'|'missing'|'modified'} for every artifact of a project."""
        out: dict[str, str] = {}
        for rec in s.scalars(select(ArtifactRecord).where(ArtifactRecord.project_id == project_id)):
            p = Path(rec.path)
            if not p.exists():
                out[rec.sha256] = "missing"
            elif sha256_path(p) != rec.sha256:
                out[rec.sha256] = "modified"
            else:
                out[rec.sha256] = "ok"
        return out
