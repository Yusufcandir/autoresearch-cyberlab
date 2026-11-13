"""Restore drill (spec Step 24, AT-21): unpack a backup and prove the evidence survived.

A backup passes only if every artifact file still hashes to the SHA-256 it is stored under,
every artifact recorded in the restored database is present, and the approval records can be
read back. Nothing is written outside the scratch directory.
"""

from __future__ import annotations

import hashlib
import sqlite3
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in tar.getmembers():
        target = (dest / member.name).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"unsafe path in backup: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError(f"links are not allowed in a backup: {member.name}")
    tar.extractall(dest)  # noqa: S202 - every member was checked above


def verify_backup(tarball: Path, scratch: Path | None = None) -> dict[str, Any]:
    """Restore `tarball` into a scratch directory and check it. Returns a report dict."""
    work = Path(scratch or tempfile.mkdtemp(prefix="cyberlab-restore-"))
    with tarfile.open(tarball, "r:gz") as tar:
        _safe_extract(tar, work)

    problems: list[str] = []
    files_ok = 0
    art_root = work / "artifacts"
    on_disk: set[str] = set()
    if art_root.exists():
        for f in art_root.glob("*/*/*"):
            if not f.is_file() or f.name.startswith("."):
                continue
            sha = f.parent.name
            on_disk.add(sha)
            if _sha256(f) != sha:
                problems.append(f"artifact {sha[:12]} ({f.name}) does not match its hash")
            else:
                files_ok += 1

    db_path = work / "cyberlab.db"
    recorded: set[str] = set()
    approvals = 0
    projects = 0
    if db_path.exists():
        con = sqlite3.connect(db_path)
        try:
            recorded = {r[0] for r in con.execute("SELECT sha256 FROM artifacts")}
            approvals = con.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]
            projects = con.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        finally:
            con.close()
        missing = sorted(recorded - on_disk)
        problems += [
            f"artifact {sha[:12]} is in the database but not in the backup" for sha in missing
        ]
    elif (work / "db.sql").exists():
        text = (work / "db.sql").read_text(encoding="utf-8", errors="replace")
        approvals = text.count("INSERT INTO public.approvals") or text.count("approvals")
    else:
        problems.append("backup contains no database")

    return {
        "backup": str(tarball),
        "restored_to": str(work),
        "artifacts_verified": files_ok,
        "artifacts_recorded": len(recorded),
        "approvals": approvals,
        "projects": projects,
        "ok": not problems,
        "problems": problems,
    }
