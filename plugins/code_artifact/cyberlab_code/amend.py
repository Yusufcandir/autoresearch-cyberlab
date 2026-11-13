"""Controlled code amendment (AT-19).

An agent may only *propose* a unified diff against an immutable source snapshot. This module:

1. checks every path the diff touches: relative, no traversal, no symlinks, inside the
   permitted globs, and never a test file unless explicitly permitted;
2. applies the diff to a fresh copy of the snapshot (the original is never written);
3. runs static security checks on the changed code (parsed, never imported or executed);
4. runs the functional tests only through an isolated runner. Without an isolated backend the
   functional step is BLOCKED: ISOLATION_REQUIRED instead of running on the shared host;
5. writes a receipt linking the snapshot hashes, the patch hash, the security report and the
   functional-test receipt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from cyberlab_schemas.plugins import (
    ISOLATION_REQUIRED,
    PATH_NOT_PERMITTED,
    SCHEMA_INVALID,
    PluginError,
)

TestRunner = Callable[[Path], dict[str, Any]]

FORBIDDEN_IMPORTS = {
    "subprocess",
    "socket",
    "ctypes",
    "multiprocessing",
    "urllib",
    "http",
    "requests",
}
FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "system",
    "popen",
    "spawnv",
    "remove",
    "rmtree",
}


def tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class FilePatch:
    path: str
    hunks: list[tuple[int, list[str]]] = field(default_factory=list)  # (old_start, lines)
    new_file: bool = False


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(diff: str) -> list[FilePatch]:
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    lines = diff.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- "):
            old = ln[4:].split("\t")[0].strip()
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise PluginError(SCHEMA_INVALID, "malformed diff: '---' without '+++'")
            new = lines[i + 1][4:].split("\t")[0].strip()
            path = new[2:] if new.startswith("b/") else new
            cur = FilePatch(path=path, new_file=old == "/dev/null")
            files.append(cur)
            i += 2
            continue
        m = _HUNK.match(ln)
        if m and cur is not None:
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ")):
                if lines[i].startswith(("+", "-", " ")) or lines[i] == "":
                    body.append(lines[i] if lines[i] else " ")
                i += 1
            cur.hunks.append((int(m.group(1)), body))
            continue
        i += 1
    if not files:
        raise PluginError(SCHEMA_INVALID, "the diff changes no files")
    return files


def check_path(rel: str, snapshot: Path, permitted: list[str], allow_tests: bool) -> None:
    p = PurePosixPath(rel)
    if not rel or p.is_absolute() or ".." in p.parts or "\\" in rel or rel.startswith("~"):
        raise PluginError(PATH_NOT_PERMITTED, f"unsafe path in diff: {rel!r}")
    if not any(fnmatchcase(rel, g) for g in permitted):
        raise PluginError(PATH_NOT_PERMITTED, f"{rel} is outside the permitted paths {permitted}")
    if not allow_tests and (p.parts[0] in {"tests", "test"} or p.name.startswith("test_")):
        raise PluginError(
            PATH_NOT_PERMITTED, f"{rel}: test files are the oracle and may not be changed"
        )
    target = snapshot / rel
    for parent in [target, *target.parents]:
        if parent == snapshot:
            break
        if parent.is_symlink():
            raise PluginError(PATH_NOT_PERMITTED, f"{rel} goes through a symbolic link")


def _apply(original: list[str], fp: FilePatch) -> list[str]:
    out: list[str] = []
    pos = 0
    for start, body in fp.hunks:
        start_idx = max(start - 1, 0)
        out.extend(original[pos:start_idx])
        pos = start_idx
        for ln in body:
            tag, text = ln[0], ln[1:]
            if tag == " ":
                if pos >= len(original) or original[pos] != text:
                    raise PluginError(
                        SCHEMA_INVALID, f"{fp.path}: context does not match the snapshot"
                    )
                out.append(text)
                pos += 1
            elif tag == "-":
                if pos >= len(original) or original[pos] != text:
                    raise PluginError(
                        SCHEMA_INVALID, f"{fp.path}: removed line does not match the snapshot"
                    )
                pos += 1
            else:
                out.append(text)
    out.extend(original[pos:])
    return out


def security_scan(source: str, name: str) -> list[str]:
    """Static checks only: the code is parsed, never imported or run."""
    try:
        tree = ast.parse(source, filename=name)
    except SyntaxError as exc:
        return [f"{name}: syntax error at line {exc.lineno}"]
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            mods = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for mod in mods:
                if mod.split(".")[0] in FORBIDDEN_IMPORTS:
                    findings.append(f"{name}:{node.lineno}: imports {mod}")
        elif isinstance(node, ast.Call):
            fn = node.func
            called = (
                fn.id
                if isinstance(fn, ast.Name)
                else fn.attr
                if isinstance(fn, ast.Attribute)
                else ""
            )
            if called in FORBIDDEN_CALLS:
                findings.append(f"{name}:{node.lineno}: calls {called}()")
    return findings


def docker_test_runner(image: str = "python:3.12-slim", timeout: int = 300) -> TestRunner:
    """Run `python -m unittest` in a locked-down container: no network, read-only, non-root.

    Suitable for reviewed fixtures (T2). It is not a substitute for a separate VM for hostile
    (T3) code; enable it only with SANDBOX_BACKEND=docker and ALLOW_RESTRICTED_EXECUTION=true.
    """

    def run(snapshot: Path) -> dict[str, Any]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--memory",
            "512m",
            "--pids-limit",
            "128",
            "--tmpfs",
            "/tmp:size=64m",  # noqa: S108 - container-internal tmpfs, not a host path
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-v",
            f"{snapshot.resolve()}:/work:ro",
            "-w",
            "/work",
            image,
            "python",
            "-I",
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-t",
            ".",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # noqa: S603
        return {
            "backend": "docker",
            "image": image,
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "output_tail": (proc.stdout + proc.stderr)[-4000:],
        }

    return run


def amend(
    snapshot: Path,
    diff: str,
    permitted: list[str],
    work_dir: Path,
    test_runner: TestRunner | None = None,
    allow_test_changes: bool = False,
) -> dict[str, Any]:
    """Apply a proposed diff to a copy of `snapshot` under policy. Returns the receipt."""
    snapshot = snapshot.resolve()
    before = tree_sha256(snapshot)
    patches = parse_unified_diff(diff)
    for fp in patches:
        check_path(fp.path, snapshot, permitted, allow_test_changes)
        if not fp.new_file and not (snapshot / fp.path).is_file():
            raise PluginError(SCHEMA_INVALID, f"{fp.path} does not exist in the snapshot")

    amendment_id = uuid.uuid4().hex[:12]
    amended = work_dir / f"amended-{amendment_id}"
    shutil.copytree(snapshot, amended, symlinks=True)
    changed: dict[str, dict[str, str | None]] = {}
    for fp in patches:
        target = amended / fp.path
        old_bytes = target.read_bytes() if target.exists() else b""
        old_text = old_bytes.decode("utf-8")
        eol = "\r\n" if "\r\n" in old_text else "\n"  # keep the file's own line endings
        old_lines = old_text.replace("\r\n", "\n").split("\n") if old_bytes else []
        trailing = bool(old_lines) and old_lines[-1] == ""
        if trailing:
            old_lines = old_lines[:-1]
        new_text = eol.join(_apply(old_lines, fp)) + (eol if trailing or fp.new_file else "")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(new_text.encode("utf-8"))
        changed[fp.path] = {
            "before_sha256": _sha(old_bytes) if old_bytes else None,
            "after_sha256": _sha(new_text.encode()),
        }

    if tree_sha256(snapshot) != before:  # defence in depth: the original must be untouched
        raise PluginError(PATH_NOT_PERMITTED, "the original snapshot changed during amendment")

    findings = [
        f
        for rel in changed
        if rel.endswith(".py")
        for f in security_scan((amended / rel).read_text(encoding="utf-8"), rel)
    ]
    security = {"passed": not findings, "findings": findings, "method": "static AST scan"}

    if test_runner is None:
        functional: dict[str, Any] = {
            "status": ISOLATION_REQUIRED,
            "passed": None,
            "reason": "tests execute proposed code; no isolated runner is configured",
        }
    elif not security["passed"]:
        functional = {"status": "skipped", "passed": None, "reason": "security check failed"}
    else:
        functional = {"status": "ran", **test_runner(amended)}
    functional["receipt_sha256"] = _sha(json.dumps(functional, sort_keys=True).encode())

    if not security["passed"]:
        status = "rejected"
    elif functional.get("passed") is True:
        status = "accepted"
    elif functional.get("passed") is False:
        status = "rejected"
    else:
        status = "blocked"
    receipt = {
        "amendment_id": amendment_id,
        "status": status,
        "snapshot_sha256": before,
        "amended_snapshot_sha256": tree_sha256(amended),
        "amended_path": str(amended),
        "patch_sha256": _sha(diff.encode()),
        "permitted_paths": permitted,
        "changed_files": changed,
        "security_check": security,
        "functional_tests": functional,
    }
    (work_dir / f"amendment-{amendment_id}.json").write_text(json.dumps(receipt, indent=2))
    return receipt
