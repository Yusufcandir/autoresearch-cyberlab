"""Package metadata extraction without execution (AT-18).

Archives are read in memory: nothing is extracted to disk, nothing is imported and no install
script (setup.py, npm lifecycle scripts) is ever run. Metadata comes only from static files:
wheel METADATA, sdist PKG-INFO, pyproject.toml, or package.json. If a package declares its
metadata only through setup.py code, the result is a typed block, not an execution.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from email.parser import HeaderParser
from pathlib import Path, PurePosixPath
from typing import Any

from cyberlab_schemas.plugins import METADATA_REQUIRES_EXECUTION, SCHEMA_INVALID, PluginError

MAX_MEMBER_BYTES = 2_000_000
MAX_MEMBERS = 20_000
INSTALL_HOOKS = ("preinstall", "install", "postinstall", "prepare")


def _safe_name(name: str) -> str:
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise PluginError(SCHEMA_INVALID, f"unsafe path inside the archive: {name}")
    return str(p)


class _Archive:
    """Read-only view of a .whl/.zip or .tar.gz/.tgz archive."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.names: list[str] = []
        self._zip: zipfile.ZipFile | None = None
        self._tar: tarfile.TarFile | None = None
        if zipfile.is_zipfile(path):
            self._zip = zipfile.ZipFile(path)
            self.names = [_safe_name(n) for n in self._zip.namelist()]
        else:
            try:
                self._tar = tarfile.open(path, "r:*")
            except tarfile.TarError as exc:
                raise PluginError(
                    SCHEMA_INVALID, f"{path.name} is not a zip or tar archive"
                ) from exc
            members = self._tar.getmembers()
            if len(members) > MAX_MEMBERS:
                raise PluginError(SCHEMA_INVALID, "archive has too many members")
            self.names = [_safe_name(m.name) for m in members if m.isfile()]

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            info = self._zip.getinfo(name)
            if info.file_size > MAX_MEMBER_BYTES:
                raise PluginError(SCHEMA_INVALID, f"{name} is too large to inspect")
            return self._zip.read(name)
        assert self._tar is not None
        member = self._tar.getmember(name)
        if member.size > MAX_MEMBER_BYTES:
            raise PluginError(SCHEMA_INVALID, f"{name} is too large to inspect")
        fh = self._tar.extractfile(member)
        return fh.read() if fh else b""

    def find(self, *suffixes: str, max_depth: int = 2) -> str | None:
        hits = [
            n
            for n in self.names
            if any(n == s or n.endswith("/" + s) for s in suffixes)
            and len(PurePosixPath(n).parts) <= max_depth + 1
        ]
        return min(hits, key=len) if hits else None

    def close(self) -> None:
        if self._zip:
            self._zip.close()
        if self._tar:
            self._tar.close()


def _core_metadata(raw: bytes) -> dict[str, Any]:
    msg = HeaderParser().parsestr(raw.decode("utf-8", errors="replace"))
    return {
        "name": msg.get("Name"),
        "version": msg.get("Version"),
        "summary": msg.get("Summary"),
        "license": msg.get("License") or msg.get("License-Expression"),
        "requires_python": msg.get("Requires-Python"),
        "dependencies": msg.get_all("Requires-Dist") or [],
    }


def extract_metadata(path: Path) -> dict[str, Any]:
    """Static metadata for a Python wheel/sdist or an npm tarball. Never executes code."""
    arc = _Archive(path)
    try:
        install_scripts = [
            n
            for n in arc.names
            if PurePosixPath(n).name in {"setup.py", "install.sh", "postinstall.js"}
        ]
        source = None
        meta: dict[str, Any] | None = None
        if (m := arc.find("METADATA")) and ".dist-info/" in m:
            source, meta = m, _core_metadata(arc.read(m))
        elif m := arc.find("PKG-INFO"):
            source, meta = m, _core_metadata(arc.read(m))
        elif m := arc.find("pyproject.toml"):
            data = tomllib.loads(arc.read(m).decode("utf-8", errors="replace"))
            proj = data.get("project") or {}
            if proj.get("name"):
                source = m
                meta = {
                    "name": proj.get("name"),
                    "version": proj.get("version"),
                    "summary": proj.get("description"),
                    "license": (proj.get("license") or {}).get("text")
                    if isinstance(proj.get("license"), dict)
                    else proj.get("license"),
                    "requires_python": proj.get("requires-python"),
                    "dependencies": proj.get("dependencies", []),
                }
        elif m := arc.find("package.json"):
            data = json.loads(arc.read(m))
            hooks = {k: v for k, v in (data.get("scripts") or {}).items() if k in INSTALL_HOOKS}
            if hooks:
                install_scripts.append(f"{m}: npm lifecycle scripts {sorted(hooks)}")
            source = m
            meta = {
                "name": data.get("name"),
                "version": data.get("version"),
                "summary": data.get("description"),
                "license": data.get("license"),
                "requires_python": None,
                "dependencies": sorted((data.get("dependencies") or {}).keys()),
            }
        if meta is None:
            raise PluginError(
                METADATA_REQUIRES_EXECUTION,
                "no static metadata (METADATA, PKG-INFO, pyproject.toml or package.json); "
                "reading it would require running setup.py, which is not allowed",
                {"install_scripts": install_scripts},
            )
        return {
            **meta,
            "archive": path.name,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "metadata_source": source,
            "install_scripts_present": install_scripts,
            "executed": False,
        }
    finally:
        arc.close()


def make_sdist_fixture(dest: Path, with_pkg_info: bool = True, canary: Path | None = None) -> Path:
    """Build a tiny sdist whose setup.py would write `canary` if it were ever executed."""
    buf = io.BytesIO()
    files = {
        "demo-1.0/setup.py": (
            "import pathlib\n"
            f"pathlib.Path({str(canary)!r}).write_text('executed')\n"
            "from setuptools import setup\nsetup(name='demo', version='1.0')\n"
        ),
        "demo-1.0/demo/__init__.py": "VALUE = 1\n",
    }
    if with_pkg_info:
        files["demo-1.0/PKG-INFO"] = (
            "Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nSummary: Demo package\n"
            "License: MIT\nRequires-Dist: requests>=2\n"
        )
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    dest.write_bytes(buf.getvalue())
    return dest
