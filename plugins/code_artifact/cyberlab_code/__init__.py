"""Source-code and software-artifact capability plugin (AT-16, AT-18, AT-19).

Everything here is static: source is parsed with `ast`, archives are read in memory, and
proposed patches are applied to copies. Running code (tests) requires an isolated runner.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cyberlab_code.amend import amend, docker_test_runner, security_scan, tree_sha256
from cyberlab_code.metadata import extract_metadata
from cyberlab_schemas.plugins import (
    SCHEMA_INVALID,
    Capability,
    PluginError,
    ValidationReport,
    require_view,
)

CAPABILITY = Capability(
    name="code_artifact",
    version="1",
    supported_modalities=("source_code", "package"),
    feature_views=("static_summary", "package_metadata"),
    requires_isolated_execution=True,  # only for running tests on amended code
    notes="Static analysis only; executing code needs an isolated runner (AT-12, AT-19).",
)


class SourceFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Literal["source_code"] = "source_code"
    root: str
    files: list[SourceFile] = Field(min_length=1)


def snapshot_manifest(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": p.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    return {"modality": "source_code", "root": str(root), "files": files}


class CodeAdapter:
    def capabilities(self) -> Capability:
        return CAPABILITY

    def _parse(self, manifest: Mapping[str, Any]) -> SourceSnapshot:
        try:
            return SourceSnapshot.model_validate(manifest)
        except ValidationError as exc:
            raise PluginError(
                SCHEMA_INVALID,
                "source snapshot does not match the schema",
                {"errors": exc.errors(include_url=False)},
            ) from exc

    def validate_dataset(self, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        snap = self._parse(dataset_manifest)
        root = Path(snap.root)
        problems = []
        for f in snap.files:
            p = root / f.path
            if not p.is_file():
                problems.append(f"{f.path}: missing")
            elif hashlib.sha256(p.read_bytes()).hexdigest() != f.sha256:
                problems.append(f"{f.path}: hash does not match the manifest")
        return ValidationReport(
            ok=not problems,
            modality="source_code",
            n_items=len(snap.files),
            groups=sorted({Path(f.path).parts[0] for f in snap.files}),
            problems=problems,
            stats={"tree_sha256": tree_sha256(root) if root.exists() else None},
        ).as_dict()

    def feature_view(self, name: str, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        require_view(CAPABILITY, name)
        if name == "package_metadata":
            raise PluginError(
                SCHEMA_INVALID, "package_metadata takes an archive: use extract_metadata()"
            )
        snap = self._parse(dataset_manifest)
        rows = {}
        for f in snap.files:
            if not f.path.endswith(".py"):
                continue
            src = (Path(snap.root) / f.path).read_text(encoding="utf-8")
            tree = ast.parse(src)
            rows[f.path] = {
                "functions": sum(
                    isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) for n in ast.walk(tree)
                ),
                "classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
                "imports": sorted(
                    {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
                    | {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
                ),
                "lines": src.count("\n") + 1,
                "security_findings": security_scan(src, f.path),
            }
        return {"view": name, "rows": rows}


def write_fixture(root: Path) -> Path:
    """A tiny project with a bug and a test that catches it (test use only)."""
    (root / "src").mkdir(parents=True, exist_ok=True)
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "src" / "__init__.py").write_text("")
    (root / "src" / "calc.py").write_text(
        "def mean(values):\n    return sum(values) / (len(values) + 1)\n"
    )
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_calc.py").write_text(
        "import unittest\n\nfrom src.calc import mean\n\n\n"
        "class T(unittest.TestCase):\n    def test_mean(self):\n"
        "        self.assertEqual(mean([2, 4]), 3)\n"
    )
    return root


__all__ = [
    "CAPABILITY",
    "CodeAdapter",
    "amend",
    "docker_test_runner",
    "extract_metadata",
    "snapshot_manifest",
    "write_fixture",
]
