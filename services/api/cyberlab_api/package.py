"""Reproducibility package of a study (brief §1.2 item 12: deliver the manuscript, source code
version, dataset references, model artifacts and reproducibility documentation).

One zip with: README (what each file is and how to reproduce), the approved protocol, the
analysis, the latest manuscript and bibliography, a table of every run (seed, dataset, code
version, data hash, scores, timing), and each run's predictions, settings and model info.
Trained models are listed with their SHA-256 (they stay in the lab's artifact store; they can
be large)."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cyberlab_common.db import (
    AgentNote,
    ArtifactRecord,
    ManuscriptRecord,
    Project,
    ProtocolVersion,
    RunRecord,
)

RUN_FILES = ("predictions.csv", "model_info.json", "tuning.json")
SCORES = ("macro_f1", "accuracy", "balanced_accuracy", "mcc", "pr_auc_macro", "pr_auc")


def _path(s: Session, project_id: str, sha: str) -> Path | None:
    rec = s.scalar(
        select(ArtifactRecord).where(
            ArtifactRecord.project_id == project_id, ArtifactRecord.sha256 == sha
        )
    )
    return Path(rec.path) if rec and Path(rec.path).exists() else None


def build_package(s: Session, project_id: str) -> tuple[str, bytes]:
    project = s.get(Project, project_id)
    if project is None:
        raise ValueError("no such study")
    spec = dict(project.spec)
    protocol = s.scalar(
        select(ProtocolVersion)
        .where(ProtocolVersion.project_id == project_id)
        .order_by(ProtocolVersion.version.desc())
    )
    ms = s.scalar(
        select(ManuscriptRecord)
        .where(ManuscriptRecord.project_id == project_id)
        .order_by(ManuscriptRecord.version.desc())
    )
    analysis = s.scalar(
        select(ArtifactRecord)
        .where(ArtifactRecord.project_id == project_id, ArtifactRecord.name == "analysis.json")
        .order_by(ArtifactRecord.id.desc())
    )
    runs = list(
        s.scalars(
            select(RunRecord)
            .where(RunRecord.project_id == project_id, RunRecord.status == "succeeded")
            .order_by(RunRecord.experiment_id)
        )
    )
    if protocol is not None:
        runs = [r for r in runs if r.experiment_id.startswith(f"p{protocol.version}-")]

    buf = io.BytesIO()
    models: list[dict[str, Any]] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if protocol is not None:
            z.writestr(
                "research_protocol.json",
                json.dumps(
                    {"version": protocol.version, "sha256": protocol.sha256, **protocol.body},
                    indent=2,
                    default=str,
                ),
            )
        if analysis is not None and Path(analysis.path).exists():
            z.write(analysis.path, "analysis.json")
        report_rec = s.scalar(
            select(ArtifactRecord)
            .where(
                ArtifactRecord.project_id == project_id,
                ArtifactRecord.name == "scientific_validation_report.json",
            )
            .order_by(ArtifactRecord.id.desc())
        )
        if report_rec is not None and Path(report_rec.path).exists():
            report = json.loads(Path(report_rec.path).read_text())
            z.writestr("scientific_validation_report.json", json.dumps(report, indent=2))
            # the professor's plan names these reports (Phase 1 deliverables)
            for rname, part in (report.get("reports") or {}).items():
                if part:
                    z.writestr(f"{rname}.json", json.dumps(part, indent=2, default=str))
        for kind, fname in (
            ("research_decision", "research_decision_log.json"),
            ("citation_verification", "citation_verification_report.json"),
            ("manuscript_critique", "manuscript_critique_log.json"),
        ):
            notes = [
                {"agent": n.agent, "at": n.created_at.isoformat(), **(n.content or {})}
                for n in s.scalars(
                    select(AgentNote)
                    .where(AgentNote.project_id == project_id, AgentNote.kind == kind)
                    .order_by(AgentNote.id)
                )
            ]
            if notes:
                z.writestr(fname, json.dumps(notes, indent=2, default=str))
        if ms is not None:
            z.writestr(f"manuscript-v{ms.version}.md", ms.markdown)
            z.writestr(f"manuscript-v{ms.version}.bib", ms.bibtex or "")
        rows = []
        for r in runs:
            rc = r.receipt or {}
            arts = rc.get("artifacts", {})
            info: dict[str, Any] = {}
            folder = f"runs/{r.experiment_id}"
            for name in RUN_FILES:
                p = _path(s, project_id, arts[name]) if name in arts else None
                if p is not None:
                    z.write(p, f"{folder}/{name}")
                    if name == "model_info.json":
                        info = json.loads(p.read_text())
            if "model.joblib" in arts:
                models.append(
                    {
                        "experiment_id": r.experiment_id,
                        "file": "model.joblib",
                        "sha256": arts["model.joblib"],
                    }  # fmt: skip
                )
            rows.append(
                {
                    "experiment_id": r.experiment_id,
                    "model": r.model_name,
                    "seed": r.seed,
                    "dataset": r.experiment_id.split("@", 1)[1]
                    if "@" in r.experiment_id
                    else (protocol.body.get("dataset_id") if protocol else ""),
                    "code_version": rc.get("code_commit") or rc.get("code_version", ""),
                    "data_sha256": rc.get("split_manifest_sha256", ""),
                    "dataset_manifest_sha256": rc.get("dataset_manifest_sha256", ""),
                    "environment": json.dumps(info.get("environment", {}), sort_keys=True),
                    "train_seconds": info.get("train_seconds", ""),
                    "inference_seconds": info.get("inference_seconds", ""),
                    **{k: r.metrics.get(k, "") for k in SCORES},
                }
            )
        if rows:
            out = io.StringIO()
            w = csv.DictWriter(out, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
            z.writestr("runs.csv", out.getvalue())
        z.writestr("trained_models.json", json.dumps(models, indent=2))
        z.writestr(
            "README.md",
            _readme(spec, protocol.version if protocol else None, len(rows), len(models)),
        )
    stem = "".join(c if c.isalnum() else "-" for c in str(spec.get("title", "study")))[:60]
    return f"{stem.strip('-') or 'study'}-reproducibility.zip", buf.getvalue()


def _readme(spec: dict[str, Any], version: int | None, n_runs: int, n_models: int) -> str:
    return f"""# Reproducibility package: {spec.get("title", "")}

Research goal: {spec.get("goal", "")}

Produced by AutoResearch-CyberLab. Every number in the manuscript was computed from the files in
this package; nothing was typed by hand.

| File | Content |
|---|---|
| `research_protocol.json` | The approved experimental protocol (version {version}): hypotheses, models, seeds, split, tuning budget, statistical tests, and its SHA-256 (the approval is bound to it). |
| `analysis.json` | All metrics, confidence intervals, hypothesis tests, rankings, explanations and generalization results, recomputed from the stored predictions. |
| `manuscript-v*.md`, `.bib` | The latest manuscript draft and its verified bibliography. |
| `scientific_validation_report.json` and `data_preprocessing_audit.json`, `leakage_validation_report.json`, `statistical_validation_report.json`, `rare_class_reliability_report.json` | Every automatic scientific check (PASS / WARNING / BLOCK / NOT_EVALUATED) behind the paper. |
| `research_decision_log.json` | The agents' research decisions: evidence, problem, proposed action, justification, cost, approval and outcome. |
| `citation_verification_report.json` | Whether each cited paper exists, matches its DOI and is cited where it is listed. |
| `manuscript_critique_log.json` | The manuscript critic's rounds and the problems it found. |
| `runs.csv` | One row per run ({n_runs} runs): model, seed, dataset, code version, data hash, environment, timing and headline scores. |
| `runs/<experiment>/predictions.csv` | Per-row test predictions (true label, predicted label, class probabilities). |
| `runs/<experiment>/model_info.json` | Settings actually used, tuning result, timing, environment, feature importances. |
| `runs/<experiment>/tuning.json` | The settings search (trials, best value) on the validation split. |
| `trained_models.json` | The {n_models} saved trained models (main dataset) with their SHA-256; they are kept in the lab's artifact store and can be provided on request. |

## How to reproduce

1. Check out the platform at the `code_version` given in `runs.csv`.
2. Prepare each dataset with `cyberlab dataset prepare --name <dataset>` (same row caps as in
   `research_protocol.json`); the prepared data must have the `data_sha256` in `runs.csv`.
3. Create a study with the protocol in `research_protocol.json` and approve it; the platform runs the same
   models, seeds and settings search.
4. Any metric can be recomputed independently from `runs/<experiment>/predictions.csv`.
"""
