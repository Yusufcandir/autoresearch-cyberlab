"""Pydantic contracts shared by every service.

Every document carries `schema_version`; unknown versions are rejected so stale clients cannot
start work. Content hashes make approvals version-specific: changing a protocol changes its hash,
and an approval for the old hash no longer authorizes the new one (AT-06).
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "1"


def utcnow() -> datetime:
    return datetime.now(UTC)


def content_hash(obj: BaseModel | dict[str, Any]) -> str:
    """Stable SHA-256 of a model's content, ignoring volatile bookkeeping fields."""
    data = obj.model_dump(mode="json") if isinstance(obj, BaseModel) else obj
    data = {k: v for k, v in data.items() if k not in {"created_at", "content_sha256"}}
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class StatusLabel(StrEnum):
    """UI labels from spec 10.3. Never show prose and measured results with the same label."""

    PROPOSED = "Proposed"
    RUNNING = "Running"
    MEASURED = "Measured"
    VERIFIED = "Verified"
    UNVERIFIED = "Unverified"
    BLOCKED = "Blocked"
    APPROVED = "Approved"


class Versioned(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1"] = "1"

    @field_validator("schema_version", mode="before")
    @classmethod
    def _known_version(cls, v: Any) -> Any:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unknown schema_version {v!r}; this server supports {SCHEMA_VERSION}")
        return v


class ResearchSpec(Versioned):
    title: str = Field(min_length=5, max_length=300)
    goal: str = Field(min_length=20, max_length=5000)
    domain: Literal["cybersecurity"] = "cybersecurity"
    dataset_hint: str = "CIC-IDS2017"
    constraints: list[str] = Field(default_factory=list)
    data_may_leave_host: bool = Field(
        default=False,
        description="If false, only the local LLM (Ollama) may see study data.",
    )
    gpu_budget_minutes: int = Field(default=480, ge=1, le=7 * 24 * 60)
    llm_token_budget: int = Field(default=1_000_000, ge=0)


class Hypothesis(Versioned):
    id: str = Field(pattern=r"^H\d+$")
    statement: str
    primary_metric: Literal["macro_f1", "accuracy", "balanced_accuracy", "mcc"] = "macro_f1"
    comparison: str = Field(description="e.g. 'xgboost > logistic_regression'")


class ModelSpec(Versioned):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    template: Literal["logistic_regression", "random_forest", "xgboost", "mlp"]
    params: dict[str, Any] = Field(default_factory=dict)
    role: Literal["baseline", "candidate", "ablation"] = "baseline"


class SplitSpec(Versioned):
    strategy: Literal["stratified_dedup", "by_day"] = "stratified_dedup"
    test_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    val_fraction: float = Field(default=0.1, ge=0.0, lt=0.4)
    seed: int = 42
    test_days: list[str] = Field(default_factory=list)


StatTest = Literal["bootstrap_ci", "mcnemar"]


def _default_tests() -> list[StatTest]:
    return ["bootstrap_ci", "mcnemar"]


class Protocol(Versioned):
    project_id: str
    version: int = Field(ge=1)
    research_questions: list[str] = Field(min_length=1)
    hypotheses: list[Hypothesis] = Field(min_length=1)
    dataset_id: str
    label_scheme: Literal["binary", "multiclass"] = "multiclass"
    split: SplitSpec = Field(default_factory=SplitSpec)
    models: list[ModelSpec] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [42])
    statistical_tests: list[StatTest] = Field(default_factory=_default_tests)
    gpu_budget_minutes: int = Field(default=480, ge=1)
    risk_tier: Literal["T0", "T1", "T2", "T3"] = "T1"
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def sha256(self) -> str:
        return content_hash(self)


class DatasetManifest(Versioned):
    dataset_id: str
    name: str
    source_url: str
    license: str
    citation: str
    files: dict[str, str] = Field(description="relative file name -> sha256")
    n_rows_raw: int = Field(ge=0)
    n_rows_clean: int = Field(ge=0)
    n_duplicates_removed: int = Field(ge=0)
    features: list[str]
    label_column: str
    class_counts: dict[str, int]
    split_counts: dict[str, int] = Field(default_factory=dict)
    split_overlap_rows: int = 0
    status: Literal["verified", "blocked"] = "blocked"
    blocked_reasons: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)


class ExperimentManifest(Versioned):
    experiment_id: str
    project_id: str
    protocol_version: int
    protocol_sha256: str
    dataset_id: str
    model: ModelSpec
    seed: int
    label_scheme: Literal["binary", "multiclass"]
    split: SplitSpec
    risk_tier: Literal["T0", "T1", "T2", "T3"]
    max_minutes: int = Field(ge=1)
    use_gpu: bool = False


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    BLOCKED = "blocked"


class Job(Versioned):
    job_id: str
    project_id: str
    experiment_id: str
    idempotency_key: str
    state: JobState = JobState.QUEUED
    external_ref: str | None = None
    failure_reason: str | None = None


class RunReceipt(Versioned):
    run_id: str
    job_id: str
    experiment_id: str
    status: Literal["succeeded", "failed"]
    started_at: datetime
    finished_at: datetime
    artifacts: dict[str, str] = Field(description="artifact name -> sha256")
    code_version: str
    environment: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


class EvidenceClaim(Versioned):
    claim_id: str
    project_id: str
    text: str
    kind: Literal["metric", "comparison", "citation", "narrative"]
    run_id: str | None = None
    metric: str | None = None
    value: float | None = None
    artifact_sha256: str | None = None
    doi: str | None = None
    status: StatusLabel = StatusLabel.UNVERIFIED


class ApprovalGate(StrEnum):
    A_PROTOCOL = "A_protocol_and_data_rights"
    B_SCOPE_CHANGE = "B_scope_or_budget_change"
    C_RESTRICTED = "C_restricted_execution"
    D_RELEASE = "D_manuscript_release"


class Approval(Versioned):
    project_id: str
    gate: ApprovalGate
    subject_version: int
    subject_sha256: str
    approver: str
    decision: Literal["approved", "rejected"]
    comment: str = ""
    created_at: datetime = Field(default_factory=utcnow)
