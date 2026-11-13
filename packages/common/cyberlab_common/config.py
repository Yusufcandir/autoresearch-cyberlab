"""Platform configuration (spec section 11.3).

Non-secret settings come from environment variables. Secrets must come from a
secret store (or a mode-600 `.env` on the host); when a required value is missing,
the affected feature is blocked rather than falling back to an insecure default.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerBackend(StrEnum):
    NONE = "none"
    SLURM = "slurm"
    LOCAL_LOCK = "local_lock"  # single-GPU file lock while no Slurm exists
    LOCAL_TEST = "local_test"


class SandboxBackend(StrEnum):
    NONE = "none"
    ISOLATED_VM = "isolated_vm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"

    # Secrets and connection strings: no defaults on purpose.
    database_url: SecretStr | None = None
    secret_key: SecretStr | None = None
    redis_url: SecretStr | None = None  # optional; the job queue lives in the database
    mlflow_tracking_uri: str | None = None
    artifact_store_uri: str | None = None

    # LLM gateway
    llm_provider: str | None = None  # "gemini", "ollama" or "fake"
    llm_token_budget: int = Field(default=0, ge=0)
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-2.5-flash"
    ollama_host: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b-instruct"
    llm_timeout_seconds: float = Field(default=120, gt=0)

    # Literature services (polite pool contact is optional and never required)
    literature_contact_email: str | None = None

    # Compute
    scheduler_backend: SchedulerBackend = SchedulerBackend.NONE
    slurm_partition: str | None = None
    gpu_job_max_minutes: int = Field(default=60, gt=0)
    max_project_storage_gb: int = Field(default=50, gt=0)
    use_gpu: bool = False
    data_dir: Path = Path("./var")

    sandbox_backend: SandboxBackend = SandboxBackend.NONE
    # Must default to false (spec 11.3).
    allow_restricted_execution: bool = False

    approval_policy_version: str = "1"
    audit_log_destination: str | None = None

    @property
    def artifact_root(self) -> Path:
        uri = self.artifact_store_uri
        if uri and uri.startswith("file:"):
            return Path(uri.removeprefix("file://").removeprefix("file:"))
        return self.data_dir / "artifacts"

    @property
    def dataset_root(self) -> Path:
        return self.data_dir / "datasets"

    def missing_required(self) -> list[str]:
        """Names of settings that must be set before the control plane is 'ready'."""
        required = {
            "DATABASE_URL": self.database_url,
            "SECRET_KEY": self.secret_key,
            "AUDIT_LOG_DESTINATION": self.audit_log_destination,
        }
        return [name for name, value in required.items() if value is None]


class IsolationRequiredError(RuntimeError):
    """Raised when restricted execution is requested without an isolated host (AT-12)."""

    code = "BLOCKED: ISOLATION_REQUIRED"


def require_isolated_execution(settings: Settings) -> None:
    """Gate for any restricted (untrusted) task. Never falls back to the shared GPU host."""
    if not settings.allow_restricted_execution or settings.sandbox_backend is SandboxBackend.NONE:
        raise IsolationRequiredError(IsolationRequiredError.code)
