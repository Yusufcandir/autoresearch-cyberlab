"""Platform configuration (spec section 11.3).

Non-secret settings come from environment variables. Secrets must come from a
secret store; when a required value is missing, the affected feature is blocked
rather than falling back to an insecure development default.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SchedulerBackend(StrEnum):
    NONE = "none"
    SLURM = "slurm"
    LOCAL_LOCK = "local_lock"  # single-GPU lock while no Slurm exists
    LOCAL_TEST = "local_test"


class SandboxBackend(StrEnum):
    NONE = "none"
    ISOLATED_VM = "isolated_vm"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = "development"

    # Secrets / connection strings: no defaults on purpose.
    database_url: SecretStr | None = None
    redis_url: SecretStr | None = None
    mlflow_tracking_uri: str | None = None
    artifact_store_uri: str | None = None

    llm_provider: str | None = None
    llm_token_budget: int = Field(default=0, ge=0)

    scheduler_backend: SchedulerBackend = SchedulerBackend.NONE
    slurm_partition: str | None = None
    gpu_job_max_minutes: int = Field(default=60, gt=0)
    max_project_storage_gb: int = Field(default=50, gt=0)

    sandbox_backend: SandboxBackend = SandboxBackend.NONE
    # Must default to false (spec 11.3).
    allow_restricted_execution: bool = False

    approval_policy_version: str = "0"
    audit_log_destination: str | None = None

    def missing_required(self) -> list[str]:
        """Names of settings that must be set before the control plane is 'ready'."""
        required = {
            "DATABASE_URL": self.database_url,
            "REDIS_URL": self.redis_url,
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
