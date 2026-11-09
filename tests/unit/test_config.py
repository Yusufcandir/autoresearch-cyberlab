import pytest

from cyberlab_common.config import (
    IsolationRequiredError,
    SandboxBackend,
    Settings,
    require_isolated_execution,
)


def test_restricted_execution_defaults_off() -> None:
    assert Settings(_env_file=None).allow_restricted_execution is False


def test_restricted_task_blocked_without_isolated_host() -> None:
    # AT-12: restricted work with no isolated host must be blocked, not run locally.
    with pytest.raises(IsolationRequiredError, match="ISOLATION_REQUIRED"):
        require_isolated_execution(Settings(_env_file=None, allow_restricted_execution=True))


def test_restricted_task_blocked_when_flag_off() -> None:
    with pytest.raises(IsolationRequiredError):
        require_isolated_execution(
            Settings(_env_file=None, sandbox_backend=SandboxBackend.ISOLATED_VM)
        )


def test_restricted_task_allowed_only_with_both() -> None:
    require_isolated_execution(
        Settings(
            _env_file=None,
            allow_restricted_execution=True,
            sandbox_backend=SandboxBackend.ISOLATED_VM,
        )
    )
