from cyberlab_authz.policy import (
    ALLOWED,
    APPROVAL_REQUIRED,
    ISOLATION_REQUIRED,
    TOOL_FORBIDDEN,
    TOOL_NOT_ALLOWED,
    UNKNOWN_TIER,
    ExecutionRequest,
    decide,
    load_policy,
    write_audit,
)
from cyberlab_common.config import SandboxBackend, Settings

SHARED_HOST = Settings(_env_file=None)  # defaults: no sandbox, restricted execution off


def test_policy_loads_with_four_tiers() -> None:
    assert set(load_policy().tiers) == {"T0", "T1", "T2", "T3"}


def test_t3_blocked_on_shared_host_at12() -> None:
    d = decide(ExecutionRequest("run_restricted_benchmark", "T3", approved=True), SHARED_HOST)
    assert not d.allowed
    assert d.code == ISOLATION_REQUIRED


def test_t3_with_flag_but_no_sandbox_still_blocked() -> None:
    s = Settings(_env_file=None, allow_restricted_execution=True)
    d = decide(ExecutionRequest("run_restricted_benchmark", "T3", approved=True), s)
    assert d.code == ISOLATION_REQUIRED


def test_t3_with_sandbox_still_needs_approval() -> None:
    s = Settings(
        _env_file=None,
        allow_restricted_execution=True,
        sandbox_backend=SandboxBackend.ISOLATED_VM,
    )
    assert decide(ExecutionRequest("run_restricted_benchmark", "T3"), s).code == APPROVAL_REQUIRED
    ok = decide(ExecutionRequest("run_restricted_benchmark", "T3", approved=True), s)
    assert ok.allowed
    assert ok.plane == "isolated"


def test_t1_template_allowed_on_trusted_compute() -> None:
    d = decide(ExecutionRequest("run_experiment_template", "T1"), SHARED_HOST)
    assert d.code == ALLOWED
    assert d.plane == "trusted_compute"


def test_t2_generated_code_needs_review() -> None:
    assert (
        decide(ExecutionRequest("run_reviewed_code", "T2"), SHARED_HOST).code == APPROVAL_REQUIRED
    )


def test_forbidden_and_unlisted_tools_denied_at07() -> None:
    assert decide(ExecutionRequest("shell", "T1"), SHARED_HOST).code == TOOL_FORBIDDEN
    assert decide(ExecutionRequest("modify_policy", "T0"), SHARED_HOST).code == TOOL_FORBIDDEN
    assert decide(ExecutionRequest("curl_anything", "T0"), SHARED_HOST).code == TOOL_NOT_ALLOWED


def test_tool_used_at_wrong_tier_denied() -> None:
    d = decide(ExecutionRequest("run_experiment_template", "T3"), SHARED_HOST)
    assert d.code == TOOL_NOT_ALLOWED


def test_unknown_tier() -> None:
    assert decide(ExecutionRequest("draft_manuscript", "T9"), SHARED_HOST).code == UNKNOWN_TIER


def test_decision_written_to_audit_log(tmp_path) -> None:  # type: ignore[no-untyped-def]
    log = tmp_path / "audit.jsonl"
    s = Settings(_env_file=None, audit_log_destination=f"file:{log}")
    d = decide(ExecutionRequest("shell", "T1"), s)
    assert write_audit(d, s) == log
    assert "TOOL_FORBIDDEN" in log.read_text()
