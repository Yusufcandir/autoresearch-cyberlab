"""Deterministic execution and tool policy. An LLM never decides these outcomes."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path
from typing import Any

from cyberlab_common.config import SandboxBackend, Settings

ALLOWED = "ALLOWED"
ISOLATION_REQUIRED = "BLOCKED: ISOLATION_REQUIRED"
APPROVAL_REQUIRED = "BLOCKED: APPROVAL_REQUIRED"
TOOL_NOT_ALLOWED = "BLOCKED: TOOL_NOT_ALLOWED"
TOOL_FORBIDDEN = "BLOCKED: TOOL_FORBIDDEN"
UNKNOWN_TIER = "BLOCKED: UNKNOWN_TIER"


@dataclass(frozen=True)
class Policy:
    version: str
    status: str
    tiers: dict[str, dict[str, Any]]
    tools: dict[str, dict[str, Any]]
    forbidden_tools: frozenset[str]


@dataclass(frozen=True)
class ExecutionRequest:
    tool: str
    tier: str
    actor: str = "agent"
    approved: bool = False
    project_id: str | None = None


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str
    reason: str
    plane: str | None
    policy_version: str
    request: ExecutionRequest
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def load_policy(path: Path | None = None) -> Policy:
    if path is None:
        raw = resources.files("cyberlab_authz").joinpath("security_policy_v1.toml").read_text()
    else:
        raw = path.read_text(encoding="utf-8")
    data = tomllib.loads(raw)
    return Policy(
        version=str(data["version"]),
        status=str(data.get("status", "draft")),
        tiers=data["tiers"],
        tools=data.get("tools", {}),
        forbidden_tools=frozenset(data.get("forbidden", {}).get("tools", [])),
    )


def decide(request: ExecutionRequest, settings: Settings, policy: Policy | None = None) -> Decision:
    policy = policy or load_policy()

    def result(allowed: bool, code: str, reason: str, plane: str | None = None) -> Decision:
        return Decision(allowed, code, reason, plane, policy.version, request)

    if request.tool in policy.forbidden_tools:
        return result(False, TOOL_FORBIDDEN, f"'{request.tool}' is forbidden for agents.")
    tier = policy.tiers.get(request.tier)
    if tier is None:
        return result(False, UNKNOWN_TIER, f"Unknown tier '{request.tier}'.")
    tool = policy.tools.get(request.tool)
    if tool is None or request.tier not in tool.get("tiers", []):
        return result(
            False, TOOL_NOT_ALLOWED, f"'{request.tool}' is not allowed at {request.tier}."
        )

    plane = str(tier["plane"])
    if plane == "isolated":
        isolated_ok = (
            settings.allow_restricted_execution
            and settings.sandbox_backend is not SandboxBackend.NONE
        )
        if not isolated_ok:
            return result(
                False,
                ISOLATION_REQUIRED,
                "No isolated evaluation host is configured; the shared workstation is never used.",
            )
        if not request.approved:
            return result(False, APPROVAL_REQUIRED, "Restricted benchmarks need PI approval.")
    if tier.get("requires_approval") and not request.approved:
        return result(False, APPROVAL_REQUIRED, f"{request.tier} work needs human review first.")
    return result(True, ALLOWED, f"Route to the {plane} plane.", plane)


def write_audit(decision: Decision, settings: Settings) -> Path | None:
    """Append the decision to a JSONL audit log when AUDIT_LOG_DESTINATION is a file: URI."""
    dest = settings.audit_log_destination
    if not dest or not dest.startswith("file:"):
        return None
    path = Path(dest.removeprefix("file://").removeprefix("file:"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(decision.to_json() + "\n")
    return path
