"""Shared runtime context handed to every agent. Agents get tools, never raw credentials."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.orm import Session

from cyberlab_authz.policy import ExecutionRequest, decide, write_audit
from cyberlab_common.config import Settings
from cyberlab_common.db import AgentNote, Database, Project, audit
from cyberlab_evidence import ArtifactStore
from cyberlab_exec import Broker
from cyberlab_literature import ScholarlyClient
from cyberlab_llm import LLMGateway


class ToolDenied(RuntimeError):
    pass


@dataclass
class LabContext:
    settings: Settings
    db: Database
    llm: LLMGateway
    scholar: ScholarlyClient
    broker: Broker
    store: ArtifactStore
    extra: dict[str, Any] = field(default_factory=dict)

    def authorize(self, agent: str, tool: str, tier: str, project_id: str) -> None:
        """Every agent tool call passes the deterministic policy and is audited (AT-07)."""
        d = decide(ExecutionRequest(tool, tier, actor=agent, project_id=project_id), self.settings)
        write_audit(d, self.settings)
        if not d.allowed:
            with self.db.session() as s:
                audit(
                    s, agent, "security.policy_denied", project_id, {"tool": tool, "code": d.code}
                )
            raise ToolDenied(f"{agent}: {d.code}: {d.reason}")

    def note(
        self, s: Session, project_id: str, agent: str, kind: str, content: dict[str, Any]
    ) -> None:
        s.add(AgentNote(project_id=project_id, agent=agent, kind=kind, content=content))

    def set_stage(
        self, project_id: str, stage: str, label: str, blocked: str | None = None
    ) -> None:
        with self.db.session() as s:
            p = s.get(Project, project_id)
            if p is not None:
                p.stage, p.status_label, p.blocked_reason = stage, label, blocked
                audit(
                    s,
                    "orchestrator",
                    "project.stage",
                    project_id,
                    {"stage": stage, "blocked": blocked},
                )
