"""A study that fails on an LLM error resumes from its checkpoint when retried."""

from __future__ import annotations

from pathlib import Path

from cyberlab_authz.rbac import Principal, create_user
from cyberlab_llm import LLMError
from cyberlab_orchestrator.lab import Lab
from cyberlab_orchestrator.offline import offline_responder
from cyberlab_schemas import ResearchSpec
from tests.fixtures.fakes import fake_gateway, fake_scholar, make_settings


def test_retry_resumes_after_error(tmp_path: Path) -> None:
    calls = {"n": 0}

    def flaky(task: str, system: str, prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise LLMError("simulated timeout")
        return offline_responder(task, system, prompt)

    lab = Lab(make_settings(tmp_path), llm=fake_gateway(flaky), scholar=fake_scholar())
    with lab.db.session() as s:
        u = create_user(s, "prof", "correct-horse-battery", "pi")
        pi = Principal(u.id, u.username, u.role)
    pid = lab.create_project(
        pi, ResearchSpec(title="Retry test study", goal="Check that retry resumes the workflow.")
    )
    try:
        lab.start(pid)
    except LLMError:
        pass
    assert lab.status(pid)["stage"] == "error"
    status = lab.start(pid)  # what the Retry button calls
    assert status["pending"]["gate"] == "A"
    lab.close()
