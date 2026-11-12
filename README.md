# AutoResearch-CyberLab

A secure, reproducible multi-agent research platform. The first pilot targets
**AI for cybersecurity** (for example, explainable intrusion detection on public
datasets). Humans stay accountable: agents propose, a PI approves, and every
reported number must trace back to a stored, hashed artifact.

> Status: **Phase A built** (Steps 01 to 05). Draft governance, security and topology docs await
> sign-off; runnable host checks and the security policy work. Nothing here runs experiments yet.

## Four planes

1. **Control plane**: dashboard, auth, LangGraph orchestration, approvals, LLM gateway.
2. **Trusted compute**: approved CPU/GPU jobs on the shared RTX 6000 Ada via the scheduler.
3. **Isolated evaluation**: separate VM for untrusted code. If it does not exist,
   those tasks stay `BLOCKED: ISOLATION_REQUIRED`.
4. **Evidence plane**: PostgreSQL, MLflow and hashed artifacts.

## Layout

| Path | Contents |
|---|---|
| `services/` | api, orchestrator, llm_gateway, data_registry, protocol_compiler, execution_broker, scientific_qa, evidence_service, manuscript_builder |
| `plugins/` | text_tabular, audio_multimodal, graph_policy, code_artifact, agent_evaluation, statistical_methods |
| `packages/` | schemas, authz, common (shared config) |
| `ui/` | PI dashboard (Next.js, Step 09) |
| `infra/` | dev compose file, env example, Slurm, monitoring, backup |
| `docs/` | architecture, threat model, governance, runbooks, Phase A discovery |
| `tests/` | unit, integration, e2e, security, reproducibility, fixtures |

## Developer quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy
uv run uvicorn cyberlab_api.main:app --reload   # http://127.0.0.1:8000/health/live
```

Optional dev databases (on a dev machine or approved VM, **not** the shared workstation):

```bash
cp infra/env.example .env   # then fill in local-only values
docker compose -f infra/compose.dev.yaml up -d
```

## On the lab server

See [docs/development_guide.md](docs/development_guide.md):
`bash scripts/setup_server.sh`, then `uv run cyberlab doctor` and
`uv run cyberlab policy-check run_restricted_benchmark T3 --approved`.

## Rules

- Never commit API keys, restricted datasets, production logs or private prompts.
- `ALLOW_RESTRICTED_EXECUTION` defaults to `false`; missing secrets block features.
- Do not change drivers, open ports or install packages on the shared host without admin approval.

See [docs/implementation_plan.md](docs/implementation_plan.md) for the step tracker.
