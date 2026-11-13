# AutoResearch-CyberLab

A multi-agent research platform for **AI-for-cybersecurity** studies. A researcher writes a
research goal; ten AI agents plan the study, review the literature, propose a protocol, check
the dataset, run experiments on the lab GPU, analyze results and draft a paper. The PI stays in
charge: nothing runs on the GPU before the PI approves the protocol, and nothing is released
without the PI.

The pilot dataset is **CIC-IDS2017**. LLMs: **Gemini API** (free tier, public data) and
**Ollama** (local model, private data).

## Core rules (enforced in code, not in prompts)

- Every number in a manuscript comes from stored, hashed predictions. Agent prose uses
  placeholders; any other number is flagged and, in the final revision, removed.
- Every reference must resolve to a real DOI with a matching title (Crossref/OpenAlex).
- Approvals are bound to the exact protocol hash; a changed protocol needs a new approval.
- One GPU job at a time, with time and budget caps. Untrusted (T3) work is always blocked,
  because there is no isolated host.
- Every agent tool call passes a deterministic security policy and is audited.

## The ten agents

| # | Agent | What it does |
|---|---|---|
| 1 | Chief Research Orchestrator | plan, search queries, risks |
| 2 | Literature Review | OpenAlex search, DOI verification, evidence matrix |
| 3 | Research Gap and Novelty | proposed gaps with honest caveats |
| 4 | Methodology Design | protocol (hypotheses, models, seeds, tests), validated against allowlists |
| 5 | Dataset | CIC-IDS2017 cleaning, de-duplication, hashes, leakage-safe split |
| 6 | Coding | compiles the approved protocol into reviewed experiment templates |
| 7 | Experiment Execution | submits jobs to the broker (GPU lock, caps, crash recovery) |
| 8 | Statistical Analysis | metrics from predictions, bootstrap CIs, McNemar tests, leakage audit |
| 9 | Scientific Writing | draft with evidence placeholders, BibTeX |
| 10 | Scientific Reviewer | evidence checks plus a scored peer review |

Workflow (LangGraph, checkpointed per study): orchestrator, literature, gaps, methodology,
dataset, **PI gate A**, coding, experiments, statistics, writing, reviewer (one revision),
**PI gate D (release)**.

## Layout

| Path | Contents |
|---|---|
| `services/orchestrator` | the 10 agents, LangGraph workflow, `Lab` facade |
| `services/api` | REST API (`/api/v1/...`) and the PI dashboard |
| `services/llm_gateway` | Gemini, Ollama and offline providers; budgets; privacy routing |
| `services/literature` | OpenAlex search, Crossref verification, BibTeX |
| `services/data_registry` | CIC-IDS2017 adapter |
| `services/protocol_compiler`, `services/execution_broker` | approved protocol to jobs; broker and runner |
| `services/scientific_qa`, `services/evidence_service`, `services/manuscript_builder` | statistics, hashed artifacts, evidence-bound drafts |
| `plugins/text_tabular` | reviewed model templates (logistic regression, random forest, XGBoost, MLP) |
| `packages/` | schemas, auth, roles and security policy, config, CLI |
| `docs/operator_runbook.md` | **how to run it on the lab server** |
| `docs/validation_matrix.md` | acceptance tests AT-01 to AT-22 and where each is covered |

## Quick start (any machine)

```bash
uv sync
cp infra/env.example .env          # set SECRET_KEY; LLM_PROVIDER=fake works offline
uv run cyberlab init-db
uv run cyberlab create-user admin --role admin
uv run cyberlab dataset fixture    # tiny synthetic sample, for trying it out only
uv run cyberlab serve              # http://127.0.0.1:8100
uv run cyberlab worker             # second terminal: runs the experiments
bash scripts/check.sh              # lint, types, tests
```

For real CIC-IDS2017 data, Gemini/Ollama and Docker on the lab server, follow
[docs/operator_runbook.md](docs/operator_runbook.md).
