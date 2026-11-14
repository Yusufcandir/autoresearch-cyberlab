# AutoResearch-CyberLab

A multi-agent research platform for **AI-for-cybersecurity** studies. A researcher writes a
research goal; ten AI agents plan the study, review the literature, propose a protocol, check
the dataset, run experiments on the lab GPU, analyse the results and draft a paper with
figures. The PI stays in charge: nothing runs on the GPU before the PI approves the protocol,
and nothing is released without the PI.

**Datasets:** CIC-IDS2017 and UNSW-NB15 (chosen per study on the New study form).
**LLMs:** Gemini API (free tier, public data only) with automatic fallback to a local
**Ollama** model (`qwen2.5:14b-instruct`), which is also the only model a private study may use.

## Status (14 Nov 2026)

- Running on the lab's GPU server (RTX 6000 Ada): PostgreSQL 16 in Docker, dashboard and
  worker as systemd user services, Ollama local model.
- Studies run end to end on real data: two on CIC-IDS2017 (one released) and one on UNSW-NB15
  with the official train/test split.
- 133 automated tests pass (`bash scripts/check.sh`); 21 of the 22 acceptance tests are done.
  AT-13 needs an isolated VM, which the pilot scope excludes.

## Core rules (enforced in code, not in prompts)

- **No invented numbers.** Every number in a manuscript (results, tables, statistics, method
  facts, figures) is generated from stored, hashed per-row predictions. The LLM writes only
  framing text; sentences that state results, compare models without a test, or guess at
  untested mechanisms are removed by deterministic checks.
- **No invented references.** Every reference must resolve to a real DOI with a matching title
  (Crossref/OpenAlex); off-topic search results are dropped.
- **The PI decides.** Approvals are bound to the exact protocol hash. Model settings are fixed,
  reviewed defaults; only the PI's comment can change them (no tuning on the test split).
- **Leakage-safe data.** Duplicates and conflicting labels are removed before splitting; each
  split (stratified, by capture day, official partition) gets its own audited copy, and a test
  row may never appear in training.
- **Safe execution.** One GPU job at a time with time and budget caps; only reviewed model
  templates run. Untrusted (T3) code is always blocked; reviewed fixtures run only in a
  locked-down Docker runner (no network, read-only, non-root).
- Every agent tool call passes a deterministic security policy and is audited.

## The ten agents

| # | Agent | What it does |
|---|---|---|
| 1 | Chief Research Orchestrator | plan, search queries, risks |
| 2 | Literature Review | OpenAlex search, relevance filter, DOI verification, evidence matrix |
| 3 | Research Gap and Novelty | proposed gaps with honest caveats |
| 4 | Methodology Design | protocol; the researcher's own hypotheses, primary metric and split are taken from the goal |
| 5 | Dataset | cleaning, de-duplication, hashes, leakage-safe split (per dataset adapter) |
| 6 | Coding | compiles the approved protocol into reviewed experiment templates |
| 7 | Experiment Execution | submits jobs to the broker (GPU lock, caps, crash recovery) |
| 8 | Statistical Analysis | metrics from predictions, seed-averaged bootstrap CIs, McNemar on every seed, Holm correction |
| 9 | Scientific Writing | framing text; Methodology, Results and findings are generated from evidence; figures; BibTeX |
| 10 | Scientific Reviewer | deterministic consistency checks, Related Work coverage of every verified paper, one revision |

Workflow (LangGraph, checkpointed per study): orchestrator, literature, gaps, methodology,
dataset, **PI gate A (protocol)**, coding, experiments, statistics, writing, reviewer,
**PI gate D (release)**. **Redraft** rewrites a paper and recomputes the statistics from the
stored predictions without re-running experiments.

## What a paper contains

Abstract, Introduction, Related Work (every verified paper covered), Methodology (generated
from the recorded preprocessing and settings), Results (means ± sd over seeds, 95% bootstrap
CIs, per-class table, recall on the smallest classes, hypothesis table with Holm-adjusted p),
three figures (CI plot, per-class recall, confusion matrix), Discussion (including false-positive
rates and classes no model detected), Limitations (small-class caveat), Conclusion, Evidence
hashes and verified References. Exports: Markdown, LaTeX zip (with figures and BibTeX), print
view for PDF.

## Layout

| Path | Contents |
|---|---|
| `services/orchestrator` | the 10 agents, LangGraph workflow, `Lab` facade |
| `services/api` | REST API (`/api/v1/...`) and the PI dashboard |
| `services/llm_gateway` | Gemini, Ollama and offline providers; handover deadline, circuit breaker, budgets, privacy routing |
| `services/literature` | OpenAlex search, Crossref verification, BibTeX |
| `services/data_registry` | dataset registry and adapters (CIC-IDS2017, UNSW-NB15), shared cleaning and split gate |
| `services/protocol_compiler`, `services/execution_broker` | research-intent parsing, approved protocol to jobs; broker and runner |
| `services/scientific_qa` | metrics, bootstrap CIs, McNemar, Holm |
| `services/evidence_service`, `services/manuscript_builder` | hashed artifacts; evidence-locked text, consistency checks, figures, LaTeX |
| `plugins/` | model templates (tabular) and capability plugins: text, audio, graph policy, code artifact |
| `packages/` | schemas, auth, roles and security policy, config, CLI, health check, migration, restore |
| `infra/` | Docker Compose (PostgreSQL), systemd units, env example |

Internal documentation (server runbook, step-by-step guide, requirements status, governance
and threat model) is kept in `docs/` locally and is not published.

## Quick start (any machine, offline)

```bash
uv sync --locked
cp infra/env.example .env          # set SECRET_KEY; LLM_PROVIDER=fake works offline
uv run cyberlab init-db
uv run cyberlab create-user admin --role admin
uv run cyberlab dataset fixture    # tiny synthetic sample, for trying it out only
uv run cyberlab serve              # http://127.0.0.1:8100
uv run cyberlab worker             # second terminal: runs the experiments
bash scripts/check.sh              # lint, types, tests
```

## Useful commands

| Command | Purpose |
|---|---|
| `cyberlab dataset prepare --name cicids2017` / `--name unsw-nb15` | clean and verify a dataset once after placing its files |
| `cyberlab health-check [--alert]` | disk, GPU, database, failed jobs, local model, backup age |
| `cyberlab llm-check` | one test prompt through Gemini/Ollama |
| `cyberlab restore-test` | restore the newest backup and re-verify every evidence hash |
| `cyberlab plugin-selftest [--docker]` | run the capability plugins on their fixtures |
| `cyberlab migrate-db` | copy all data into another database (SQLite to PostgreSQL) |
| `bash scripts/install_services.sh` | install serve, worker, Ollama, health and backup as user services |
| `bash scripts/backup.sh` | backup (database dump + evidence), retention, optional off-site copy |
