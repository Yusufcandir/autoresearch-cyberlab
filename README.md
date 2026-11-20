# AutoResearch-CyberLab

A multi-agent research platform for **AI-for-cybersecurity** studies. A researcher writes a
research goal; ten AI agents plan the study, review the literature, propose a protocol, check
the dataset, run experiments on the lab GPU, analyse the results and draft a paper with
figures. The PI stays in charge: nothing runs on the GPU before the PI approves the protocol,
and nothing is released without the PI.

**Datasets** (chosen per study): CIC-IDS2017, UNSW-NB15, CTU-13 (real botnet traffic), the
NetFlow-v2 family (NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2, NF-BoT-IoT-v2, with
cross-dataset testing) and PhiUSIIL phishing URLs (text modality).
**Models:** logistic regression, random forest, XGBoost, LightGBM, MLP; deep PyTorch models
(ResNet-MLP, FT-Transformer, TabNet, 1D-CNN); TF-IDF text models. Optional Optuna tuning on the
validation split only, with equal budgets per model.
**LLMs:** Claude (Anthropic API) or Gemini, with automatic fallback to a local **Ollama** model
(`qwen2.5:14b-instruct`), which is also the only model a private study may use. Spend is
tracked in US dollars per study.

## Status (20 Nov 2026)

- Running on the lab's GPU server (RTX 6000 Ada): PostgreSQL 16 in Docker, dashboard and
  worker as systemd user services, Ollama local model.
- Studies run end to end on real data (CIC-IDS2017, UNSW-NB15). The final-version features
  (deep models, tuning, new datasets, text modality, Claude writing pipeline) are built and
  tested on synthetic fixtures; the flagship study runs next on the server.
- All automated tests pass (`bash scripts/check.sh`); 21 of the 22 acceptance tests are done.
  AT-13 needs an isolated VM, which the pilot scope excludes.

## Core rules (enforced in code, not in prompts)

- **No invented numbers.** Every number in a manuscript (results, tables, statistics, method
  facts, figures) is rendered from stored, hashed per-row predictions; the writer can only use
  placeholders, and a typed number is rejected. The writer plans a story from an evidence brief,
  a referee pass criticises the draft and it is revised; sentences that compare models without a
  test, contradict a hypothesis result or guess at untested mechanisms are removed, and wording
  repeated from earlier papers is rewritten.
- **No invented references.** Every reference must resolve to a real DOI with a matching title
  (Crossref/OpenAlex); off-topic search results are dropped.
- **The PI decides.** Approvals are bound to the exact protocol hash. Model settings are
  reviewed defaults or tuned on the validation split only; the test split is used once. Release
  needs the PI and a science, ethics and security checklist.
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
hashes, an AI-use disclosure and verified References. Exports: Markdown, LaTeX zip (with
figures and BibTeX), Word, print view for PDF. Protocol and draft versions can be compared side
by side.

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
| `cyberlab dataset prepare --name <dataset>` | clean and verify a dataset once after placing its files (cicids2017, unsw-nb15, ctu13, nf-unsw-nb15-v2, nf-cse-cic-ids2018-v2, nf-ton-iot-v2, nf-bot-iot-v2, phishing-urls) |
| `cyberlab health-check [--alert]` | disk, GPU, database, failed jobs, local model, backup age |
| `cyberlab llm-check` | one test prompt through Gemini/Ollama |
| `cyberlab restore-test` | restore the newest backup and re-verify every evidence hash |
| `cyberlab plugin-selftest [--docker]` | run the capability plugins on their fixtures |
| `cyberlab migrate-db` | copy all data into another database (SQLite to PostgreSQL) |
| `cyberlab db-upgrade` | apply schema migrations (Alembic) after a code update |
| `bash scripts/install_services.sh` | install serve, worker, Ollama, health and backup as user services |
| `bash scripts/backup.sh` | backup (database dump + evidence), retention, optional off-site copy |
