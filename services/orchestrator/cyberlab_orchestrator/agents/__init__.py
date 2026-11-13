"""The ten research agents (phase-one brief, section 4).

| # | Agent | Uses the LLM for | Deterministic tools it may call |
|---|---|---|---|
| 1 | Chief Research Orchestrator | study plan, search queries | none (routes work) |
| 2 | Literature Review | evidence matrix from verified abstracts | OpenAlex search, DOI verification |
| 3 | Research Gap and Novelty | gap analysis (labelled Proposed) | none |
| 4 | Methodology Design | protocol proposal | schema validation, protocol store |
| 5 | Dataset | none | CIC-IDS2017 cleaning, hashing, split checks |
| 6 | Coding | parameter suggestions within allowlists | protocol compiler, template registry |
| 7 | Experiment Execution | none | job broker (policy-checked, GPU lock) |
| 8 | Statistical Analysis | interpretation note (Proposed) | metrics, bootstrap CI, McNemar, leakage audit |
| 9 | Scientific Writing | prose with evidence placeholders | manuscript builder |
| 10 | Scientific Reviewer | review against the brief's dimensions | evidence report, citation status |
"""

from cyberlab_orchestrator.agents.coding import coding_agent
from cyberlab_orchestrator.agents.dataset import dataset_agent
from cyberlab_orchestrator.agents.experiment import experiment_agent
from cyberlab_orchestrator.agents.gap import gap_agent
from cyberlab_orchestrator.agents.literature import literature_agent
from cyberlab_orchestrator.agents.methodology import methodology_agent
from cyberlab_orchestrator.agents.orchestrator import orchestrator_agent
from cyberlab_orchestrator.agents.reviewer import reviewer_agent
from cyberlab_orchestrator.agents.statistics import statistics_agent
from cyberlab_orchestrator.agents.writing import writing_agent

__all__ = [
    "coding_agent",
    "dataset_agent",
    "experiment_agent",
    "gap_agent",
    "literature_agent",
    "methodology_agent",
    "orchestrator_agent",
    "reviewer_agent",
    "statistics_agent",
    "writing_agent",
]
