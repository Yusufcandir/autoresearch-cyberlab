"""Graph and policy capability plugin (AT-16, AT-17).

* Graph adapter: typed vertices/edges, structural validation, degree features and a
  component-aware split (no connected component is split across train and test).
* Policy evaluator: exact, deterministic access decisions. Explicit deny always wins, the
  default is deny, and group/role membership is resolved through the graph. A machine-learning
  risk score is recorded as advice only: it can flag an allowed request for review, but it can
  never turn a deny into an allow (spec 7.2, "ML cannot replace formal policy semantics").
"""

from __future__ import annotations

from cyberlab_graph.graph import CAPABILITY, GraphAdapter, fixture
from cyberlab_graph.policy import Decision, decide, evaluate, load_policy

__all__ = ["CAPABILITY", "Decision", "GraphAdapter", "decide", "evaluate", "fixture", "load_policy"]
