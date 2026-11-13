# graph_policy (`cyberlab_graph`)

Typed graph datasets (vertices/edges) with structural validation and a component-aware split,
plus the exact policy evaluator used for access decisions (AT-17): membership resolved over
`member_of` edges, explicit deny wins, default deny. A model risk score is advice only: it can
flag an allowed request for review but never changes the decision.
