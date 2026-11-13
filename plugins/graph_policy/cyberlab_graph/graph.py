"""Typed graph datasets with structural validation (no external graph library needed)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cyberlab_schemas.plugins import (
    SCHEMA_INVALID,
    Capability,
    PluginError,
    ValidationReport,
    group_split,
    require_view,
)

CAPABILITY = Capability(
    name="graph_policy",
    version="1",
    supported_modalities=("graph", "policy"),
    feature_views=("degree_features", "component_split", "effective_permissions"),
    notes="Access decisions come from the exact policy evaluator, never from a model score.",
)


class Vertex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    label: str | None = None


class Edge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    src: str
    dst: str
    type: str = Field(min_length=1)


class GraphDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modality: Literal["graph"] = "graph"
    vertices: list[Vertex] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)


def parse_graph(manifest: Mapping[str, Any]) -> GraphDataset:
    try:
        return GraphDataset.model_validate(manifest)
    except ValidationError as exc:
        raise PluginError(
            SCHEMA_INVALID,
            "graph dataset does not match the schema",
            {"errors": exc.errors(include_url=False)},
        ) from exc


def components(g: GraphDataset) -> dict[str, str]:
    """vertex id -> component id (undirected), deterministic."""
    adj: dict[str, set[str]] = defaultdict(set)
    for e in g.edges:
        adj[e.src].add(e.dst)
        adj[e.dst].add(e.src)
    comp: dict[str, str] = {}
    for v in sorted(x.id for x in g.vertices):
        if v in comp:
            continue
        stack, root = [v], v
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp[cur] = root
            stack.extend(n for n in adj[cur] if n not in comp)
    return comp


class GraphAdapter:
    def capabilities(self) -> Capability:
        return CAPABILITY

    def validate_dataset(self, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        g = parse_graph(dataset_manifest)
        ids = [v.id for v in g.vertices]
        problems = []
        dup = [k for k, n in Counter(ids).items() if n > 1]
        if dup:
            problems.append(f"duplicate vertex ids: {dup}")
        known = set(ids)
        dangling = [
            f"{e.src}->{e.dst}" for e in g.edges if e.src not in known or e.dst not in known
        ]
        if dangling:
            problems.append(f"edges to unknown vertices: {dangling}")
        loops = [e.src for e in g.edges if e.src == e.dst]
        if loops:
            problems.append(f"self-loops: {loops}")
        comp = components(g) if not problems else {}
        return ValidationReport(
            ok=not problems,
            modality="graph",
            n_items=len(ids),
            groups=sorted(set(comp.values())),
            problems=problems,
            stats={
                "edges": len(g.edges),
                "vertex_types": dict(Counter(v.type for v in g.vertices)),
                "edge_types": dict(Counter(e.type for e in g.edges)),
            },
        ).as_dict()

    def feature_view(self, name: str, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]:
        require_view(CAPABILITY, name)
        g = parse_graph(dataset_manifest)
        if name == "component_split":
            comp = components(g)
            split = group_split(list(comp.values()))
            test_groups = set(split["test"])
            return {
                "view": name,
                "train": sorted(v for v, c in comp.items() if c not in test_groups),
                "test": sorted(v for v, c in comp.items() if c in test_groups),
            }
        if name == "effective_permissions":
            raise PluginError(
                SCHEMA_INVALID,
                "effective_permissions needs a policy and a request: use cyberlab_graph.decide()",
            )
        out_deg: Counter[str] = Counter(e.src for e in g.edges)
        in_deg: Counter[str] = Counter(e.dst for e in g.edges)
        return {
            "view": name,
            "rows": {v.id: {"in": in_deg[v.id], "out": out_deg[v.id]} for v in g.vertices},
        }


def fixture() -> dict[str, Any]:
    """Users, groups and roles of a small lab (test use only)."""
    v = [
        ("user:alice", "user"),
        ("user:bob", "user"),
        ("user:mallory", "user"),
        ("group:analysts", "group"),
        ("group:contractors", "group"),
        ("role:reader", "role"),
        ("role:admin", "role"),
    ]
    e = [
        ("user:alice", "group:analysts"),
        ("user:bob", "group:analysts"),
        ("user:mallory", "group:contractors"),
        ("group:analysts", "role:reader"),
        ("group:contractors", "role:reader"),
    ]
    return {
        "modality": "graph",
        "vertices": [{"id": i, "type": t} for i, t in v],
        "edges": [{"src": s, "dst": d, "type": "member_of"} for s, d in e],
    }
