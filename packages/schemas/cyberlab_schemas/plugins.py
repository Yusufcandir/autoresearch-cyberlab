"""Plugin contract (spec 7.2 and 7.3).

Every capability plugin validates its input against its own typed schema and either returns a
typed result or raises a PluginError with a machine-readable code. A plugin never silently
substitutes a missing view: it raises UNAVAILABLE_FEATURE_VIEW and the protocol must be
revised and approved again.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

UNAVAILABLE_FEATURE_VIEW = "UNAVAILABLE_FEATURE_VIEW"
SCHEMA_INVALID = "SCHEMA_INVALID"
ISOLATION_REQUIRED = "BLOCKED: ISOLATION_REQUIRED"
METADATA_REQUIRES_EXECUTION = "BLOCKED: METADATA_REQUIRES_EXECUTION"
PATH_NOT_PERMITTED = "BLOCKED: PATH_NOT_PERMITTED"
POLICY_INVALID = "POLICY_INVALID"


class PluginError(Exception):
    """Typed, user-visible plugin failure."""

    def __init__(self, code: str, message: str, details: Mapping[str, Any] | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass(frozen=True)
class Capability:
    name: str
    version: str
    supported_modalities: tuple[str, ...]
    feature_views: tuple[str, ...]
    requires_isolated_execution: bool = False
    notes: str = ""


@runtime_checkable
class ResearchAdapter(Protocol):
    """Reference interface from spec 7.3."""

    def capabilities(self) -> Capability: ...

    def validate_dataset(self, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]: ...

    def feature_view(self, name: str, dataset_manifest: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass
class ValidationReport:
    ok: bool
    modality: str
    n_items: int
    groups: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "modality": self.modality,
            "n_items": self.n_items,
            "groups": self.groups,
            "problems": self.problems,
            "stats": self.stats,
        }


def require_view(capability: Capability, view: str) -> None:
    if view not in capability.feature_views:
        raise PluginError(
            UNAVAILABLE_FEATURE_VIEW,
            f"{capability.name} has no '{view}' view; available: {', '.join(capability.feature_views)}",
            {"requested": view, "available": list(capability.feature_views)},
        )


def group_split(groups: list[str], test_fraction: float = 0.25) -> dict[str, list[str]]:
    """Deterministic group-aware split: every group lands wholly in train or test (spec 7.5)."""
    import hashlib

    uniq = sorted(set(groups))
    ranked = sorted(uniq, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
    n_test = max(1, round(len(ranked) * test_fraction)) if len(ranked) > 1 else 0
    test = set(ranked[:n_test])
    return {"train": [g for g in uniq if g not in test], "test": [g for g in uniq if g in test]}


# Registry: plugins are found by name; the orchestrator's state schema does not change when a
# plugin is added (FR-013). Only reviewed, installed packages listed here can be loaded.
KNOWN_ADAPTERS: dict[str, str] = {
    "text": "cyberlab_tabular.text:TextAdapter",
    "audio": "cyberlab_audio:AudioAdapter",
    "graph_policy": "cyberlab_graph.graph:GraphAdapter",
    "code_artifact": "cyberlab_code:CodeAdapter",
}


def load_adapter(name: str) -> Any:
    import importlib

    target = KNOWN_ADAPTERS.get(name)
    if target is None:
        raise PluginError(UNAVAILABLE_FEATURE_VIEW, f"no installed plugin named {name!r}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)()


def list_capabilities() -> list[Capability]:
    out = []
    for name in KNOWN_ADAPTERS:
        try:
            out.append(load_adapter(name).capabilities())
        except ImportError:
            continue
    return out
