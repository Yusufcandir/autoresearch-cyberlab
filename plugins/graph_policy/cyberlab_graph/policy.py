"""Exact policy evaluator (AT-17).

Semantics (fixed, documented, tested):
1. Identities of the requester = the principal plus everything reachable over `member_of`
   edges in the membership graph (user -> group -> role ...).
2. A statement matches when one of its principals, actions and resources matches (shell-style
   wildcards) and every condition attribute equals the request attribute.
3. Any matching deny -> DENY. Otherwise any matching allow -> ALLOW. Otherwise DENY (default).
4. An optional model risk score is advice only. It may flag an ALLOW for human review; it can
   never change the effect.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cyberlab_graph.graph import parse_graph
from cyberlab_schemas.plugins import POLICY_INVALID, PluginError


class Statement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sid: str = Field(min_length=1)
    effect: Literal["allow", "deny"]
    principals: list[str] = Field(min_length=1)
    actions: list[str] = Field(min_length=1)
    resources: list[str] = Field(min_length=1)
    conditions: dict[str, str | int | bool] = Field(default_factory=dict)


class PolicyDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal["1"] = "1"
    statements: list[Statement] = Field(min_length=1)


class AccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    principal: str
    action: str
    resource: str
    attributes: dict[str, str | int | bool] = Field(default_factory=dict)


@dataclass
class Decision:
    effect: Literal["allow", "deny"]
    reason: str
    matched_allow: list[str] = field(default_factory=list)
    matched_deny: list[str] = field(default_factory=list)
    identities: list[str] = field(default_factory=list)
    model_score: float | None = None
    model_advice: str | None = None
    flagged_for_review: bool = False
    evaluator: str = "exact-policy-v1"

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_policy(doc: Mapping[str, Any]) -> PolicyDoc:
    try:
        return PolicyDoc.model_validate(doc)
    except ValidationError as exc:
        raise PluginError(
            POLICY_INVALID,
            "policy does not match the schema",
            {"errors": exc.errors(include_url=False)},
        ) from exc


def identities(principal: str, graph: Mapping[str, Any] | None) -> list[str]:
    if graph is None:
        return [principal]
    g = parse_graph(graph)
    member: dict[str, set[str]] = defaultdict(set)
    for e in g.edges:
        if e.type == "member_of":
            member[e.src].add(e.dst)
    seen, stack = {principal}, [principal]
    while stack:
        for nxt in member[stack.pop()]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return sorted(seen)


def _matches(stmt: Statement, ids: list[str], req: AccessRequest) -> bool:
    return (
        any(fnmatchcase(i, p) for i in ids for p in stmt.principals)
        and any(fnmatchcase(req.action, a) for a in stmt.actions)
        and any(fnmatchcase(req.resource, r) for r in stmt.resources)
        and all(req.attributes.get(k) == v for k, v in stmt.conditions.items())
    )


def evaluate(
    policy: Mapping[str, Any] | PolicyDoc,
    request: Mapping[str, Any] | AccessRequest,
    graph: Mapping[str, Any] | None = None,
) -> Decision:
    pol = policy if isinstance(policy, PolicyDoc) else load_policy(policy)
    try:
        req = (
            request if isinstance(request, AccessRequest) else AccessRequest.model_validate(request)
        )
    except ValidationError as exc:
        raise PluginError(POLICY_INVALID, "access request does not match the schema") from exc
    ids = identities(req.principal, graph)
    allow = [s.sid for s in pol.statements if s.effect == "allow" and _matches(s, ids, req)]
    deny = [s.sid for s in pol.statements if s.effect == "deny" and _matches(s, ids, req)]
    if deny:
        return Decision("deny", f"explicit deny by {', '.join(deny)}", allow, deny, ids)
    if allow:
        return Decision("allow", f"allowed by {', '.join(allow)}", allow, deny, ids)
    return Decision("deny", "no statement allows this request (default deny)", allow, deny, ids)


def decide(
    policy: Mapping[str, Any] | PolicyDoc,
    request: Mapping[str, Any] | AccessRequest,
    graph: Mapping[str, Any] | None = None,
    model_score: float | None = None,
    risk_threshold: float = 0.5,
) -> Decision:
    """Exact decision, with a model risk score attached as advice only."""
    d = evaluate(policy, request, graph)
    if model_score is not None:
        d.model_score = float(model_score)
        risky = model_score >= risk_threshold
        d.model_advice = "risky" if risky else "safe"
        # The score never changes the effect. It can only ask a human to look at an allow.
        d.flagged_for_review = d.effect == "allow" and risky
    return d
