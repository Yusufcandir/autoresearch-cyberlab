"""Citation verification report (professor's plan, Step 4.3).

Checks 1-3 (the publication exists, the DOI resolves to it, authors/title/year/venue match)
happen when the literature agent verifies each reference against Crossref/OpenAlex; only
Verified references can be cited. This module adds:

  4. claim support per citing sentence: supported / partially_supported / contradicted /
     insufficient_evidence / unable_to_verify (judged against the paper's title and abstract;
     a valid DOI does not prove that a paper supports a sentence)
  5. every citation in the text appears in the reference list
  6. every listed reference is cited
  7. citation keys are consistent (one key per paper)
  8. no duplicate references (same DOI or same title under two keys)
"""

from __future__ import annotations

import re
from typing import Any

SUPPORT_LEVELS = (
    "supported",
    "partially_supported",
    "contradicted",
    "insufficient_evidence",
    "unable_to_verify",
)
# levels that must be revised or removed before release; the others are flagged for authors
MUST_FIX = {"contradicted", "insufficient_evidence"}


def _norm_title(t: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def reference_checks(markdown: str, citations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    body, _, refs = markdown.partition("\n## References")
    cited = set(re.findall(r"@([A-Za-z0-9_]+)", " ".join(re.findall(r"\[@[^\]]+\]", body))))
    listed = set(re.findall(r"^- \[([A-Za-z0-9_]+)\]", refs, re.M))
    verified = {k for k, c in citations.items() if c.get("status") == "Verified"}
    by_doi: dict[str, list[str]] = {}
    by_title: dict[str, list[str]] = {}
    for k in verified:
        c = citations[k]
        if c.get("doi"):
            by_doi.setdefault(str(c["doi"]).lower(), []).append(k)
        if c.get("title"):
            by_title.setdefault(_norm_title(c["title"]), []).append(k)
    dups = sorted({tuple(sorted(v)) for v in [*by_doi.values(), *by_title.values()] if len(v) > 1})
    return {
        "cited_not_in_reference_list": sorted(cited - listed),
        "listed_but_not_cited": sorted(listed - cited),
        "cited_but_unverified": sorted(cited - verified),
        "duplicate_references": [list(d) for d in dups],
        "consistent": not (cited - listed) and not (cited - verified) and not dups,
    }


def support_level(item: dict[str, Any]) -> str:
    level = str(item.get("support", "")).strip().lower().replace(" ", "_")
    if level in SUPPORT_LEVELS:
        return level
    if item.get("supported") is True:
        return "supported"
    if item.get("supported") is False:
        return "insufficient_evidence"
    return "unable_to_verify"


def build_report(
    markdown: str,
    citations: dict[str, dict[str, Any]],
    judged: list[dict[str, Any]],
) -> dict[str, Any]:
    refs = reference_checks(markdown, citations)
    counts = {lvl: sum(1 for j in judged if j["level"] == lvl) for lvl in SUPPORT_LEVELS}
    must_fix = [j for j in judged if j["level"] in MUST_FIX]
    status = (
        "BLOCK"
        if refs["cited_not_in_reference_list"] or refs["cited_but_unverified"]
        else "WARNING"
        if must_fix
        or refs["duplicate_references"]
        or counts["partially_supported"]
        or counts["unable_to_verify"]
        else "PASS"
    )
    return {
        "status": status,
        "reference_list": refs,
        "claim_support_counts": counts,
        "claims": judged,
        "note": "metadata and abstracts alone cannot verify detailed claims about a paper's "
        "methods; sentences marked unable_to_verify or partially_supported need author review",
    }
