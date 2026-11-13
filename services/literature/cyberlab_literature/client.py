"""OpenAlex search plus Crossref/OpenAlex DOI verification.

A citation is `Verified` only when its DOI resolves at Crossref (or OpenAlex) and the
registered title matches the claimed title. Anything an LLM proposes without a resolvable
DOI stays `Unverified` and the manuscript builder refuses to treat it as a reference (AT-03).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import httpx

OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works/"


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    d = doi.strip().lower()
    d = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", d)
    return d if re.match(r"^10\.\d{4,9}/\S+$", d) else None


def _norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKD", t).lower()
    return re.sub(r"[^a-z0-9 ]+", " ", t).strip()


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_title(a), _norm_title(b)).ratio()


@dataclass
class Paper:
    title: str
    doi: str | None
    year: int | None
    authors: list[str] = field(default_factory=list)
    venue: str | None = None
    abstract: str | None = None
    cited_by: int = 0
    source: str = "openalex"
    status: str = "Unverified"
    verification_note: str = ""


def _abstract_from_index(idx: dict[str, list[int]] | None) -> str | None:
    if not idx:
        return None
    words: dict[int, str] = {}
    for word, positions in idx.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words))


def citation_key(p: Paper) -> str:
    first = p.authors[0].split()[-1] if p.authors else "anon"
    word = next((w for w in _norm_title(p.title).split() if len(w) > 3), "paper")
    key = f"{first}{p.year or 'nd'}{word}"
    return re.sub(r"[^A-Za-z0-9]", "", unicodedata.normalize("NFKD", key))


def bibtex_entry(p: Paper, key: str | None = None) -> str:
    key = key or citation_key(p)
    fields = {
        "title": "{" + p.title + "}",
        "author": " and ".join(p.authors) or "Unknown",
        "year": str(p.year or ""),
        "journal": p.venue or "",
        "doi": p.doi or "",
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@article{{{key},\n{body}\n}}"


class ScholarlyClient:
    def __init__(self, contact_email: str | None = None, client: httpx.Client | None = None):
        headers = {"User-Agent": "autoresearch-cyberlab/0.1 (academic research platform)"}
        self.params: dict[str, str] = {}
        if contact_email:
            self.params["mailto"] = contact_email
        self.client = client or httpx.Client(timeout=30, headers=headers, follow_redirects=True)

    def search(self, query: str, limit: int = 10, from_year: int = 2015) -> list[Paper]:
        params = {
            "search": query,
            "per-page": str(min(limit, 50)),
            "filter": f"from_publication_date:{from_year}-01-01,type:article",
            "sort": "relevance_score:desc",
            **self.params,
        }
        r = self.client.get(OPENALEX, params=params)
        r.raise_for_status()
        return [self._from_openalex(w) for w in r.json().get("results", [])]

    @staticmethod
    def _from_openalex(w: dict[str, Any]) -> Paper:
        loc = (w.get("primary_location") or {}).get("source") or {}
        return Paper(
            title=w.get("display_name") or w.get("title") or "",
            doi=normalize_doi(w.get("doi")),
            year=w.get("publication_year"),
            authors=[
                a.get("author", {}).get("display_name", "") for a in w.get("authorships", [])[:12]
            ],
            venue=loc.get("display_name"),
            abstract=_abstract_from_index(w.get("abstract_inverted_index")),
            cited_by=int(w.get("cited_by_count") or 0),
        )

    def resolve_doi(self, doi: str) -> Paper | None:
        d = normalize_doi(doi)
        if d is None:
            return None
        r = self.client.get(CROSSREF + d, params=self.params)
        if r.status_code == 200:
            m = r.json().get("message", {})
            year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            return Paper(
                title=(m.get("title") or [""])[0],
                doi=d,
                year=year,
                authors=[
                    f"{a.get('given', '')} {a.get('family', '')}".strip()
                    for a in m.get("author", [])[:12]
                ],
                venue=(m.get("container-title") or [None])[0],
                source="crossref",
            )
        r2 = self.client.get(f"{OPENALEX}/doi:{d}", params=self.params)
        if r2.status_code == 200:
            return self._from_openalex(r2.json())
        return None

    def verify(self, claimed: Paper, min_similarity: float = 0.85) -> Paper:
        """Return a copy of `claimed` with status Verified or Unverified and a reason."""
        if not normalize_doi(claimed.doi):
            claimed.status, claimed.verification_note = "Unverified", "no valid DOI"
            return claimed
        try:
            found = self.resolve_doi(claimed.doi or "")
        except httpx.HTTPError as exc:
            claimed.status, claimed.verification_note = "Unverified", f"lookup failed: {exc}"
            return claimed
        if found is None:
            claimed.status, claimed.verification_note = "Unverified", "DOI does not resolve"
            return claimed
        sim = title_similarity(claimed.title, found.title)
        if sim < min_similarity:
            claimed.status = "Unverified"
            claimed.verification_note = f"title mismatch ({sim:.2f}): registry says {found.title!r}"
            return claimed
        found.abstract = found.abstract or claimed.abstract
        found.cited_by = claimed.cited_by
        found.status, found.verification_note = "Verified", f"{found.source} title match {sim:.2f}"
        return found
