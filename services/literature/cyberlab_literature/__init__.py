"""Literature retrieval and citation verification (spec Step 12)."""

from cyberlab_literature.client import (
    Paper,
    ScholarlyClient,
    bibtex_entry,
    citation_key,
    normalize_doi,
    title_similarity,
)

__all__ = [
    "Paper",
    "ScholarlyClient",
    "bibtex_entry",
    "citation_key",
    "normalize_doi",
    "title_similarity",
]
