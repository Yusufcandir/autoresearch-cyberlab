"""Registry records with missing fields must not crash the literature step."""

from __future__ import annotations

import httpx

from cyberlab_literature.client import Paper, ScholarlyClient, citation_key


def _client(handler) -> ScholarlyClient:  # type: ignore[no-untyped-def]
    return ScholarlyClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_citation_key_with_blank_or_missing_author_names() -> None:
    assert citation_key(Paper("Deep intrusion detection", None, 2020, authors=["", "  "]))
    assert citation_key(Paper("Deep intrusion detection", None, None, authors=[]))
    assert citation_key(Paper("Deep intrusion detection", None, 2020, authors=["Ada Lovelace"]))


def test_crossref_record_with_empty_date_and_nameless_authors() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "title": [],
                    "issued": {"date-parts": [[]]},
                    "author": [{"name": "CICIDS Team"}, {"given": None, "family": None}],
                    "container-title": [],
                }
            },
        )

    p = _client(handler).resolve_doi("10.1234/abc")
    assert p is not None and p.year is None and p.authors == ["CICIDS Team"]
    citation_key(p)


def test_semantic_scholar_and_openalex_null_names() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "semanticscholar" in str(req.url):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "title": "T",
                            "externalIds": {"DOI": "10.1234/x"},
                            "authors": [{"name": None}],
                        }
                    ]
                },
            )
        return httpx.Response(
            200, json={"results": [{"display_name": "T", "authorships": [{"author": None}]}]}
        )

    c = _client(handler)
    assert c.search_semantic_scholar("q")[0].authors == []
    assert c.search("q")[0].authors == []
