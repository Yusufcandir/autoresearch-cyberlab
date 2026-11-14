"""Offline fakes for the scholarly APIs and a ready-to-use test Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from cyberlab_common.config import Settings
from cyberlab_data import write_fixture
from cyberlab_data.unsw_nb15 import write_fixture as unsw_fixture
from cyberlab_literature import ScholarlyClient
from cyberlab_llm import FakeProvider, LLMGateway
from cyberlab_orchestrator.offline import offline_responder

PAPERS = {
    "10.1000/real.one": "Deep learning for network intrusion detection on CIC-IDS2017",
    "10.1000/real.two": "Gradient boosting for flow-based attack classification",
    "10.5220/0006639801080116": (
        "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization"
    ),
    "10.1109/MilCIS.2015.7348942": (
        "UNSW-NB15: a comprehensive data set for network intrusion detection systems "
        "(UNSW-NB15 network data set)"
    ),
}
FAKE_DOI = "10.9999/invented.by.llm"


def scholarly_handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if "api.openalex.org/works" in url and "search=" in url:
        results: list[dict[str, Any]] = [
            {
                "display_name": PAPERS["10.1000/real.one"],
                "doi": "https://doi.org/10.1000/real.one",
                "publication_year": 2021,
                "authorships": [{"author": {"display_name": "Ada Lovelace"}}],
                "primary_location": {"source": {"display_name": "Computers & Security"}},
                "abstract_inverted_index": {"We": [0], "study": [1], "IDS.": [2]},
                "cited_by_count": 50,
            },
            {
                "display_name": PAPERS["10.1000/real.two"],
                "doi": "https://doi.org/10.1000/real.two",
                "publication_year": 2022,
                "authorships": [{"author": {"display_name": "Alan Turing"}}],
                "cited_by_count": 20,
            },
            {
                "display_name": "Intrusion detection results that do not exist",
                "doi": f"https://doi.org/{FAKE_DOI}",
                "publication_year": 2023,
                "authorships": [],
                "cited_by_count": 99,
            },
        ]
        return httpx.Response(200, json={"results": results})
    if "api.crossref.org/works/" in url:
        doi = url.split("/works/", 1)[1].split("?")[0]
        doi = next(
            (d for d in PAPERS if d.lower() == doi.lower()), doi
        )  # DOIs are case-insensitive
        if doi in PAPERS:
            return httpx.Response(
                200,
                json={
                    "message": {
                        "title": [PAPERS[doi]],
                        "author": [{"given": "A", "family": "Author"}],
                        "issued": {"date-parts": [[2021]]},
                        "container-title": ["Journal"],
                    }
                },
            )
        return httpx.Response(404)
    if "api.openalex.org/works/doi:" in url:
        return httpx.Response(404)
    return httpx.Response(404)


def fake_scholar() -> ScholarlyClient:
    return ScholarlyClient(client=httpx.Client(transport=httpx.MockTransport(scholarly_handler)))


def fake_gateway(responder: Any = offline_responder) -> LLMGateway:
    fake = FakeProvider(responder)
    return LLMGateway(fake, fake)


def make_settings(tmp: Path, **kw: Any) -> Settings:
    write_fixture(tmp / "var" / "datasets" / "cicids2017" / "raw")
    unsw_fixture(tmp / "var" / "datasets" / "unsw_nb15" / "raw")
    base: dict[str, Any] = {
        "_env_file": None,
        "database_url": f"sqlite:///{(tmp / 'lab.db').as_posix()}",
        "secret_key": "test-secret-key-0123456789",
        "audit_log_destination": f"file:{(tmp / 'audit.jsonl').as_posix()}",
        "data_dir": tmp / "var",
        "llm_provider": "fake",
        "gpu_job_max_minutes": 5,
    }
    base.update(kw)
    return Settings(**base)
