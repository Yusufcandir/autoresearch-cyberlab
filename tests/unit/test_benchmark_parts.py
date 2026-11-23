"""Pieces of the journal-scale features: dataset detection, snowballing, Ollama context."""

from __future__ import annotations

import httpx

from cyberlab_literature.client import ScholarlyClient
from cyberlab_orchestrator.datasets import datasets_named


def test_datasets_named_reads_several_and_keeps_longer_names_apart() -> None:
    assert datasets_named("Compare on CIC-IDS2017 and UNSW-NB15") == [
        "cicids2017-mlcsv-v1",
        "unsw-nb15-v1",
    ]
    assert datasets_named("UNSW-NB15 then CICIDS2017 and NF-ToN-IoT-v2") == [
        "unsw-nb15-v1",
        "cicids2017-mlcsv-v1",
        "nf-ton-iot-v2",
    ]
    got = datasets_named("train on NF-UNSW-NB15-v2 only")
    assert len(got) == 1 and "nf" in got[0].lower()
    assert datasets_named("no dataset named here") == []


def test_snowball_returns_the_referenced_papers() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        url = str(req.url)
        if "doi:10.1234/seed" in url:
            return httpx.Response(200, json={"referenced_works": ["https://openalex.org/W1"]})
        if "openalex_id" in url:
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "display_name": "Intrusion detection with flows",
                            "doi": "https://doi.org/10.5555/ref.one",
                            "publication_year": 2020,
                            "cited_by_count": 9,
                        }
                    ]
                },
            )
        return httpx.Response(404)

    c = ScholarlyClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    got = c.snowball(["10.1234/seed"])
    assert [p.doi for p in got] == ["10.5555/ref.one"]
    assert c.snowball(["not-a-doi"]) == []


def test_ollama_sends_the_context_window() -> None:
    from cyberlab_llm import OllamaProvider

    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"message": {"content": "ok"}})

    p = OllamaProvider(
        "http://x",
        "qwen",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        num_ctx=16384,
    )
    p.generate("s", "p", False, 0.2)
    assert seen["options"]["num_ctx"] == 16384


def test_ollama_reasoning_models_answer_without_a_think_block() -> None:
    import json

    from cyberlab_llm import OllamaProvider

    sent: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        sent.update(json.loads(req.content))
        return httpx.Response(
            200, json={"message": {"content": '<think>plan {x}</think>\n{"ok": true}'}}
        )

    p = OllamaProvider(
        "http://x", "qwen3:32b", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    res = p.generate("s", "p", True, 0.2)
    assert sent["think"] is False and res.json() == {"ok": True}


def test_values_beyond_float32_are_removed_not_turned_into_infinity(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import numpy as np
    import pandas as pd

    from cyberlab_data.common import DatasetInfo, finalize
    from cyberlab_schemas import SplitSpec

    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "Class": ["BENIGN"] * (n // 2) + ["DoS"] * (n // 2),
        }
    )
    df.loc[3, "a"] = 1e300  # like a corrupt NetFlow-v2 byte-rate value
    info = DatasetInfo(name="t", source_url="x", license="x", citation="x", citation_doi=None)
    prepared = finalize(
        df,
        root=tmp_path,
        split=SplitSpec(),
        dataset_id="t-v1",
        info=info,
        files={},
        group_column=None,
        benign_cap=None,
        class_cap=None,
    )
    clean = pd.read_parquet(prepared.root / "clean.parquet")
    feats = [c for c in clean.columns if c in ("a", "b")]
    assert np.isfinite(clean[feats].to_numpy(dtype=float)).all()
    assert len(clean) == n - 1
