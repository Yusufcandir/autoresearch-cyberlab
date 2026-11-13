"""Request/response shape of the real LLM providers, checked against mocked HTTP."""

from __future__ import annotations

import json

import httpx

from cyberlab_llm import GeminiProvider, OllamaProvider


def test_gemini_request_and_parse() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["key"] = req.headers.get("x-goog-api-key")
        seen["body"] = json.loads(req.content)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"ok": true}'}]}}],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3},
            },
        )

    p = GeminiProvider(
        "test-key", "gemini-3.8-flash", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    res = p.generate("sys", "hello", json_mode=True, temperature=0.1)
    assert res.json() == {"ok": True} and res.total_tokens == 15
    assert str(seen["url"]).endswith("/models/gemini-3.8-flash:generateContent")
    assert seen["key"] == "test-key"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == "sys"


def test_ollama_request_and_parse() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert req.url.path == "/api/chat" and body["format"] == "json" and body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        return httpx.Response(
            200, json={"message": {"content": "{}"}, "prompt_eval_count": 5, "eval_count": 2}
        )

    p = OllamaProvider(
        "http://127.0.0.1:11434",
        "qwen2.5:7b-instruct",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    res = p.generate("sys", "hi", json_mode=True, temperature=0)
    assert res.provider == "ollama" and res.total_tokens == 7


def test_gemini_falls_back_to_next_model_when_overloaded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        if "busy-model" in str(req.url):
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    p = GeminiProvider(
        "k",
        "busy-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        fallback_models=["spare-model"],
    )
    res = p.generate("s", "p", json_mode=False, temperature=0)
    assert res.text == "ok" and res.model == "spare-model"


def test_gemini_busy_status_is_reported(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw
    from cyberlab_llm import LLMGateway

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)

    def handler(req: httpx.Request) -> httpx.Response:
        if "busy-model" in str(req.url):
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    p = GeminiProvider(
        "k",
        "busy-model",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        fallback_models=["spare-model"],
    )
    seen: list[str] = []
    g = LLMGateway(p, None, on_status=lambda pid, task, msg: seen.append(msg))
    assert g.complete("t", "s", "p", project_id="x").text == "ok"
    assert any("busy" in m for m in seen) and any("switching to spare-model" in m for m in seen)


def test_busy_gemini_hands_over_to_ready_local_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw
    from cyberlab_llm import LLMGateway

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    clock = iter(range(10_000))  # each call advances the clock one second
    monkeypatch.setattr(gw.time, "monotonic", lambda: float(next(clock)))
    gemini_calls = {"n": 0}

    def gemini(req: httpx.Request) -> httpx.Response:
        gemini_calls["n"] += 1
        return httpx.Response(429, json={})

    def ollama(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:14b-instruct"}]})
        return httpx.Response(200, json={"message": {"content": "local ok"}})

    g = GeminiProvider(
        "k",
        "m1",
        client=httpx.Client(transport=httpx.MockTransport(gemini)),
        fallback_models=["m2", "m3"],
    )
    o = OllamaProvider(
        "http://x",
        "qwen2.5:14b-instruct",
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
    )
    gate = LLMGateway(g, o, handover_seconds=1.5)
    res = gate.complete("t", "s", "p")
    assert res.provider == "ollama" and res.text == "local ok"
    assert gemini_calls["n"] <= 2  # handed over quickly instead of trying every model


def test_gemini_breaker_keeps_later_calls_on_local_model(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw
    from cyberlab_llm import LLMGateway

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    now = {"t": 0.0}

    def tick() -> float:
        now["t"] += 1.0
        return now["t"]

    monkeypatch.setattr(gw.time, "monotonic", tick)
    gemini_calls = {"n": 0}

    def gemini(req: httpx.Request) -> httpx.Response:
        gemini_calls["n"] += 1
        return httpx.Response(429, json={})

    def ollama(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:14b-instruct"}]})
        return httpx.Response(200, json={"message": {"content": "local ok"}})

    g = GeminiProvider("k", "m1", client=httpx.Client(transport=httpx.MockTransport(gemini)))
    o = OllamaProvider(
        "http://x",
        "qwen2.5:14b-instruct",
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
    )
    seen: list[str] = []
    gate = LLMGateway(
        g,
        o,
        handover_seconds=1.5,
        cooldown_seconds=600,
        on_status=lambda pid, task, msg: seen.append(msg),
    )
    assert gate.complete("t", "s", "p", project_id="x").provider == "ollama"
    first = gemini_calls["n"]
    for _ in range(5):  # later calls must not go back to Gemini while the breaker is open
        assert gate.complete("t", "s", "p", project_id="x").provider == "ollama"
    assert gemini_calls["n"] == first
    assert any("until" in m for m in seen)

    now["t"] += 10_000  # cooldown over: Gemini is tried again
    gate.complete("t", "s", "p", project_id="x")
    assert gemini_calls["n"] > first
