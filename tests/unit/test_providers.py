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
