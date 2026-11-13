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
        "test-key", "gemini-2.5-flash", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    res = p.generate("sys", "hello", json_mode=True, temperature=0.1)
    assert res.json() == {"ok": True} and res.total_tokens == 15
    assert str(seen["url"]).endswith("/models/gemini-2.5-flash:generateContent")
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
