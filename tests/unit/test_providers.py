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
    # every model has its own quota, so each is tried once, without waiting, before handover
    assert gemini_calls["n"] == 3
    assert "429" in gate.last_error  # the reason is kept and shown with the fallback


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


def test_slow_gemini_request_is_capped_by_the_handover_deadline(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A hanging Gemini request must time out at the handover deadline, not after minutes."""
    import cyberlab_llm.gateway as gw
    from cyberlab_llm import LLMGateway

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    seen: list[float | None] = []

    class HangingClient:
        def post(self, *a, **kw):  # type: ignore[no-untyped-def]
            seen.append(kw.get("timeout"))
            raise httpx.ReadTimeout("hung")

    def ollama(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:14b-instruct"}]})
        return httpx.Response(200, json={"message": {"content": "local ok"}})

    g = GeminiProvider("k", "m1", client=HangingClient())  # type: ignore[arg-type]
    o = OllamaProvider(
        "http://x",
        "qwen2.5:14b-instruct",
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
    )
    res = LLMGateway(g, o, handover_seconds=20).complete("t", "s", "p")
    assert res.provider == "ollama"
    assert seen and seen[0] is not None and seen[0] <= 20.5


class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class _FakeAnthropic:
    def __init__(self, msg):
        from types import SimpleNamespace

        self.kwargs = {}
        outer = self

        def stream(**kw):
            outer.kwargs = kw
            return _FakeStream(msg)

        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=stream))


def _msg(stop="end_turn", text='{"ok": true}'):
    from types import SimpleNamespace

    return SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category="cyber") if stop == "refusal" else None,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_read_input_tokens=2000,
            cache_creation_input_tokens=0,
        ),
        model="claude-opus-5-5",
    )


def test_claude_provider_request_and_cost() -> None:
    from cyberlab_llm.gateway import ClaudeProvider

    fake = _FakeAnthropic(_msg())
    p = ClaudeProvider("k", client=fake)
    r = p.generate("sys", "prompt", True, 0.2)
    assert r.json() == {"ok": True} and r.provider == "claude"
    assert "temperature" not in fake.kwargs and fake.kwargs["fallbacks"] == "default"
    assert fake.kwargs["output_config"] == {"effort": "high"}
    # 1000*4 + 2000*0.2 + 500*20 per million
    assert abs(r.usd - (4000 + 400 + 10000) / 1e6) < 1e-12
    assert r.prompt_tokens == 3000 and r.output_tokens == 500


def test_claude_refusal_and_cutoff_are_errors() -> None:
    import pytest

    from cyberlab_llm.gateway import ClaudeProvider, LLMError

    for stop in ("refusal", "max_tokens"):
        with pytest.raises(LLMError):
            ClaudeProvider("k", client=_FakeAnthropic(_msg(stop))).generate("s", "p", False, 0.2)


def test_build_gateway_claude_needs_key(tmp_path) -> None:
    import pytest

    from cyberlab_common.config import Settings
    from cyberlab_llm.gateway import LLMError, build_gateway

    with pytest.raises(LLMError):
        build_gateway(Settings(_env_file=None, llm_provider="claude"))
    gw = build_gateway(Settings(_env_file=None, llm_provider="claude", anthropic_api_key="x"))
    assert gw.primary.name == "claude" and gw.handover_seconds == 0


def test_claude_model_override_and_no_fallback_beta_on_haiku() -> None:
    from cyberlab_llm.gateway import ClaudeProvider

    fake = _FakeAnthropic(_msg())
    p = ClaudeProvider("k", client=fake)
    p.generate("s", "p", False, 0.2, model="claude-haiku-5-5")
    assert fake.kwargs["model"] == "claude-haiku-5-5"
    assert "fallbacks" not in fake.kwargs and "betas" not in fake.kwargs
    p.generate("s", "p", False, 0.2, model="claude-sonnet-5-5")
    assert fake.kwargs["model"] == "claude-sonnet-5-5" and fake.kwargs["fallbacks"] == "default"


def test_gateway_uses_dashboard_model_choice() -> None:
    from cyberlab_llm.gateway import ClaudeProvider, LLMGateway

    fake = _FakeAnthropic(_msg())
    gw = LLMGateway(ClaudeProvider("k", client=fake))
    gw.complete("t", "s", "p")
    assert fake.kwargs["model"] == "claude-opus-5-5"  # nothing chosen: the CLAUDE_MODEL default
    gw.model_lookup = lambda: "claude-haiku-5-5"
    gw.complete("t", "s", "p")
    assert fake.kwargs["model"] == "claude-haiku-5-5"
    gw.model_lookup = lambda: "gpt-oops"  # unknown values are ignored, never sent
    gw.complete("t", "s", "p")
    assert fake.kwargs["model"] == "claude-opus-5-5"


def test_gemini_rate_limit_moves_to_next_model_without_waiting(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw

    slept: list[float] = []
    monkeypatch.setattr(gw.time, "sleep", slept.append)
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if "flash-lite-a" in str(req.url):
            return httpx.Response(429, json={"error": {"message": "quota"}})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    p = GeminiProvider(
        "k",
        "flash-lite-a",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        fallback_models=["flash-lite-b"],
    )
    res = p.generate("s", "p", json_mode=False, temperature=0)
    assert res.model == "flash-lite-b" and len(calls) == 2 and not slept


def test_default_tries_every_gemini_model_then_local_and_retries_gemini_next_call(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw
    from cyberlab_llm import LLMGateway

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    tried: list[str] = []
    down = {"on": True}

    def gemini(req: httpx.Request) -> httpx.Response:
        tried.append(req.url.path.split("/")[-1])
        if down["on"]:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "g"}]}}]})

    def ollama(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:14b-instruct"}]})
        return httpx.Response(200, json={"message": {"content": "local"}})

    g = GeminiProvider(
        "k",
        "a",
        client=httpx.Client(transport=httpx.MockTransport(gemini)),
        fallback_models=["b", "c"],
    )
    o = OllamaProvider(
        "http://x",
        "qwen2.5:14b-instruct",
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
    )
    gate = LLMGateway(g, o)  # defaults: no early handover, no cooldown
    assert gate.complete("t", "s", "p").text == "local"
    assert {t.split(":")[0] for t in tried} == {"a", "b", "c"}
    down["on"] = False
    assert gate.complete("t", "s", "p").text == "g"  # Gemini is tried again at once


def test_busy_gemini_model_is_skipped_without_waiting(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw

    slept: list[float] = []
    monkeypatch.setattr(gw.time, "sleep", slept.append)
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req.url.path)
        if "busy" in str(req.url):
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    p = GeminiProvider(
        "k",
        "busy-a",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        fallback_models=["busy-b", "free"],
    )
    res = p.generate("s", "p", json_mode=False, temperature=0)
    assert res.model == "free" and len(calls) == 3 and not slept


def test_busy_model_goes_to_the_back_for_the_next_calls(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import cyberlab_llm.gateway as gw

    monkeypatch.setattr(gw.time, "sleep", lambda _s: None)
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        model = req.url.path.split("/")[-1].split(":")[0]
        calls.append(model)
        if model == "a":
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    p = GeminiProvider(
        "k", "a", client=httpx.Client(transport=httpx.MockTransport(handler)), fallback_models=["b"]
    )
    p.generate("s", "p", json_mode=False, temperature=0)
    assert calls == ["a", "b"]
    calls.clear()
    p.generate("s", "p", json_mode=False, temperature=0)
    assert calls == ["b"]  # the busy model is not asked first again right away


def test_model_calls_wait_while_the_study_is_paused() -> None:
    import threading

    from cyberlab_llm import LLMGateway
    from cyberlab_llm.gateway import FakeProvider

    paused = {"on": True}
    seen: list[str] = []
    gate = LLMGateway(
        FakeProvider(),
        None,
        paused_lookup=lambda pid: paused["on"],
        on_status=lambda pid, task, msg: seen.append(msg),
        pause_poll_seconds=0.01,
    )
    out: list[str] = []
    t = threading.Thread(
        target=lambda: out.append(gate.complete("t", "s", "p", project_id="x").text)
    )
    t.start()
    t.join(0.2)
    assert t.is_alive() and not out and any("Paused" in m for m in seen)
    paused["on"] = False
    t.join(2)
    assert out  # resumed: the call went through
