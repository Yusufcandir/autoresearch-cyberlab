from pathlib import Path
from types import SimpleNamespace

import httpx

from cyberlab_common import health


def test_disk_reports_usage(tmp_path: Path) -> None:
    r = health.disk(tmp_path)
    assert r["check"] == "disk" and "GiB free" in r["detail"]


def test_backup_age(tmp_path: Path) -> None:
    s = SimpleNamespace(data_dir=tmp_path)
    assert health.backup_age(s)["ok"] is False
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups" / "cyberlab-1.tar.gz").write_bytes(b"x")
    assert health.backup_age(s)["ok"] is True


def test_local_model_unreachable_is_warning() -> None:
    s = SimpleNamespace(ollama_host="http://127.0.0.1:9", ollama_model="qwen")
    r = health.local_model(s)
    assert r["ok"] is False and r["level"] == "warning"


def test_broken_check_is_reported(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("x")

    monkeypatch.setattr(health, "gpu", boom)
    s = SimpleNamespace(
        data_dir=Path("."), database_url=None, ollama_host="http://127.0.0.1:9", ollama_model="q"
    )
    monkeypatch.setattr(health, "database", lambda _s: health._check("database", True, "ok"))
    monkeypatch.setattr(health, "failed_jobs", lambda _s: health._check("failed_jobs", True, "0"))
    out = health.run_all(s)
    assert any(not r["ok"] and "check failed" in r["detail"] for r in out)


def test_alert_payloads(monkeypatch) -> None:
    sent = []
    monkeypatch.setattr(httpx, "post", lambda url, **kw: sent.append((url, kw)))
    p = [{"check": "disk", "detail": "96% used"}]
    health.alert("https://hooks.slack.com/x", p, "srv")
    health.alert("https://ntfy.sh/topic", p, "srv")
    assert sent[0][1]["json"]["text"].endswith("disk: 96% used")
    assert b"disk: 96% used" in sent[1][1]["content"]


def test_alert_never_raises(monkeypatch) -> None:
    def fail(*_a, **_k):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "post", fail)
    health.alert("https://ntfy.sh/t", [{"check": "x", "detail": "y"}], "h")
