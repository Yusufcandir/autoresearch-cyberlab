"""Operational health check (spec Step 23): disk, GPU, failed jobs, database, local model,
backup age. Read-only; prints a report and can post alerts to a webhook (ntfy/Slack/Discord).

Run it from a timer (infra/systemd/cyberlab-health.timer) so problems surface without anyone
watching the dashboard.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from cyberlab_common.config import Settings

DISK_WARN, DISK_STOP = 0.85, 0.95  # spec 3.4: warning at 85%, emergency at 95%


def _check(name: str, ok: bool, detail: str, level: str = "error") -> dict[str, Any]:
    return {"check": name, "ok": ok, "level": "ok" if ok else level, "detail": detail}


def disk(path: Path) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    u = shutil.disk_usage(path)
    used = u.used / u.total
    level = "error" if used >= DISK_STOP else "warning"
    return _check(
        "disk", used < DISK_WARN, f"{used:.0%} used, {u.free / 2**30:.0f} GiB free at {path}", level
    )


def gpu() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return _check("gpu", True, "no nvidia-smi on this host")
    out = subprocess.run(  # noqa: S603 - fixed read-only query
        [
            exe,
            "--query-gpu=memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).stdout.strip()
    if not out:
        return _check("gpu", False, "nvidia-smi returned nothing", "warning")
    used, total, temp = (float(x) for x in out.splitlines()[0].split(","))
    ok = temp < 85
    return _check("gpu", ok, f"{used:.0f}/{total:.0f} MiB used, {temp:.0f} C", "warning")


def failed_jobs(settings: Settings, hours: int = 24) -> dict[str, Any]:
    from sqlalchemy import func, select

    from cyberlab_common.db import Database, JobRecord

    db = Database(settings.database_url.get_secret_value() if settings.database_url else "")
    since = datetime.now(UTC) - timedelta(hours=hours)
    with db.session() as s:
        n = s.scalar(
            select(func.count())
            .select_from(JobRecord)
            .where(JobRecord.state == "failed", JobRecord.updated_at >= since)
        )
    return _check("failed_jobs", not n, f"{n or 0} failed job(s) in the last {hours} h", "warning")


def database(settings: Settings) -> dict[str, Any]:
    from sqlalchemy import text

    from cyberlab_common.db import Database

    try:
        db = Database(settings.database_url.get_secret_value() if settings.database_url else "")
        with db.engine.connect() as c:
            c.execute(text("SELECT 1"))
        return _check("database", True, "reachable")
    except Exception as exc:  # noqa: BLE001 - any failure is the finding
        return _check("database", False, f"not reachable: {type(exc).__name__}")


def local_model(settings: Settings) -> dict[str, Any]:
    try:
        r = httpx.get(f"{settings.ollama_host.rstrip('/')}/api/tags", timeout=3)
        names = [m.get("name", "") for m in r.json().get("models", [])]
        ok = any(n.split(":")[0] == settings.ollama_model.split(":")[0] for n in names)
        return _check(
            "local_model",
            ok,
            f"{settings.ollama_model} {'ready' if ok else 'not pulled'}",
            "warning",
        )
    except (httpx.HTTPError, ValueError):
        return _check(
            "local_model", False, f"Ollama not reachable at {settings.ollama_host}", "warning"
        )


def backup_age(settings: Settings, max_hours: int = 30) -> dict[str, Any]:
    found = sorted((settings.data_dir / "backups").glob("cyberlab-*.tar.gz"))
    if not found:
        return _check("backup", False, "no backup yet", "warning")
    age_h = (time.time() - found[-1].stat().st_mtime) / 3600
    return _check("backup", age_h <= max_hours, f"newest backup is {age_h:.0f} h old", "warning")


def run_all(settings: Settings) -> list[dict[str, Any]]:
    checks = [
        lambda: disk(settings.data_dir),
        gpu,
        lambda: database(settings),
        lambda: failed_jobs(settings),
        lambda: local_model(settings),
        lambda: backup_age(settings),
    ]
    out = []
    for fn in checks:
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001 - a broken check is itself reported
            out.append(_check(getattr(fn, "__name__", "check"), False, f"check failed: {exc}"))
    return out


def alert(webhook: str, problems: list[dict[str, Any]], host: str) -> None:
    """POST a plain-text alert. Works with ntfy topics, and Slack/Discord incoming webhooks."""
    text = f"AutoResearch-CyberLab on {host}: " + "; ".join(
        f"{p['check']}: {p['detail']}" for p in problems
    )
    payload: dict[str, Any] | str
    if "hooks.slack.com" in webhook:
        payload = {"text": text}
    elif "discord.com" in webhook:
        payload = {"content": text}
    else:
        payload = text
    try:
        if isinstance(payload, str):
            httpx.post(webhook, content=payload.encode(), timeout=10)
        else:
            httpx.post(webhook, json=payload, timeout=10)
    except httpx.HTTPError:
        pass  # alerting must never crash the check
