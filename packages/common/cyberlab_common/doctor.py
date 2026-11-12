"""Read-only host check (Step 02). Installs nothing, kills nothing, changes nothing."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess  # noqa: S404 - fixed read-only commands only
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cyberlab_common.config import Settings

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


def _run(cmd: list[str], timeout: float = 10) -> str | None:
    exe = shutil.which(cmd[0])
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - absolute path, fixed arguments
            [exe, *cmd[1:]], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _http_ok(url: str, timeout: float = 5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed URLs
            status: int = resp.status
            return 200 <= status < 400
    except OSError:
        return False


def check_python() -> Check:
    v = sys.version_info
    status = OK if (v.major, v.minor) == (3, 12) else WARN
    return Check("python", status, f"{platform.python_version()} ({sys.executable})")


def check_gpu() -> list[Check]:
    q = "name,memory.total,memory.used,utilization.gpu,driver_version"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    if out is None:
        return [Check("gpu", INFO, "nvidia-smi not available (expected on a dev PC)")]
    checks = []
    for line in out.splitlines():
        name, total, used, util, driver = (p.strip() for p in line.split(","))
        busy = int(used) > 1024 or int(util) > 10
        status = WARN if busy else OK
        detail = f"{name}, {used}/{total} MiB used, {util}% util, driver {driver}"
        checks.append(Check("gpu", status, detail + (" (in use by someone)" if busy else "")))
    procs = _run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]) or ""
    n = len([p for p in procs.splitlines() if p.strip()])
    checks.append(Check("gpu processes", WARN if n else OK, f"{n} running"))
    return checks


def check_disk(path: Path) -> Check:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    pct_used = 100 * usage.used / usage.total
    status = FAIL if free_gb < 20 else WARN if pct_used > 85 else OK
    return Check("disk", status, f"{free_gb:.0f} GB free, {pct_used:.0f}% used at {path}")


def check_tools() -> list[Check]:
    wanted = {
        "git": True,
        "conda": False,
        "ollama": False,
        "docker": False,
        "apptainer": False,
        "sbatch": False,
    }
    checks = []
    for tool, required in wanted.items():
        found = shutil.which(tool)
        if found:
            checks.append(Check(f"tool {tool}", OK, found))
        else:
            checks.append(Check(f"tool {tool}", FAIL if required else INFO, "missing"))
    return checks


def check_network() -> list[Check]:
    internet = _http_ok("https://pypi.org/simple/")
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    if not ollama_url.startswith("http"):
        ollama_url = "http://" + ollama_url
    ollama = _http_ok(ollama_url + "/api/tags", timeout=2)
    return [
        Check("internet", OK if internet else WARN, "pypi.org reachable" if internet else "no"),
        Check("ollama api", OK if ollama else INFO, f"{ollama_url} {'up' if ollama else 'down'}"),
    ]


def check_config(settings: Settings) -> list[Check]:
    checks = [
        Check(
            "restricted execution",
            OK if not settings.allow_restricted_execution else WARN,
            f"ALLOW_RESTRICTED_EXECUTION={settings.allow_restricted_execution}, "
            f"SANDBOX_BACKEND={settings.sandbox_backend.value}",
        ),
        Check("scheduler", INFO, f"SCHEDULER_BACKEND={settings.scheduler_backend.value}"),
    ]
    missing = settings.missing_required()
    checks.append(
        Check(
            "required config",
            OK if not missing else INFO,
            "complete" if not missing else "missing " + ", ".join(missing) + " (API not ready)",
        )
    )
    return checks


def run_all(settings: Settings, path: Path | None = None) -> list[Check]:
    return [
        check_python(),
        *check_gpu(),
        check_disk(path or Path.home()),
        *check_tools(),
        *check_network(),
        *check_config(settings),
    ]
