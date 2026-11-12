"""Read-only GPU usage sampler for the Step 02 baseline (who uses the GPU, and when)."""

from __future__ import annotations

import csv
import shutil
import subprocess  # noqa: S404 - fixed read-only nvidia-smi query
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "util_pct", "mem_used_mib", "mem_total_mib", "processes"]


def sample() -> dict[str, str] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    q = "utilization.gpu,memory.used,memory.total"
    gpu = subprocess.run(  # noqa: S603
        [exe, f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).stdout.splitlines()
    apps = subprocess.run(  # noqa: S603
        [exe, "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).stdout.splitlines()
    if not gpu:
        return None
    util, used, total = (p.strip() for p in gpu[0].split(","))
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "util_pct": util,
        "mem_used_mib": used,
        "mem_total_mib": total,
        "processes": str(len([a for a in apps if a.strip()])),
    }


def record(out: Path, interval_s: int, hours: float) -> int:
    """Append samples to a CSV until `hours` have passed. Returns the number of samples."""
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    end = time.monotonic() + hours * 3600
    n = 0
    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        while time.monotonic() < end:
            row = sample()
            if row is None:
                raise RuntimeError("nvidia-smi not available on this machine")
            writer.writerow(row)
            fh.flush()
            n += 1
            time.sleep(interval_s)
    return n


def summarize(path: Path) -> str:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        return "no samples"
    busy = [r for r in rows if int(r["util_pct"]) > 10 or int(r["mem_used_mib"]) > 1024]
    by_hour: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_hour[datetime.fromisoformat(r["timestamp"]).hour].append(int(r["util_pct"]))
    lines = [
        f"samples: {len(rows)} from {rows[0]['timestamp']} to {rows[-1]['timestamp']}",
        f"busy samples (>10% util or >1 GiB used): {100 * len(busy) / len(rows):.1f}%",
        f"peak memory used: {max(int(r['mem_used_mib']) for r in rows)} MiB",
        "mean utilisation by hour of day:",
    ]
    for hour in sorted(by_hour):
        vals = by_hour[hour]
        mean = sum(vals) / len(vals)
        lines.append(f"  {hour:02d}:00  {mean:5.1f}%  {'#' * int(mean / 5)}")
    return "\n".join(lines)
