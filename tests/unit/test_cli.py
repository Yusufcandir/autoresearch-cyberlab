import csv
from pathlib import Path

import pytest

from cyberlab_common import gpu_baseline
from cyberlab_common.cli import main
from cyberlab_common.doctor import check_disk


def test_policy_check_blocks_restricted_on_shared_host(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALLOW_RESTRICTED_EXECUTION", raising=False)
    monkeypatch.delenv("AUDIT_LOG_DESTINATION", raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # no .env here
    assert main(["policy-check", "run_restricted_benchmark", "T3", "--approved"]) == 2
    assert "BLOCKED: ISOLATION_REQUIRED" in capsys.readouterr().out


def test_policy_check_allows_template(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUDIT_LOG_DESTINATION", raising=False)
    monkeypatch.chdir(Path(__file__).parent)
    assert main(["policy-check", "run_experiment_template", "T1"]) == 0
    assert "ALLOWED" in capsys.readouterr().out


def test_disk_check_reports_free_space(tmp_path: Path) -> None:
    c = check_disk(tmp_path)
    assert "GB free" in c.detail


def test_gpu_baseline_summary(tmp_path: Path) -> None:
    out = tmp_path / "b.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=gpu_baseline.FIELDS)
        w.writeheader()
        w.writerow(
            {
                "timestamp": "2026-11-12T09:00:00",
                "util_pct": "0",
                "mem_used_mib": "12",
                "mem_total_mib": "49140",
                "processes": "0",
            }
        )
        w.writerow(
            {
                "timestamp": "2026-11-12T10:00:00",
                "util_pct": "90",
                "mem_used_mib": "30000",
                "mem_total_mib": "49140",
                "processes": "1",
            }
        )
    text = gpu_baseline.summarize(out)
    assert "busy samples" in text
    assert "50.0%" in text
    assert "peak memory used: 30000 MiB" in text
