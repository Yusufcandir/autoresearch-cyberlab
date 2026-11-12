"""`cyberlab` command line: read-only checks for Phase A."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from cyberlab_common.config import Settings


def _doctor(args: argparse.Namespace) -> int:
    from cyberlab_common.doctor import FAIL, run_all

    checks = run_all(Settings(), Path(args.path) if args.path else None)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=2))
    else:
        for c in checks:
            print(f"[{c.status:4}] {c.name:22} {c.detail}")
    return 1 if any(c.status == FAIL for c in checks) else 0


def _policy_check(args: argparse.Namespace) -> int:
    from cyberlab_authz.policy import ExecutionRequest, decide, write_audit

    settings = Settings()
    decision = decide(ExecutionRequest(args.tool, args.tier, approved=args.approved), settings)
    print(f"{decision.code}: {decision.reason} (policy v{decision.policy_version})")
    logged = write_audit(decision, settings)
    if logged:
        print(f"audit: {logged}")
    return 0 if decision.allowed else 2


def _gpu_baseline(args: argparse.Namespace) -> int:
    from cyberlab_common.gpu_baseline import record, summarize

    out = Path(args.out)
    if args.summarize:
        print(summarize(out))
        return 0
    n = record(out, args.interval, args.hours)
    print(f"wrote {n} samples to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cyberlab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="read-only check of this machine and config")
    p.add_argument("--path", help="disk path to check (default: home)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_doctor)

    p = sub.add_parser("policy-check", help="ask the security policy about a tool request")
    p.add_argument("tool", help="e.g. run_restricted_benchmark, run_experiment_template, shell")
    p.add_argument("tier", help="T0, T1, T2 or T3")
    p.add_argument("--approved", action="store_true", help="pretend a human approved it")
    p.set_defaults(func=_policy_check)

    p = sub.add_parser("gpu-baseline", help="sample GPU usage to CSV (read-only)")
    p.add_argument("--out", default="docs/inventory/gpu_baseline.csv")
    p.add_argument("--interval", type=int, default=60, help="seconds between samples")
    p.add_argument("--hours", type=float, default=24)
    p.add_argument("--summarize", action="store_true", help="summarize an existing CSV")
    p.set_defaults(func=_gpu_baseline)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
