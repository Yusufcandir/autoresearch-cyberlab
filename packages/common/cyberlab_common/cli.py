"""`cyberlab` command line: host checks, setup, worker and server."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

    p = sub.add_parser("init-db", help="create database tables")
    p.set_defaults(func=_init_db)

    p = sub.add_parser("create-user", help="create a user (prompts for the password)")
    p.add_argument("username")
    p.add_argument(
        "--role", default="researcher", choices=["admin", "pi", "researcher", "reviewer"]
    )
    p.set_defaults(func=_create_user)

    p = sub.add_parser("dataset", help="CIC-IDS2017: download (official URL only) or prepare")
    p.add_argument("action", choices=["download", "unzip", "prepare", "fixture"])
    p.add_argument("--benign-cap", type=int, default=400_000)
    p.set_defaults(func=_dataset)

    p = sub.add_parser("worker", help="run the job broker loop (one GPU job at a time)")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--once", action="store_true")
    p.set_defaults(func=_worker)

    p = sub.add_parser("serve", help="start the API and dashboard (localhost only)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.set_defaults(func=_serve)

    p = sub.add_parser("llm-check", help="send one tiny prompt through the configured LLM gateway")
    p.set_defaults(func=_llm_check)

    p = sub.add_parser("gemini-models", help="list Gemini models your API key can use")
    p.set_defaults(func=_gemini_models)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


def _db() -> Any:
    from cyberlab_common.db import Database

    s = Settings()
    if s.database_url is None:
        raise SystemExit("DATABASE_URL is not set (see .env)")
    db = Database(s.database_url.get_secret_value())
    db.create_all()
    return db


def _init_db(args: argparse.Namespace) -> int:
    _db()
    print("database ready")
    return 0


def _create_user(args: argparse.Namespace) -> int:
    import getpass
    import os

    from cyberlab_authz.rbac import create_user

    pw = os.environ.get("CYBERLAB_NEW_PASSWORD") or getpass.getpass("Password (10+ characters): ")
    if "CYBERLAB_NEW_PASSWORD" not in os.environ and pw != getpass.getpass("Repeat password: "):
        print("passwords differ")
        return 1
    with _db().session() as s:
        u = create_user(s, args.username, pw, args.role)
        print(f"created {u.username} ({u.role})")
    return 0


def _dataset(args: argparse.Namespace) -> int:
    from cyberlab_data import cicids2017

    s = Settings()
    root = s.dataset_root / "cicids2017"
    if args.action == "download":
        print("trying the official CIC-IDS2017 download ...")
        try:
            cicids2017.download(root)
        except cicids2017.DatasetBlocked as exc:
            print(exc)
            return 2
    elif args.action == "unzip":
        files = cicids2017.unzip(root)
        print(f"extracted {len(files)} CSV files into {root / 'raw'}:")
        for f in files:
            print(f"  {f.name}")
        print("next: cyberlab dataset prepare")
        return 0 if files else 2
    elif args.action == "fixture":
        cicids2017.write_fixture(root / "raw")
        print(f"wrote a small SYNTHETIC test fixture to {root / 'raw'} (not real data)")
        return 0
    prepared = cicids2017.prepare(root, benign_cap=args.benign_cap)
    print(json.dumps(cicids2017.manifest_summary(prepared.manifest), indent=2))
    return 0 if prepared.manifest.status == "verified" else 2


def _worker(args: argparse.Namespace) -> int:
    import time

    from cyberlab_orchestrator.lab import Lab

    lab = Lab(Settings())
    print("worker started; Ctrl+C to stop", flush=True)
    try:
        while True:
            info = lab.worker_tick(wait=args.once)
            if info["started"] or info["advanced"] or info["reconciled"]:
                print(json.dumps(info), flush=True)
            if args.once:
                return 0
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("cyberlab_api.main:app", host=args.host, port=args.port)
    return 0


def _llm_check(args: argparse.Namespace) -> int:
    from cyberlab_llm import LLMError, build_gateway

    gw = build_gateway(Settings())
    try:
        res = gw.complete("health.check", "Answer in five words or fewer.", "Say hello to the lab.")
    except LLMError as exc:
        print(f"LLM check failed: {exc}")
        return 1
    print(
        f"{res.provider}/{res.model}: {res.text.strip()!r} ({res.total_tokens} tokens, {res.seconds:.1f}s)"
    )
    return 0


def _gemini_models(args: argparse.Namespace) -> int:
    import httpx

    s = Settings()
    if s.gemini_api_key is None:
        print("GEMINI_API_KEY is not set")
        return 1
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": s.gemini_api_key.get_secret_value()},
        params={"pageSize": 200},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"HTTP {r.status_code}: {r.text[:300]}")
        return 1
    for m in r.json().get("models", []):
        if "generateContent" in m.get("supportedGenerationMethods", []):
            print(m["name"].removeprefix("models/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
