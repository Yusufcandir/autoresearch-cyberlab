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

    p = sub.add_parser("dataset", help="download (official URL only), unzip or prepare a dataset")
    p.add_argument("action", choices=["download", "unzip", "prepare", "fixture"])
    p.add_argument("--name", default="cicids2017", help="cicids2017 (default) or unsw-nb15")
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

    p = sub.add_parser(
        "restore-test", help="restore a backup into a scratch folder and verify its evidence"
    )
    p.add_argument("backup", nargs="?", help="backup .tar.gz (default: the newest one)")
    p.set_defaults(func=_restore_test)

    p = sub.add_parser(
        "plugin-selftest", help="run every capability plugin on its synthetic fixture (AT-16..19)"
    )
    p.add_argument(
        "--docker", action="store_true", help="also run the amended code's tests in Docker"
    )
    p.set_defaults(func=_plugin_selftest)

    p = sub.add_parser(
        "health-check", help="disk, GPU, database, failed jobs, local model, backup age"
    )
    p.add_argument("--alert", action="store_true", help="post problems to ALERT_WEBHOOK")
    p.set_defaults(func=_health_check)

    p = sub.add_parser(
        "migrate-db", help="copy every table from DATABASE_URL into another (empty) database"
    )
    p.add_argument(
        "target", help="target URL, e.g. postgresql+psycopg://cyberlab:PW@127.0.0.1:5433/cyberlab"
    )
    p.set_defaults(func=_migrate_db)

    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


def _plugin_selftest(args: argparse.Namespace) -> int:
    import tempfile

    from cyberlab_schemas.plugins import PluginError, list_capabilities

    results: dict[str, Any] = {"plugins": [c.name for c in list_capabilities()]}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        from cyberlab_tabular.text import TextAdapter
        from cyberlab_tabular.text import fixture as text_fixture

        results["text"] = TextAdapter().validate_dataset(text_fixture())["ok"]
        from cyberlab_audio import AudioAdapter
        from cyberlab_audio import write_fixture as audio_fixture

        results["audio"] = AudioAdapter().validate_dataset(audio_fixture(root / "audio"))["ok"]
        from cyberlab_graph import GraphAdapter, decide
        from cyberlab_graph import fixture as graph_fixture

        results["graph"] = GraphAdapter().validate_dataset(graph_fixture())["ok"]
        policy = {
            "version": "1",
            "statements": [
                {
                    "sid": "deny-all",
                    "effect": "deny",
                    "principals": ["*"],
                    "actions": ["*"],
                    "resources": ["*"],
                }
            ],
        }
        d = decide(
            policy,
            {"principal": "user:alice", "action": "read", "resource": "x"},
            graph_fixture(),
            model_score=0.0,
        )
        results["policy_deny_kept_despite_safe_score"] = d.effect == "deny"
        from cyberlab_code import amend, extract_metadata
        from cyberlab_code import write_fixture as code_fixture
        from cyberlab_code.metadata import make_sdist_fixture

        canary = root / "EXECUTED"
        meta = extract_metadata(make_sdist_fixture(root / "demo.tar.gz", canary=canary))
        results["metadata_read_without_execution"] = meta["name"] == "demo" and not canary.exists()
        diff = (
            "--- a/src/calc.py\n+++ b/src/calc.py\n@@ -1,2 +1,2 @@\n def mean(values):\n"
            "-    return sum(values) / (len(values) + 1)\n+    return sum(values) / len(values)\n"
        )
        runner = None
        if args.docker:
            if not Settings().allow_restricted_execution:
                print("--docker needs ALLOW_RESTRICTED_EXECUTION=true in .env (spec 11.3)")
                return 1
            from cyberlab_code import docker_test_runner

            runner = docker_test_runner()
        try:
            receipt = amend(
                code_fixture(root / "snap"), diff, ["src/*.py"], root, test_runner=runner
            )
            results["amendment"] = {
                "status": receipt["status"],
                "functional_tests": receipt["functional_tests"].get("status"),
                "tests_passed": receipt["functional_tests"].get("passed"),
            }
        except (PluginError, OSError) as exc:
            results["amendment"] = {"error": str(exc)}
    print(json.dumps(results, indent=2))
    ok = all(v for k, v in results.items() if isinstance(v, bool))
    return 0 if ok else 1


def _health_check(args: argparse.Namespace) -> int:
    import socket

    from cyberlab_common.health import alert, run_all

    settings = Settings()
    results = run_all(settings)
    for r in results:
        mark = "OK  " if r["ok"] else ("WARN" if r["level"] == "warning" else "FAIL")
        print(f"{mark} {r['check']:<12} {r['detail']}")
    problems = [r for r in results if not r["ok"]]
    if problems and args.alert and settings.alert_webhook:
        alert(settings.alert_webhook, problems, socket.gethostname())
    return 1 if any(r["level"] == "error" for r in problems) else 0


def _migrate_db(args: argparse.Namespace) -> int:
    from cyberlab_common.migrate import copy_database

    url = Settings().database_url
    source = url.get_secret_value() if url is not None else None
    if not source:
        print("DATABASE_URL is not set")
        return 1
    counts = copy_database(source, args.target)
    print(json.dumps(counts, indent=2))
    print("done: now set DATABASE_URL in .env to the target and restart serve and worker")
    return 0


def _restore_test(args: argparse.Namespace) -> int:
    from cyberlab_common.restore import verify_backup

    if args.backup:
        tarball = Path(args.backup)
    else:
        backups = Settings().data_dir / "backups"
        found = sorted(backups.glob("cyberlab-*.tar.gz"))
        if not found:
            print(f"no backups in {backups}; run scripts/backup.sh first")
            return 1
        tarball = found[-1]
    report = verify_backup(tarball)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


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
    from cyberlab_data import DatasetBlocked, get_adapter, manifest_summary

    adapter = get_adapter(args.name)
    s = Settings()
    root = adapter.root(s.dataset_root)
    try:
        if args.action == "download":
            if adapter.download is None:
                print(adapter.manual_steps.format(raw=root / "raw"))
                return 2
            print(f"trying the official {adapter.display} download ...")
            adapter.download(root)
        elif args.action == "unzip":
            if adapter.unzip is None:
                print(f"{adapter.display} is distributed as CSV files; no unzip needed")
                return 0
            files = adapter.unzip(root)
            print(f"extracted {len(files)} CSV files into {root / 'raw'}:")
            for f in files:
                print(f"  {f.name}")
            print(f"next: cyberlab dataset prepare --name {adapter.key}")
            return 0 if files else 2
        elif args.action == "fixture":
            adapter.write_fixture(root / "raw")
            print(f"wrote a small SYNTHETIC test fixture to {root / 'raw'} (not real data)")
            return 0
        prepared = adapter.prepare(root, benign_cap=args.benign_cap)
    except DatasetBlocked as exc:
        print(exc)
        return 2
    print(json.dumps(manifest_summary(prepared.manifest), indent=2))
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
