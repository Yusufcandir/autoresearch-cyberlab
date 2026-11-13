# code_artifact (`cyberlab_code`)

* Static source analysis (`ast`, never imported or executed) with hash-checked snapshots.
* `extract_metadata()`: wheel/sdist/npm metadata read in memory; install scripts never run;
  setup.py-only packages are blocked with `METADATA_REQUIRES_EXECUTION` (AT-18).
* `amend()`: applies a proposed unified diff to a copy of a snapshot inside permitted paths
  (tests are read-only), static security scan, and functional tests only through an isolated
  runner; otherwise `BLOCKED: ISOLATION_REQUIRED`. A receipt links all hashes (AT-19).
* `docker_test_runner()`: no network, read-only, non-root, capped container for reviewed
  fixtures (T2). Not a substitute for a separate VM for hostile (T3) code.

Run `uv run cyberlab plugin-selftest` (add `--docker` on a host with Docker and
`ALLOW_RESTRICTED_EXECUTION=true`).
