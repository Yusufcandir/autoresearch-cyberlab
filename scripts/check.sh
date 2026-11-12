#!/usr/bin/env bash
# Same checks as CI.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
echo "All checks passed."
