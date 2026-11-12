#!/usr/bin/env bash
# User-space setup on the lab workstation. No sudo, no system packages, no drivers, no open ports.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Installing uv into ~/.local (user only) =="
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  python3 -m pip install --user --quiet uv
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== Creating the project environment (.venv, Python 3.12) =="
uv sync --locked

if [ ! -f .env ]; then
  echo "== Creating .env from infra/env.example (edit it; it is git-ignored) =="
  cp infra/env.example .env
  chmod 600 .env
fi

echo "== Read-only machine check =="
uv run cyberlab doctor || true

echo
echo "Done. Add this to ~/.bashrc so 'uv' is always found:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
