#!/usr/bin/env bash
# Install the CyberLab user services (no root needed). Replaces the tmux setup:
# serve, worker and ollama restart on failure, health check every 15 min, backup nightly.
#
#   bash scripts/install_services.sh
#
# For the services to keep running after you log out (and start after a reboot) an admin runs
# ONCE:  sudo loginctl enable-linger $USER
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
[ "$REPO" = "$HOME/autoresearch-cyberlab" ] || {
  echo "units assume the repo at ~/autoresearch-cyberlab (found $REPO); edit infra/systemd/*.service"; exit 1; }
[ -f .env ] || { echo ".env missing"; exit 1; }
UNITS="$HOME/.config/systemd/user"
mkdir -p "$UNITS"
install_unit() { cp "infra/systemd/$1" "$UNITS/$1"; }
for u in cyberlab-serve.service cyberlab-worker.service cyberlab-health.service \
         cyberlab-health.timer cyberlab-backup.service cyberlab-backup.timer; do install_unit "$u"; done
# Ollama: use whichever binary this user runs (PATH, ~/.local/bin or ~/ollama/bin).
OLLAMA=""
OLLAMA_BIN="$(command -v ollama || true)"
for c in "$HOME/bin/ollama" "$HOME/.local/bin/ollama" "$HOME/ollama/bin/ollama"; do
  [ -z "$OLLAMA_BIN" ] && [ -x "$c" ] && OLLAMA_BIN="$c"
done
if [ -n "$OLLAMA_BIN" ]; then
  sed "s|%h/.local/bin/ollama|$OLLAMA_BIN|" infra/systemd/ollama.service > "$UNITS/ollama.service"
  OLLAMA=ollama.service
  tmux kill-session -t ollama 2>/dev/null || true  # the service replaces a tmux copy
else
  echo "WARNING: ollama binary not found; no local fallback model until it is installed"
fi
systemctl --user daemon-reload
echo "Stop the tmux copies first (tmux kill-session -t lab), otherwise port 8000 is taken."
systemctl --user enable --now cyberlab-serve.service cyberlab-worker.service \
  cyberlab-health.timer cyberlab-backup.timer $OLLAMA
systemctl --user --no-pager status cyberlab-serve cyberlab-worker | grep -E "Active|●" || true
loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes || \
  echo "WARNING: linger is off; services stop when you log out. Ask an admin: sudo loginctl enable-linger $USER"
echo "Logs: journalctl --user -u cyberlab-worker -f"
