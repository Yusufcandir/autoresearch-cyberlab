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
OLLAMA=""
if [ -x "$HOME/.local/bin/ollama" ]; then install_unit ollama.service; OLLAMA=ollama.service
else echo "note: ~/.local/bin/ollama not found; ollama.service skipped (keep running it as before)"; fi
systemctl --user daemon-reload
echo "Stop the tmux copies first (tmux kill-session -t lab), otherwise port 8000 is taken."
systemctl --user enable --now cyberlab-serve.service cyberlab-worker.service \
  cyberlab-health.timer cyberlab-backup.timer $OLLAMA
systemctl --user --no-pager status cyberlab-serve cyberlab-worker | grep -E "Active|●" || true
loginctl show-user "$USER" -p Linger 2>/dev/null | grep -q yes || \
  echo "WARNING: linger is off; services stop when you log out. Ask an admin: sudo loginctl enable-linger $USER"
echo "Logs: journalctl --user -u cyberlab-worker -f"
