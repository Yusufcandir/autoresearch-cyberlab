#!/usr/bin/env bash
# Back up the database, evidence artifacts, workflow checkpoints and the dataset manifest.
# Raw datasets are not included (they can be downloaded again).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
[ -f .env ] && . ./.env
set +a
DATA="${DATA_DIR:-./var}"
OUT="$DATA/backups"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d-%H%M%S)
TMP=$(mktemp -d)

case "${DATABASE_URL:-}" in
  postgresql*)
    docker compose -f infra/compose.server.yaml exec -T postgres pg_dump -U cyberlab cyberlab > "$TMP/db.sql"
    ;;
  sqlite*)
    DB="${DATABASE_URL#sqlite:///}"
    python3 -c "import sqlite3,sys; sqlite3.connect(sys.argv[1]).backup(sqlite3.connect(sys.argv[2]))" "$DB" "$TMP/cyberlab.db"
    ;;
esac
[ -f "$DATA/checkpoints.sqlite" ] && cp "$DATA/checkpoints.sqlite" "$TMP/"
[ -d "$DATA/artifacts" ] && cp -r "$DATA/artifacts" "$TMP/"
[ -f "$DATA/datasets/cicids2017/prepared/manifest.json" ] && cp "$DATA/datasets/cicids2017/prepared/manifest.json" "$TMP/"
tar -czf "$OUT/cyberlab-$STAMP.tar.gz" -C "$TMP" .
rm -rf "$TMP"
echo "backup written: $OUT/cyberlab-$STAMP.tar.gz"
