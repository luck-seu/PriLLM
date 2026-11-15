#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT_DIR/output"
LOG_DIR="$OUTPUT_DIR/logs"
LOCK_FILE="$OUTPUT_DIR/.run.lock"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date -u +%Y-%m-%d).log"

run() {
  cd "$ROOT_DIR"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -m openrouter_harvester.cli \
    --root "$ROOT_DIR" \
    --workers "${HARVEST_WORKERS:-6}" \
    run
}

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    printf '%s another harvest is already running\n' "$(date -u +%FT%TZ)" >>"$LOG_FILE"
    exit 0
  fi
fi

run >>"$LOG_FILE" 2>&1

