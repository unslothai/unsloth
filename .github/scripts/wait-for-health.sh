#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Poll a booted Unsloth's /api/health until healthy; on timeout print the server log tail.
# Usage:
#   wait-for-health.sh --port 18888 [--log logs/studio.log] [--tmp /tmp/health.json]
# Retries on any not-yet-healthy answer, not only on a refused connection.

set -uo pipefail

# Cold runners with venv warm-up and lazy imports can exceed 60s.
TIMEOUT_SECONDS=180

PORT=""
LOG="logs/studio.log"
TMP="/tmp/health.json"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --log)  LOG="$2";  shift 2 ;;
    --tmp)  TMP="$2";  shift 2 ;;
    *) echo "wait-for-health.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$PORT" ] || { echo "wait-for-health.sh: --port is required" >&2; exit 2; }

# Use a wall-clock deadline plus curl --max-time: a wedged server can hang a probe forever,
# and counting iterations would stretch 180s to minutes once each probe can take --max-time.
deadline=$(( SECONDS + TIMEOUT_SECONDS ))
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -fs --connect-timeout 3 --max-time 5 \
       "http://127.0.0.1:${PORT}/api/health" > "$TMP" \
     && jq -e '.status == "healthy"' "$TMP" > /dev/null; then
    echo "[health] 127.0.0.1:${PORT} reported healthy"
    exit 0
  fi
  sleep 1
done

echo "Unsloth did not become healthy in ${TIMEOUT_SECONDS}s"
if [ -f "$LOG" ]; then
  tail -200 "$LOG"
else
  echo "wait-for-health.sh: no log at '$LOG' to tail" >&2
fi
exit 1
