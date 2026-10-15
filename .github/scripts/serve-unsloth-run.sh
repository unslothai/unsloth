#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Boot `unsloth run --disable-tools` in the background, wait for health, parse the API key
# from the banner and resolve the /v1/models id. Exports to $GITHUB_ENV (or prints locally).
# Usage:
#   serve-unsloth-run.sh --model REPO --gguf-variant VAR --port PORT \
#       [--gguf-file PATH] [--extra "--seed 3407 --temp 0"] \
#       [--log-dir logs] [--health-timeout 300] [--banner-timeout 300]
# Outputs: UNSLOTH_API_KEY, UNSLOTH_STUDIO_URL, UNSLOTH_BASE_URL, UNSLOTH_MODEL_ID,
# UNSLOTH_SERVER_PID, UNSLOTH_LLAMA_LOG_DIR.

set -uo pipefail

MODEL=""
GGUF_VARIANT=""
GGUF_FILE=""
PORT=""
EXTRA=""
LOG_DIR="logs"
HEALTH_TIMEOUT="300"
BANNER_TIMEOUT="300"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)          MODEL="$2"; shift 2 ;;
    --gguf-variant)   GGUF_VARIANT="$2"; shift 2 ;;
    --gguf-file)      GGUF_FILE="$2"; shift 2 ;;
    --port)           PORT="$2"; shift 2 ;;
    --extra)          EXTRA="$2"; shift 2 ;;
    --log-dir)        LOG_DIR="$2"; shift 2 ;;
    --health-timeout) HEALTH_TIMEOUT="$2"; shift 2 ;;
    --banner-timeout) BANNER_TIMEOUT="$2"; shift 2 ;;
    *) echo "serve-unsloth-run.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$PORT" ] || { echo "serve-unsloth-run.sh: --port is required" >&2; exit 2; }
if [ -z "$MODEL" ] && [ -z "$GGUF_FILE" ]; then
  echo "serve-unsloth-run.sh: one of --model or --gguf-file is required" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/unsloth-run-${PORT}.log"
BASE_URL="http://127.0.0.1:${PORT}"
STUDIO_HOME_DIR="${STUDIO_HOME:-$HOME/.unsloth/studio}"
LLAMA_LOG_DIR="${STUDIO_HOME_DIR}/logs/llama-server"

emit() {
  echo "$1=$2"
  if [ -n "${GITHUB_ENV:-}" ]; then
    echo "$1=$2" >> "$GITHUB_ENV"
  fi
}

server_fail() {
  echo "::error::Unsloth server/API regression: $*" >&2
  echo "---- last 200 lines of $SERVER_LOG ----" >&2
  tail -200 "$SERVER_LOG" 2>/dev/null || true
  exit 1
}

# Fail fast if the port is taken, or we would test the wrong server.
if command -v ss >/dev/null 2>&1; then
  if ss -tln 2>/dev/null | grep -q ":${PORT}\b"; then
    server_fail "port ${PORT} already has a listener before we started (collision)"
  fi
fi

# --disable-tools is required so the agent's own tools relay instead of the server's.
CMD=(unsloth run -H 127.0.0.1 -p "$PORT" --disable-tools --no-cloudflare)
if [ -n "$GGUF_FILE" ]; then
  CMD+=(--model "$GGUF_FILE")
else
  CMD+=(--model "$MODEL")
  [ -n "$GGUF_VARIANT" ] && CMD+=(--gguf-variant "$GGUF_VARIANT")
fi
# shellcheck disable=SC2206  # intentional word-split of caller-controlled flags
[ -n "$EXTRA" ] && CMD+=($EXTRA)

echo "[serve] launching: ${CMD[*]}"
echo "[serve] server log: $SERVER_LOG"

setsid "${CMD[@]}" > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PID=$!
emit UNSLOTH_SERVER_PID "$SERVER_PID"

HEALTHY=0
for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    server_fail "process exited before becoming healthy (pid $SERVER_PID)"
  fi
  if curl -fs "${BASE_URL}/api/health" -o "$LOG_DIR/health-${PORT}.json" 2>/dev/null; then
    if jq -e '.status == "healthy"' "$LOG_DIR/health-${PORT}.json" >/dev/null 2>&1; then
      HEALTHY=1
      break
    fi
  fi
  sleep 1
done
[ "$HEALTHY" = "1" ] || server_fail "did not report /api/health healthy within ${HEALTH_TIMEOUT}s"
echo "[serve] /api/health healthy"

# The key prints only after the model loads, which /api/health does not wait for.
API_KEY=""
for _ in $(seq 1 "$BANNER_TIMEOUT"); do
  API_KEY="$(grep -aoE 'sk-unsloth-[A-Za-z0-9_-]+' "$SERVER_LOG" 2>/dev/null | head -1 || true)"
  [ -n "$API_KEY" ] && break
  grep -aq 'API Key:' "$SERVER_LOG" 2>/dev/null && break
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    server_fail "process exited before printing its banner (the model load failed)"
  fi
  sleep 1
done
if [ -z "$API_KEY" ] && ! grep -aq 'API Key:' "$SERVER_LOG" 2>/dev/null; then
  server_fail "no banner within ${BANNER_TIMEOUT}s of /api/health: the model is still loading"
fi
if [ -z "$API_KEY" ]; then
  API_KEY="$(grep -aE 'API Key:' "$SERVER_LOG" 2>/dev/null \
    | sed -E 's/.*API Key:[[:space:]]*//' | head -1 || true)"
fi
[ -n "$API_KEY" ] || server_fail "could not parse an API key from the banner (banner-parse fragility -- check the 'API Key:' line in unsloth_cli/commands/studio.py)"
echo "::add-mask::${API_KEY}"
emit UNSLOTH_API_KEY "$API_KEY"

if ! curl -fs "${BASE_URL}/v1/models" \
    -H "Authorization: Bearer ${API_KEY}" -o "$LOG_DIR/models-${PORT}.json" 2>/dev/null; then
  server_fail "/v1/models did not respond (or rejected the banner key)"
fi
MODEL_ID="$(jq -r '.data[0].id // empty' "$LOG_DIR/models-${PORT}.json" 2>/dev/null || true)"
[ -n "$MODEL_ID" ] || server_fail "/v1/models returned no model id (model failed to load)"
echo "[serve] resolved model id: $MODEL_ID"

emit UNSLOTH_MODEL_ID "$MODEL_ID"
emit UNSLOTH_STUDIO_URL "$BASE_URL"
emit UNSLOTH_BASE_URL "$BASE_URL"
emit UNSLOTH_LLAMA_LOG_DIR "$LLAMA_LOG_DIR"

echo "[serve] server is up: ${BASE_URL} (model ${MODEL_ID})"
