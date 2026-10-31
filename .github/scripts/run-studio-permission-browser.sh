#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

set -euo pipefail

port="${1:?usage: $0 PORT BROWSER [CHANNEL]}"
browser="${2:?usage: $0 PORT BROWSER [CHANNEL]}"
channel="${3:-}"
slug="$browser${channel:+-$channel}"
artifact_dir="logs/playwright-permissions-$slug"
server_log="logs/studio-permissions-$slug.log"
studio_home="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}"
set --
if [ -n "${STUDIO_PERMISSION_FRONTEND:-}" ]; then
  set -- -f "$STUDIO_PERMISSION_FRONTEND"
fi

mkdir -p "$artifact_dir"

# The Windows UI lane calls this right after stopping its own server on the same home, and a
# killed server on Windows can hold auth.db open for a few seconds after `kill` returns. The
# wipe then failed with "Device or resource busy" and the lane went red after every suite had
# passed (runs 39669502772, staging 39764533102). Retry for a bounded time, then fail loudly.
wipe_auth() {
  local tries=0
  until rm -rf "$studio_home/auth" 2>/dev/null; do
    tries=$((tries + 1))
    if [ "$tries" -ge "${STUDIO_AUTH_WIPE_TRIES:-30}" ]; then
      # One last attempt with rm's own error showing; it can still be the one that succeeds.
      if rm -rf "$studio_home/auth"; then
        return 0
      fi
      echo "::error::$studio_home/auth is still held open after $((tries + 1)) tries; a server on this home did not exit" >&2
      return 1
    fi
    sleep 1
  done
}

# Wipe (not reset-password): the boot below must re-seed a fresh .bootstrap_password.
wipe_auth
UNSLOTH_API_ONLY=1 unsloth studio -H 127.0.0.1 -p "$port" "$@" \
  >"$server_log" 2>&1 &
studio_pid=$!

cleanup() {
  kill "$studio_pid" 2>/dev/null || true
  wait "$studio_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Report which signal killed us and whether the suite had already passed (suite_done).
suite_done=0
_on_signal() {
  name="$1"; number="$2"
  echo "[permissions] SIG${name} received at $(date -u +%H:%M:%S) after suite_done=${suite_done}" >&2
  # comm, not args: a command line in this public log could carry an unmasked token.
  ps -o pid,ppid,comm 2>/dev/null | tail -20 >&2 || true
  exit $((128 + number))
}
trap '_on_signal TERM 15' TERM
trap '_on_signal INT 2' INT
trap '_on_signal HUP 1' HUP

healthy=0
# --max-time plus a real deadline, so a wedged server cannot stall the loop.
health_deadline=$(( SECONDS + 180 ))
while [ "$SECONDS" -lt "$health_deadline" ]; do
  if curl -fs --connect-timeout 3 --max-time 5 \
       "http://127.0.0.1:$port/api/health" >/dev/null; then
    healthy=1
    break
  fi
  if ! kill -0 "$studio_pid" 2>/dev/null; then
    tail -100 "$server_log" || true
    exit 1
  fi
  sleep 1
done
if [ "$healthy" -ne 1 ]; then
  tail -100 "$server_log" || true
  exit 1
fi

old_password=$(cat "$studio_home/auth/.bootstrap_password")
new_password="CIPerm-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "::add-mask::$old_password"
  echo "::add-mask::$new_password"
fi

export BASE_URL="http://127.0.0.1:$port"
export STUDIO_OLD_PW="$old_password"
export STUDIO_NEW_PW="$new_password"
export STUDIO_UI_STRICT=1
export STUDIO_UI_PERMISSION_ONLY=1
# Per-browser cap between progress reports, not a total for the step.
export STUDIO_UI_WALL_TIMEOUT_S=240
export STUDIO_UI_TOTAL_TIMEOUT_S=240
export STUDIO_PLAYWRIGHT_BROWSER="$browser"
export PW_ART_DIR="$artifact_dir"
if [ -n "$channel" ]; then
  export STUDIO_PLAYWRIGHT_CHANNEL="$channel"
else
  unset STUDIO_PLAYWRIGHT_CHANNEL || true
fi

python tests/studio/playwright_chat_ui.py
# Set only after the suite passes, to tell a teardown signal from a mid-run one.
suite_done=1
