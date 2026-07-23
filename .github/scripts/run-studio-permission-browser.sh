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
unsloth studio reset-password
UNSLOTH_API_ONLY=1 unsloth studio -H 127.0.0.1 -p "$port" "$@" \
  >"$server_log" 2>&1 &
studio_pid=$!

cleanup() {
  kill "$studio_pid" 2>/dev/null || true
  wait "$studio_pid" 2>/dev/null || true
}
trap cleanup EXIT

healthy=0
for _ in $(seq 1 180); do
  if curl -fs "http://127.0.0.1:$port/api/health" >/dev/null; then
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
export STUDIO_UI_WALL_TIMEOUT_S=240
export STUDIO_PLAYWRIGHT_BROWSER="$browser"
export PW_ART_DIR="$artifact_dir"
if [ -n "$channel" ]; then
  export STUDIO_PLAYWRIGHT_CHANNEL="$channel"
else
  unset STUDIO_PLAYWRIGHT_CHANNEL || true
fi

python tests/studio/playwright_chat_ui.py
