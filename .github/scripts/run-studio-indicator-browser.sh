#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Boot an API-only Unsloth and run the loaded-models indicator suite against it
# in one browser engine. Sibling of run-studio-permission-browser.sh, same shape
# and same bootstrap; the suite stubs the four /status endpoints with
# page.route, so it needs no model, no GPU and no llama.cpp build.

set -euo pipefail

port="${1:?usage: $0 PORT BROWSER [CHANNEL]}"
browser="${2:?usage: $0 PORT BROWSER [CHANNEL]}"
channel="${3:-}"
slug="$browser${channel:+-$channel}"
artifact_dir="logs/playwright-indicator-$slug"
server_log="logs/studio-indicator-$slug.log"
studio_home="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}"
set --
if [ -n "${STUDIO_INDICATOR_FRONTEND:-}" ]; then
  set -- -f "$STUDIO_INDICATOR_FRONTEND"
fi

mkdir -p "$artifact_dir"
# Wipe rather than reset: the boot below must mint a fresh .bootstrap_password.
rm -rf "$studio_home/auth"
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
new_password="CIInd-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "::add-mask::$old_password"
  echo "::add-mask::$new_password"
fi

export BASE_URL="http://127.0.0.1:$port"
export STUDIO_OLD_PW="$old_password"
export STUDIO_NEW_PW="$new_password"
export STUDIO_PLAYWRIGHT_BROWSER="$browser"
export PW_ART_DIR="$artifact_dir"
if [ -n "$channel" ]; then
  export STUDIO_PLAYWRIGHT_CHANNEL="$channel"
else
  unset STUDIO_PLAYWRIGHT_CHANNEL || true
fi

python tests/studio/playwright_loaded_models_indicator.py
