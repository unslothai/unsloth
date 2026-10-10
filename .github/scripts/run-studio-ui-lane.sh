#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# One lane of the Windows Chat UI Tests job: boot Unsloth on its own port and home,
# rotate the bootstrap password, drive its Playwright suites, stop the server.
# Usage:  run-studio-ui-lane.sh chat|extra
# Each lane needs its own port, UNSLOTH_STUDIO_HOME, health tmp and log paths, or they race.
# The lane home links the installed venv, and UNSLOTH_LLAMA_CPP_PATH is set explicitly because
# a custom home would otherwise point llama.cpp at $UNSLOTH_STUDIO_HOME/llama.cpp.

set -euo pipefail

LANE="${1:?usage: $0 chat|extra}"

case "$LANE" in
  chat)  PORT=18896 ;;
  extra) PORT=18897 ;;
  *) echo "run-studio-ui-lane.sh: unknown lane '$LANE'" >&2; exit 2 ;;
esac

installed_home="${UNSLOTH_INSTALLED_STUDIO_HOME:-$HOME/.unsloth/studio}"
if [ -x "$installed_home/unsloth_studio/Scripts/python.exe" ]; then
  installed_py="$installed_home/unsloth_studio/Scripts/python.exe"
elif [ -x "$installed_home/unsloth_studio/bin/python" ]; then
  installed_py="$installed_home/unsloth_studio/bin/python"
else
  echo "::error::no studio venv under $installed_home; nothing for lane $LANE to link"
  ls -la "$installed_home" 2>/dev/null || true
  exit 1
fi
echo "[lane $LANE] linking venv from $installed_py"

home="${GITHUB_WORKSPACE:-$PWD}/.studio-lane-$LANE"
mkdir -p "$home"

# A junction, not `ln -s`: MSYS copies by default and native symlinks need privileges.
if [ ! -e "$home/unsloth_studio" ]; then
  if [ "${OS:-}" = "Windows_NT" ]; then
    cmd //c mklink //J "$(cygpath -w "$home/unsloth_studio")" \
                       "$(cygpath -w "$installed_home/unsloth_studio")" >/dev/null
  else
    ln -sfn "$installed_home/unsloth_studio" "$home/unsloth_studio"
  fi
fi

export UNSLOTH_STUDIO_HOME="$home"
export UNSLOTH_LLAMA_CPP_PATH="${UNSLOTH_LLAMA_CPP_PATH:-$HOME/.unsloth/llama.cpp}"

server_log="logs/studio-lane-$LANE.log"
mkdir -p logs

boot() {
  local port="$1" log="$2"
  # `env -u GITHUB_ENV` so concurrent lanes do not both append LANE_PID to the shared file.
  env -u GITHUB_ENV bash .github/scripts/boot-studio-api-only.sh \
    --port "$port" --log "$log" --pid-var "LANE_PID" > "logs/boot-$LANE.out" 2>&1
  LANE_PID="$(sed -n 's/^LANE_PID=//p' "logs/boot-$LANE.out" | tail -1)"
  cat "logs/boot-$LANE.out"
  [ -n "$LANE_PID" ] || { echo "::error::lane $LANE: boot returned no pid"; return 1; }
  bash .github/scripts/wait-for-health.sh --port "$port" --log "$log" \
    --tmp "logs/health-$LANE.json"
}

stop() {
  [ -n "${LANE_PID:-}" ] || return 0
  kill "$LANE_PID" 2>/dev/null || true
  sleep 2
  LANE_PID=""
}
trap stop EXIT

# Read from this lane's home; the legacy path would hand over the other lane's password.
mint() {
  STUDIO_OLD_PW="$(cat "$home/auth/.bootstrap_password")"
  STUDIO_NEW_PW="CIUi-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
  STUDIO_NEW2_PW="CIUi-$(python -c 'import secrets; print(secrets.token_urlsafe(16))')"
  echo "::add-mask::$STUDIO_OLD_PW"
  echo "::add-mask::$STUDIO_NEW_PW"
  echo "::add-mask::$STUDIO_NEW2_PW"
  export STUDIO_OLD_PW STUDIO_NEW_PW STUDIO_NEW2_PW
}

export STUDIO_UI_STRICT=1
export STUDIO_UI_TURN_TIMEOUT_MS=540000

if [ "$LANE" = "chat" ]; then
  boot "$PORT" "$server_log"
  mint
  mkdir -p logs/playwright
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright \
    python tests/studio/playwright_chat_ui.py
  stop

  # Real Edge, the same engine as the desktop app's WebView2.
  bash .github/scripts/run-studio-indicator-browser.sh 18899 chromium msedge
else
  boot "$PORT" "$server_log"
  mint
  mkdir -p logs/playwright_extra
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright_extra \
    python tests/studio/playwright_extra_ui.py

  mkdir -p logs/playwright_update_banner
  BASE_URL="http://127.0.0.1:$PORT" PW_ART_DIR=logs/playwright_update_banner \
    python tests/studio/playwright_update_banner_layout.py
  stop

  bash .github/scripts/run-studio-permission-browser.sh 18895 chromium msedge
fi

echo "[lane $LANE] done"
