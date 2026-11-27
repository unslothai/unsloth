#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Wipe Studio auth state and boot `unsloth studio` in the background, exporting the pid.
# Usage: boot-studio-api-only.sh --port 18888 [--log FILE] [--pid-var VAR] [--api-only]
# --api-only is opt-in: UNSLOTH_API_ONLY does not control UI serving, and UI smokes need the UI.
# Does not wait for health.

set -uo pipefail

PORT=""
LOG="logs/studio.log"
PID_VAR="STUDIO_PID"
API_ONLY=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)     PORT="$2"; shift 2 ;;
    --log)      LOG="$2"; shift 2 ;;
    --pid-var)  PID_VAR="$2"; shift 2 ;;
    --api-only) API_ONLY="--api-only"; shift ;;
    *) echo "boot-studio-api-only.sh: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

[ -n "$PORT" ] || { echo "boot-studio-api-only.sh: --port is required" >&2; exit 2; }

# Wipe rather than reset: the boot re-seeds .bootstrap_password only when the dir is gone.
# Honour UNSLOTH_STUDIO_HOME so concurrent lanes do not share one home.
studio_home="${UNSLOTH_STUDIO_HOME:-$HOME/.unsloth/studio}"
rm -rf "$studio_home/auth"
mkdir -p "$(dirname "$LOG")"

# shellcheck disable=SC2086  # $API_ONLY is one flag or empty, and must not become ''
UNSLOTH_API_ONLY=1 unsloth studio -H 127.0.0.1 -p "$PORT" $API_ONLY > "$LOG" 2>&1 &
SERVER_PID=$!

echo "[boot] unsloth studio ${API_ONLY:---with-frontend} on 127.0.0.1:${PORT}, pid ${SERVER_PID}, log ${LOG}"
if [ -n "${GITHUB_ENV:-}" ]; then
  echo "${PID_VAR}=${SERVER_PID}" >> "$GITHUB_ENV"
else
  echo "${PID_VAR}=${SERVER_PID}"
fi
