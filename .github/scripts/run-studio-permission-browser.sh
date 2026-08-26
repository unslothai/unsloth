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
# Wipe (not reset-password): the boot below must re-seed a fresh .bootstrap_password.
rm -rf "$studio_home/auth"
UNSLOTH_API_ONLY=1 unsloth studio -H 127.0.0.1 -p "$port" "$@" \
  >"$server_log" 2>&1 &
studio_pid=$!

cleanup() {
  kill "$studio_pid" 2>/dev/null || true
  wait "$studio_pid" 2>/dev/null || true
}
trap cleanup EXIT

# Say who died and when, for a failure mode that currently reports nothing.
#
# Observed on windows-latest at roughly one run in twenty-five: the suite prints
# "permission-only run passed" and the STEP then ends with "Process completed with exit
# code 143". 143 is 128+SIGTERM, so something signalled this script after its work had
# already succeeded, and neither the step log nor the server log records what. The
# server log simply stops mid-request, which is what cleanup killing it looks like, so
# it cannot distinguish the two.
#
# suite_done is the fact worth capturing: it separates "signalled while driving the
# browser" (a real timeout worth chasing) from "signalled after passing" (a teardown
# ordering problem, and what the one observed instance was). Without it the next
# occurrence is as unreadable as this one.
#
# `exit` explicitly, because the whole point is to be unambiguous about the status
# rather than to rely on what bash would have chosen for a trapped signal.
suite_done=0
_on_signal() {
  name="$1"; number="$2"
  echo "[permissions] SIG${name} received at $(date -u +%H:%M:%S) after suite_done=${suite_done}" >&2
  # Best effort: what was still alive. `ps` differs across MSYS and Linux and its
  # absence must not replace the signal report with an error about ps.
  # comm, not args: this lands in a public CI log, and a command line can carry a
  # token that ::add-mask:: never saw. Process names answer "what was still alive"
  # without quoting anyone's argv.
  ps -o pid,ppid,comm 2>/dev/null | tail -20 >&2 || true
  exit $((128 + number))
}
trap '_on_signal TERM 15' TERM
trap '_on_signal INT 2' INT
trap '_on_signal HUP 1' HUP

healthy=0
# --max-time, or only the loop counter is bounded and a server that binds the
# port then wedges parks the first iteration forever. And a real deadline
# rather than an iteration count, because once a probe can cost --max-time,
# 180 iterations is up to 18 minutes rather than the 180s it reads as. See
# wait-for-health.sh, which had both halves of the same hole.
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
export STUDIO_UI_WALL_TIMEOUT_S=240
export STUDIO_PLAYWRIGHT_BROWSER="$browser"
export PW_ART_DIR="$artifact_dir"
if [ -n "$channel" ]; then
  export STUDIO_PLAYWRIGHT_CHANNEL="$channel"
else
  unset STUDIO_PLAYWRIGHT_CHANNEL || true
fi

python tests/studio/playwright_chat_ui.py
# Set only after the suite returns 0 (set -e means a failure never reaches here), so a
# signal arriving during teardown is distinguishable from one that interrupted the run.
suite_done=1
