#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Wipe Unsloth's auth state and boot `unsloth studio` in the background, exporting
# the pid so a later step can stop it.
#
# Usage:
#   boot-studio-api-only.sh --port 18888 [--log logs/studio.log] [--pid-var STUDIO_PID]
#                           [--api-only]
#
# On the name, and on --api-only being a flag rather than the default. The
# UNSLOTH_API_ONLY=1 below reads like the switch and is not: nothing in unsloth_cli
# takes it as input. Whether the web UI is served is decided by the CLI's --api-only
# flag alone, and the backend only ever reads that variable back out (main.py, to pick
# a CORS profile) after run.py has set it from the flag. So every caller here has in
# fact been booting a server that serves the frontend, and the Playwright UI smokes
# depend on exactly that -- passing --api-only unconditionally would leave them
# driving a browser at a backend with no UI. Hence opt-in: callers with no built
# `studio/frontend/dist` (mlx-ci.yml boots on a bare pip install) must ask for it, or
# the server prints "Unsloth frontend build not found" and exits before it binds.
#
# Twenty steps across eight workflows ran this same five-line body, varying only
# in those three values. Extracted so the three easy-to-get-wrong parts have one
# definition:
#
#  * `rm -rf`, not `unsloth studio reset-password`. The boot below has to re-seed
#    a fresh `.bootstrap_password`, and it only does that when the auth directory
#    is absent. A caller that "resets" instead silently keeps the old password and
#    the test then authenticates against stale state.
#  * The pid has to reach `$GITHUB_ENV`, because the step that stops the server is
#    a different step with a different shell. `$!` alone dies with the step.
#  * stdout AND stderr go to the log. The server writes its startup diagnostics to
#    stderr, so a `>` without `2>&1` produces an empty log on exactly the failure
#    a reader needs it for.
#
# NOT for `unsloth run`: that is serve-unsloth-run.sh, which boots a different
# command with a different contract (banner API key, /v1/models resolution) and
# has nothing to share with this beyond the word "boot".
#
# Deliberately does not wait for health. The callers' waits differ -- some poll
# /api/health and stop, others go on to rotate the bootstrap password and load a
# model in the same step -- and folding the simple case in here would leave the
# rest calling a script that does half their work.

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

# Wipe rather than reset: the boot below re-seeds .bootstrap_password only when
# the directory is gone. See the header.
#
# Through $UNSLOTH_STUDIO_HOME, matching run-studio-permission-browser.sh and
# run-studio-indicator-browser.sh, which have read it since #9158. This script was
# the odd one out, hardcoding the legacy path, and that is what forced every
# Playwright step sharing this boot to run one after another: two concurrent lanes
# on one home race destructively, one wiping the .bootstrap_password the other just
# minted and is about to log in with. Unset, this is byte-for-byte the old path.
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
