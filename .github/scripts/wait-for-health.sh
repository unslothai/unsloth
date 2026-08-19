#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Poll a booted Studio's /api/health until it reports healthy, and on timeout
# print the tail of that server's log before failing.
#
# Usage:
#   wait-for-health.sh --port 18888 [--log logs/studio.log] [--tmp /tmp/health.json]
#
# Fourteen steps across eight workflows ran this same poll, in two dialects that
# disagreed about what a failure looks like. This is the union of the two, taking
# the better half of each:
#
#  * Retry on ANY not-yet-healthy answer, not just on a refused connection. The
#    "exit-0" dialect ran `jq -e` as a bare command under `bash -e`, so a server
#    that answered the very first poll with `status != "healthy"` failed the step
#    on the spot instead of giving the model loader the remaining seconds. The
#    other eleven copies already retried; now all fourteen do.
#  * Tail the log when the deadline passes. Eleven copies ended on a bare
#    `jq -e '.status == "healthy"'`, whose entire output on failure is a non-zero
#    exit code -- no message, and nothing about why the server never came up. The
#    log tail is the only place that says.
#
# Parameters, and why each one is a parameter:
#
#  * --port  differs per call site; several workflows run two or three servers in
#            one job on different ports.
#  * --log   is whichever log the matching boot-studio-api-only.sh call was told
#            to write. Five distinct names are in use, and tailing the wrong one
#            on failure is worse than tailing none.
#  * --tmp   is where the last poll's response body is left for later steps and
#            for a human reading the runner. Five distinct names are in use so
#            that a later phase in the same job does not overwrite the evidence
#            an earlier phase left behind.
#
# The deadline is fixed at 180s because all fourteen converted call sites used
# 180. The three "boot briefly to confirm the install is still usable" steps poll
# for 60s, but they also boot and kill the server in the same shell and keep the
# pid in a local variable, so they are not call sites for this and no --timeout
# flag exists yet to serve them.

set -uo pipefail

# Every converted call site polled for 180s, and studio-ui-smoke.yml carried the
# reason: a cold runner with venv warm-up plus lazy imports has been seen to
# exceed 60s, and failing the wait costs more than waiting two more minutes.
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

for _ in $(seq 1 "$TIMEOUT_SECONDS"); do
  if curl -fs "http://127.0.0.1:${PORT}/api/health" > "$TMP" \
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
