#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Run install.sh and SIGTERM its process GROUP partway through, like quitting the desktop app.
# Usage: bash .github/scripts/interrupt-install.sh "<marker>" "<logfile>" [-- install args]
#   <marker>  log regex to wait for before killing; "" kills at deadline.
# Env: KILL_AT_SECONDS deadline (default 900), KILL_GRACE grace before SIGKILL (default 10)
set -uo pipefail

MARKER="${1:-}"
LOG="${2:-logs/install.log}"
shift 2 || true
[ "${1:-}" = "--" ] && shift
KILL_AT_SECONDS="${KILL_AT_SECONDS:-900}"
KILL_GRACE="${KILL_GRACE:-10}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

# The desktop writes this marker before spawning the installer. Never cleared, by design.
for _marker_dir in "${UNSLOTH_STUDIO_HOME:-}" "$HOME/.unsloth/studio"; do
  [ -n "$_marker_dir" ] || continue
  mkdir -p "$_marker_dir" 2>/dev/null || continue
  : > "$_marker_dir/.desktop-install-in-progress" 2>/dev/null || true
done

# Job control makes $! the pgid, so `kill -- -$!` reaches every descendant.
set -m
bash install.sh "$@" > "$LOG" 2>&1 &
PID=$!
set +m
echo "[interrupt] installer pid/pgid=$PID marker='${MARKER}' deadline=${KILL_AT_SECONDS}s"

# The dep pass rewrites one line with \r, so sub-steps are CR-separated segments.
SUB_RE='\[[=-]+\][[:space:]]*[0-9]+/[0-9]+[[:space:]]'
phase_lines() { tr '\r' '\n' < "$LOG" 2>/dev/null || true; }

# Results go through variables, not `| grep -q`, which can report SIGPIPE through pipefail.
marked_phase_over() {
  [ -n "$MARKER" ] || return 1
  local lines steps subs last
  lines="$(phase_lines)"
  steps="$(printf '%s\n' "$lines" | grep -aE '^\[TAURI:STEP\]')" || true
  subs="$(printf '%s\n' "$lines" | grep -aE "$SUB_RE")" || true
  if [ -n "$subs" ] && [[ $subs =~ $MARKER ]]; then
    last="$(printf '%s\n' "$lines" | grep -aE "^\[TAURI:STEP\]|$SUB_RE" | tail -1)" || true
    [[ $last =~ $SUB_RE && $last =~ $MARKER ]] && return 1
    return 0
  fi
  if [ -n "$steps" ] && [[ $steps =~ $MARKER ]]; then
    last="$(printf '%s\n' "$steps" | tail -1)"
    [[ $last =~ $MARKER ]] && return 1
    return 0
  fi
  return 1
}

killed=false
reason=""
for i in $(seq 1 $(( KILL_AT_SECONDS * 5 ))); do
  if ! kill -0 "$PID" 2>/dev/null; then
    reason="exited-before-marker"
    break
  fi
  if [ -n "$MARKER" ] && grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    # Signal at detection, never after a delay: labels print before the work they name.
    if ! kill -0 "$PID" 2>/dev/null; then
      reason="exited-before-signal"
      break
    fi
    reason="marker-hit"
    killed=true
    break
  fi
  sleep 0.2
done

if [ "$killed" != "true" ] && kill -0 "$PID" 2>/dev/null; then
  reason="${reason:-deadline}"
  killed=true
fi

if [ "$killed" = "true" ]; then
  echo "[interrupt] SIGTERM to process group -$PID ($reason)"
  kill -TERM -- -"$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null || true
  for _ in $(seq 1 "$KILL_GRACE"); do
    kill -0 "$PID" 2>/dev/null || break
    sleep 1
  done
  # Unconditional, and to the group: uv/python children survive SIGTERM to the leader.
  echo "[interrupt] SIGKILL to process group -$PID"
  kill -KILL -- -"$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
fi

wait "$PID" 2>/dev/null
rc=$?

# Only after the reap: an unreaped leader keeps its group alive.
if [ "$killed" = "true" ]; then
  for _ in $(seq 1 "$KILL_GRACE"); do
    kill -0 -- -"$PID" 2>/dev/null || break
    kill -KILL -- -"$PID" 2>/dev/null || true
    sleep 1
  done
  if kill -0 -- -"$PID" 2>/dev/null; then
    echo "::warning::processes from installer group -$PID outlived SIGKILL"
  fi
fi
echo "[interrupt] installer exit=$rc reason=$reason killed=$killed"
echo "[interrupt] last log lines:"
tail -15 "$LOG" || true

if [ -n "$MARKER" ] && ! grep -qE "$MARKER" "$LOG" 2>/dev/null; then
  echo "::warning::marker '$MARKER' never appeared -- this leg killed at the deadline, not at the intended step"
fi
_last_phase="$(phase_lines | grep -aE "^\[TAURI:STEP\]|$SUB_RE" | tail -1)" || true
echo "[interrupt] phase at kill: $_last_phase"
mismatch=false
if marked_phase_over; then
  mismatch=true
  echo "::warning::killed in '$_last_phase', not the marked phase -- that phase was already over"
fi
# Only simple values: the workflow sources this file.
{
  echo "interrupt_reason=$reason"
  echo "interrupt_killed=$killed"
  echo "installer_exit=$rc"
  echo "interrupt_phase_mismatch=$mismatch"
} > "$(dirname "$LOG")/interrupt.env"
exit 0
