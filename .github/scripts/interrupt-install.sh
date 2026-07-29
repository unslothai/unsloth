#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Run install.sh and SIGTERM it partway through, reproducing a user quitting the desktop
# app mid-install: main.rs cleanup_child_processes() -> install::stop_install() kills the
# installer PROCESS GROUP (install.rs:798-807). Group, not leader: `uv`/`python` children
# would otherwise finish the dep pass and the leg would prove nothing.
#
# Usage: bash .github/scripts/interrupt-install.sh "<marker>" "<logfile>" [-- install args]
#   <marker>  log regex to wait for before killing, e.g. "studio deps"; "" kills at deadline.
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

# The desktop writes this before spawning the installer (install.rs); we kill the installer
# directly, so without it #7490's marker is absent for an unrelated reason. Both roots: Rust
# hardcodes ~/.unsloth/studio, CI overrides UNSLOTH_STUDIO_HOME. Never cleared by design.
for _marker_dir in "${UNSLOTH_STUDIO_HOME:-}" "$HOME/.unsloth/studio"; do
  [ -n "$_marker_dir" ] || continue
  mkdir -p "$_marker_dir" 2>/dev/null || continue
  : > "$_marker_dir/.desktop-install-in-progress" 2>/dev/null || true
done

# Job control puts the child in its own group, so $! is the pgid and `kill -- -$!`
# reaches every descendant, as the Rust side does.
set -m
bash install.sh "$@" > "$LOG" 2>&1 &
PID=$!
set +m
echo "[interrupt] installer pid/pgid=$PID marker='${MARKER}' deadline=${KILL_AT_SECONDS}s"

# True when the marker names a [TAURI:STEP] line that is no longer the last one: the step
# ended before the poll noticed it. Sub-step markers ("studio deps") print no step line of
# their own, so they are never judged here.
marked_step_over() {
  [ -n "$MARKER" ] || return 1
  grep -E '^\[TAURI:STEP\]' "$LOG" 2>/dev/null | grep -qE "$MARKER" || return 1
  grep -E '^\[TAURI:STEP\]' "$LOG" 2>/dev/null | tail -1 | grep -qE "$MARKER" && return 1
  return 0
}

killed=false
reason=""
# Half-second slices: a sub-second step is over before a 1s poll sees its line.
for i in $(seq 1 $(( KILL_AT_SECONDS * 2 ))); do
  if ! kill -0 "$PID" 2>/dev/null; then
    reason="exited-before-marker"
    break
  fi
  if [ -n "$MARKER" ] && grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    # A beat into the step so the kill lands mid-work, cut short the moment a later
    # [TAURI:STEP] line appears: the venv takes ~0.1s, so a flat 3s sleep put the venv
    # leg's signal in "Installing PyTorch", a duplicate of the torch leg. Sub-step markers
    # ("studio deps") print no step line and keep the whole beat. Skipped when the step is
    # ALREADY over: the cut-short cannot help once the next line is logged and beating on
    # only pushes the signal deeper, so kill now and let the mismatch check below report.
    if ! marked_step_over; then
      _steps_at_marker="$(grep -cE '^\[TAURI:STEP\]' "$LOG" 2>/dev/null || true)"
      for _ in $(seq 1 $(( ${KILL_AFTER_MARKER_SECONDS:-3} * 5 ))); do
        sleep 0.2
        kill -0 "$PID" 2>/dev/null || break
        [ "$(grep -cE '^\[TAURI:STEP\]' "$LOG" 2>/dev/null || true)" = "$_steps_at_marker" ] || break
      done
    fi
    # ...but a cached step can FINISH inside the beat. Recording marker-hit before it
    # handed the landing assertion a COMPLETED install that interrupted nothing.
    if ! kill -0 "$PID" 2>/dev/null; then
      reason="exited-during-marker-delay"
      break
    fi
    reason="marker-hit"
    killed=true
    break
  fi
  sleep 0.5
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
  # Unconditional, and to the GROUP: the leader exits on SIGTERM while a uv or python
  # descendant does not, and gating on `kill -0 "$PID"` left that child finishing the dep
  # pass under the probe. Signalling an empty group is a no-op.
  echo "[interrupt] SIGKILL to process group -$PID"
  kill -KILL -- -"$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
fi

wait "$PID" 2>/dev/null
rc=$?

# Only after the reap: an unreaped leader is still a member of its own group, so this
# poll would report it alive forever. No installer may still run when the probe starts.
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

# A leg that never reached the target step must be visible, not quietly green.
if [ -n "$MARKER" ] && ! grep -qE "$MARKER" "$LOG" 2>/dev/null; then
  echo "::warning::marker '$MARKER' never appeared -- this leg killed at the deadline, not at the intended step"
fi
# Where the signal actually landed. A sub-second step can end before any poll sees its
# line, and the leg then kills the NEXT step while its label claims otherwise. Sub-step
# markers print no [TAURI:STEP] line, so the test skips them instead of always warning.
_last_step="$(grep -E '^\[TAURI:STEP\]' "$LOG" 2>/dev/null | tail -1)"
echo "[interrupt] step at kill: $_last_step"
mismatch=false
if marked_step_over; then
  mismatch=true
  echo "::warning::killed in '$_last_step', not the marked step -- that step was already over"
fi
# Only simple values: the workflow sources this file, so the step text stays out of it.
{
  echo "interrupt_reason=$reason"
  echo "interrupt_killed=$killed"
  echo "installer_exit=$rc"
  echo "interrupt_step_mismatch=$mismatch"
} > "$(dirname "$LOG")/interrupt.env"
exit 0
