#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Run install.sh and SIGTERM it partway through, reproducing a user quitting the desktop
# app mid-install: main.rs cleanup_child_processes() -> install::stop_install() -> kill
# the installer PROCESS GROUP (install.rs:798-807). It must be the group: killing only
# the leader leaves `uv`/`python` children to finish the dep pass, proving nothing.
#
# Usage: bash .github/scripts/interrupt-install.sh "<marker>" "<logfile>" [-- install args]
#   <marker>  regex to wait for in the install log before killing, e.g. "studio deps"
#             or "\[TAURI:STEP\] Installing PyTorch". Use "" to kill after --at-seconds.
# Env:
#   KILL_AT_SECONDS  hard deadline; kill even if the marker never appears (default 900)
#   KILL_GRACE       seconds to wait for the group to die before SIGKILL (default 10)
set -uo pipefail

MARKER="${1:-}"
LOG="${2:-logs/install.log}"
shift 2 || true
[ "${1:-}" = "--" ] && shift
KILL_AT_SECONDS="${KILL_AT_SECONDS:-900}"
KILL_GRACE="${KILL_GRACE:-10}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

# Stand in for the desktop app, which writes this before spawning the installer
# (install.rs). We kill the installer directly, so without it the marker #7490 relies on
# is absent for a reason unrelated to #7490. Both locations because the Rust side
# hardcodes ~/.unsloth/studio while CI overrides UNSLOTH_STUDIO_HOME. Never cleared:
# being killed is the whole point.
for _marker_dir in "${UNSLOTH_STUDIO_HOME:-}" "$HOME/.unsloth/studio"; do
  [ -n "$_marker_dir" ] || continue
  mkdir -p "$_marker_dir" 2>/dev/null || continue
  : > "$_marker_dir/.desktop-install-in-progress" 2>/dev/null || true
done

# Job control puts the child in its own process group, so $! is the pgid leader and
# `kill -- -$!` reaches every descendant, matching the Rust side.
set -m
bash install.sh "$@" > "$LOG" 2>&1 &
PID=$!
set +m
echo "[interrupt] installer pid/pgid=$PID marker='${MARKER}' deadline=${KILL_AT_SECONDS}s"

killed=false
reason=""
# Half-second slices, not one-second: a step that lasts under a second is over by the time
# a 1s poll notices its line, and the beat below then cannot help.
for i in $(seq 1 $(( KILL_AT_SECONDS * 2 ))); do
  if ! kill -0 "$PID" 2>/dev/null; then
    reason="exited-before-marker"
    break
  fi
  if [ -n "$MARKER" ] && grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    # A beat into the step, so the kill lands mid-work rather than on the boundary
    # before the step has touched the venv. Waited in slices and cut short the moment a
    # later [TAURI:STEP] line appears: creating the venv takes ~0.1s, so a flat 3s sleep
    # sent the venv leg's signal into "Installing PyTorch" and made it a duplicate of the
    # torch leg. Sub-step markers ("studio deps") print no step line of their own, so
    # they still get the whole beat.
    _steps_at_marker="$(grep -cE '^\[TAURI:STEP\]' "$LOG" 2>/dev/null || true)"
    for _ in $(seq 1 $(( ${KILL_AFTER_MARKER_SECONDS:-3} * 5 ))); do
      sleep 0.2
      kill -0 "$PID" 2>/dev/null || break
      [ "$(grep -cE '^\[TAURI:STEP\]' "$LOG" 2>/dev/null || true)" = "$_steps_at_marker" ] || break
    done
    # ...but a cached step can FINISH inside that beat. Recording marker-hit before the
    # sleep handed the landing assertion a COMPLETED install: the signal reached no
    # process and the leg passed green having interrupted nothing.
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
  # Unconditional, and to the GROUP. The leader can exit on SIGTERM while a uv or python
  # descendant does not; gating on `kill -0 "$PID"` skipped this escalation and left that
  # descendant free to finish the dependency pass while the probe ran. Signalling an
  # already-empty group is a no-op.
  echo "[interrupt] SIGKILL to process group -$PID"
  kill -KILL -- -"$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
fi

wait "$PID" 2>/dev/null
rc=$?

# Only after the reap: an unreaped leader is still a member of its own group, so polling
# the group before `wait` would report it alive forever. The probe must not start while
# an installer process is still running.
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

# Report how far it got, so a leg that never reached the target step is visible rather
# than passing for the wrong reason.
if [ -n "$MARKER" ] && ! grep -qE "$MARKER" "$LOG" 2>/dev/null; then
  echo "::warning::marker '$MARKER' never appeared -- this leg killed at the deadline, not at the intended step"
fi
# Which step the signal actually landed in. A sub-second step (creating the venv takes
# ~0.1s) can be over before any log poll notices its line, and the leg then silently kills
# the NEXT step while its matrix label still claims the marked one. Only steps say where
# they landed: sub-step markers ("studio deps") print no [TAURI:STEP] line of their own, so
# the test skips them rather than warning on every leg.
_last_step="$(grep -E '^\[TAURI:STEP\]' "$LOG" 2>/dev/null | tail -1)"
echo "[interrupt] step at kill: $_last_step"
if [ -n "$MARKER" ] &&
   grep -E '^\[TAURI:STEP\]' "$LOG" 2>/dev/null | grep -qE "$MARKER" &&
   ! printf '%s\n' "$_last_step" | grep -qE "$MARKER"; then
  echo "::warning::killed in '$_last_step', not the marked step -- that step was already over"
fi
{
  echo "interrupt_reason=$reason"
  echo "interrupt_killed=$killed"
  echo "installer_exit=$rc"
} > "$(dirname "$LOG")/interrupt.env"
exit 0
