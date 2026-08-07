#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Run install.sh and SIGTERM it partway through, reproducing a user quitting the desktop
# app mid-install: main.rs cleanup_child_processes() -> install::stop_install() kills the
# installer PROCESS GROUP (install.rs:798-807). Group, not leader: the `uv`/`python`
# children would otherwise finish the dep pass and the leg would prove nothing.
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
# hardcodes ~/.unsloth/studio, CI overrides UNSLOTH_STUDIO_HOME. Never cleared, by design.
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

# A leg can be aimed at either kind of phase, and only one of them is a line. install.sh
# prints "[TAURI:STEP] <name>" lines, while the dependency pass rewrites ONE physical line
# with \r (install_python_stack.py:2499), so its ten sub-steps are CR-separated SEGMENTS.
# Splitting on \r is what makes a sub-step's END observable: without it the "studio deps"
# leg of staging run 30419729244 killed at "7/10 data designer deps" with backend_ok=true,
# having installed the structlog it exists to remove.
SUB_RE='\[[=-]+\][[:space:]]*[0-9]+/[0-9]+[[:space:]]'
phase_lines() { tr '\r' '\n' < "$LOG" 2>/dev/null || true; }

# True when the phase the marker named is no longer the running one. A sub-step marker is
# judged against the running sub-step, a step marker against the running step -- a step is
# not "over" because the sub-steps beneath it advanced; a marker naming neither is not
# judged. Results go through variables, never `| grep -q`, which can report SIGPIPE through
# pipefail on a long log.
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
# Fifth-of-a-second slices: every phase label prints BEFORE its work, so the poll delay is
# the whole distance between the label and the signal.
for i in $(seq 1 $(( KILL_AT_SECONDS * 5 ))); do
  if ! kill -0 "$PID" 2>/dev/null; then
    reason="exited-before-marker"
    break
  fi
  if [ -n "$MARKER" ] && grep -qE "$MARKER" "$LOG" 2>/dev/null; then
    # Signal at detection, never after a delay: every label prints BEFORE its work, so the
    # kill is inside the phase the moment the line appears, and any wait is a bet on how
    # long that phase runs. The bet lost twice -- a flat 3s wait put 5 of the 12 legs of
    # staging run 30419729244 into a LATER phase, and in 30426111484 it carried the macOS
    # torch leg from "Installing PyTorch" into "Installing Unsloth", a step the workflow
    # called minutes long that finished in under three seconds.
    #
    # Between the grep and the signal the installer can still exit on its own, which would
    # record marker-hit over an install that interrupted nothing.
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
  # Unconditional, and to the GROUP: the leader exits on SIGTERM while a uv or python
  # descendant does not, and gating on `kill -0 "$PID"` left that child finishing the dep
  # pass under the probe. Signalling an empty group is a no-op.
  echo "[interrupt] SIGKILL to process group -$PID"
  kill -KILL -- -"$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null || true
fi

wait "$PID" 2>/dev/null
rc=$?

# Only after the reap: an unreaped leader is still a member of its own group, so this poll
# would report it alive forever. No installer may still run when the probe starts.
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
# Where the signal actually landed. A phase that ended before the poll saw the marker sends
# the kill into a LATER phase, so the leg duplicates whichever leg owns that phase while its
# own label claims otherwise.
_last_phase="$(phase_lines | grep -aE "^\[TAURI:STEP\]|$SUB_RE" | tail -1)" || true
echo "[interrupt] phase at kill: $_last_phase"
mismatch=false
if marked_phase_over; then
  mismatch=true
  echo "::warning::killed in '$_last_phase', not the marked phase -- that phase was already over"
fi
# Only simple values: the workflow sources this file, so the phase text stays out of it.
{
  echo "interrupt_reason=$reason"
  echo "interrupt_killed=$killed"
  echo "installer_exit=$rc"
  echo "interrupt_phase_mismatch=$mismatch"
} > "$(dirname "$LOG")/interrupt.env"
exit 0
