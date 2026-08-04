#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards how .github/scripts/agent-guides-drive.sh classifies an agent invoke
# that hits its `timeout` cap.
#
# History: every timeout was reported as class-(c) guide drift, "headless-TTY
# hang -- the recipe likely needs a non-interactive/print flag", blaming the
# recipe in unsloth_cli/commands/start.py. On 2026-08-03 that fired on
# opencode's file-edit turn 2, and the uploaded artifact showed the opposite:
# the agent had run the tool, printed 'Hello', and then sat idle for the
# remaining 18 minutes with llama-server serving nothing -- the same two turns
# it had completed in 635s a week earlier. The recipe worked; the CLI did not
# exit.
#
# The contract now: a timeout that printed nothing is still guide drift and
# still fatal, a timeout that printed a transcript warns and defers to the
# caller's own assertions, and neither disposition touches the non-timeout
# exit paths.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVE_SH="$SCRIPT_DIR/../../.github/scripts/agent-guides-drive.sh"
PASS=0
FAIL=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Extract run_timed() alone: the rest of the script needs a served model, an
# installed agent CLI and a real `unsloth start`, none of which belong in a
# unit test.
sed -n '/^run_timed() {/,/^}/p' "$DRIVE_SH" > "$WORK/run_timed.sh"
if [ ! -s "$WORK/run_timed.sh" ]; then
    echo "  FAIL: could not extract run_timed() from $DRIVE_SH"
    exit 1
fi

assert_eq() {
    _label="$1"; _got="$2"; _want="$3"
    if [ "$_got" = "$_want" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (got '$_got', want '$_want')"
        FAIL=$((FAIL + 1))
    fi
}

# Run run_timed() against a stand-in command with guide_fail/redact stubbed.
# Echoes the captured stdout plus RC= and TIMED_OUT= trailers.
run_case() {  # $1 = TIMEOUT, rest = command
    _t="$1"; shift
    cat > "$WORK/case.sh" <<EOF
set -uo pipefail
AGENT=testagent
TIMEOUT=$_t
redact() { :; }
guide_fail() { echo "GUIDE_FAIL: \$*"; exit 9; }
$(cat "$WORK/run_timed.sh")
run_timed "$WORK/out.txt" "\$@"
echo "RC=\$?"
echo "TIMED_OUT=\${TIMED_OUT:-unset}"
EOF
    bash "$WORK/case.sh" "$@" 2>&1 || true
}

count() { echo "$1" | grep -c -- "$2" || true; }
field() { echo "$1" | sed -n "s/^$2=//p"; }

echo "1. timeout with no output at all is still guide drift, and still fatal"
OUT="$(run_case 1 sleep 5)"
assert_eq "guide_fail fired"            "$(count "$OUT" 'GUIDE_FAIL')" 1
assert_eq "still names the TTY hang"    "$(count "$OUT" 'headless-TTY hang')" 1
assert_eq "exited before returning"     "$(count "$OUT" '^RC=')" 0

echo "2. timeout after printing a transcript defers to the caller"
OUT="$(run_case 1 bash -c 'echo did the work; sleep 5')"
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "warns instead"               "$(count "$OUT" '::warning::')" 1
assert_eq "does not blame the recipe"   "$(count "$OUT" 'headless-TTY hang')" 0
assert_eq "rc is still the timeout"     "$(field "$OUT" RC)" 124
assert_eq "TIMED_OUT set for callers"   "$(field "$OUT" TIMED_OUT)" 1

echo "3. a command that ignores SIGTERM is killed, then classified the same way"
OUT="$(run_case 1 bash -c 'trap "" TERM; echo still here; sleep 40')"
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "TIMED_OUT set"               "$(field "$OUT" TIMED_OUT)" 1

echo "4. a clean run is untouched"
OUT="$(run_case 5 bash -c 'echo hi')"
assert_eq "rc 0"                        "$(field "$OUT" RC)" 0
assert_eq "TIMED_OUT cleared"           "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no warning"                  "$(count "$OUT" '::warning::')" 0

echo "5. a non-timeout failure is untouched and stays the caller's call"
OUT="$(run_case 5 bash -c 'echo boom; exit 3')"
assert_eq "rc preserved"                "$(field "$OUT" RC)" 3
assert_eq "TIMED_OUT cleared"           "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0

echo "6. the callers that can rescue a soft timeout check TIMED_OUT, resume does not"
# connection + both file-edit turns judge the turn on their own assertions;
# resume computes RESULT from a session-store delta, which a partial turn would
# corrupt, so it must stay fatal.
RESCUERS="$(grep -c 'TIMED_OUT:-0}" = 1 \] \\$' "$DRIVE_SH" || true)"
assert_eq "connection + 2 file-edit turns exempt" "$RESCUERS" 3
assert_eq "resume keeps a hang fatal" \
    "$(grep -c 'a resume pass cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
