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
#
# Two call sites deliberately do NOT get the waiver. `connection` has no
# assertion that can tell a completed reply from a startup banner (assert_reply
# checks for non-empty text without the connection/auth error strings, not for
# the requested "pong"), and `resume` reads a session-store delta that a partial
# turn would corrupt. And only exit 124 counts as an expiry: 137 is what an
# external SIGKILL produces too, so it must never waive a crash.
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

echo "3. a command that ignores SIGTERM still reports the expiry as 124"
OUT="$(run_case 1 bash -c 'trap "" TERM; echo still here; sleep 4')"
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "rc is the timeout, not 137"  "$(field "$OUT" RC)" 124
assert_eq "TIMED_OUT set"               "$(field "$OUT" TIMED_OUT)" 1

echo "3b. an external SIGKILL is a crash, never an expiry"
# 137 is 128+9 whether --kill-after fired or the OOM killer struck, so it must
# not reach the waiver -- otherwise a killed run whose side effects happen to
# look right would pass.
OUT="$(run_case 30 bash -c 'echo partial work; kill -9 $$')"
assert_eq "rc 137 preserved"            "$(field "$OUT" RC)" 137
assert_eq "not treated as a timeout"    "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no warning"                  "$(count "$OUT" '::warning::')" 0
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "no --kill-after in run_timed" "$(grep -c 'kill-after' "$WORK/run_timed.sh")" 0

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

echo "6. only the file-edit turns rescue a soft timeout"
# Both file-edit turns judge the turn on real side effects (hello.py exists,
# contains Hello, and the run output contains Hello). connection and resume have
# no such assertion, so a cap stays fatal for them.
# Count every waiver regardless of how the statement is wrapped: an escape
# rewritten onto one line must still be counted, or this guard checks nothing.
RESCUERS="$(grep -c 'TIMED_OUT:-0}" = 1 \]' "$DRIVE_SH" || true)"
# 2 file-edit turns + the turn-2 bare-line check + resume + attribution-ab.
assert_eq "exactly the expected TIMED_OUT sites" "$RESCUERS" 5
assert_eq "resume keeps a hang fatal" \
    "$(grep -c 'a resume pass cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "attribution-ab keeps a hang fatal" \
    "$(grep -c 'the A/B cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "and all four of its invokes go through that guard" \
    "$(grep -c 'ab_invoke "\$LOGS_DIR/claude-ab-' "$DRIVE_SH" || true)" 4

# The connection guard must stay a bare rc test: assert_reply cannot tell a
# completed reply from a startup banner, so it cannot adjudicate a cap. Read the
# whole statement, not one line of it -- a one-line rewrite is the easy mistake.
CONN_LINE="$(grep -n 'documented launch command exited non-zero' "$DRIVE_SH" | cut -d: -f1)"
CONN_STMT="$(sed -n "$((CONN_LINE - 2)),${CONN_LINE}p" "$DRIVE_SH")"
assert_eq "connection guard has no TIMED_OUT escape" \
    "$(echo "$CONN_STMT" | grep -c 'TIMED_OUT' || true)" 0
assert_eq "connection guard still fails on a bare rc" \
    "$(echo "$CONN_STMT" | grep -c '\[ "\$rc" -eq 0 \] || guide_fail' || true)" 1

echo "7. a waived file-edit turn 2 is not satisfied by hello.py's own source"
# `print("Hello")` contains the string, so an agent that read the file back and
# then blocked would pass a bare `grep -q Hello`. The waived path must demand
# the bare line that actually running it produces.
printf '$ cat hello.py\nprint("Hello")\n(waiting for approval)\n' > "$WORK/read_back.txt"
printf '\033[0m$ python hello.py\nHello\n\033[0m\n' > "$WORK/really_ran.txt"
bare_line() {  # mirrors the check in the script, CSI-stripped
    _esc=$(printf '\033')
    sed -E "s/${_esc}\[[0-9;]*[a-zA-Z]//g" "$1" | grep -cE '^[[:space:]]*Hello[[:space:]]*$' || true
}
assert_eq "reading the file back does not count" "$(bare_line "$WORK/read_back.txt")" 0
assert_eq "actually running it does"             "$(bare_line "$WORK/really_ran.txt")" 1
assert_eq "and the script applies that check"    \
    "$(grep -c "grep -qE '\^\[\[:space:\]\]\*Hello\[\[:space:\]\]\*\\\$'" "$DRIVE_SH" || true)" 1

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
