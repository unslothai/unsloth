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
# Four call sites deliberately do NOT get the waiver. `connection` has no
# assertion that can tell a completed reply from a startup banner (assert_reply
# checks for non-empty text without the connection/auth error strings, not for
# the requested "pong"), `resume` reads a session-store delta that a partial turn
# would corrupt, `attribution-ab` judges by a llama-server log slice, and
# file-edit turn 2 has only its transcript, which cannot separate the program's
# stdout from a narration of it. Turn 1 is the only site that qualifies, because
# the harness re-runs hello.py itself.
#
# Expiry is read off the wall clock, not the exit status: --kill-after bounds a
# TERM-resistant CLI but makes it exit 137, which is also what an unrelated
# SIGKILL (the OOM killer) produces, so only a 137 at or after the deadline
# counts.
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
TURN_DONE_RE='${TURN_DONE_RE:-}'
EXIT_GRACE=${EXIT_GRACE:-30}
redact() { :; }
guide_fail() { echo "GUIDE_FAIL: \$*"; exit 9; }
$(cat "$WORK/${RT:-run_timed}.sh")
run_timed "$WORK/out.txt" "\$@"
echo "RC=\$?"
echo "TIMED_OUT=\${TIMED_OUT:-unset}"
echo "TURN_DONE=\${TURN_DONE:-unset}"
EOF
    bash "$WORK/case.sh" "$@" 2>&1 || true
}

# Like run_case but with a hard outer bound, for a command that only terminates
# if run_timed's own kill fallback works.
run_case_bounded() {  # $1 = outer bound, $2 = TIMEOUT, rest = command
    _outer="$1"; shift
    _t="$1"; shift
    cat > "$WORK/case.sh" <<EOF
set -uo pipefail
AGENT=testagent
TIMEOUT=$_t
TURN_DONE_RE='${TURN_DONE_RE:-}'
EXIT_GRACE=${EXIT_GRACE:-30}
redact() { :; }
guide_fail() { echo "GUIDE_FAIL: \$*"; exit 9; }
$(cat "$WORK/${RT:-run_timed}.sh")
run_timed "$WORK/out.txt" "\$@"
echo "RC=\$?"
echo "TIMED_OUT=\${TIMED_OUT:-unset}"
echo "TURN_DONE=\${TURN_DONE:-unset}"
EOF
    timeout --kill-after=5 "$_outer" bash "$WORK/case.sh" "$@" 2>&1 || true
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
assert_eq "the kill was well inside the cap" "$(field "$OUT" RC)" 137

echo "3c. a kill-after that fires AT the deadline is an expiry"
# Same 137, opposite verdict from 3b: here the wall clock says the cap elapsed.
# Run against a copy with a 2s kill-after so this takes seconds, not 31.
sed 's/--kill-after=30/--kill-after=2/' "$WORK/run_timed.sh" > "$WORK/run_timed_fast.sh"
# Bounded from the outside: if the kill fallback is ever removed, run_timed has
# nothing to stop a TERM-ignoring loop, and this case would hang the suite
# instead of failing it.
OUT="$(RT=run_timed_fast run_case_bounded 25 1 bash -c 'trap "" TERM; echo working; while true; do sleep 1; done')"
assert_eq "rc 137"                      "$(field "$OUT" RC)" 137
assert_eq "counted as a timeout"        "$(field "$OUT" TIMED_OUT)" 1
assert_eq "warned, not guide drift"     "$(count "$OUT" '::warning::')" 1

echo "3d. a CLI that exits 124 on its own is not an expiry"
# timeout(1) otherwise returns "the exit status of COMMAND", so 124 can come
# from the agent's own internal request timeout. Landing far short of the cap
# means it is a real failure, and must not reach the file-edit waiver.
OUT="$(run_case 30 bash -c 'echo partial work; exit 124')"
assert_eq "rc 124 preserved"            "$(field "$OUT" RC)" 124
assert_eq "not treated as a timeout"    "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no warning"                  "$(count "$OUT" '::warning::')" 0
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0

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

echo "6. only file-edit turn 1 rescues a soft timeout"
# Turn 1 is judged on a side effect the harness verifies itself (it runs
# hello.py and compares the output). Everything else -- turn 2, connection,
# resume, attribution-ab -- has only text or a partial store to go on, so a cap
# stays fatal there.
# This heading says "only turn 1 RESCUES", so count rescues, not mentions. The
# previous form counted every consultation of TIMED_OUT and pinned the total at
# 3, which conflated the one waiver with the fatal checks beside it -- so adding
# a fourth site that makes a cap MORE explicitly fatal failed a guard named for
# waivers. Splitting them enforces the sentence above instead of a magic number,
# and keeps the real power: a new waiver anywhere still fails.
#
# The waiver is the `||` form, and it is the only shape that lets execution
# continue past a cap. Read as a whole statement, not per line: an escape
# rewritten onto one line must still be counted, or this guard checks nothing.
WAIVERS="$(grep -c '|| \[ "${TIMED_OUT:-0}" = 1 \]' "$DRIVE_SH" || true)"
assert_eq "exactly one TIMED_OUT waiver (file-edit turn 1)" "$WAIVERS" 1

# The rest must consult it only to STOP: resume, attribution-ab and connection.
# Counted separately so a waiver can never masquerade as one of them.
TOTAL_SITES="$(grep -c 'TIMED_OUT:-0}" = 1 \]' "$DRIVE_SH" || true)"
assert_eq "every other TIMED_OUT site is a fatal check" "$((TOTAL_SITES - WAIVERS))" 3
assert_eq "resume keeps a hang fatal" \
    "$(grep -c 'a resume pass cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "attribution-ab keeps a hang fatal" \
    "$(grep -c 'the A/B cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "and all four of its invokes go through that guard" \
    "$(grep -c 'ab_invoke "\$LOGS_DIR/claude-ab-' "$DRIVE_SH" || true)" 4

# The connection guard still refuses a bare cap. What changed on 2026-09-06 is
# that it no longer has to: openclaw answered `pong` and logged
# `ended with stopReason=stop`, then held its session write lock for the rest of
# the 1200s cap, and the job reported "never completed a turn" over a transcript
# that showed the turn completing. The old reasoning stands -- assert_reply
# cannot tell a completed reply from a startup banner -- so the fix is not to
# waive the cap but to give connection the assertion it was missing: a line the
# agent prints only when a run ends. A banner carries no such line and still
# fails. Read the whole statement, not one line of it.
CONN_LINE="$(grep -n 'documented launch command exited non-zero' "$DRIVE_SH" | cut -d: -f1)"
CONN_STMT="$(sed -n "$((CONN_LINE - 2)),${CONN_LINE}p" "$DRIVE_SH")"
assert_eq "connection guard has no TIMED_OUT escape" \
    "$(echo "$CONN_STMT" | grep -c 'TIMED_OUT' || true)" 0
assert_eq "connection guard still fails on a bare rc" \
    "$(echo "$CONN_STMT" | grep -c '\[ "\$rc" -eq 0 \] || guide_fail' || true)" 1
assert_eq "a cap with no end-of-run marker is still fatal for connection" \
    "$(grep -c 'TIMED_OUT:-0}" = 1 \] && \[ "${TURN_DONE:-0}" != 1 \]' "$DRIVE_SH" || true)" 1
# TURN_DONE is only ever set where the marker was actually seen, so the guard
# above cannot be satisfied by a hang that printed nothing but a banner.
assert_eq "TURN_DONE is set only behind a marker match" \
    "$(grep -c 'TURN_DONE=1' "$WORK/run_timed.sh")" 2
assert_eq "and both sites grep the transcript for it" \
    "$(grep -c 'grep -qF -- "$TURN_DONE_RE"' "$WORK/run_timed.sh")" 2
# Only openclaw opts in today, and only to its own end-of-run line.
assert_eq "openclaw declares the marker" \
    "$(grep -c "TURN_DONE_RE='ended with stopReason='" "$DRIVE_SH" || true)" 1

echo "7. file-edit turn 2 keeps a cap fatal, and asks one thing"
# The two-part T2 that would have made a waived turn 2 verifiable degraded the
# agents: opencode narrated the tool call instead of running it and created no
# file, having executed the one-part prompt for real on every prior run.
assert_eq "T2 is a single instruction" \
    "$(grep -c "T2='Run hello.py with python and show me the exact output.'" "$DRIVE_SH" || true)" 1
# Only the comment explaining why it was dropped may mention it; no live line
# may ask for it or read it.
assert_eq "no ran.txt artifact in live code" \
    "$(grep -v '^[[:space:]]*#' "$DRIVE_SH" | grep -c 'ran.txt' || true)" 0
TURN2_LINE="$(grep -n 'turn 2 (run hello.py) exited non-zero' "$DRIVE_SH" | cut -d: -f1)"
TURN2_STMT="$(sed -n "$((TURN2_LINE - 2)),${TURN2_LINE}p" "$DRIVE_SH")"
assert_eq "turn 2 has no TIMED_OUT escape" \
    "$(echo "$TURN2_STMT" | grep -c 'TIMED_OUT' || true)" 0
assert_eq "turn 2 still fails on a bare rc" \
    "$(echo "$TURN2_STMT" | grep -c '\[ "\$rc" -eq 0 \] \\' || true)" 1

echo "8. the cap keeps a finite kill fallback, but expiry comes from the clock"
# Cases 3b/3c/3d prove the behaviour; these pin the shape, so a refactor cannot
# quietly go back to trusting an exit status.
# Both invocations carry it -- the plain blocking one and the watched one. A
# marker-watching run that dropped the fallback would leave a TERM-resistant CLI
# unbounded exactly where the watcher is meant to bound it.
assert_eq "kill-after restored on both call sites" \
    "$(grep -c 'kill-after=30' "$WORK/run_timed.sh")" 2
assert_eq "neither status alone decides" \
    "$(grep -c 'rc" -eq 124 \] || \[ "$rc" -eq 137 \]' "$WORK/run_timed.sh")" 1
assert_eq "the clock decides, with 1s of slack for truncation" \
    "$(grep -c 'elapsed" -ge \$(( TIMEOUT - 1 ))' "$WORK/run_timed.sh")" 1
assert_eq "a suffixed cap falls back to 124 alone" \
    "$(grep -c '\*\[!0-9\]\*) \[ "$rc" -eq 124 \] && expired=1' "$WORK/run_timed.sh")" 1

echo "9. an agent that finishes its run and then will not exit is released early"
# The openclaw case: the marker lands, the CLI keeps running, and without this
# the job burns the whole cap and then calls a completed turn a hang. The stand-in
# ignores TERM so the escalation to KILL is exercised too. Cap 60 with a 2s grace:
# a pass has to come back in seconds, so a regression here shows up as a slow
# test, not a green one.
OUT="$(TURN_DONE_RE='ended with stopReason=' EXIT_GRACE=2 \
    run_case_bounded 45 60 bash -c 'trap "" TERM; echo pong; echo run 1 ended with stopReason=stop; while true; do sleep 1; done')"
assert_eq "TURN_DONE set"                "$(field "$OUT" TURN_DONE)" 1
assert_eq "not reported as a cap"        "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no guide_fail"                "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "says the run ended but the CLI would not exit" \
    "$(count "$OUT" 'would not exit')" 1
assert_eq "the transcript survived the kill" \
    "$(grep -c '^pong$' "$WORK/out.txt" || true)" 1

echo "9f. the agent under the wrapper is killed too, not orphaned"
# invoke_via_connect runs the CLI from a generated bash script, so timeout(1)'s
# direct child is that wrapper and the agent is a grandchild. Signalling the
# wrapper alone would leave the CLI running for the rest of the job with the
# transcript's fd still open. The stand-in records the grandchild's pid and
# ignores TERM, so a surviving process is visible after run_timed returns.
rm -f "$WORK/kid.pid"
# Two statements, so bash cannot exec-optimize the wrapper away and the agent is
# a real grandchild -- the shape invoke_via_connect produces. A one-liner wrapper
# collapses into a single process and the case silently stops testing anything.
cat > "$WORK/agent.sh" <<AGENT
trap "" TERM
echo \$\$ > "$WORK/kid.pid"
echo pong
echo run 1 ended with stopReason=stop
while true; do sleep 1; done
AGENT
cat > "$WORK/wrapper.sh" <<WRAP
export STANDIN=1
bash "$WORK/agent.sh"
WRAP
OUT="$(TURN_DONE_RE='ended with stopReason=' EXIT_GRACE=2 run_case_bounded 60 90 \
    bash "$WORK/wrapper.sh")"
assert_eq "TURN_DONE set"                "$(field "$OUT" TURN_DONE)" 1
KID="$(cat "$WORK/kid.pid" 2>/dev/null || echo 0)"
assert_eq "the grandchild recorded its pid" "$([ "$KID" -gt 0 ] && echo yes || echo no)" yes
sleep 1
assert_eq "and no descendant survived"   "$(kill -0 "$KID" 2>/dev/null && echo alive || echo gone)" gone

echo "9b. a banner-then-hang carries no marker and stays a cap"
# The failure the connection guard exists to catch. Same watcher, same grace:
# the only difference is that nothing ever printed the end-of-run line.
OUT="$(TURN_DONE_RE='ended with stopReason=' EXIT_GRACE=2 \
    run_case_bounded 45 3 bash -c 'echo Welcome to the agent; sleep 30')"
assert_eq "TURN_DONE stays clear"        "$(field "$OUT" TURN_DONE)" 0
assert_eq "still a cap"                  "$(field "$OUT" TIMED_OUT)" 1
assert_eq "no early-release warning"     "$(count "$OUT" 'would not exit')" 0

echo "9c. a marker that lands inside the last poll interval still counts"
# The watcher samples; a run that ends just before the cap can expire before the
# next look. Reading the transcript after the fact keeps the two paths agreeing.
OUT="$(TURN_DONE_RE='ended with stopReason=' EXIT_GRACE=600 \
    run_case_bounded 45 3 bash -c 'echo pong; echo run 1 ended with stopReason=stop; sleep 30')"
assert_eq "cap was hit"                  "$(field "$OUT" TIMED_OUT)" 1
assert_eq "and the finished turn is still recognized" "$(field "$OUT" TURN_DONE)" 1

echo "9d. declaring a marker does not change a CLI that exits on its own"
OUT="$(TURN_DONE_RE='ended with stopReason=' run_case 10 bash -c 'echo pong; echo run 1 ended with stopReason=stop')"
assert_eq "rc 0"                         "$(field "$OUT" RC)" 0
assert_eq "TIMED_OUT cleared"            "$(field "$OUT" TIMED_OUT)" 0
assert_eq "TURN_DONE cleared"            "$(field "$OUT" TURN_DONE)" 0
assert_eq "no warning"                   "$(count "$OUT" '::warning::')" 0

echo "9e. with no marker declared, every path is byte-for-byte the old one"
OUT="$(run_case 1 bash -c 'echo did the work; sleep 5')"
assert_eq "still a cap"                  "$(field "$OUT" TIMED_OUT)" 1
assert_eq "TURN_DONE never set"          "$(field "$OUT" TURN_DONE)" 0
assert_eq "rc is still the timeout"      "$(field "$OUT" RC)" 124

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
