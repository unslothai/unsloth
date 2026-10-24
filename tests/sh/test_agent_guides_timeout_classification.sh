#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A timeout that printed nothing is fatal guide drift; one that printed a transcript warns and
# defers to the caller. Only file-edit turn 1 gets the waiver (the harness reruns hello.py).
# Expiry is read off the wall clock: 137 also comes from an unrelated SIGKILL (OOM killer).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DRIVE_SH="$SCRIPT_DIR/../../.github/scripts/agent-guides-drive.sh"
PASS=0
FAIL=0

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Extract run_timed() alone: the rest needs a served model and a real agent CLI.
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

# Hard outer bound, for a command that only ends if run_timed's kill fallback works.
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
# 137 before the deadline (e.g. OOM kill) must not reach the waiver.
OUT="$(run_case 30 bash -c 'echo partial work; kill -9 $$')"
assert_eq "rc 137 preserved"            "$(field "$OUT" RC)" 137
assert_eq "not treated as a timeout"    "$(field "$OUT" TIMED_OUT)" 0
assert_eq "no warning"                  "$(count "$OUT" '::warning::')" 0
assert_eq "no guide_fail"               "$(count "$OUT" 'GUIDE_FAIL')" 0
assert_eq "the kill was well inside the cap" "$(field "$OUT" RC)" 137

echo "3c. a kill-after that fires AT the deadline is an expiry"
# Same 137 after the cap elapsed. A 2s kill-after copy keeps this fast.
sed 's/--kill-after=30/--kill-after=2/' "$WORK/run_timed.sh" > "$WORK/run_timed_fast.sh"
# Bounded from outside so a removed kill fallback fails instead of hanging the suite.
OUT="$(RT=run_timed_fast run_case_bounded 25 1 bash -c 'trap "" TERM; echo working; while true; do sleep 1; done')"
assert_eq "rc 137"                      "$(field "$OUT" RC)" 137
assert_eq "counted as a timeout"        "$(field "$OUT" TIMED_OUT)" 1
assert_eq "warned, not guide drift"     "$(count "$OUT" '::warning::')" 1

echo "3d. a CLI that exits 124 on its own is not an expiry"
# 124 far short of the cap is the agent's own timeout, not ours, so it must stay fatal.
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
# Only turn 1 is judged on a side effect the harness verifies, so only it may waive a cap.
# Count `||` waivers on whole statements, separately from the fatal checks.
WAIVERS="$(grep -c '|| \[ "${TIMED_OUT:-0}" = 1 \]' "$DRIVE_SH" || true)"
assert_eq "exactly one TIMED_OUT waiver (file-edit turn 1)" "$WAIVERS" 1

# The other sites consult TIMED_OUT only to stop: resume, attribution-ab, connection.
TOTAL_SITES="$(grep -c 'TIMED_OUT:-0}" = 1 \]' "$DRIVE_SH" || true)"
assert_eq "every other TIMED_OUT site is a fatal check" "$((TOTAL_SITES - WAIVERS))" 3
assert_eq "resume keeps a hang fatal" \
    "$(grep -c 'a resume pass cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "attribution-ab keeps a hang fatal" \
    "$(grep -c 'the A/B cannot be judged from a partial turn' "$DRIVE_SH" || true)" 1
assert_eq "and all four of its invokes go through that guard" \
    "$(grep -c 'ab_invoke "\$LOGS_DIR/claude-ab-' "$DRIVE_SH" || true)" 4

# Connection still refuses a bare cap; it now also accepts an end-of-run marker line,
# which a startup banner never prints.
CONN_LINE="$(grep -n 'documented launch command exited non-zero' "$DRIVE_SH" | cut -d: -f1)"
CONN_STMT="$(sed -n "$((CONN_LINE - 2)),${CONN_LINE}p" "$DRIVE_SH")"
assert_eq "connection guard has no TIMED_OUT escape" \
    "$(echo "$CONN_STMT" | grep -c 'TIMED_OUT' || true)" 0
assert_eq "connection guard still fails on a bare rc" \
    "$(echo "$CONN_STMT" | grep -c '\[ "\$rc" -eq 0 \] || guide_fail' || true)" 1
assert_eq "a cap with no end-of-run marker is still fatal for connection" \
    "$(grep -c 'TIMED_OUT:-0}" = 1 \] && \[ "${TURN_DONE:-0}" != 1 \]' "$DRIVE_SH" || true)" 1
assert_eq "TURN_DONE is set only behind a marker match" \
    "$(grep -c 'TURN_DONE=1' "$WORK/run_timed.sh")" 2
assert_eq "and both sites grep the transcript for it" \
    "$(grep -c 'grep -qF -- "$TURN_DONE_RE"' "$WORK/run_timed.sh")" 2
assert_eq "openclaw declares the marker" \
    "$(grep -c "TURN_DONE_RE='ended with stopReason='" "$DRIVE_SH" || true)" 1

echo "7. file-edit turn 2 keeps a cap fatal, and asks one thing"
# A two-part T2 made agents narrate the tool call instead of running it.
assert_eq "T2 is a single instruction" \
    "$(grep -c "T2='Run hello.py with python and show me the exact output.'" "$DRIVE_SH" || true)" 1
assert_eq "no ran.txt artifact in live code" \
    "$(grep -v '^[[:space:]]*#' "$DRIVE_SH" | grep -c 'ran.txt' || true)" 0
TURN2_LINE="$(grep -n 'turn 2 (run hello.py) exited non-zero' "$DRIVE_SH" | cut -d: -f1)"
TURN2_STMT="$(sed -n "$((TURN2_LINE - 2)),${TURN2_LINE}p" "$DRIVE_SH")"
assert_eq "turn 2 has no TIMED_OUT escape" \
    "$(echo "$TURN2_STMT" | grep -c 'TIMED_OUT' || true)" 0
assert_eq "turn 2 still fails on a bare rc" \
    "$(echo "$TURN2_STMT" | grep -c '\[ "\$rc" -eq 0 \] \\' || true)" 1

echo "8. the cap keeps a finite kill fallback, but expiry comes from the clock"
# Both call sites (plain and watched) must keep --kill-after, or a TERM-resistant CLI is unbounded.
assert_eq "kill-after restored on both call sites" \
    "$(grep -c 'kill-after=30' "$WORK/run_timed.sh")" 2
assert_eq "neither status alone decides" \
    "$(grep -c 'rc" -eq 124 \] || \[ "$rc" -eq 137 \]' "$WORK/run_timed.sh")" 1
assert_eq "the clock decides, with 1s of slack for truncation" \
    "$(grep -c 'elapsed" -ge \$(( TIMEOUT - 1 ))' "$WORK/run_timed.sh")" 1
assert_eq "a suffixed cap falls back to 124 alone" \
    "$(grep -c '\*\[!0-9\]\*) \[ "$rc" -eq 124 \] && expired=1' "$WORK/run_timed.sh")" 1

echo "9. an agent that finishes its run and then will not exit is released early"
# Marker lands but the CLI keeps running and ignores TERM. A pass must return in seconds.
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
# The agent is a grandchild of timeout(1) via a wrapper script; killing only the wrapper
# would leave it running. The stand-in records its pid and ignores TERM.
rm -f "$WORK/kid.pid"
# Two statements so bash cannot exec-optimize the wrapper away.
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
OUT="$(TURN_DONE_RE='ended with stopReason=' EXIT_GRACE=2 \
    run_case_bounded 45 3 bash -c 'echo Welcome to the agent; sleep 30')"
assert_eq "TURN_DONE stays clear"        "$(field "$OUT" TURN_DONE)" 0
assert_eq "still a cap"                  "$(field "$OUT" TIMED_OUT)" 1
assert_eq "no early-release warning"     "$(count "$OUT" 'would not exit')" 0

echo "9c. a marker that lands inside the last poll interval still counts"
# The watcher samples, so a run ending just before the cap is caught by reading the transcript after.
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
