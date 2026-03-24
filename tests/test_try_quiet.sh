#!/usr/bin/env bash
# test_try_quiet.sh -- Standalone tests for try_quiet stderr redirect (Fix 2, PR #4494)
#
# Usage:  bash tests/test_try_quiet.sh
#
# Tests that the fixed try_quiet sends failure logs to stderr (not stdout),
# and contrasts with the broken pre-fix behavior.
set -uo pipefail

PASS=0
FAIL=0

# ── Helpers ───────────────────────────────────────────────────────────

# Minimal step() matching setup.sh signature (stdout, no color)
step() {
    local _label="$1" _value="$2"
    echo "  $_label: $_value"
}

# Fixed try_quiet (PR #4494)
try_quiet_fixed() {
    local label="$1"; shift
    local tmplog; tmplog=$(mktemp)
    if "$@" > "$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        if [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; then
            step "error" "$label failed (exit code $exit_code)"
            cat "$tmplog" >&2
        fi
        rm -f "$tmplog"
        return $exit_code
    fi
}

# Broken try_quiet (pre-fix: log dumps to stdout)
try_quiet_broken() {
    local label="$1"; shift
    local tmplog; tmplog=$(mktemp)
    if "$@" > "$tmplog" 2>&1; then
        rm -f "$tmplog"
        return 0
    else
        local exit_code=$?
        if [ "${UNSLOTH_VERBOSE:-0}" = "1" ]; then
            step "error" "$label failed (exit code $exit_code)"
            cat "$tmplog"   # BUG: no >&2, leaks to stdout
        fi
        rm -f "$tmplog"
        return $exit_code
    fi
}

assert_eq() {
    local name="$1" expected="$2" actual="$3"
    if [ "$expected" = "$actual" ]; then
        PASS=$((PASS + 1))
        echo "PASS:$name"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL:$name:expected '$(echo "$expected" | head -c 60)' got '$(echo "$actual" | head -c 60)'"
    fi
}

assert_contains() {
    local name="$1" haystack="$2" needle="$3"
    if printf '%s' "$haystack" | grep -qF "$needle"; then
        PASS=$((PASS + 1))
        echo "PASS:$name"
    else
        FAIL=$((FAIL + 1))
        echo "FAIL:$name:should contain '$needle'"
    fi
}

assert_not_contains() {
    local name="$1" haystack="$2" needle="$3"
    if printf '%s' "$haystack" | grep -qF "$needle"; then
        FAIL=$((FAIL + 1))
        echo "FAIL:$name:should NOT contain '$needle'"
    else
        PASS=$((PASS + 1))
        echo "PASS:$name"
    fi
}

tmpout=$(mktemp)
tmperr=$(mktemp)
trap 'rm -f "$tmpout" "$tmperr"' EXIT

# ── Test 1: Success -- no output anywhere ─────────────────────────────
rc=0
UNSLOTH_VERBOSE=0 try_quiet_fixed "test" true > "$tmpout" 2> "$tmperr" || rc=$?
assert_eq "1:success_rc" "0" "$rc"

# ── Test 2: Failure, VERBOSE=0 -- silent ──────────────────────────────
rc=0
UNSLOTH_VERBOSE=0 try_quiet_fixed "test" false > "$tmpout" 2> "$tmperr" || rc=$?
stdout_content=$(cat "$tmpout")
stderr_content=$(cat "$tmperr")
assert_eq "2:fail_quiet_stdout" "" "$stdout_content"
assert_eq "2:fail_quiet_stderr" "" "$stderr_content"

# ── Test 3: Failure, VERBOSE=1 -- log on stderr only ─────────────────
rc=0
UNSLOTH_VERBOSE=1 try_quiet_fixed "test" bash -c "echo logline_here; exit 1" > "$tmpout" 2> "$tmperr" || rc=$?
stdout_content=$(cat "$tmpout")
stderr_content=$(cat "$tmperr")
assert_contains    "3:fail_verbose_status"  "$stdout_content" "failed"
assert_contains    "3:fail_verbose_stderr"  "$stderr_content" "logline_here"
assert_not_contains "3:fail_verbose_clean"  "$stdout_content" "logline_here"

# ── Test 4: Multi-line failure, VERBOSE=1 -- all lines on stderr ──────
rc=0
UNSLOTH_VERBOSE=1 try_quiet_fixed "multi" bash -c "echo line_A; echo line_B; echo line_C; exit 1" > "$tmpout" 2> "$tmperr" || rc=$?
stdout_content=$(cat "$tmpout")
stderr_content=$(cat "$tmperr")
assert_contains     "4:multi_stderr_A" "$stderr_content" "line_A"
assert_contains     "4:multi_stderr_B" "$stderr_content" "line_B"
assert_contains     "4:multi_stderr_C" "$stderr_content" "line_C"
assert_not_contains "4:multi_stdout_A" "$stdout_content" "line_A"

# ── Test 5: Exit code preservation ────────────────────────────────────
rc=0
UNSLOTH_VERBOSE=0 try_quiet_fixed "test" bash -c "exit 42" > "$tmpout" 2> "$tmperr" || rc=$?
assert_eq "5:exit_code" "42" "$rc"

# ── Test 6: Binary output in tmplog -- no crash ──────────────────────
rc=0
UNSLOTH_VERBOSE=1 try_quiet_fixed "bintest" bash -c "printf '\\x00\\xff\\xfe'; exit 3" > "$tmpout" 2> "$tmperr" || rc=$?
assert_eq "6:binary_rc" "3" "$rc"

# ── Test 7: Success, VERBOSE=1 -- no error output ────────────────────
rc=0
UNSLOTH_VERBOSE=1 try_quiet_fixed "test" true > "$tmpout" 2> "$tmperr" || rc=$?
stdout_content=$(cat "$tmpout")
stderr_content=$(cat "$tmperr")
assert_eq "7:success_verbose_stdout" "" "$stdout_content"
assert_eq "7:success_verbose_stderr" "" "$stderr_content"

# ── Test 8: Broken version contrast -- log leaks to stdout ───────────
rc=0
UNSLOTH_VERBOSE=1 try_quiet_broken "test" bash -c "echo leaked_data; exit 1" > "$tmpout" 2> "$tmperr" || rc=$?
stdout_content=$(cat "$tmpout")
assert_contains "8:broken_leaks" "$stdout_content" "leaked_data"

# ── Summary ──────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo ""
if [ "$FAIL" -eq 0 ]; then
    echo "test_try_quiet.sh: $PASS/$TOTAL passed -- ALL PASSED"
    exit 0
else
    echo "test_try_quiet.sh: $PASS/$TOTAL passed, $FAIL FAILED"
    exit 1
fi
