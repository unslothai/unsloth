#!/bin/bash
# Unit tests for parallel setup helpers from studio/setup.sh (issue #8818).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
sed -n '/^_setup_parallel_reset()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"
sed -n '/^_setup_parallel_run()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
sed -n '/^_setup_parallel_wait()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
sed -n '/^_setup_parallel_wait_fail_open()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"
sed -n '/^step()/,/^}/p' "$SETUP_SH" | head -n 1 >> "$_FUNC_FILE"
sed -n '/^setup_fail()/,/^}/p' "$SETUP_SH" >> "$_FUNC_FILE"

if [ ! -s "$_FUNC_FILE" ]; then
    echo "FAIL: could not extract parallel helpers from $SETUP_SH"
    exit 1
fi

# shellcheck disable=SC1090
. "$_FUNC_FILE"

setup_fail() {
    return 1
}

assert_parallel_ok() {
    local label="$1"
    _setup_parallel_reset
    _setup_parallel_run "$label" true
    if _setup_parallel_wait; then
        echo "  PASS: $label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $label"
        FAIL=$((FAIL + 1))
    fi
}

assert_parallel_fail() {
    local label="$1"
    _setup_parallel_reset
    _setup_parallel_run "$label" false
    if _setup_parallel_wait; then
        echo "  FAIL: $label (expected failure)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $label (failed as expected)"
        PASS=$((PASS + 1))
    fi
}

echo "setup parallel helpers"
assert_parallel_ok "single successful job"
assert_parallel_fail "single failing job"

_setup_parallel_reset
_setup_parallel_run "job-a" true
_setup_parallel_run "job-b" true
if _setup_parallel_wait; then
    echo "  PASS: two successful jobs"
    PASS=$((PASS + 1))
else
    echo "  FAIL: two successful jobs"
    FAIL=$((FAIL + 1))
fi

rm -f "$_FUNC_FILE"
echo ""
echo "Passed: $PASS  Failed: $FAIL"
[ "$FAIL" -eq 0 ] || exit 1
