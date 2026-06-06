#!/bin/bash
# Unit tests for _classify_smoke_exit() from studio/setup.sh.
#
# After a llama.cpp source build, setup.sh runs install_llama_prebuilt.py
# --smoke-test against the fresh binary and maps its exit code to a decision:
#   2 (EXIT_FALLBACK) -> "cpu_only"     : GPU was requested but the model ran
#                                         on CPU -> rebuild CPU (#5807 / #5854).
#   0                 -> "ok"           : GPU offload confirmed -> keep build.
#   1 / signals / etc -> "inconclusive" : could not validate -> keep GPU build
#                                         (never downgrade on uncertainty).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

# Extract just the helper (same approach as test_nvcc_meets_llama_minimum.sh).
# The function and its closing brace sit at 8-space indent inside setup.sh.
_FUNC_FILE=$(mktemp)
sed -n '/        _classify_smoke_exit() {/,/^        }/p' "$SETUP_SH" > "$_FUNC_FILE"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"
        FAIL=$((FAIL + 1))
    fi
}

run_classify() {
    bash -c ". '$_FUNC_FILE'; _classify_smoke_exit '$1'"
}

echo "=== test_llama_gpu_smoke (_classify_smoke_exit) ==="

assert_eq "exit 0 -> ok"            "ok"           "$(run_classify 0)"
assert_eq "exit 2 -> cpu_only"      "cpu_only"     "$(run_classify 2)"
assert_eq "exit 1 -> inconclusive"  "inconclusive" "$(run_classify 1)"
assert_eq "exit 3 -> inconclusive"  "inconclusive" "$(run_classify 3)"
assert_eq "exit 137 -> inconclusive" "inconclusive" "$(run_classify 137)"

# Sanity: the extracted function is non-empty and well-formed.
if [ -s "$_FUNC_FILE" ] && grep -q 'cpu_only' "$_FUNC_FILE"; then
    echo "  PASS: _classify_smoke_exit extracted from setup.sh"
    PASS=$((PASS + 1))
else
    echo "  FAIL: _classify_smoke_exit could not be extracted from setup.sh"
    FAIL=$((FAIL + 1))
fi

rm -f "$_FUNC_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
