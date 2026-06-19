#!/bin/bash
# Unit tests for _resolve_cuda_archs() from studio/setup.sh.
# A source CUDA build with no explicit -DCMAKE_CUDA_ARCHITECTURES ships PTX
# only and fails at runtime on a driver older than the toolkit (#5854). The
# helper turns `nvidia-smi --query-gpu=compute_cap` text into a deduped,
# ';'-separated arch list; an empty result is the signal the caller uses to
# build CPU instead of a PTX-only binary. An explicit override wins.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

# Extract just the helper function. Same sed range pattern the other
# setup.sh / install.sh function tests use.
_FUNC_FILE=$(mktemp)
sed -n '/^_resolve_cuda_archs()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"

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

# $1 = raw compute_cap text, $2 = override
run_resolve() {
    bash -c ". '$_FUNC_FILE'; _resolve_cuda_archs \"\$1\" \"\$2\"" _ "$1" "$2"
}

echo "=== test_resolve_cuda_archs ==="

# 1) Single GPU -> single arch.
assert_eq "single 8.6" "86" "$(run_resolve "8.6" "")"

# 2) Two distinct GPUs -> both archs, order preserved.
assert_eq "distinct 8.6 + 9.0" "86;90" "$(run_resolve "$(printf '8.6\n9.0\n')" "")"

# 3) Duplicate caps (multi-GPU same model) -> deduped.
assert_eq "dedup 12.0 x2" "120" "$(run_resolve "$(printf '12.0\n12.0\n')" "")"

# 4) Empty input -> empty (the CPU-fallback signal, the crux of #5854).
assert_eq "empty input" "" "$(run_resolve "" "")"

# 5) Garbage / N/A lines are ignored -> empty.
assert_eq "garbage N/A" "" "$(run_resolve "$(printf 'N/A\n[Not Supported]\n')" "")"

# 6) Mixed valid + junk -> only the valid caps survive.
assert_eq "mixed valid+junk" "86;90" "$(run_resolve "$(printf '8.6\nfoo\n9.0\n')" "")"

# 7) Whitespace / CR around a cap is stripped.
assert_eq "whitespace stripped" "86" "$(run_resolve "$(printf '  8.6 \r\n')" "")"

# 8) Override wins and is passed through verbatim, ignoring detection.
assert_eq "override wins" "120" "$(run_resolve "8.6" "120")"

# 9) Override works even with no detected caps.
assert_eq "override no detection" "86;90" "$(run_resolve "" "86;90")"

# 10) Three-digit-ish future arch (e.g. compute 10.0 -> 100) parses.
assert_eq "future 10.0" "100" "$(run_resolve "10.0" "")"

rm -f "$_FUNC_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
