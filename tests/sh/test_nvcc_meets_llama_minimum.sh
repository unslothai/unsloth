#!/bin/bash
# Unit tests for _nvcc_meets_llama_minimum() from studio/setup.sh.
#
# Background: llama.cpp requires CUDA toolkit >= 12.4. setup.ps1 already
# aborts on older toolkits via PR #4517 (issue #4437); the Linux side was
# silent and surfaced as a generic cmake failure. This test covers the
# follow-up Linux fix.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

# Extract just the helper function. The sed range is the same pattern the
# install.sh tests use.
_FUNC_FILE=$(mktemp)
sed -n '/^_nvcc_meets_llama_minimum()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"

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

# Make a fake nvcc binary that prints "release X.Y" in the canonical nvcc -V
# layout. The helper greps for "release X.Y", which is stable across CUDA
# 9.x through 13.x.
make_mock_nvcc() {
    _ver=$1
    _dir=$(mktemp -d)
    cat > "$_dir/nvcc" <<MOCK
#!/bin/sh
cat <<NV
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2026 NVIDIA Corporation
Cuda compilation tools, release $_ver, V${_ver}.0
NV
MOCK
    chmod +x "$_dir/nvcc"
    echo "$_dir/nvcc"
}

run_check() {
    _nvcc=$1
    bash -c ". '$_FUNC_FILE'; _nvcc_meets_llama_minimum '$_nvcc'"
}

echo "=== test_nvcc_meets_llama_minimum ==="

# 1) CUDA 12.4 is the minimum supported -> ok
_bin=$(make_mock_nvcc "12.4")
_out=$(run_check "$_bin")
assert_eq "12.4 status" "ok" "$(echo "$_out" | sed -n '1p')"
assert_eq "12.4 version" "12.4" "$(echo "$_out" | sed -n '2p')"
rm -rf "$(dirname "$_bin")"

# 2) CUDA 12.3 is the highest version that should be rejected.
_bin=$(make_mock_nvcc "12.3")
_out=$(run_check "$_bin")
assert_eq "12.3 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 3) CUDA 12.1 (matches the original bug report in #4437).
_bin=$(make_mock_nvcc "12.1")
_out=$(run_check "$_bin")
assert_eq "12.1 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 4) CUDA 11.8 -> too_old (anything < 12.0 is rejected).
_bin=$(make_mock_nvcc "11.8")
_out=$(run_check "$_bin")
assert_eq "11.8 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 5) CUDA 12.8 -> ok (mid-range supported).
_bin=$(make_mock_nvcc "12.8")
_out=$(run_check "$_bin")
assert_eq "12.8 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 6) CUDA 13.0 -> ok.
_bin=$(make_mock_nvcc "13.0")
_out=$(run_check "$_bin")
assert_eq "13.0 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 7) CUDA 13.3 -> ok. This is the freshly shipped toolkit the reddit thread
#    is hitting; the helper must accept it.
_bin=$(make_mock_nvcc "13.3")
_out=$(run_check "$_bin")
assert_eq "13.3 status" "ok" "$(echo "$_out" | sed -n '1p')"
assert_eq "13.3 version" "13.3" "$(echo "$_out" | sed -n '2p')"
rm -rf "$(dirname "$_bin")"

# 8) Future CUDA 14.0 -> ok (no upper bound).
_bin=$(make_mock_nvcc "14.0")
_out=$(run_check "$_bin")
assert_eq "14.0 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# 9) Empty argument -> unknown (defensive; do not block the build on
#    detection failure).
_out=$(run_check "")
assert_eq "empty path status" "unknown" "$(echo "$_out" | sed -n '1p')"

# 10) Mock nvcc that prints garbage -> unknown.
_dir=$(mktemp -d)
cat > "$_dir/nvcc" <<'MOCK'
#!/bin/sh
echo "totally not nvcc output"
MOCK
chmod +x "$_dir/nvcc"
_out=$(run_check "$_dir/nvcc")
assert_eq "garbage output status" "unknown" "$(echo "$_out" | sed -n '1p')"
rm -rf "$_dir"

rm -f "$_FUNC_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
