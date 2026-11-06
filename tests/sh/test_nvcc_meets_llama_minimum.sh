#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# _nvcc_meets_llama_minimum() from studio/setup.sh: llama.cpp needs CUDA toolkit >= 12.4.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
_FUNC_FILE=$(mktemp)
sed -n '/^_nvcc_meets_llama_minimum()/,/^}/p' "$SETUP_SH" > "$_FUNC_FILE"

# Fake nvcc printing "release X.Y" in the canonical nvcc -V layout.
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

_bin=$(make_mock_nvcc "12.4")
_out=$(run_check "$_bin")
assert_eq "12.4 status" "ok" "$(echo "$_out" | sed -n '1p')"
assert_eq "12.4 version" "12.4" "$(echo "$_out" | sed -n '2p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "12.3")
_out=$(run_check "$_bin")
assert_eq "12.3 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "12.1")
_out=$(run_check "$_bin")
assert_eq "12.1 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "11.8")
_out=$(run_check "$_bin")
assert_eq "11.8 status" "too_old" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "12.8")
_out=$(run_check "$_bin")
assert_eq "12.8 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "13.0")
_out=$(run_check "$_bin")
assert_eq "13.0 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "13.3")
_out=$(run_check "$_bin")
assert_eq "13.3 status" "ok" "$(echo "$_out" | sed -n '1p')"
assert_eq "13.3 version" "13.3" "$(echo "$_out" | sed -n '2p')"
rm -rf "$(dirname "$_bin")"

_bin=$(make_mock_nvcc "14.0")
_out=$(run_check "$_bin")
assert_eq "14.0 status" "ok" "$(echo "$_out" | sed -n '1p')"
rm -rf "$(dirname "$_bin")"

# Empty -> unknown: never block the build on detection.
_out=$(run_check "")
assert_eq "empty path status" "unknown" "$(echo "$_out" | sed -n '1p')"

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

summary
