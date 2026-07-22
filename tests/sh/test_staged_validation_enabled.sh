#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for setup.sh staged-validation helpers (#5854 gap 2).
# Opt-in GPU smoke after a source build; default off (Blackwell JIT stall).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
{
    sed -n '/^_staged_validation_enabled()/,/^}/p' "$SETUP_SH"
    sed -n '/^_source_smoke_install_kind()/,/^}/p' "$SETUP_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

assert_rc() {
    _label="$1"; _expected="$2"
    shift 2
    set +e
    "$@" >/dev/null 2>&1
    _rc=$?
    set -e
    if [ "$_rc" -eq "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected rc $_expected, got $_rc)"; FAIL=$((FAIL + 1))
    fi
}

echo "=== _staged_validation_enabled ==="
unset UNSLOTH_LLAMA_STAGED_VALIDATION
assert_rc "default off" 1 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=0
assert_rc "0 is off" 1 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=1
assert_rc "1 is on" 0 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=true
assert_rc "true is on" 0 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=yes
assert_rc "yes is on" 0 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=on
assert_rc "on is on" 0 _staged_validation_enabled

UNSLOTH_LLAMA_STAGED_VALIDATION=maybe
assert_rc "maybe is off" 1 _staged_validation_enabled
unset UNSLOTH_LLAMA_STAGED_VALIDATION

echo "=== _source_smoke_install_kind ==="
_TRY_METAL_CPU_FALLBACK=true
GPU_BACKEND=""
assert_eq "metal" "macos-arm64" "$(_source_smoke_install_kind)"

_TRY_METAL_CPU_FALLBACK=false
GPU_BACKEND=cuda
_kind="$(_source_smoke_install_kind)"
case "$(uname -m)" in
    aarch64|arm64) assert_eq "cuda arm" "linux-arm64-cuda" "$_kind" ;;
    *) assert_eq "cuda x86" "linux-cuda" "$_kind" ;;
esac

GPU_BACKEND=rocm
assert_eq "rocm" "linux-rocm" "$(_source_smoke_install_kind)"

GPU_BACKEND=""
assert_eq "cpu empty" "" "$(_source_smoke_install_kind)"

echo "=== setup.sh source smoke contract ==="
assert_contains() {
    _label="$1"; _hay="$2"; _needle="$3"
    case "$_hay" in
        *"$_needle"*) echo "  PASS: $_label"; PASS=$((PASS + 1)) ;;
        *) echo "  FAIL: $_label (missing '$_needle')"; FAIL=$((FAIL + 1)) ;;
    esac
}
_src=$(cat "$SETUP_SH")
assert_contains "env gate present" "$_src" "UNSLOTH_LLAMA_STAGED_VALIDATION"
assert_contains "calls validate-install" "$_src" "--validate-install"
assert_contains "smoke fail retries CPU" "$_src" "source build failed smoke test; retrying CPU build"
# Smoke must run before the install swap.
_smoke_pos=$(printf '%s' "$_src" | awk '/validate source llama.cpp/{print NR; exit}')
_swap_pos=$(printf '%s' "$_src" | awk '/mv "\$_BUILD_TMP" "\$LLAMA_CPP_DIR"/{print NR; exit}')
if [ -n "$_smoke_pos" ] && [ -n "$_swap_pos" ] && [ "$_smoke_pos" -lt "$_swap_pos" ]; then
    echo "  PASS: smoke before install swap"; PASS=$((PASS + 1))
else
    echo "  FAIL: smoke before install swap (smoke=$_smoke_pos swap=$_swap_pos)"; FAIL=$((FAIL + 1))
fi

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
