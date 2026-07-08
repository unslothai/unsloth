#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for select_cuda_jit_tools() from docker/entrypoint.sh.
#
# The image bakes CUDA 13 ptxas + NVRTC, but a cu13 cubin cannot LOAD on a
# 570-579 driver even when it targets an old arch like sm_80 (CUDA has forward,
# not backward, driver compatibility across major versions). So cu12.8 is the
# IMMUTABLE baked default (libnvrtc.so.12 -> .cu128.orig), and the cu13 tools are
# switched on ONLY for the two Blackwell datacenter arches that require them --
# sm_103 (B300 / GB300) and sm_121 (GB10 / DGX Spark), which only ship on >= 580
# drivers. Every other supported arch (Turing..sm_120) keeps the cu12.8 default,
# untouched, so a 570+ driver host -- including a non-root --user container that
# cannot rewrite the symlink -- is never broken.
#
# The function picks per device via nvidia-smi compute_cap: DC -> retarget
# libnvrtc.so.12 -> the staged .cu13 alias (and point Triton at cu13 ptxas);
# anything else -> leave the cu12.8 default in place and ptxas unset.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENTRYPOINT_SH="$SCRIPT_DIR/../../docker/entrypoint.sh"
PASS=0
FAIL=0

# Extract just the helper function (same sed range as the other function tests).
_FUNC_FILE=$(mktemp)
sed -n '/^select_cuda_jit_tools()/,/^}/p' "$ENTRYPOINT_SH" > "$_FUNC_FILE"

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

# $1 = compute_cap(s) the mock nvidia-smi reports, ONE PER LINE ("none" -> no
# nvidia-smi on PATH). A multi-line value models a mixed-GPU host so we can check
# that every visible cap is scanned, not just the first.
# Builds a fake Studio venv NVRTC dir exactly as the build stages it: the real
# cu12.8 lib as .cu128.orig, libnvrtc.so.12 -> it (the immutable default), and a
# .cu13 alias pointing at a stand-in cu13 lib. Runs the function against it via
# UNSLOTH_STUDIO_HOME. The hardcoded base venv path does not exist on the test
# host, so its glob is skipped. Prints "<PTXAS_STATE> <NVRTC_TARGET>".
run_select() {
    _cap="$1"
    _tmp=$(mktemp -d)
    mkdir -p "$_tmp/bin"
    if [ "$_cap" != "none" ]; then
        # nvidia-smi --query-gpu=compute_cap prints one cap per line; cat a file
        # so an embedded newline in $_cap survives into the mock's output.
        printf '%s\n' "$_cap" > "$_tmp/caps.txt"
        printf '#!/bin/sh\ncat "%s"\n' "$_tmp/caps.txt" > "$_tmp/bin/nvidia-smi"
        chmod +x "$_tmp/bin/nvidia-smi"
    fi
    _nvrtc="$_tmp/studio/unsloth_studio/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib"
    mkdir -p "$_nvrtc"
    : > "$_nvrtc/libnvrtc.so.12.cu128.orig"                       # real cu12.8 lib
    : > "$_nvrtc/libnvrtc.so.13.stub"                             # stand-in cu13 lib
    ln -sf libnvrtc.so.13.stub "$_nvrtc/libnvrtc.so.12.cu13"      # staged cu13 alias
    ln -sf libnvrtc.so.12.cu128.orig "$_nvrtc/libnvrtc.so.12"     # immutable cu12.8 default
    bash -c '
        set -euo pipefail
        export PATH="'"$_tmp"'/bin:/usr/bin:/bin"
        export UNSLOTH_STUDIO_HOME="'"$_tmp"'/studio"
        unset TRITON_PTXAS_PATH || true
        . "'"$_FUNC_FILE"'"
        select_cuda_jit_tools || true
        printf "%s %s\n" "${TRITON_PTXAS_PATH:-UNSET}" "$(readlink "'"$_nvrtc"'/libnvrtc.so.12")"
    '
    rm -rf "$_tmp"
}

echo "=== test_select_cuda_jit_tools ==="

# Non-DC arches: cu12.8 default is left untouched (no write) and ptxas unset
# (Triton keeps its bundled cu12.8 ptxas), so a 570-579 driver host -- root or
# --user -- is unaffected.
assert_eq "sm_80 Ampere -> cu128 default kept"  "UNSET libnvrtc.so.12.cu128.orig" "$(run_select 8.0)"
assert_eq "sm_90 Hopper -> cu128 default kept"  "UNSET libnvrtc.so.12.cu128.orig" "$(run_select 9.0)"
assert_eq "sm_100 B200 -> cu128 default kept"   "UNSET libnvrtc.so.12.cu128.orig" "$(run_select 10.0)"
assert_eq "sm_120 RTX50 -> cu128 default kept"  "UNSET libnvrtc.so.12.cu128.orig" "$(run_select 12.0)"
assert_eq "no nvidia-smi -> cu128 default kept" "UNSET libnvrtc.so.12.cu128.orig" "$(run_select none)"

# Blackwell datacenter: retarget libnvrtc.so.12 -> the .cu13 alias. ptxas stays
# UNSET here only because the test host has no /usr/local/cuda-13.0/bin/ptxas;
# the assertion that matters is that the NVRTC switched to cu13 for these arches.
assert_eq "sm_103 B300 -> cu13 NVRTC selected"      "UNSET libnvrtc.so.12.cu13" "$(run_select 10.3)"
assert_eq "sm_121 DGX Spark -> cu13 NVRTC selected" "UNSET libnvrtc.so.12.cu13" "$(run_select 12.1)"

# Mixed-GPU hosts: a datacenter Blackwell (sm_103 / sm_121) sitting BEHIND an
# H100/B200 in the nvidia-smi ordering must still switch to cu13 -- every visible
# cap is scanned, not just the first. A host with no datacenter Blackwell at all
# keeps the cu12.8 default regardless of order.
assert_eq "H100 then B300 -> cu13 NVRTC selected" "UNSET libnvrtc.so.12.cu13"       "$(run_select "$(printf '9.0\n10.3')")"
assert_eq "B200 then GB10 -> cu13 NVRTC selected" "UNSET libnvrtc.so.12.cu13"       "$(run_select "$(printf '10.0\n12.1')")"
assert_eq "B300 then H100 -> cu13 NVRTC selected" "UNSET libnvrtc.so.12.cu13"       "$(run_select "$(printf '10.3\n9.0')")"
assert_eq "H100 then A100 -> cu128 default kept"  "UNSET libnvrtc.so.12.cu128.orig" "$(run_select "$(printf '9.0\n8.0')")"

rm -f "$_FUNC_FILE"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
