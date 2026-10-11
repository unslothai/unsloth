#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# install.sh warns only when CUDA_VISIBLE_DEVICES="" hid NVIDIA on a mixed host and ROCm torch was picked.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"

_FN_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE"' EXIT
for _fn in _cvd_hides_nvidia _torch_index_url_leaf _torch_index_url_is_rocm \
           _warn_if_cuda_mask_hides_amd; do
    sed -n "/^${_fn}()/,/^}/p" "$INSTALL_SH" >> "$_FN_FILE"
done
if ! grep -q '^_warn_if_cuda_mask_hides_amd()' "$_FN_FILE"; then
    echo "FAIL: could not extract _warn_if_cuda_mask_hides_amd from install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
ROCM_URL="https://download.pytorch.org/whl/rocm7.1"
# $1 = physical NVIDIA card (1/0), $2 = index url, $3 = env assignments, $4 = AMD card on PCI (default 1). Stub honours the mask like the real probe.
_run() {
    env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
        $3 "$_SH" -c "
        _has_usable_nvidia_gpu() {
            _cvd_hides_nvidia && return 1
            [ '$1' = 1 ]
        }
        _amd_gpu_present_via_pci() { [ '${4:-1}' = 1 ]; }
        . '$_FN_FILE'
        _warn_if_cuda_mask_hides_amd '$2'
    " 2>&1
}

echo "=== The reporter's shape: 3080 + R9700, CUDA_VISIBLE_DEVICES emptied, ROCm index ==="
_out=$(_run 1 "$ROCM_URL" "CUDA_VISIBLE_DEVICES=")
assert_contains "warns about the mask"          "$_out" 'CUDA_VISIBLE_DEVICES=""'
assert_contains "names the supported switch"    "$_out" "UNSLOTH_FORCE_ROCM_TORCH=1"
assert_contains "names the HIP override"        "$_out" "HIP_VISIBLE_DEVICES=0"
_out=$(_run 1 "$ROCM_URL" "CUDA_VISIBLE_DEVICES=-1")
assert_contains "-1 is the same mask"           "$_out" 'CUDA_VISIBLE_DEVICES="-1"'
_out=$(_run 1 "https://repo.amd.com/rocm/whl/gfx120X-all/" "CUDA_VISIBLE_DEVICES=")
assert_contains "a per-arch gfx index counts"   "$_out" "UNSLOTH_FORCE_ROCM_TORCH=1"

echo "=== Everything else stays quiet ==="
assert_eq "AMD-only host: the mask is deliberate" "" "$(_run 0 "$ROCM_URL" "CUDA_VISIBLE_DEVICES=")"
assert_eq "NVIDIA-only host with a pinned ROCm index" "" "$(_run 1 "$ROCM_URL" "CUDA_VISIBLE_DEVICES=" 0)"
assert_eq "HIP_VISIBLE_DEVICES set: HIP ignores CUDA_VISIBLE_DEVICES" "" \
    "$(_run 1 "$ROCM_URL" "CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=0")"
assert_eq "no mask" "" "$(_run 1 "$ROCM_URL" "")"
assert_eq "a mask naming a device" "" "$(_run 1 "$ROCM_URL" "CUDA_VISIBLE_DEVICES=0")"
assert_eq "CUDA index" "" "$(_run 1 "https://download.pytorch.org/whl/cu128" "CUDA_VISIBLE_DEVICES=")"
assert_eq "cpu index" "" "$(_run 1 "https://download.pytorch.org/whl/cpu" "CUDA_VISIBLE_DEVICES=")"
assert_eq "mirror whose base path says rocm" "" \
    "$(_run 1 "https://mirror.example/rocm/whl/cpu" "CUDA_VISIBLE_DEVICES=")"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
