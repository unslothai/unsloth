#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The manifest fast path must not strand an Intel venv.
#
# `unsloth studio update` skips install_python_stack entirely when the package version is
# current, and that pass is the ONLY thing that repairs an XPU wheel or swaps generic Triton
# out. Two ways that goes wrong:
#
# * the pin is ONE-SHOT. `UNSLOTH_TORCH_INDEX_FAMILY=xpu ./install.sh` leaves nothing in the
#   environment, so every later update sees no pin. Keying the escape on the pin alone means a
#   migrated +xpu venv that picked generic triton back up keeps it forever, and torch.compile
#   silently loads the CUDA-oriented library on an Arc GPU.
# * the leaf must match EXACTLY. The shared index parsers treat only a bare `xpu` leaf as the
#   curated family, so a custom mirror ending in `-xpu` would clear the skip flag on every
#   single up-to-date run while the repair helpers decline to act on it.
#
# The block is extracted and EXECUTED against built venv trees; nothing here is a grep for the
# code's own text, and no interpreter is launched (a wedged Intel driver would hang one).
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${1:-$SCRIPT_DIR/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# From the pin read to the end of the acting if/elif chain.
awk '/_setup_pin="\$\{UNSLOTH_TORCH_INDEX_URL/{on=1} on{print} on && /_SKIP_PYTHON_DEPS=false$/{n++} n==2{exit}' \
    "$SETUP_SH" > "$WORK/blk.sh"
echo "        fi" >> "$WORK/blk.sh"
[ -s "$WORK/blk.sh" ] || { echo "FATAL: escape block not found in $SETUP_SH" >&2; exit 1; }
# An extraction that lost any of the three moving parts would make cases below pass vacuously.
for _need in _setup_pin_leaf _setup_pin_is_xpu _setup_generic_triton; do
    grep -q "$_need" "$WORK/blk.sh" || { echo "FATAL: extraction lost $_need" >&2; exit 1; }
done
bash -n "$WORK/blk.sh" || { echo "FATAL: extracted block does not parse" >&2; exit 1; }

PASS=0
FAIL=0
check() {
    if [ "$2" = "$3" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  %-46s got=%s want=%s\n' "$1" "$2" "$3"
        FAIL=$((FAIL + 1))
    fi
}

# venv with torch label $1 (empty for none) and generic triton present when $2 is yes.
make_venv() {
    _v="$WORK/venv_$3"
    rm -rf "$_v"
    _sp="$_v/lib/python3.12/site-packages"
    mkdir -p "$_sp"
    if [ -n "$1" ]; then
        mkdir -p "$_sp/torch"
        printf "from typing import Optional\n__version__ = '%s'\ndebug = False\n" "$1" \
            > "$_sp/torch/version.py"
    fi
    # The XPU distributions must NOT count as the shadowing one.
    [ "$2" = "yes" ] && mkdir -p "$_sp/triton-3.7.1.dist-info"
    [ "$2" = "xpuonly" ] && mkdir -p "$_sp/pytorch_triton_xpu-3.5.0.dist-info" "$_sp/triton_xpu-3.6.0.dist-info"
    printf '%s' "$_v"
}

# Echoes the resulting _SKIP_PYTHON_DEPS. $2 is the pin (empty for none).
escape() {
    (
        VENV_DIR="$1"
        UNSLOTH_TORCH_INDEX_URL="$2"
        UNSLOTH_TORCH_INDEX_FAMILY="${3-}"
        _SKIP_PYTHON_DEPS=true
        # shellcheck disable=SC2317
        substep() { :; }
        # shellcheck disable=SC1091
        . "$WORK/blk.sh"
        echo "$_SKIP_PYTHON_DEPS"
    )
}

XPU=https://download.pytorch.org/whl/xpu

echo "the installed wheel is the pin -- no environment variable needed"
# The headline case: one-shot pin, gone by the next update, generic triton crept back in.
check "no pin, +xpu wheel, generic triton" \
    "$(escape "$(make_venv '2.9.1+xpu' yes a)" "")" false
check "no pin, +xpu wheel, clean venv" \
    "$(escape "$(make_venv '2.9.1+xpu' no b)" "")" true
# Only the generic distribution shadows; torch's own XPU builds must not retrigger forever.
check "no pin, +xpu wheel, xpu triton only" \
    "$(escape "$(make_venv '2.9.1+xpu' xpuonly c)" "")" true
# A CUDA or CPU venv owns its generic triton; removing it would break torch.compile there.
check "no pin, cuda wheel, generic triton" \
    "$(escape "$(make_venv '2.9.1+cu128' yes d)" "")" true
check "no pin, cpu wheel, generic triton" \
    "$(escape "$(make_venv '2.9.1+cpu' yes e)" "")" true
check "no pin, untagged wheel"  "$(escape "$(make_venv '2.9.1' yes f)" "")" true
check "no pin, no torch at all" "$(escape "$(make_venv '' yes g)" "")" true
check "no pin, no venv at all"  "$(escape "$WORK/nope" "")" true

echo "an explicit xpu pin still repairs a mismatched wheel"
check "pin + cpu wheel"        "$(escape "$(make_venv '2.9.1+cpu' no h)" "$XPU")" false
check "pin + no torch"         "$(escape "$(make_venv '' no i)" "$XPU")" false
# 2.6 is the floor (unsloth raises at import for an XPU device below it) and 2.11 the ceiling.
check "pin + 2.5+xpu below floor" "$(escape "$(make_venv '2.5.1+xpu' no j)" "$XPU")" false
check "pin + 2.11+xpu above ceiling" "$(escape "$(make_venv '2.11.0+xpu' no k)" "$XPU")" false
check "pin + supported wheel, clean" "$(escape "$(make_venv '2.9.1+xpu' no l)" "$XPU")" true
check "pin + supported wheel, generic triton" \
    "$(escape "$(make_venv '2.9.1+xpu' yes m)" "$XPU")" false
# The FAMILY spelling is a bare leaf with no slashes at all.
check "FAMILY=xpu + cpu wheel" "$(escape "$(make_venv '2.9.1+cpu' no n)" "" "xpu")" false
# Authenticated / fragmented / double-slashed mirrors are supported pin shapes.
check "pin with query"    "$(escape "$(make_venv '2.9.1+cpu' no o)" "$XPU?token=x")" false
check "pin with fragment" "$(escape "$(make_venv '2.9.1+cpu' no p)" "$XPU#f")" false
check "pin with two trailing slashes" "$(escape "$(make_venv '2.9.1+cpu' no q)" "$XPU//")" false
# Case. Every other leaf parser lowercases (install.sh _torch_index_url_leaf, setup.ps1
# Get-TorchIndexLeaf, install_python_stack _torch_index_leaf), so those accept FAMILY=XPU and
# would migrate the wheel; this one kept the fast path on and the repair never ran.
check "FAMILY=XPU uppercase"  "$(escape "$(make_venv '2.9.1+cpu' no w)" "" "XPU")" false
check "FAMILY=Xpu mixed case" "$(escape "$(make_venv '2.9.1+cpu' no x)" "" "Xpu")" false
check "pin URL ending /XPU"   "$(escape "$(make_venv '2.9.1+cpu' no y)" "https://download.pytorch.org/whl/XPU")" false
# Lowercasing must not widen the match: a custom leaf that merely ends in xpu stays unknown
# whatever its case.
check "custom leaf PRIVATE-XPU" "$(escape "$(make_venv '2.9.1+cpu' no z)" "https://mirror/simple/PRIVATE-XPU")" true

echo "a leaf that merely ends in xpu is NOT the curated family"
# The shared parsers call this an unknown family and refuse to act, so clearing the skip flag
# would mean a full dependency pass on every update that repairs nothing.
for _odd in "https://mirror/simple/private-xpu" "https://mirror/whl/cu128-xpu" "https://mirror/whl/rocm-xpu"; do
    check "custom leaf $(basename "$_odd") + cpu wheel" \
        "$(escape "$(make_venv '2.9.1+cpu' no r)" "$_odd")" true
    # ...but such a venv must still not be judged by the pin: if the WHEEL is xpu and generic
    # triton is there, the swap is owed regardless of what the odd pin says.
    check "custom leaf $(basename "$_odd") + xpu wheel + triton" \
        "$(escape "$(make_venv '2.9.1+xpu' yes s)" "$_odd")" false
done
# Non-XPU families must not be touched by any of this.
check "cuda pin + cuda wheel"  "$(escape "$(make_venv '2.9.1+cu128' yes t)" "https://download.pytorch.org/whl/cu128")" true
check "rocm pin + rocm wheel"  "$(escape "$(make_venv '2.9.1+rocm6.4' yes u)" "https://download.pytorch.org/whl/rocm6.4")" true
check "cpu pin + cpu wheel"    "$(escape "$(make_venv '2.9.1+cpu' yes v)" "https://download.pytorch.org/whl/cpu")" true

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
