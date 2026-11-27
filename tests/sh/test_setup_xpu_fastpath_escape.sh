#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The manifest fast path must not strand an Intel venv: the pin is one-shot, and only a bare
# `xpu` leaf is the curated family. Executed against built venv trees, no interpreter.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${1:-$SCRIPT_DIR/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Sliced to _fast_path_escapes()'s closing brace; counting assignments truncates on new arms.
awk '/_setup_pin="\$\{UNSLOTH_TORCH_INDEX_URL/{on=1} on && /^}$/{exit} on{print}' \
    "$SETUP_SH" > "$WORK/blk.sh"
[ -s "$WORK/blk.sh" ] || { echo "FATAL: escape block not found in $SETUP_SH" >&2; exit 1; }
grep -q '^_fast_path_escapes() {$' "$SETUP_SH" \
    || { echo "FATAL: _fast_path_escapes is no longer a top-level function in $SETUP_SH" >&2; exit 1; }
grep -q 'INSTALLED_VER' "$WORK/blk.sh" \
    && { echo "FATAL: extraction ran past the end of _fast_path_escapes" >&2; exit 1; }
# A lossy extraction would make the cases below pass vacuously.
_arms=$(grep -c '_SKIP_PYTHON_DEPS=false' "$WORK/blk.sh")
[ "$_arms" = "4" ] || { echo "FATAL: expected 4 escape arms, extracted $_arms" >&2; exit 1; }
for _need in _setup_pin_leaf _setup_pin_is_xpu _setup_generic_triton _setup_pin_known_nonxpu \
             _setup_known_nonxpu_leaf; do
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

echo "an explicit pin of another curated family over a labelled wheel forces the pass (setup.ps1 parity)"
check "cu128 pin, +cpu wheel"          "$(escape "$(make_venv '2.9.1+cpu' no h)" "" cu128)" false
check "cu128 pin, +cu130 wheel"        "$(escape "$(make_venv '2.9.1+cu130' no i)" "" cu128)" true
check "rocm7.1 pin, +cpu wheel"        "$(escape "$(make_venv '2.9.1+cpu' no j)" "" rocm7.1)" false
check "gfx1201 pin, +cu128 wheel"      "$(escape "$(make_venv '2.9.1+cu128' no k)" "" gfx1201)" false
check "cpu pin, +cu130 wheel"          "$(escape "$(make_venv '2.9.1+cu130' no l)" "" cpu)" false
check "cu128 pin, untagged wheel"      "$(escape "$(make_venv '2.9.1' no m)" "" cu128)" true
check "cu128 pin, no torch at all"     "$(escape "$(make_venv '' no n)" "" cu128)" true
check "cu128-private pin, +cpu wheel"  "$(escape "$(make_venv '2.9.1+cpu' no o)" "" cu128-private)" true
check "cu128 URL pin, +cpu wheel"      "$(escape "$(make_venv '2.9.1+cpu' no p)" "https://download.pytorch.org/whl/cu128")" false

echo "an explicit xpu pin still repairs a mismatched wheel"
check "pin + cpu wheel"        "$(escape "$(make_venv '2.9.1+cpu' no h)" "$XPU")" false
check "pin + no torch"         "$(escape "$(make_venv '' no i)" "$XPU")" false
# 2.6 is the floor (unsloth raises below it on XPU) and 2.11 the ceiling.
check "pin + 2.5+xpu below floor" "$(escape "$(make_venv '2.5.1+xpu' no j)" "$XPU")" false
check "pin + 2.11+xpu above ceiling" "$(escape "$(make_venv '2.11.0+xpu' no k)" "$XPU")" false
check "pin + supported wheel, clean" "$(escape "$(make_venv '2.9.1+xpu' no l)" "$XPU")" true
check "pin + supported wheel, generic triton" \
    "$(escape "$(make_venv '2.9.1+xpu' yes m)" "$XPU")" false
check "FAMILY=xpu + cpu wheel" "$(escape "$(make_venv '2.9.1+cpu' no n)" "" "xpu")" false
check "pin with query"    "$(escape "$(make_venv '2.9.1+cpu' no o)" "$XPU?token=x")" false
check "pin with fragment" "$(escape "$(make_venv '2.9.1+cpu' no p)" "$XPU#f")" false
check "pin with two trailing slashes" "$(escape "$(make_venv '2.9.1+cpu' no q)" "$XPU//")" false
# Other leaf parsers lowercase, so this one must accept FAMILY=XPU too.
check "FAMILY=XPU uppercase"  "$(escape "$(make_venv '2.9.1+cpu' no w)" "" "XPU")" false
check "FAMILY=Xpu mixed case" "$(escape "$(make_venv '2.9.1+cpu' no x)" "" "Xpu")" false
check "pin URL ending /XPU"   "$(escape "$(make_venv '2.9.1+cpu' no y)" "https://download.pytorch.org/whl/XPU")" false
check "custom leaf PRIVATE-XPU" "$(escape "$(make_venv '2.9.1+cpu' no z)" "https://mirror/simple/PRIVATE-XPU")" true

echo "migrating AWAY from xpu escapes too -- the pin is authoritative"
# Only install_python_stack applies a changed pin, so a family switch must escape.
check "cu128 pin over xpu wheel"  "$(escape "$(make_venv '2.9.1+xpu' no A)" "https://download.pytorch.org/whl/cu128")" false
check "rocm pin over xpu wheel"   "$(escape "$(make_venv '2.9.1+xpu' no B)" "https://download.pytorch.org/whl/rocm6.4")" false
check "cpu pin over xpu wheel"    "$(escape "$(make_venv '2.9.1+xpu' no C)" "https://download.pytorch.org/whl/cpu")" false
check "gfx pin over xpu wheel"    "$(escape "$(make_venv '2.9.1+xpu' no D)" "https://repo.radeon.com/whl/gfx1151")" false
check "FAMILY=cu128 over xpu wheel" "$(escape "$(make_venv '2.9.1+xpu' no E)" "" "cu128")" false
check "rocm7.2 dotted pin over xpu wheel" "$(escape "$(make_venv '2.9.1+xpu' no B2)" "https://download.pytorch.org/whl/rocm7.2")" false
# gfx is a PREFIX family because gfx120x-all is a real Radeon index leaf.
check "gfx120x-all pin over xpu wheel" "$(escape "$(make_venv '2.9.1+xpu' no D2)" "https://repo.radeon.com/whl/gfx120x-all")" false
# EXACT families only: a leaf that merely starts with one is a custom pin never repaired.
check "custom rocm-current over xpu"    "$(escape "$(make_venv '2.9.1+xpu' no F)" "https://mirror/whl/rocm-current")" true
check "custom cu-private over xpu"      "$(escape "$(make_venv '2.9.1+xpu' no G)" "https://mirror/whl/cu-private")" true
check "custom cu128-private over xpu"   "$(escape "$(make_venv '2.9.1+xpu' no G2)" "https://mirror/whl/cu128-private")" true
check "custom cu128rc1 over xpu"        "$(escape "$(make_venv '2.9.1+xpu' no G3)" "https://mirror/whl/cu128rc1")" true
check "custom rocm7.2-private over xpu" "$(escape "$(make_venv '2.9.1+xpu' no F2)" "https://mirror/whl/rocm7.2-private")" true
check "custom rocm7. over xpu"          "$(escape "$(make_venv '2.9.1+xpu' no F3)" "https://mirror/whl/rocm7.")" true
check "custom rocm7.2.1 over xpu"       "$(escape "$(make_venv '2.9.1+xpu' no F4)" "https://mirror/whl/rocm7.2.1")" true
check "custom cpu-private over xpu"     "$(escape "$(make_venv '2.9.1+xpu' no F5)" "https://mirror/whl/cpu-private")" true
check "cu128 pin over cuda wheel"  "$(escape "$(make_venv '2.9.1+cu128' no H)" "https://download.pytorch.org/whl/cu128")" true

echo "a leaf that merely ends in xpu is NOT the curated family"
# Unknown family to the shared parsers, so clearing the skip flag would repair nothing.
for _odd in "https://mirror/simple/private-xpu" "https://mirror/whl/cu128-xpu" "https://mirror/whl/rocm-xpu"; do
    check "custom leaf $(basename "$_odd") + cpu wheel" \
        "$(escape "$(make_venv '2.9.1+cpu' no r)" "$_odd")" true
    check "custom leaf $(basename "$_odd") + xpu wheel + triton" \
        "$(escape "$(make_venv '2.9.1+xpu' yes s)" "$_odd")" false
done
check "cuda pin + cuda wheel"  "$(escape "$(make_venv '2.9.1+cu128' yes t)" "https://download.pytorch.org/whl/cu128")" true
check "rocm pin + rocm wheel"  "$(escape "$(make_venv '2.9.1+rocm6.4' yes u)" "https://download.pytorch.org/whl/rocm6.4")" true
check "cpu pin + cpu wheel"    "$(escape "$(make_venv '2.9.1+cpu' yes v)" "https://download.pytorch.org/whl/cpu")" true

# Parity: the leaf this calls known must be one install_python_stack acts on.
PY_STACK="$SCRIPT_DIR/../../studio/install_python_stack.py"
if [ -f "$PY_STACK" ] && command -v python3 >/dev/null 2>&1; then
    awk '/^    _setup_known_nonxpu_leaf\(\) \{/{on=1} on{print} on && /^    \}$/{exit}' \
        "$WORK/blk.sh" > "$WORK/leaf.sh"
    grep -q 'rocm\[0-9\]' "$WORK/leaf.sh" || { echo "FATAL: predicate not extracted" >&2; exit 1; }
    # shellcheck disable=SC1091
    . "$WORK/leaf.sh"
    _corpus="cpu cu118 cu126 cu128 cu130 rocm6.3 rocm6.4 rocm7.0 rocm7.2 gfx90a gfx1151
             gfx120x-all xpu cu128-private cu128rc1 cu128.1 cu-private rocm-current rocm7.
             rocm7.2.1 rocm7.2-private rocm.4 cpu-private gfx-private private-xpu cu rocm gfx"
    _pyout=$(cd "$(dirname "$PY_STACK")" && python3 -c '
import sys, install_python_stack as m
for leaf in sys.argv[1:]:
    known = m._is_pip_rocm_family_leaf(leaf) or leaf == "cpu" or m._is_cuda_family_leaf(leaf)
    print(leaf, "true" if known else "false")
' $_corpus) || { echo "FATAL: python predicate did not run" >&2; exit 1; }
    echo "the shell predicate agrees with install_python_stack leaf for leaf"
    while read -r _leaf _want; do
        [ -n "$_leaf" ] || continue
        _got=false
        _setup_known_nonxpu_leaf "$_leaf" && _got=true
        check "leaf $_leaf" "$_got" "$_want"
    done <<EOF
$_pyout
EOF
fi

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
