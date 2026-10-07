#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The Intel XPU bitsandbytes pass must run on both install paths, including migrated envs,
# and fire only on the xpu leaf.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="${1:-$SCRIPT_DIR/../../install.sh}"
BANNER='# ── Intel XPU: bitsandbytes with XPU kernels ──'
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

awk -v b="^[[:space:]]*$BANNER\$" '$0 ~ b, /^[[:space:]]*fi$/' "$INSTALL_SH" > "$WORK/blk.sh"
[ -s "$WORK/blk.sh" ] || { echo "FATAL: XPU bitsandbytes block not found in $INSTALL_SH" >&2; exit 1; }
# An extraction that lost the payload would make every case below pass vacuously.
grep -q '_BNB_XPU_SPEC' "$WORK/blk.sh" || { echo "FATAL: extraction lost the spec" >&2; exit 1; }

awk '/^_torch_index_url_leaf\(\) \{/,/^\}/' "$INSTALL_SH" > "$WORK/leaf.sh"
[ -s "$WORK/leaf.sh" ] || { echo "FATAL: could not extract _torch_index_url_leaf" >&2; exit 1; }

PASS=0
FAIL=0

_chain_start=$(grep -n '^if \[ "\$_MIGRATED" = true \]; then$' "$INSTALL_SH" | head -1 | cut -d: -f1)
_chain_fi=$(grep -n '^fi$' "$INSTALL_SH" | awk -F: -v s="$_chain_start" '$1 > s { print $1; exit }')
_blk_line=$(grep -n "^[[:space:]]*$BANNER\$" "$INSTALL_SH" | head -1 | cut -d: -f1)
if [ -n "$_chain_fi" ] && [ -n "$_blk_line" ] && [ "$_blk_line" -gt "$_chain_fi" ]; then
    echo "  PASS  placed past the install chain (block $_blk_line, chain closes $_chain_fi)"
    PASS=$((PASS + 1))
else
    echo "  FAIL  block at ${_blk_line:-?} is inside an install arm (chain closes ${_chain_fi:-?})"
    FAIL=$((FAIL + 1))
fi

run_case() {
    _migrated="$1"; _skip="$2"; _url="$3"; _want="$4"
    _got=$(
        export _MIGRATED="$_migrated" SKIP_TORCH="$_skip" TORCH_INDEX_URL="$_url"
        _VENV_PY=/nonexistent/python; C_WARN=""; _BNB_XPU_SPEC="bitsandbytes>=0.50.0"
        substep() { :; }
        run_install_cmd() { shift; echo "FIRED: $*"; }
        # shellcheck disable=SC1090
        . "$WORK/leaf.sh"; . "$WORK/blk.sh"
    )
    _fired=no
    case "$_got" in *FIRED:*) _fired=yes ;; esac
    if [ "$_fired" = "$_want" ]; then
        PASS=$((PASS + 1))
    else
        printf '  FAIL  migrated=%s skip_torch=%s index=%s fired=%s want=%s\n' \
            "$_migrated" "$_skip" "${_url:-<empty>}" "$_fired" "$_want"
        FAIL=$((FAIL + 1))
    fi
}

# [migrated, fresh] x [xpu, mirrored xpu, cuda, rocm, cpu, none] x [torch, no-torch].
for m in true false; do
    for s in false true; do
        want_xpu=yes
        [ "$s" = true ] && want_xpu=no
        run_case "$m" "$s" "https://download.pytorch.org/whl/xpu"     "$want_xpu"
        run_case "$m" "$s" "https://mirror.internal/pytorch/whl/xpu/" "$want_xpu"
        run_case "$m" "$s" "https://download.pytorch.org/whl/cu128"   no
        run_case "$m" "$s" "https://download.pytorch.org/whl/rocm6.4" no
        run_case "$m" "$s" "https://download.pytorch.org/whl/cpu"     no
        run_case "$m" "$s" ""                                         no
    done
done

echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
