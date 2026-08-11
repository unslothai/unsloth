#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# Guards the "arch computes incorrectly under ROCm" gate in get_torch_index_url.
#
# History: the ROCm wheel index was keyed purely off the ROCm runtime VERSION, never off
# the GPU's gfx arch. gfx1033 (Van Gogh, the Steam Deck APU) is mapped to gfx103X-all in
# _GFX_TO_AMD_INDEX_ARCH, so a Deck with any ROCm runtime present was routed to ROCm
# wheels -- which install and then return wrong answers rather than refusing: training
# diverges to a negative MSE loss and then NaN, and torch.autograd.gradcheck fails in
# float64, reproduced on rocm7.1/torch 2.10, rocm7.2/torch 2.11 and AMD's own native
# gfx1033 build. Forward math matches CPU. See studio/ROCM_RDNA2_APU.md.
#
# The contract: gfx1033 routes to the cpu index; every other arch is untouched.
#
# Scope note, and the reason this is not gated on AMD's hardware support table: unsloth
# deliberately serves archs AMD does not list -- gfx906 (MI50) via the rocm6.3 legacy
# index, verified there with torch 2.7.0, and gfx1031-gfx1036 via gfx103X-all (#7277).
# A table-wide gate would drop support that is known to work, so this list holds only
# archs measured to be wrong.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
PASS=0
FAIL=0

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

# The gate lives inline in get_torch_index_url on purpose (several harnesses extract that
# function alone; a helper they missed would be an undefined command whose negation sent
# every ROCm case to cpu). So exercise the real case block lifted from the function.
_FN_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE"' EXIT
awk '/# Archs measured to compute INCORRECTLY under ROCm/,/^        esac$/' "$INSTALL_SH" > "$_FN_FILE"

if ! grep -q 'gfx1033)' "$_FN_FILE"; then
    echo "FAIL: could not extract the gfx gate from get_torch_index_url in install.sh"
    exit 1
fi

_SH="${BASH:-/bin/bash}"
# Build a real function whose BODY is the extracted block, then call it. Sourcing the
# block instead would not do: `return` in a dot-script unwinds the source, not the caller,
# so execution would fall through to the trailing echo and print both lines. Embedding it
# reproduces exactly how `return` behaves inside get_torch_index_url.
_GATE_FILE=$(mktemp)
{
    echo '_gate() {'
    cat "$_FN_FILE"
    echo '    echo rocm'
    echo '}'
} > "$_GATE_FILE"
trap 'rm -f "$_FN_FILE" "$_GATE_FILE"' EXIT

_SH="${BASH:-/bin/bash}"
_route() {  # gfx -> the cpu index url when the gate intercepts, else "rocm"
    "$_SH" -c "
        _base=https://download.pytorch.org/whl
        _amd_gfx_probe='$1'
        . '$_GATE_FILE'
        _gate
    " 2>/dev/null | tail -1
}

echo "=== The measured-bad arch routes to CPU ==="
assert_eq "gfx1033 -> cpu"                 "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route gfx1033)"
assert_eq "gfx1033 with feature suffix"    "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route 'gfx1033:xnack-')"
assert_eq "GFX1033 uppercase"              "$(printf 'https://download.pytorch.org/whl/cpu')" "$(_route GFX1033)"

echo "=== Everything else falls through to the ROCm path, unchanged ==="
# gfx906 and gfx1031-1036 are the deliberately-supported-beyond-AMD's-table archs; if
# this gate ever swallows them it has silently dropped verified support.
for _gfx in gfx906 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1036 \
            gfx908 gfx90a gfx942 gfx950 gfx1100 gfx1101 gfx1102 gfx1103 \
            gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201; do
    assert_eq "$_gfx not intercepted" "rocm" "$(_route "$_gfx")"
done
assert_eq "empty probe not intercepted"   "rocm" "$(_route '')"
assert_eq "garbage not intercepted"       "rocm" "$(_route 'not-a-gfx')"
assert_eq "gfx10330 is not gfx1033"       "rocm" "$(_route gfx10330)"

echo "=== The runtime-less reroute honours the same gate ==="
# get_torch_index_url's gate is not enough on its own. The reroute below it fires when
# UNSLOTH_ROCM_GFX_ARCH is set, takes that value as the inferred arch, and rewrites a
# */cpu index to the arch family index -- so setting UNSLOTH_ROCM_GFX_ARCH=gfx1033 sent
# a Deck straight back to gfx103X-all, undoing the gate. The documented escape hatch is
# UNSLOTH_TORCH_INDEX_URL, which returns long before either point.
_REROUTE_FILE=$(mktemp)
trap 'rm -f "$_FN_FILE" "$_GATE_FILE" "$_REROUTE_FILE"' EXIT
{
    awk '/^_amd_arch_index_family_for_gfx\(\)/,/^}/' "$INSTALL_SH"
    awk '/^_infer_linux_amd_gfx_arch\(\)/,/^}/' "$INSTALL_SH"
    echo '_reroute() {'
    awk '/_linux_inferred_gfx=\$\(_infer_linux_amd_gfx_arch/,/^            if \[ -n "\$_linux_inferred_gfx" \]; then$/' "$INSTALL_SH"
    echo '    _amd_arch_index_family_for_gfx "$_linux_inferred_gfx" || echo cpu'
    echo '    else echo cpu'   # gated out: no arch survives, so no reroute happens
    echo '    fi'
    echo '}'
} > "$_REROUTE_FILE"

_reroute_family() {  # UNSLOTH_ROCM_GFX_ARCH -> family index, or cpu when gated out
    "$_SH" -c "
        UNSLOTH_ROCM_GFX_ARCH='$1'; export UNSLOTH_ROCM_GFX_ARCH
        . '$_REROUTE_FILE'
        _reroute
    " 2>/dev/null | tail -1
}
assert_eq "gfx1033 override does not reroute"  "cpu"          "$(_reroute_family gfx1033)"
assert_eq "GFX1033 override does not reroute"  "cpu"          "$(_reroute_family GFX1033)"
assert_eq "gfx1032 override still reroutes"    "gfx103X-all"  "$(_reroute_family gfx1032)"
assert_eq "gfx1030 override still reroutes"    "gfx103X-all"  "$(_reroute_family gfx1030)"
assert_eq "gfx1151 override still reroutes"    "gfx1151"      "$(_reroute_family gfx1151)"

echo "=== Structural: the gate precedes the version-keyed index selection ==="
_gate_line=$(grep -n 'Archs measured to compute INCORRECTLY under ROCm' "$INSTALL_SH" | head -1 | cut -d: -f1)
_idx_line=$(grep -n 'rocm7.2|rocm7.2.\*) echo "\$_base/rocm7.2"' "$INSTALL_SH" | head -1 | cut -d: -f1)
assert_eq "gate is before the rocm index case" "yes" \
    "$([ -n "$_gate_line" ] && [ -n "$_idx_line" ] && [ "$_gate_line" -lt "$_idx_line" ] && echo yes || echo no)"

echo ""
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" -eq 0 ]
