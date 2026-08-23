#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Drive install.sh's AMD summary with stub rocminfo and amd-smi outputs.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0

{
    sed -n '/^_rocminfo_gpu_records()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_amd_smi_gpu_records()/,/^}/p' "$INSTALL_SH"
    sed -n '/^_amd_smi_hip_order()/,/^}/p'  "$INSTALL_SH"
    echo ""
    # Extract the probe and selection block.
    awk '/^    _gpu_disp_gfx_all=""/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$INSTALL_SH"
} > "$WORK/block.sh"
grep -q '_gpu_disp_record=' "$WORK/block.sh" || {
    echo "FATAL: GPU summary probe block not found in $INSTALL_SH" >&2; exit 1; }
grep -q '_amd_smi_gpu_records' "$WORK/block.sh" || {
    echo "FATAL: the amd-smi record parser is not wired into the block" >&2; exit 1; }
grep -q '_amd_smi_hip_order' "$WORK/block.sh" || {
    echo "FATAL: the amd-smi HIP reorder is not wired into the block" >&2; exit 1; }

# Build PATH from scratch so host AMD tools cannot leak into the test.
mkdir -p "$WORK/base" "$WORK/roc" "$WORK/smi"
for _tool in awk grep sed cat tr sort wc head tail; do
    _p=$(command -v "$_tool") || { echo "FATAL: $_tool not found" >&2; exit 1; }
    ln -sf "$_p" "$WORK/base/$_tool"
done
cat > "$WORK/roc/rocminfo" <<'STUB'
#!/bin/sh
[ -s "$STUB_ROCMINFO" ] || exit 1
cat "$STUB_ROCMINFO"
STUB
# `amd-smi list` prints BDF / UUID / KFD_ID and no gfx token, so the arch has to come
# from `static --asic`. A stub that leaks it into `list` never tests the fallthrough.
cat > "$WORK/smi/amd-smi" <<'STUB'
#!/bin/sh
[ -s "$STUB_AMDSMI" ] || exit 1
case "$1 $2" in
    # `list -e` carries HIP_ID and nothing else of interest; an older CLI rejects the
    # flag outright, which is the STUB_AMDSMI_E="" case.
    "list -e")  [ -z "${STUB_AMDSMI_E:-}" ] || cat "$STUB_AMDSMI_E" ;;
    "list "*)   sed -n 's/^\(GPU: [0-9]*\).*/\1  BDF: 0000:03:00.0  UUID: aaaa-bbbb  KFD_ID: 1/p' "$STUB_AMDSMI" ;;
    "static "*) cat "$STUB_AMDSMI" ;;
esac
STUB
chmod +x "$WORK/roc/rocminfo" "$WORK/smi/amd-smi"

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

# $1 rocminfo fixture ("-" = the tool is not installed, empty file = installed but silent)
# $2 amd-smi fixture, same convention. $3 visible-device mask. Prints "gfx|name".
summary() {
    _path="$WORK/base"
    [ "$1" != "-" ] && _path="$WORK/roc:$_path"
    [ "$2" != "-" ] && _path="$WORK/smi:$_path"
    # install.sh runs this block under `set -e`; match it.
    env -i PATH="$_path" \
        STUB_ROCMINFO="$1" STUB_AMDSMI="$2" \
        ${STUB_AMDSMI_E:+STUB_AMDSMI_E="$STUB_AMDSMI_E"} \
        ${3:+HIP_VISIBLE_DEVICES="$3"} \
        /bin/bash -c 'set -eu; . "$1"; printf "%s|%s\n" "$_gpu_disp_gfx" "$_gpu_disp_mkt"' _ "$WORK/block.sh"
}

cat > "$WORK/roc_gpu" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 5950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 5950X 16-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1151
  Marketing Name:          AMD Radeon Graphics
  Vendor Name:             AMD
  Device Type:             GPU
EOF
cat > "$WORK/roc_cpu_only" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 5950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 5950X 16-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
EOF
cat > "$WORK/roc_blank_name" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 5950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 5950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1030
  Device Type:             GPU
EOF
cat > "$WORK/roc_two_gpus" <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 7763 64-Core Processor
  Marketing Name:          AMD EPYC 7763 64-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx90a
  Marketing Name:          AMD Instinct MI210
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
EOF
# MARKET_NAME is what a real host prints. Spelling it "Market Name" -- the one form the
# old parser matched -- kept this green while the naming path was dead in production.
cat > "$WORK/smi_fixture" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
        TARGET_GRAPHICS_VERSION: gfx1100
EOF
# Older amd-smi builds print it title-cased; both spellings must still work.
cat > "$WORK/smi_titlecase" <<'EOF'
GPU: 0
    ASIC:
        Market Name: 		 AMD Radeon RX 7900 XTX
        TARGET_GRAPHICS_VERSION: gfx1100
EOF
# The Instinct OAM SKUs put ": " inside the name, which -F'[:|]' truncated.
cat > "$WORK/smi_colon" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Instinct MI300X OAM: 750W SKU
        TARGET_GRAPHICS_VERSION: gfx942
EOF
# Three adapters: the arch followed the mask, the name was always adapter 0's.
cat > "$WORK/smi_three" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Instinct MI210
        VENDOR_ID: 0x1002
        TARGET_GRAPHICS_VERSION: gfx90a
GPU: 1
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
        VENDOR_ID: 0x1002
        TARGET_GRAPHICS_VERSION: gfx1100
GPU: 2
    ASIC:
        MARKET_NAME: AMD Radeon AI PRO R9700
        VENDOR_ID: 0x1002
        TARGET_GRAPHICS_VERSION: gfx1201
EOF
# amd-smi 6.1.1 has MARKET_NAME and no TARGET_GRAPHICS_VERSION, so the name picks the wheel.
cat > "$WORK/smi_two_nogfx" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
        VENDOR_ID: 0x1002
GPU: 1
    ASIC:
        MARKET_NAME: AMD Radeon AI PRO R9700
        VENDOR_ID: 0x1002
EOF
# The #7307 shape: a Ryzen iGPU plus two IDENTICAL R9700s. The two dGPU records are
# byte-identical, so a selector that folds duplicates resolves device 2 to the iGPU.
cat > "$WORK/roc_twins" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 9950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 9950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
*******
Agent 4
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
EOF
# Three devices, two sharing an arch: the ordinal past the duplicate is the one the
# old deduplicating selector could not reach.
cat > "$WORK/roc_three_dup" <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Marketing Name:          AMD EPYC 9654 96-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx90a:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI210
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
*******
Agent 4
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon PRO W7900
  Device Type:             GPU
EOF
: > "$WORK/empty"

echo "=== rocminfo enumerates the device ==="
assert_eq "its arch and its own name are used" \
    "gfx1151|AMD Radeon Graphics" "$(summary "$WORK/roc_gpu" "$WORK/smi_fixture")"
assert_eq "and amd-smi is not consulted for the name" \
    "gfx1151|AMD Radeon Graphics" "$(summary "$WORK/roc_gpu" "$WORK/empty")"

echo "=== rocminfo names something but enumerates no device ==="
# The CPU-only record must not survive to beat the arch amd-smi found.
assert_eq "amd-smi supplies the arch" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_cpu_only" "$WORK/smi_fixture")"
assert_eq "with no amd-smi, the APU fallback name still shows" \
    "|AMD Ryzen 9 5950X 16-Core Processor" "$(summary "$WORK/roc_cpu_only" "$WORK/empty")"

echo "=== a device that reported no name of its own ==="
# Borrowing amd-smi's unindexed first row would label this card as another one.
assert_eq "keeps its arch and stays unnamed" \
    "gfx1030|" "$(summary "$WORK/roc_blank_name" "$WORK/smi_fixture")"

# amd-smi discovery order is not HIP order; `amd-smi list -e` publishes HIP_ID as the map.
# Discovery 0/1/2 is HIP 2/1/0 here, so an untranslated mask names the wrong card.
cat > "$WORK/smi_e_reversed" <<'EOF'
GPU: 0
    HIP_ID: 2
GPU: 1
    HIP_ID: 1
GPU: 2
    HIP_ID: 0
EOF
# A partial map (hip_id reads N/A when a KFD node is unreachable) is not a mapping.
cat > "$WORK/smi_e_partial" <<'EOF'
GPU: 0
    HIP_ID: 2
GPU: 1
    HIP_ID: N/A
GPU: 2
    HIP_ID: 0
EOF

cat > "$WORK/smi_e_identity" <<'EOF'
GPU: 0
    HIP_ID: 0
GPU: 1
    HIP_ID: 1
GPU: 2
    HIP_ID: 2
EOF
# Two of the same card: every ordinal yields the same arch, so there is nothing to decline.
cat > "$WORK/smi_two_same" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Instinct MI300X
        TARGET_GRAPHICS_VERSION: gfx942
GPU: 1
    ASIC:
        MARKET_NAME: AMD Instinct MI300X
        TARGET_GRAPHICS_VERSION: gfx942
EOF

echo "=== amd-smi only ==="
# The arch comes from `static --asic`: `amd-smi list` carries no gfx token at all.
assert_eq "arch and name both come from amd-smi" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_fixture")"
assert_eq "an older title-cased Market Name still works" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_titlecase")"
assert_eq "an amd-smi name survives a colon in the middle" \
    "gfx942|AMD Instinct MI300X OAM: 750W SKU" "$(summary "$WORK/empty" "$WORK/smi_colon")"
assert_eq "each adapter is announced with its own name, device 0" \
    "gfx90a|AMD Instinct MI210" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "device 1" "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 1)"
assert_eq "device 2" "gfx1201|AMD Radeon AI PRO R9700" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 2)"
assert_eq "an out-of-range mask falls back to adapter 0" \
    "gfx90a|AMD Instinct MI210" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 9)"
# No gfx token anywhere: the name is all there is, so it must be the selected card's.
assert_eq "a nameless-arch build still names the selected adapter" \
    "|AMD Radeon AI PRO R9700" "$(summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"
assert_eq "neither tool installed reports nothing" "|" "$(summary - -)"
assert_eq "both installed but silent reports nothing" "|" "$(summary "$WORK/empty" "$WORK/empty")"

echo "=== the mask selects a device, and its name follows ==="
assert_eq "device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_two_gpus" "$WORK/empty" 0)"
assert_eq "device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_two_gpus" "$WORK/empty" 1)"
# Driven through the real block, not a restated selector: a deduplicating selector
# resolves both gfx1100 slots to one entry and sends device 2 back to device 0.
assert_eq "an ordinal past a duplicated arch still selects its own device" \
    "gfx1100|AMD Radeon PRO W7900" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 2)"
assert_eq "and the duplicate before it keeps its own name" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 1)"
# The second identical card must still be a device, not a line to fold away.
assert_eq "two identical cards are still two devices" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(summary "$WORK/roc_twins" "$WORK/empty" 2)"

echo "=== amd-smi ordinals are translated into HIP order ==="
assert_eq "HIP 0 is discovery 2" "gfx1201|AMD Radeon AI PRO R9700" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "HIP 2 is discovery 0" "gfx90a|AMD Instinct MI210" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/empty" "$WORK/smi_three" 2)"
# No usable map and the adapters disagree on arch: report nothing rather than name one
# card while the mask selects another.
assert_eq "an older CLI that rejects -e declines on unlike adapters" "|" \
    "$(summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "a partial map is declined, not half-applied" "|" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_partial" summary "$WORK/empty" "$WORK/smi_three" 0)"
# Identical adapters, and a single adapter, are never ambiguous.
assert_eq "identical adapters still resolve without a map" \
    "gfx942|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same" 1)"
assert_eq "one adapter resolves without a map" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_fixture" 0)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
