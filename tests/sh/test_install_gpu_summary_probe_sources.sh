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
    echo ""
    # Extract the probe and selection block.
    awk '/^    _gpu_disp_gfx_all=""/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$INSTALL_SH"
} > "$WORK/block.sh"
grep -q '_gpu_disp_record=' "$WORK/block.sh" || {
    echo "FATAL: GPU summary probe block not found in $INSTALL_SH" >&2; exit 1; }

# Build PATH from scratch so host AMD tools cannot leak into the test.
mkdir -p "$WORK/base" "$WORK/roc" "$WORK/smi"
for _tool in awk grep sed cat tr; do
    _p=$(command -v "$_tool") || { echo "FATAL: $_tool not found" >&2; exit 1; }
    ln -sf "$_p" "$WORK/base/$_tool"
done
cat > "$WORK/roc/rocminfo" <<'STUB'
#!/bin/sh
[ -s "$STUB_ROCMINFO" ] || exit 1
cat "$STUB_ROCMINFO"
STUB
cat > "$WORK/smi/amd-smi" <<'STUB'
#!/bin/sh
[ -s "$STUB_AMDSMI" ] || exit 1
case "$1" in
    list)   sed -n '/^GPU/p;/TARGET_GRAPHICS_VERSION/p' "$STUB_AMDSMI" ;;
    static) cat "$STUB_AMDSMI" ;;
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
    env -i PATH="$_path" \
        STUB_ROCMINFO="$1" STUB_AMDSMI="$2" \
        ${3:+HIP_VISIBLE_DEVICES="$3"} \
        /bin/bash -c '. "$1"; printf "%s|%s\n" "$_gpu_disp_gfx" "$_gpu_disp_mkt"' _ "$WORK/block.sh"
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
# "Market Name" with a space is the spelling install.sh's amd-smi parser matches.
cat > "$WORK/smi_fixture" <<'EOF'
GPU: 0
    ASIC:
        Market Name: 		 AMD Radeon RX 7900 XTX
        TARGET_GRAPHICS_VERSION: gfx1100
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

echo "=== amd-smi only ==="
assert_eq "arch and name both come from amd-smi" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_fixture")"
assert_eq "neither tool installed reports nothing" "|" "$(summary - -)"
assert_eq "both installed but silent reports nothing" "|" "$(summary "$WORK/empty" "$WORK/empty")"

echo "=== the mask selects a device, and its name follows ==="
assert_eq "device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_two_gpus" "$WORK/empty" 0)"
assert_eq "device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_two_gpus" "$WORK/empty" 1)"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
