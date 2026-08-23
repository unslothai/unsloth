#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Drive studio/setup.sh's AMD detection and selection with stub rocminfo and amd-smi.
#
# The sibling test_install_gpu_summary_probe_sources.sh covers install.sh, where the
# result is a display label. Here it is not: $_setup_gfx is forwarded as --rocm-gfx to
# install_llama_prebuilt.py and to the whisper installer, so a wrong ordinal picks the
# wrong binary. This file drives the real block rather than the parser alone.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
PASS=0
FAIL=0
SKIP=0

{
    sed -n '/^_setup_run_smi()/,/^}/p'              "$SETUP_SH"
    sed -n '/^_setup_rocminfo_gpu_records()/,/^}/p' "$SETUP_SH"
    sed -n '/^_setup_amd_smi_gpu_records()/,/^}/p'  "$SETUP_SH"
    echo '_setup_amd_detected=false'
    echo '_setup_nvidia_usable=false'
    echo '_setup_gfx_all=""; _setup_mkt=""; _setup_amd_records=""; _setup_gfx=""'
    # Detection through selection. NVIDIA is classified above this point and pinned
    # false here: this file is about which AMD device gets picked, not about priority.
    awk '/^if \[ "\$_setup_nvidia_usable" != true \]; then/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$SETUP_SH"
    echo 'fi'
} > "$WORK/block.sh"
grep -q '_setup_amd_record=' "$WORK/block.sh" || {
    echo "FATAL: AMD detection/selection block not found in $SETUP_SH" >&2; exit 1; }
grep -q '_setup_amd_smi_gpu_records' "$WORK/block.sh" || {
    echo "FATAL: the amd-smi record parser is not wired into the block" >&2; exit 1; }
bash -n "$WORK/block.sh" || { echo "FATAL: extracted block does not parse" >&2; exit 1; }

# PATH from scratch: the host running this may itself have a real rocminfo/amd-smi.
mkdir -p "$WORK/base" "$WORK/roc" "$WORK/smi"
for _tool in awk grep sed cat tr timeout; do
    _p=$(command -v "$_tool") || { echo "FATAL: $_tool not found" >&2; exit 1; }
    ln -sf "$_p" "$WORK/base/$_tool"
done
cat > "$WORK/roc/rocminfo" <<'STUB'
#!/bin/sh
echo "rocminfo" >> "$PROBE_LOG"
[ -s "$STUB_ROCMINFO" ] || exit 1
cat "$STUB_ROCMINFO"
STUB
# `amd-smi list` prints BDF / UUID / KFD_ID and no gfx token, so the arch has to come
# from `static --asic`; keeping the two subcommands distinct is what tests that.
cat > "$WORK/smi/amd-smi" <<'STUB'
#!/bin/sh
echo "amd-smi $1" >> "$PROBE_LOG"
[ -s "$STUB_AMDSMI" ] || exit 1
case "$1" in
    list)   sed -n 's/^\(GPU: [0-9]*\).*/\1  BDF: 0000:03:00.0  UUID: aaaa-bbbb  KFD_ID: 1/p' "$STUB_AMDSMI" ;;
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

# $1 rocminfo fixture ("-" = not installed), $2 amd-smi fixture, $3 visible-device mask.
# Prints "gfx|name". The probe log is left in $WORK/probes for the call-count asserts.
summary() {
    _path="$WORK/base"
    [ "$1" != "-" ] && _path="$WORK/roc:$_path"
    [ "$2" != "-" ] && _path="$WORK/smi:$_path"
    : > "$WORK/probes"
    # setup.sh runs under `set -euo pipefail`; match it so a guard that would abort
    # `unsloth studio update` cannot pass here.
    env -i PATH="$_path" PROBE_LOG="$WORK/probes" \
        STUB_ROCMINFO="$1" STUB_AMDSMI="$2" \
        ${3:+HIP_VISIBLE_DEVICES="$3"} \
        /bin/bash -c 'set -euo pipefail; . "$1"; printf "%s|%s\n" "$_setup_gfx" "$_setup_mkt"' \
        _ "$WORK/block.sh"
}
# grep -c prints 0 and exits 1 when there is no match, so the status is discarded.
probe_count() { _n=$(grep -c "^$1" "$WORK/probes" 2>/dev/null) || true; echo "${_n:-0}"; }

cat > "$WORK/roc_gpu" <<'EOF'
Agent 1
*******
  Name:                    AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
  Marketing Name:          AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
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
# The #7307 reporter's shape: a Ryzen iGPU plus two IDENTICAL R9700s.
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
cat > "$WORK/smi_fixture" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
        TARGET_GRAPHICS_VERSION: gfx1100
EOF
# Three adapters: the arch was indexed by the mask while the name was always the first
# adapter's, so a later card was announced with another card's name.
cat > "$WORK/smi_three" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Instinct MI210
        TARGET_GRAPHICS_VERSION: gfx90a
GPU: 1
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
        TARGET_GRAPHICS_VERSION: gfx1100
GPU: 2
    ASIC:
        MARKET_NAME: AMD Radeon AI PRO R9700
        TARGET_GRAPHICS_VERSION: gfx1201
EOF
# amd-smi 6.1.1 ships MARKET_NAME with no TARGET_GRAPHICS_VERSION, so the arch is
# inferred from the name and $_setup_gfx -- hence --rocm-gfx -- follows the name.
cat > "$WORK/smi_two_nogfx" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Radeon RX 7900 XTX
GPU: 1
    ASIC:
        MARKET_NAME: AMD Radeon AI PRO R9700
EOF
: > "$WORK/empty"

echo "=== rocminfo enumerates the device ==="
assert_eq "the GPU agent names the GPU, not the processor" \
    "gfx1151|AMD Radeon Graphics" "$(summary "$WORK/roc_gpu" "$WORK/empty")"
assert_eq "rocminfo is run once, not once per field" 1 "$(probe_count rocminfo)"
assert_eq "and amd-smi is not consulted at all" 0 "$(probe_count amd-smi)"
assert_eq "the same answer with amd-smi installed and disagreeing" \
    "gfx1151|AMD Radeon Graphics" "$(summary "$WORK/roc_gpu" "$WORK/smi_fixture")"

echo "=== the mask selects a device, and its name follows ==="
assert_eq "device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 0)"
assert_eq "device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 1)"
# This is the ordinal that decides --rocm-gfx on a duplicate-arch host: a selector
# that collapses the two gfx1100 slots sends device 2 back to device 0's arch.
assert_eq "device 2, past a duplicated arch, keeps its own arch and name" \
    "gfx1100|AMD Radeon PRO W7900" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 2)"
assert_eq "an out-of-range mask falls back to device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 9)"
# Two of the same card produce two byte-identical records. A selector that folds
# duplicate lines away resolves device 2 to the iGPU at ordinal 0, and --rocm-gfx
# then carries gfx1036 for a host whose selected card is gfx1201.
assert_eq "two identical cards are still two devices" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(summary "$WORK/roc_twins" "$WORK/empty" 2)"

echo "=== rocminfo names something but enumerates no device ==="
# amd-smi owns the device list here, so the leftover CPU-only record must be dropped
# rather than win the selection and discard amd-smi's arch.
assert_eq "amd-smi supplies both the arch and the name" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_cpu_only" "$WORK/smi_fixture")"
# `list` runs twice: once to detect the GPU, once to look for a gfx token it does not
# carry. `static --asic` runs ONCE and yields both fields, because they now come from
# one record parse instead of two independent greps. Pinned rather than tidied: this
# file is a regression net, and the counts are what would change if an arm were
# reordered or the parse were split again.
assert_eq "list is asked first, and carries no gfx" 2 "$(probe_count 'amd-smi list')"
assert_eq "one static --asic parse supplies both the arch and the name" \
    1 "$(probe_count 'amd-smi static')"

echo "=== a device that reported no name of its own ==="
assert_eq "keeps its arch and stays unnamed" \
    "gfx1030|" "$(summary "$WORK/roc_blank_name" "$WORK/smi_fixture")"

echo "=== amd-smi owns the device list ==="
assert_eq "each adapter is announced with its own name, device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_three" 1)"
assert_eq "device 2" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(summary "$WORK/empty" "$WORK/smi_three" 2)"
assert_eq "an out-of-range mask falls back to adapter 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/empty" "$WORK/smi_three" 9)"
# No gfx token anywhere, so the name is what --rocm-gfx is inferred from: it has to be
# the selected adapter's, not the first one's.
assert_eq "a nameless-arch build still names the selected adapter" \
    "|AMD Radeon AI PRO R9700" "$(summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"

echo "=== neither tool reports a device ==="
# The KFD sysfs arm below these two would also fire on a host that really has an AMD
# GPU, which is where this suite is most likely to be run by hand.
if [ -e /dev/kfd ]; then
    echo "  SKIP: /dev/kfd exists on this host, so the KFD arm is reachable"; SKIP=$((SKIP + 1))
else
    assert_eq "neither tool installed reports nothing" "|" "$(summary - -)"
    assert_eq "both installed but silent reports nothing" "|" "$(summary "$WORK/empty" "$WORK/empty")"
fi

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
