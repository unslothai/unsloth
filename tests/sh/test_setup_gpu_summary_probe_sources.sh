#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# $_setup_gfx is forwarded as --rocm-gfx to the llama and whisper installers, so a wrong
# ordinal picks the wrong binary. Drives the real block.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/_harness.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
SKIP=0


{
    # Name an extraction that fell behind setup.sh the first time a missing helper is actually
    # reached; helpers on arms this test never takes are not defects.
    cat <<'GUARD'
command_not_found_handle() {
    printf '%s\n' "FATAL: the extracted block called '$1', which studio/setup.sh defines" >&2
    printf '%s\n' "       but no sed line in this test pulled in. Add:" >&2
    printf '%s\n' "         sed -n '/^$1()/,/^}/p' \"\$SETUP_SH\"" >&2
    exit 127
}
GUARD
    sed -n '/^_setup_run_smi()/,/^}/p'              "$SETUP_SH"
    sed -n '/^_setup_rocminfo_gpu_records()/,/^}/p' "$SETUP_SH"
    sed -n '/^_setup_amd_smi_gpu_records()/,/^}/p'  "$SETUP_SH"
    sed -n '/^_setup_amd_smi_hip_order()/,/^}/p'  "$SETUP_SH"
    # Called from the selection block below, so they must come with it.
    sed -n '/^_amd_gfx_is_shadowing_integrated()/,/^}/p'  "$SETUP_SH"
    sed -n '/^_amd_prefer_discrete_gfx()/,/^}/p'  "$SETUP_SH"
    # The real initialiser group: under set -u a variable no arm assigns aborts the update.
    sed -n '/^_setup_amd_detected=false$/,/^_setup_amd_records=""$/p' "$SETUP_SH"
    # NVIDIA is pinned false: this is about which AMD device gets picked.
    awk '/^if \[ "\$_setup_nvidia_usable" != true \]; then/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$SETUP_SH"
    echo 'fi'
} > "$WORK/block.sh"
grep -q '_setup_amd_record=' "$WORK/block.sh" || {
    echo "FATAL: AMD detection/selection block not found in $SETUP_SH" >&2; exit 1; }
grep -q '_setup_amd_smi_gpu_records' "$WORK/block.sh" || {
    echo "FATAL: the amd-smi record parser is not wired into the block" >&2; exit 1; }
grep -q '_setup_amd_smi_hip_order' "$WORK/block.sh" || {
    echo "FATAL: the amd-smi HIP reorder is not wired into the block" >&2; exit 1; }
grep -q '^_amd_prefer_discrete_gfx()' "$WORK/block.sh" || {
    echo "FATAL: the discrete-GPU preference helper is not in the block" >&2; exit 1; }
grep -q '^_amd_gfx_is_shadowing_integrated()' "$WORK/block.sh" || {
    echo "FATAL: the integrated-GPU test the preference reads is not in the block" >&2
    exit 1; }
bash -n "$WORK/block.sh" || { echo "FATAL: extracted block does not parse" >&2; exit 1; }

sed -n '/^_setup_amd_detected=false$/,/^_setup_amd_records=""$/p' "$SETUP_SH" > "$WORK/init.sh"
{
    # Same missing-helper guard as above, for the split halves.
    cat <<'GUARD'
command_not_found_handle() {
    printf '%s\n' "FATAL: the extracted block called '$1', which studio/setup.sh defines" >&2
    printf '%s\n' "       but no sed line in this test pulled in. Add:" >&2
    printf '%s\n' "         sed -n '/^$1()/,/^}/p' \"\$SETUP_SH\"" >&2
    exit 127
}
GUARD
    sed -n '/^_amd_gfx_is_shadowing_integrated()/,/^}/p'  "$SETUP_SH"
    sed -n '/^_amd_prefer_discrete_gfx()/,/^}/p'  "$SETUP_SH"
    awk '/^if \[ "\$_setup_nvidia_usable" = true \]; then/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$SETUP_SH"
    echo 'fi'
} > "$WORK/select.sh"
grep -q '_setup_nvidia_usable=false' "$WORK/init.sh" || {
    echo "FATAL: initialiser group not found in $SETUP_SH" >&2; exit 1; }
grep -q '_setup_amd_record=' "$WORK/select.sh" || {
    echo "FATAL: selection block not found in $SETUP_SH" >&2; exit 1; }
grep -q '^_amd_prefer_discrete_gfx()' "$WORK/select.sh" || {
    echo "FATAL: the discrete-GPU preference helper is not in the split half" >&2; exit 1; }
bash -n "$WORK/init.sh" && bash -n "$WORK/select.sh" || {
    echo "FATAL: extracted halves do not parse" >&2; exit 1; }

# PATH from scratch: the host may have a real rocminfo/amd-smi.
mkdir -p "$WORK/base" "$WORK/roc" "$WORK/smi"
for _tool in awk grep sed cat tr timeout sort wc head tail; do
    _p=$(command -v "$_tool") || { echo "FATAL: $_tool not found" >&2; exit 1; }
    ln -sf "$_p" "$WORK/base/$_tool"
done
cat > "$WORK/roc/rocminfo" <<'STUB'
#!/bin/sh
echo "rocminfo" >> "$PROBE_LOG"
[ -s "$STUB_ROCMINFO" ] || exit 1
cat "$STUB_ROCMINFO"
STUB
# `amd-smi list` has no gfx token. `list -e` carries HIP_ID; an older CLI rejects -e
# (STUB_AMDSMI_E="").
cat > "$WORK/smi/amd-smi" <<'STUB'
#!/bin/sh
echo "amd-smi $*" >> "$PROBE_LOG"
[ -s "$STUB_AMDSMI" ] || exit 1
case "$1 $2" in
    "list -e") [ -z "${STUB_AMDSMI_E:-}" ] || cat "$STUB_AMDSMI_E" ;;
    "list "*)  sed -n 's/^\(GPU: [0-9]*\).*/\1  BDF: 0000:03:00.0  UUID: aaaa-bbbb  KFD_ID: 1/p' "$STUB_AMDSMI" ;;
    # A driver that answers `list` but not `static --asic`: detected, zero records.
    "static "*) [ -n "${STUB_AMDSMI_MUTE_STATIC:-}" ] || cat "$STUB_AMDSMI" ;;
esac
STUB
chmod +x "$WORK/roc/rocminfo" "$WORK/smi/amd-smi"

# $1 rocminfo fixture ("-" = not installed), $2 amd-smi fixture, $3 mask. Prints "gfx|name";
# the probe log stays in $WORK/probes.
summary() {
    _path="${PREPATH:+$PREPATH:}$WORK/base"
    [ "$1" != "-" ] && _path="$WORK/roc:$_path"
    [ "$2" != "-" ] && _path="$WORK/smi:$_path"
    : > "$WORK/probes"
    env -i PATH="$_path" PROBE_LOG="$WORK/probes" \
        STUB_ROCMINFO="$1" STUB_AMDSMI="$2" \
        ${STUB_AMDSMI_MUTE_STATIC:+STUB_AMDSMI_MUTE_STATIC=1} \
        ${STUB_AMDSMI_E:+STUB_AMDSMI_E="$STUB_AMDSMI_E"} \
        ${3:+HIP_VISIBLE_DEVICES="$3"} \
        ${STUB_ROCR:+ROCR_VISIBLE_DEVICES="$STUB_ROCR"} \
        ${STUB_CUDA:+CUDA_VISIBLE_DEVICES="$STUB_CUDA"} \
        ${STUB_HIP_EMPTY:+HIP_VISIBLE_DEVICES=} \
        /bin/bash -c 'set -euo pipefail; . "$1"; printf "%s|%s\n" "$_setup_gfx" "$_setup_mkt"' \
        _ "$WORK/block.sh"
}

# The KFD sysfs arm needs a real /dev/kfd and sets only _setup_amd_detected, so drive the
# selection block directly from that state.
kfd_shape_summary() {
    env -i PATH="$WORK/base" \
        /bin/bash -c 'set -euo pipefail
                      step() { :; }
                      . "$1"
                      _setup_amd_detected=true
                      . "$2"
                      printf "%s|%s\n" "$_setup_gfx" "$_setup_mkt"' \
        _ "$WORK/init.sh" "$WORK/select.sh" 2>&1
}
# grep -c exits 1 on no match. Anchored so `amd-smi list` does not count `list -e`.
probe_count() { _n=$(grep -c "^$1\$" "$WORK/probes" 2>/dev/null) || true; echo "${_n:-0}"; }
probe_prefix_count() { _n=$(grep -c "^$1" "$WORK/probes" 2>/dev/null) || true; echo "${_n:-0}"; }

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
# Ryzen iGPU plus two identical R9700s.
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
# amd-smi 6.1.1 has no TARGET_GRAPHICS_VERSION, so --rocm-gfx follows the name.
cat > "$WORK/smi_e_two_identity" <<'EOF'
GPU: 0
    HIP_ID: 0
GPU: 1
    HIP_ID: 1
EOF
cat > "$WORK/smi_two_same_nogfx" <<'EOF'
GPU: 0
    ASIC:
        MARKET_NAME: AMD Instinct MI300X
GPU: 1
    ASIC:
        MARKET_NAME: AMD Instinct MI300X
EOF
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
assert_eq "and amd-smi is not consulted at all" 0 "$(probe_prefix_count amd-smi)"
assert_eq "the same answer with amd-smi installed and disagreeing" \
    "gfx1151|AMD Radeon Graphics" "$(summary "$WORK/roc_gpu" "$WORK/smi_fixture")"

echo "=== the mask selects a device, and its name follows ==="
assert_eq "device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 0)"
assert_eq "device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 1)"
assert_eq "device 2, past a duplicated arch, keeps its own arch and name" \
    "gfx1100|AMD Radeon PRO W7900" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 2)"
assert_eq "an out-of-range mask falls back to device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 9)"
# Folding byte-identical records sends device 2 to the iGPU's gfx1036.
assert_eq "two identical cards are still two devices" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(summary "$WORK/roc_twins" "$WORK/empty" 2)"

echo "=== rocminfo names something but enumerates no device ==="
# amd-smi owns the device list, so the CPU-only rocminfo record must be dropped.
assert_eq "amd-smi supplies both the arch and the name" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_cpu_only" "$WORK/smi_fixture")"
# Probe counts are pinned: they move if an arm is reordered or the parse split.
assert_eq "list is asked first, and carries no gfx" 2 "$(probe_count 'amd-smi list')"
assert_eq "and list -e is asked once, for the HIP mapping" \
    1 "$(probe_count 'amd-smi list -e')"
assert_eq "one static --asic parse supplies both the arch and the name" \
    1 "$(probe_count 'amd-smi static --asic')"

echo "=== a device that reported no name of its own ==="
assert_eq "keeps its arch and stays unnamed" \
    "gfx1030|" "$(summary "$WORK/roc_blank_name" "$WORK/smi_fixture")"

# amd-smi discovery order is not HIP order; `amd-smi list -e` publishes HIP_ID as the map.
cat > "$WORK/smi_e_reversed" <<'EOF'
GPU: 0
    HIP_ID: 2
GPU: 1
    HIP_ID: 1
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
# hip_id reads N/A when a KFD node is unreachable; a partial map keeps discovery order.
cat > "$WORK/smi_e_partial" <<'EOF'
GPU: 0
    HIP_ID: 2
GPU: 1
    HIP_ID: N/A
GPU: 2
    HIP_ID: 0
EOF
cat > "$WORK/smi_e_collide" <<'EOF'
GPU: 0
    HIP_ID: 1
GPU: 1
    HIP_ID: 1
GPU: 2
    HIP_ID: 0
EOF

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

echo "=== amd-smi owns the device list ==="
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
# With no gfx token the name feeds --rocm-gfx, so it must be the selected adapter's.
assert_eq "a nameless-arch build still names the selected adapter" \
    "|AMD Radeon AI PRO R9700" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_two_identity" summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"
assert_eq "and declines without a map, since the name is what the arch comes from" "|" \
    "$(summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"
assert_eq "two archless adapters of the same model are not ambiguous" \
    "|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same_nogfx" 1)"

echo "=== neither tool reports a device ==="
# The KFD arm would fire on a host with a real AMD GPU.
if [ -e /dev/kfd ]; then
    echo "  SKIP: /dev/kfd exists on this host, so the KFD arm is reachable"; SKIP=$((SKIP + 1))
else
    assert_eq "neither tool installed reports nothing" "|" "$(summary - -)"
    assert_eq "both installed but silent reports nothing" "|" "$(summary "$WORK/empty" "$WORK/empty")"
fi

echo "=== amd-smi ordinals are translated into HIP order ==="
assert_eq "HIP 0 is discovery 2" "gfx1201|AMD Radeon AI PRO R9700" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "HIP 2 is discovery 0" "gfx90a|AMD Instinct MI210" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/empty" "$WORK/smi_three" 2)"
assert_eq "the middle device is unmoved" "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/empty" "$WORK/smi_three" 1)"
assert_eq "an identity map changes nothing" "gfx90a|AMD Instinct MI210" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 0)"
# No usable map and unlike adapters: any ordinal would be a guess, so report nothing.
assert_eq "an older CLI that rejects -e declines on unlike adapters" "|" \
    "$(summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "a partial map is declined, not half-applied" "|" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_partial" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "colliding hip ids are declined" "|" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_collide" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "identical adapters still resolve without a map, device 0" \
    "gfx942|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same" 0)"
assert_eq "and device 1" \
    "gfx942|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same" 1)"
assert_eq "one adapter resolves without a map" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_fixture" 0)"
# rocminfo is an HSA client, so its agent list is already in ROCr order.
assert_eq "the rocminfo path is not reordered by a HIP map" "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/roc_three_dup" "$WORK/empty" 1)"

# ROCr already filtered and renumbered these.
cat > "$WORK/roc_rocr_1_0" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1036
  Marketing Name:          AMD Radeon Graphics
  Device Type:             GPU
EOF
cat > "$WORK/roc_rocr_1_2" <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 7950X 16-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1201
  Marketing Name:          AMD Radeon AI PRO R9700
  Device Type:             GPU
EOF
echo "=== ROCR_VISIBLE_DEVICES over rocminfo ==="
assert_eq "ROCR=1,0 selects the first rocminfo survivor, not the iGPU at ordinal 1" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(STUB_ROCR=1,0 summary "$WORK/roc_rocr_1_0" "$WORK/empty")"
assert_eq "ROCR=1,2 on three cards selects survivor 0 (physical 1)" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(STUB_ROCR=1,2 summary "$WORK/roc_rocr_1_2" "$WORK/empty")"
assert_eq "HIP still indexes the ROCr survivors" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(STUB_ROCR=1,2 summary "$WORK/roc_rocr_1_2" "$WORK/empty" 1)"
assert_eq "CUDA_VISIBLE_DEVICES, HIP's alias, also indexes the ROCr survivors" \
    "gfx1036|AMD Radeon Graphics" "$(STUB_ROCR=1,0 STUB_CUDA=1 summary "$WORK/roc_rocr_1_0" "$WORK/empty")"
assert_eq "a set-but-empty HIP mask shadows CUDA, as install.sh" \
    "gfx1201|AMD Radeon AI PRO R9700" \
    "$(STUB_ROCR=1,0 STUB_CUDA=1 STUB_HIP_EMPTY=1 summary "$WORK/roc_rocr_1_0" "$WORK/empty")"
assert_eq "amd-smi is not ROCr-filtered, so its list is still indexed by the ROCr ordinal" \
    "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_ROCR=1 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three")"
assert_eq "amd-smi: HIP=1 under ROCR=2,0 selects survivor 1 (card 0)" \
    "gfx90a|AMD Instinct MI210" \
    "$(STUB_ROCR=2,0 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 1)"
assert_eq "amd-smi: CUDA_VISIBLE_DEVICES, HIP's alias, indexes the list" \
    "gfx1201|AMD Radeon AI PRO R9700" \
    "$(STUB_CUDA=2 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three")"
assert_eq "amd-smi: an empty HIP mask still leaves ROCR=1's survivor" \
    "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_ROCR=1 STUB_HIP_EMPTY=1 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three")"
assert_eq "amd-smi: a repeated ROCr ordinal ends the survivors (ROCR=0,0,1 leaves card 0 only)" \
    "gfx90a|AMD Instinct MI210" \
    "$(STUB_ROCR=0,0,1 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 2)"
assert_eq "amd-smi: an out-of-range ROCr ordinal ends the survivors (ROCR=0,99,1)" \
    "gfx90a|AMD Instinct MI210" \
    "$(STUB_ROCR=0,99,1 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three" 1)"
assert_eq "amd-smi: a UUID in ROCR over unlike adapters declines instead of guessing" \
    "|" \
    "$(STUB_ROCR=GPU-4b2c1a9f8d3e6f7a,1 STUB_AMDSMI_E="$WORK/smi_e_identity" summary "$WORK/empty" "$WORK/smi_three")"

echo "=== detected, but no arm produced a record ==="
# Under set -u an unassigned variable is fatal, killing `unsloth studio update`.
assert_eq "the KFD-shaped path reaches the end instead of aborting on set -u" \
    "|" "$(kfd_shape_summary)"
assert_eq "every variable the selection block reads is initialised up front" \
    "" "$(grep -oE '\$\{?_setup_(gfx|gfx_all|mkt|amd_records|amd_detected|amd_probe|rocr_uuid_declined|nvidia_usable)\b' \
              "$WORK/select.sh" | tr -d '${' | sort -u \
          | while read -r _v; do grep -q "^$_v=" "$WORK/init.sh" || echo "$_v"; done | tr '\n' ' ' | sed 's/ $//')"
assert_eq "amd-smi answers list but not static --asic" \
    "|" "$(STUB_AMDSMI_MUTE_STATIC=1 summary "$WORK/empty" "$WORK/smi_three")"

echo "=== the index-space line is read without SIGPIPE ==="
mkdir -p "$WORK/head1"
cat > "$WORK/head1/head" <<'STUB'
#!/bin/sh
IFS= read -r _l && printf '%s\n' "$_l"
STUB
chmod +x "$WORK/head1/head"
# Doubling, not sprintf("%200000s"): mawk 1.3.4 caps sprintf at 8192 bytes.
awk '/MARKET_NAME: AMD Radeon RX 7900 XTX/ { s = "X"; while (length(s) < 200000) s = s s; sub(/AMD Radeon RX 7900 XTX/, substr(s, 1, 200000)) } { print }' \
    "$WORK/smi_three" > "$WORK/smi_three_long"
assert_eq "an amd-smi answer larger than a pipe does not abort the block" \
    "gfx1100|200000" \
    "$(PREPATH="$WORK/head1" STUB_AMDSMI_E="$WORK/smi_e_reversed" \
        summary "$WORK/empty" "$WORK/smi_three_long" 1 2>/dev/null \
        | awk -F'|' '{ print $1 "|" length($2) }')"

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
