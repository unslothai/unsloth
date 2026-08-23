#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Drive studio/setup.sh's AMD detection and selection with stub rocminfo and amd-smi.
#
# install.sh's value is a display label; this one is not. $_setup_gfx is forwarded as
# --rocm-gfx to install_llama_prebuilt.py and the whisper installer, so a wrong ordinal
# picks the wrong binary. Drives the real block, not the parser alone.
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
    sed -n '/^_setup_amd_smi_hip_order()/,/^}/p'  "$SETUP_SH"
    # The real initialiser group, not a restated one. Seeding these here would hide the
    # thing that matters under `set -u`: a variable the selection block reads but no arm
    # assigns aborts `unsloth studio update` outright.
    sed -n '/^_setup_amd_detected=false$/,/^_setup_amd_records=""$/p' "$SETUP_SH"
    # Detection through selection. NVIDIA is pinned false: this is about which AMD
    # device gets picked, not about vendor priority.
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
bash -n "$WORK/block.sh" || { echo "FATAL: extracted block does not parse" >&2; exit 1; }

# The same two halves, split, for the arms that reach selection without probing anything.
sed -n '/^_setup_amd_detected=false$/,/^_setup_amd_records=""$/p' "$SETUP_SH" > "$WORK/init.sh"
{
    awk '/^if \[ "\$_setup_nvidia_usable" = true \]; then/ {on=1}
         on && /UNSLOTH_ROCM_GFX_ARCH env override/ {exit}
         on {print}' "$SETUP_SH"
    echo 'fi'
} > "$WORK/select.sh"
grep -q '_setup_nvidia_usable=false' "$WORK/init.sh" || {
    echo "FATAL: initialiser group not found in $SETUP_SH" >&2; exit 1; }
grep -q '_setup_amd_record=' "$WORK/select.sh" || {
    echo "FATAL: selection block not found in $SETUP_SH" >&2; exit 1; }
bash -n "$WORK/init.sh" && bash -n "$WORK/select.sh" || {
    echo "FATAL: extracted halves do not parse" >&2; exit 1; }

# PATH from scratch: the host running this may itself have a real rocminfo/amd-smi.
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
# `amd-smi list` carries no gfx token; keeping the subcommands distinct tests that.
# `list -e` is a third answer again: it carries HIP_ID and nothing else of interest, and
# an older CLI rejects the flag outright, which is the STUB_AMDSMI_E="" case.
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
    # setup.sh runs under `set -euo pipefail`; match it.
    env -i PATH="$_path" PROBE_LOG="$WORK/probes" \
        STUB_ROCMINFO="$1" STUB_AMDSMI="$2" \
        ${STUB_AMDSMI_MUTE_STATIC:+STUB_AMDSMI_MUTE_STATIC=1} \
        ${STUB_AMDSMI_E:+STUB_AMDSMI_E="$STUB_AMDSMI_E"} \
        ${3:+HIP_VISIBLE_DEVICES="$3"} \
        /bin/bash -c 'set -euo pipefail; . "$1"; printf "%s|%s\n" "$_setup_gfx" "$_setup_mkt"' \
        _ "$WORK/block.sh"
}

# The KFD sysfs arm sets _setup_amd_detected=true and nothing else, so it reaches the
# selection block with no records and no gfx list. That arm needs a real /dev/kfd, so
# drive the selection block directly from the same starting state.
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
# grep -c prints 0 and exits 1 when there is no match, so the status is discarded.
# Anchored at both ends: `amd-smi list` must not also count `amd-smi list -e`.
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
# Three adapters: the arch followed the mask, the name was always adapter 0's.
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
# Two of the same card on an amd-smi with no TARGET_GRAPHICS_VERSION: the names match, so
# whichever ordinal wins infers the same arch. Not ambiguous, so not declined.
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
# The ordinal that decides --rocm-gfx: collapsing the gfx1100 slots sends it to device 0.
assert_eq "device 2, past a duplicated arch, keeps its own arch and name" \
    "gfx1100|AMD Radeon PRO W7900" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 2)"
assert_eq "an out-of-range mask falls back to device 0" \
    "gfx90a|AMD Instinct MI210" "$(summary "$WORK/roc_three_dup" "$WORK/empty" 9)"
# Byte-identical records: folding them resolves device 2 to the iGPU, so --rocm-gfx
# carries gfx1036 on a host whose selected card is gfx1201.
assert_eq "two identical cards are still two devices" \
    "gfx1201|AMD Radeon AI PRO R9700" "$(summary "$WORK/roc_twins" "$WORK/empty" 2)"

echo "=== rocminfo names something but enumerates no device ==="
# amd-smi owns the device list here, so the leftover CPU-only record must be dropped
# rather than win the selection and discard amd-smi's arch.
assert_eq "amd-smi supplies both the arch and the name" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/roc_cpu_only" "$WORK/smi_fixture")"
# `list` runs twice (detect, then look for a gfx token it does not carry); one
# `static --asic` parse yields both fields. Pinned: the counts move if an arm is
# reordered or the parse is split again.
assert_eq "list is asked first, and carries no gfx" 2 "$(probe_count 'amd-smi list')"
assert_eq "and list -e is asked once, for the HIP mapping" \
    1 "$(probe_count 'amd-smi list -e')"
assert_eq "one static --asic parse supplies both the arch and the name" \
    1 "$(probe_count 'amd-smi static --asic')"

echo "=== a device that reported no name of its own ==="
assert_eq "keeps its arch and stays unnamed" \
    "gfx1030|" "$(summary "$WORK/roc_blank_name" "$WORK/smi_fixture")"

# amd-smi discovery order is not HIP order; `amd-smi list -e` publishes HIP_ID as the map.
# On this host discovery 0/1/2 is HIP 2/1/0, so an untranslated mask picks the wrong arch,
# and _setup_gfx is what becomes --rocm-gfx.
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
# hip_id reads N/A when the library cannot reach a device's KFD node. A partial map is
# not a mapping, so discovery order is kept rather than half-translated.
cat > "$WORK/smi_e_partial" <<'EOF'
GPU: 0
    HIP_ID: 2
GPU: 1
    HIP_ID: N/A
GPU: 2
    HIP_ID: 0
EOF
# Two devices claiming one HIP id describe something other than a 1:1 device mapping.
cat > "$WORK/smi_e_collide" <<'EOF'
GPU: 0
    HIP_ID: 1
GPU: 1
    HIP_ID: 1
GPU: 2
    HIP_ID: 0
EOF

# Two of the same card. Discovery order and HIP order may disagree here too, but every
# ordinal yields the same arch either way, so there is nothing to decline.
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
# With the map present each adapter is announced with its own name.
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
# With no gfx token the name is what --rocm-gfx comes from, so it must be the selected
# adapter's. With the map present that is answerable; without it, it is not.
assert_eq "a nameless-arch build still names the selected adapter" \
    "|AMD Radeon AI PRO R9700" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_two_identity" summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"
assert_eq "and declines without a map, since the name is what the arch comes from" "|" \
    "$(summary "$WORK/empty" "$WORK/smi_two_nogfx" 1)"
assert_eq "two archless adapters of the same model are not ambiguous" \
    "|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same_nogfx" 1)"

echo "=== neither tool reports a device ==="
# The KFD arm below would fire on a host that really has an AMD GPU.
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
# No usable map and the adapters disagree on arch: any ordinal would be a guess, and the
# guess becomes --rocm-gfx, so nothing is reported rather than the wrong thing.
assert_eq "an older CLI that rejects -e declines on unlike adapters" "|" \
    "$(summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "a partial map is declined, not half-applied" "|" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_partial" summary "$WORK/empty" "$WORK/smi_three" 0)"
assert_eq "colliding hip ids are declined" "|" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_collide" summary "$WORK/empty" "$WORK/smi_three" 0)"
# Identical adapters are not ambiguous, so the absent map costs nothing.
assert_eq "identical adapters still resolve without a map, device 0" \
    "gfx942|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same" 0)"
assert_eq "and device 1" \
    "gfx942|AMD Instinct MI300X" "$(summary "$WORK/empty" "$WORK/smi_two_same" 1)"
# A single adapter can never be ambiguous either.
assert_eq "one adapter resolves without a map" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(summary "$WORK/empty" "$WORK/smi_fixture" 0)"
# rocminfo is an HSA client, so its agent list is already in ROCr order.
assert_eq "the rocminfo path is not reordered by a HIP map" "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(STUB_AMDSMI_E="$WORK/smi_e_reversed" summary "$WORK/roc_three_dup" "$WORK/empty" 1)"

echo "=== detected, but no arm produced a record ==="
# Under `set -euo pipefail` an unassigned variable is not an empty string, it is a fatal
# error: `unsloth studio update` dies here instead of falling through to a source build.
assert_eq "the KFD-shaped path reaches the end instead of aborting on set -u" \
    "|" "$(kfd_shape_summary)"
assert_eq "every variable the selection block reads is initialised up front" \
    "" "$(grep -oE '\$\{?_setup_(gfx|gfx_all|mkt|amd_records|amd_detected|nvidia_usable)\b' \
              "$WORK/select.sh" | tr -d '${' | sort -u \
          | while read -r _v; do grep -q "^$_v=" "$WORK/init.sh" || echo "$_v"; done | tr '\n' ' ' | sed 's/ $//')"
assert_eq "amd-smi answers list but not static --asic" \
    "|" "$(STUB_AMDSMI_MUTE_STATIC=1 summary "$WORK/empty" "$WORK/smi_three")"

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
