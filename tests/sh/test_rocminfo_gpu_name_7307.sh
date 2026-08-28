#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Test both rocminfo parsers against CPU-first and multi-GPU outputs (#7307).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_SH="$SCRIPT_DIR/../../install.sh"
SETUP_SH="$SCRIPT_DIR/../../studio/setup.sh"
PASS=0
FAIL=0

_FUNC_FILE=$(mktemp)
{
    sed -n '/^_rocminfo_gpu_records()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_setup_rocminfo_gpu_records()/,/^}/p' "$SETUP_SH"
    echo ""
    sed -n '/^_amd_smi_gpu_records()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_gfx_arch_slots()/,/^}/p' "$INSTALL_SH"
    echo ""
    sed -n '/^_amd_smi_hip_order()/,/^}/p' "$INSTALL_SH"
} > "$_FUNC_FILE"
# shellcheck disable=SC1090
. "$_FUNC_FILE"
rm -f "$_FUNC_FILE"

# The visible-device pick, read OUT OF the scripts: a restated copy cannot fail when
# the real one changes (re-adding `!seen[$0]++` to production left this file green).
_extract_pick() { awk '/_[a-z_]*record=\$\(printf/ {
                           getline
                           sub(/^[[:space:]]*'"'"'/, "")
                           sub(/'"'"'\)$/, "")
                           print; exit
                       }' "$1"; }
_PICK_PROG=$(_extract_pick "$INSTALL_SH")
_PICK_PROG_SETUP=$(_extract_pick "$SETUP_SH")
[ -n "$_PICK_PROG" ] || { echo "FATAL: no record selector found in $INSTALL_SH" >&2; exit 1; }
_pick() { awk -v idx="$1" "$_PICK_PROG"; }

assert_eq() {
    _label="$1"; _expected="$2"; _actual="$3"
    if [ "$_actual" = "$_expected" ]; then
        echo "  PASS: $_label"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_label (expected '$_expected', got '$_actual')"; FAIL=$((FAIL + 1))
    fi
}

# Installer and updater must use the same parser.
_body_install=$(sed -n '/^_rocminfo_gpu_records()/,/^}/p' "$INSTALL_SH" | tail -n +2)
_body_setup=$(sed -n '/^_setup_rocminfo_gpu_records()/,/^}/p' "$SETUP_SH" | tail -n +2)
assert_eq "install.sh and setup.sh parsers are identical" "$_body_install" "$_body_setup"
# The scripts must also select the same record, or one names a card the other does not.
assert_eq "install.sh and setup.sh record selectors are identical" \
    "$_PICK_PROG" "$_PICK_PROG_SETUP"
# The amd-smi side emits the same record shape and must not drift either.
_smi_install=$(sed -n '/^_amd_smi_gpu_records()/,/^}/p' "$INSTALL_SH" | tail -n +2)
_smi_setup=$(sed -n '/^_setup_amd_smi_gpu_records()/,/^}/p' "$SETUP_SH" | tail -n +2)
assert_eq "install.sh and setup.sh amd-smi parsers are identical" "$_smi_install" "$_smi_setup"
# So does the amd-smi-to-HIP reorder: it decides which record the mask lands on.
_hip_install=$(sed -n '/^_amd_smi_hip_order()/,/^}/p' "$INSTALL_SH" | tail -n +2)
_hip_setup=$(sed -n '/^_setup_amd_smi_hip_order()/,/^}/p' "$SETUP_SH" | tail -n +2)
assert_eq "install.sh and setup.sh amd-smi HIP reorders are identical" "$_hip_install" "$_hip_setup"
[ -n "$_hip_install" ] || { echo "FATAL: no amd-smi HIP reorder found" >&2; exit 1; }

# POSIX awk forbids a physical newline in a -v value, and gawk --posix makes it fatal, so
# the multi-line record list has to reach awk some other way. Run the real helper under a
# strict awk when the host has one; a host without it records a skip rather than a pass.
if gawk --posix 'BEGIN { exit 0 }' >/dev/null 2>&1; then
    _POSIX_AWK_DIR=$(mktemp -d)
    printf '#!/bin/sh\nexec %s --posix "$@"\n' "$(command -v gawk)" > "$_POSIX_AWK_DIR/awk"
    chmod +x "$_POSIX_AWK_DIR/awk"
    _HIP_FILE=$(mktemp)
    sed -n '/^_amd_smi_hip_order()/,/^}/p' "$INSTALL_SH" > "$_HIP_FILE"
    # shellcheck disable=SC1090
    . "$_HIP_FILE"
    _RECS="gfx90a|AMD Instinct MI210
gfx1100|AMD Radeon RX 7900 XTX"
    echo "=== strict awk ==="
    # The first line is the index space the records came back in, then the records.
    assert_eq "the HIP reorder runs under a POSIX awk" \
        "hip gfx1100|AMD Radeon RX 7900 XTX gfx90a|AMD Instinct MI210" \
        "$(printf 'GPU: 0\n    HIP_ID: 1\nGPU: 1\n    HIP_ID: 0\n' \
           | PATH="$_POSIX_AWK_DIR:$PATH" _amd_smi_hip_order "$_RECS" | tr '\n' ' ' | sed 's/ $//')"
    assert_eq "and declines cleanly there too, rather than dying" \
        "discovery gfx90a|AMD Instinct MI210 gfx1100|AMD Radeon RX 7900 XTX" \
        "$(printf '' | PATH="$_POSIX_AWK_DIR:$PATH" _amd_smi_hip_order "$_RECS" | tr '\n' ' ' | sed 's/ $//')"
    rm -rf "$_POSIX_AWK_DIR" "$_HIP_FILE"
else
    echo "  SKIP: no gawk --posix on this host, strict-awk check not run"
fi
[ -n "$_smi_install" ] || { echo "FATAL: no amd-smi record parser found" >&2; exit 1; }

# Strix Halo lists the misleading CPU marketing name first.
STRIX=$(cat <<'EOF'
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
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1151
      Name:                    amdgcn-amd-amdhsa--gfx11-generic
*******
Agent 3
*******
  Name:                    aie2p
  Marketing Name:          NPU Strix Halo
  Vendor Name:             AMD
  Device Type:             DSP
EOF
)
echo "=== CPU agent first ==="
assert_eq "the GPU agent names the GPU, not the processor" \
    "gfx1151|AMD Radeon Graphics" "$(printf '%s\n' "$STRIX" | _rocminfo_gpu_records)"
assert_eq "setup.sh copy agrees" \
    "gfx1151|AMD Radeon Graphics" "$(printf '%s\n' "$STRIX" | _setup_rocminfo_gpu_records)"
assert_eq "the ISA section and a non-GPU agent add no records" \
    "1" "$(printf '%s\n' "$STRIX" | _rocminfo_gpu_records | wc -l | tr -d ' ')"

# Two discrete cards behind the same CPU agent.
DUAL=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 7763 64-Core Processor
  Marketing Name:          AMD EPYC 7763 64-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx90a:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI210
  Vendor Name:             AMD
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Device Type:             GPU
EOF
)
echo "=== multi-GPU ==="
assert_eq "one record per device, in agent order" \
    "gfx90a|AMD Instinct MI210 gfx1100|AMD Radeon RX 7900 XTX" \
    "$(printf '%s\n' "$DUAL" | _rocminfo_gpu_records | tr '\n' ' ' | sed 's/ $//')"
assert_eq "a target id still registers as its arch" \
    "gfx90a|AMD Instinct MI210" "$(printf '%s\n' "$DUAL" | _rocminfo_gpu_records | _pick 0)"
assert_eq "index 1 selects the second card and its own name" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(printf '%s\n' "$DUAL" | _rocminfo_gpu_records | _pick 1)"
assert_eq "an out-of-range index falls back to device 0" \
    "gfx90a|AMD Instinct MI210" "$(printf '%s\n' "$DUAL" | _rocminfo_gpu_records | _pick 9)"

# Two cards of the same arch: the ordinals must not collapse.
SAME=$(cat <<'EOF'
Agent 1
*******
  Name:                    Intel(R) Xeon(R) Gold 6338
  Marketing Name:          Intel(R) Xeon(R) Gold 6338
  Vendor Name:             CPU
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
  Name:                    gfx1100
  Marketing Name:          AMD Radeon PRO W7900
  Device Type:             GPU
EOF
)
assert_eq "identical arches keep separate ordinals" \
    "gfx1100|AMD Radeon PRO W7900" "$(printf '%s\n' "$SAME" | _rocminfo_gpu_records | _pick 1)"

# A GPU agent that reports no marketing name.
BLANK=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 9 5950X 16-Core Processor
  Marketing Name:          AMD Ryzen 9 5950X 16-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx1030
  Vendor Name:             AMD
  Device Type:             GPU
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Device Type:             GPU
EOF
)
echo "=== blank marketing name ==="
assert_eq "a nameless GPU keeps its arch and an empty name" \
    "gfx1030|" "$(printf '%s\n' "$BLANK" | _rocminfo_gpu_records | _pick 0)"
assert_eq "and keeps its slot, so device 1 is still device 1" \
    "gfx1100|AMD Radeon RX 7900 XTX" "$(printf '%s\n' "$BLANK" | _rocminfo_gpu_records | _pick 1)"
assert_eq "a nameless GPU does not borrow the processor name" \
    "" "$(printf '%s\n' "$BLANK" | _rocminfo_gpu_records | _pick 0 | cut -d'|' -f2-)"

# Marketing names contain ": " on the Instinct OAM SKUs, which -F": " truncated.
COLON=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Marketing Name:          AMD EPYC 9654 96-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X OAM: 750W SKU
  Vendor Name:             AMD
  Device Type:             GPU
EOF
)
echo "=== embedded colon ==="
assert_eq "the name survives the colon in the middle" \
    "gfx942|AMD Instinct MI300X OAM: 750W SKU" "$(printf '%s\n' "$COLON" | _rocminfo_gpu_records)"

# No gfx agent at all: unchanged from before, the first marketing name is reported
# with an empty arch so the APU still gets named.
APU=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD Ryzen 7 5700U with Radeon Graphics
  Marketing Name:          AMD Ryzen 7 5700U with Radeon Graphics
  Vendor Name:             CPU
  Device Type:             CPU
EOF
)
# The ISA section repeats the target id verbatim; the leading ^ is what rejects it.
# Unanchored, match() finds gfx90a at offset 20 while substr() still cuts from
# RLENGTH+1, emitting a bogus "amdgcn" device and shifting every later ordinal.
ISA_TARGET_ID=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Marketing Name:          AMD EPYC 9654 96-Core Processor
  Vendor Name:             CPU
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx90a:sramecc+:xnack-
  Marketing Name:          AMD Instinct MI210
  Vendor Name:             AMD
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
*******
Agent 3
*******
  Name:                    gfx1100
  Marketing Name:          AMD Radeon RX 7900 XTX
  Vendor Name:             AMD
  Device Type:             GPU
  ISA Info:
    ISA 1
      Name:                    amdgcn-amd-amdhsa--gfx1100
EOF
)
echo "=== ISA lines repeating a target id ==="
assert_eq "an ISA target id adds no device" \
    "gfx90a|AMD Instinct MI210 gfx1100|AMD Radeon RX 7900 XTX" \
    "$(printf '%s\n' "$ISA_TARGET_ID" | _rocminfo_gpu_records | tr '\n' ' ' | sed 's/ $//')"
assert_eq "so device 1 is still the second card" \
    "gfx1100|AMD Radeon RX 7900 XTX" \
    "$(printf '%s\n' "$ISA_TARGET_ID" | _rocminfo_gpu_records | _pick 1)"

# ROCr's processor table tops out at four characters after "gfx", so the cap costs
# nothing today. A longer one must drop the device, not report a truncated arch.
WIDE=$(cat <<'EOF'
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Marketing Name:          AMD EPYC 9654 96-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx12500
  Marketing Name:          AMD Future Accelerator
  Device Type:             GPU
EOF
)
echo "=== arch wider than the cap ==="
assert_eq "an over-long gfx token is dropped, not truncated" \
    "|AMD EPYC 9654 96-Core Processor" "$(printf '%s\n' "$WIDE" | _rocminfo_gpu_records)"

echo "=== no GPU agent ==="
assert_eq "falls back to the first marketing name with no arch" \
    "|AMD Ryzen 7 5700U with Radeon Graphics" "$(printf '%s\n' "$APU" | _rocminfo_gpu_records)"
assert_eq "empty input yields nothing" "" "$(printf '' | _rocminfo_gpu_records)"

# amd-smi heads a device either way. `GPU: N` opens a keyed block and the arch comes
# later; `GPU[N] : gfx...` is the whole record with the arch on the header. Matching only
# the first shape returned NOTHING on the second, so a host with no rocminfo answered no
# arch at all and every arch policy downstream was skipped in silence -- a gfx1200 box on
# ROCm 6.1 keeps the kernel-less rocm6.1 wheels instead of being floored to rocm6.4.
SMI_BRACKET="GPU[0]  : gfx1100
GPU[1]  : gfx1100
GPU[2]  : gfx1200"
SMI_KEYED="GPU: 0
    ASIC:
        MARKET_NAME: AMD Radeon RX 9070 XT
        TARGET_GRAPHICS_VERSION: gfx1201"
echo "=== amd-smi headers ==="
assert_eq "the bracketed one-line form keeps its arch, one record per device" \
    "gfx1100| gfx1100| gfx1200|" \
    "$(printf '%s\n' "$SMI_BRACKET" | _amd_smi_gpu_records | tr '\n' ' ' | sed 's/ $//')"
assert_eq "the keyed form still pairs arch with name" \
    "gfx1201|AMD Radeon RX 9070 XT" \
    "$(printf '%s\n' "$SMI_KEYED" | _amd_smi_gpu_records)"
assert_eq "a header with no arch stays empty rather than borrowing the next device's" \
    "|" "$(printf 'GPU: 0\n    BDF: 0000:03:00.0\n' | _amd_smi_gpu_records)"

# A visible-device mask indexes the arch list, so an adapter the probe could not read has
# to keep its slot. Dropping it shifted every device after it and handed the next card's
# arch to the mask, which is how a Strix or rocm6.4 policy lands on the wrong adapter.
echo "=== arch slots ==="
assert_eq "an unreadable adapter keeps its ordinal instead of shifting the list" \
    "unknown gfx1151" \
    "$(printf '%s\n' "|Unknown adapter
gfx1151|Radeon 8060S" | _gfx_arch_slots | tr '\n' ' ' | sed 's/ $//')"
assert_eq "and in the last slot, where command substitution used to eat it" \
    "gfx1151 unknown" \
    "$(printf '%s\n' "gfx1151|Radeon 8060S
|Unknown adapter" | _gfx_arch_slots | tr '\n' ' ' | sed 's/ $//')"
assert_eq "no arch anywhere prints nothing, so the caller tries the next probe" \
    "" "$(printf '%s\n' "|A
|B" | _gfx_arch_slots)"
assert_eq "a fully readable list is unchanged" "gfx1100 gfx1200" \
    "$(printf '%s\n' "gfx1100|A
gfx1200|B" | _gfx_arch_slots | tr '\n' ' ' | sed 's/ $//')"

# amd-smi enumerates in discovery order, HIP numbers in its own. The arch routing indexes
# the list with a HIP/ROCR ordinal, so it has to translate first or say it cannot.
HIP_MAP="GPU: 0
    HIP_ID: 1
GPU: 1
    HIP_ID: 0"
MIXED="gfx90a|AMD Instinct MI210
gfx1200|AMD Radeon RX 9070"
echo "=== amd-smi to HIP order ==="
assert_eq "a full HIP_ID map reorders the records and says so" \
    "hip gfx1200|AMD Radeon RX 9070 gfx90a|AMD Instinct MI210" \
    "$(printf '%s\n' "$HIP_MAP" | _amd_smi_hip_order "$MIXED" | tr '\n' ' ' | sed 's/ $//')"
assert_eq "no map at all keeps discovery order and says THAT" \
    "discovery gfx90a|AMD Instinct MI210 gfx1200|AMD Radeon RX 9070" \
    "$(printf '' | _amd_smi_hip_order "$MIXED" | tr '\n' ' ' | sed 's/ $//')"
assert_eq "HIP_ID: N/A is not a map either" \
    "discovery" \
    "$(printf 'GPU: 0\n    HIP_ID: N/A\nGPU: 1\n    HIP_ID: N/A\n' \
        | _amd_smi_hip_order "$MIXED" | head -n 1)"

# install.sh is a sequential script and the GPU probing runs at top level, so a helper
# defined below its first call is simply not in scope: the shell says command not found,
# the caller's `|| true` swallows it, and the probe silently answers nothing. That is not
# a hypothetical, it shipped once. Pin the order rather than the line numbers, which move.
echo "=== helpers are defined before they are called ==="
for _fn in _rocminfo_gpu_records _amd_smi_gpu_records _gfx_arch_slots _amd_smi_hip_order; do
    _def=$(grep -n "^$_fn() {" "$INSTALL_SH" | head -n 1 | cut -d: -f1)
    # Calls only, never the definition line and never a comment.
    _first_use=$(grep -n "[|( ]$_fn\b" "$INSTALL_SH" \
        | grep -v "^[0-9]*: *#" | head -n 1 | cut -d: -f1)
    if [ -z "$_first_use" ]; then
        echo "  FAIL: $_fn is defined but never called"; FAIL=$((FAIL + 1)); continue
    fi
    if [ "$_def" -lt "$_first_use" ]; then
        echo "  PASS: $_fn defined at $_def, first called at $_first_use"; PASS=$((PASS + 1))
    else
        echo "  FAIL: $_fn is called at $_first_use but not defined until $_def"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
