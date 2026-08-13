#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# On an RDNA 1 or Polaris host the AMD summary must say Unsloth ships no ROCm PyTorch wheels
# for the card, and must NOT send the user after the amdgpu kernel driver and the render and
# video groups. That advice cannot succeed there, which is what unslothai#8577 removed on the
# PowerShell side; the POSIX report grew a torch.cuda.is_available() probe in parallel, and on
# these cards torch is CPU-only by construction, so every "torch cannot use it" arm fires.
#
# The chain is run for real: the report block is extracted from studio/setup.sh and sourced,
# with lspci and the venv interpreter scripted per case. Only step/substep are stubs, so the
# assertions below read the lines a user would actually see.
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SETUP_SH="${1:-$SCRIPT_DIR/../../studio/setup.sh}"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# The supported table lives at top level; the report block calls it as its peer guard, so a
# run without it would let a covered card fall into the unsupported arm.
awk '/^_setup_supported_gfx_from_name\(\) \{/, /^\}$/' "$SETUP_SH" > "$WORK/blk.sh"
# The report chain: the two unsupported helpers, the single lookup, the torch probe and every
# arm, ending just before the ROCm path lines that follow the chain.
awk '/^    # GPU name -> gfx arch for AMD generations/, /^    _setup_rocm_root=/' "$SETUP_SH" \
    | sed '$d' >> "$WORK/blk.sh"

# An extraction that lost any of these would make the cases below pass vacuously.
for _need in \
    '_setup_supported_gfx_from_name() {' \
    '_setup_unsupported_gfx_any() {' \
    '_setup_unsup_gfx=$(_setup_unsupported_gfx_any' \
    'no ROCm PyTorch wheels Unsloth installs' \
    'render and video groups'
do
    grep -qF "$_need" "$WORK/blk.sh" || {
        echo "FATAL: extraction lost: $_need" >&2
        exit 1
    }
done

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

# lspci naming one adapter, as a real KFD-only host reports it (pci.ids spellings).
_LSPCI_NAVI10='0a:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 10 [Radeon RX 5600 OEM/5600 XT / 5700/5700 XT] [1002:731f] (rev c1)'
_LSPCI_POLARIS='01:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Ellesmere [Radeon RX 470/480/570/570X/580/580X/590] [1002:67df] (rev e7)'
_LSPCI_NAVI31='03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M] [1002:744c] (rev cc)'

# Stands in for the venv interpreter. The find_spec call must succeed (torch IS installed on
# these hosts, it is simply a CPU wheel); the availability probe answers per $1.
make_venv() {
    _v="$WORK/venv_$2"
    rm -rf "$_v"
    mkdir -p "$_v/bin"
    {
        echo '#!/bin/sh'
        echo 'case "$*" in'
        echo '    *find_spec*) exit 0 ;;'
        printf '    *is_available*) exit %s ;;\n' "$1"
        echo 'esac'
        echo 'exit 0'
    } > "$_v/bin/python"
    chmod +x "$_v/bin/python"
    printf '%s' "$_v"
}

# Runs the report chain and echoes every reported line.
# $1 lspci output ("" means the binary is absent), $2 cuda probe exit, $3 _setup_gfx,
# $4 _setup_mkt, $5 _setup_amd_hidden, $6 case tag. Remaining env comes from the caller.
run_case() {
    _bin="$WORK/bin_$6"
    rm -rf "$_bin"
    mkdir -p "$_bin"
    if [ -n "$1" ]; then
        printf '#!/bin/sh\ncat <<"LSPCI_EOF"\n%s\nLSPCI_EOF\n' "$1" > "$_bin/lspci"
        chmod +x "$_bin/lspci"
    fi
    (
        # The scripted lspci first, but the real grep/awk/timeout still resolve: dropping the
        # rest of PATH would break the block itself rather than the case under test.
        PATH="$_bin:$PATH"
        VENV_DIR="$(make_venv "$2" "$6")"
        _setup_gfx="$3"
        _setup_mkt="$4"
        _setup_amd_hidden="$5"
        _setup_vis_name="${_setup_vis_name:-}"
        C_WARN=""
        # shellcheck disable=SC2317
        step() { echo "STEP: $2"; }
        # shellcheck disable=SC2317
        substep() { echo "SUB: $1"; }
        # shellcheck disable=SC1091
        . "$WORK/blk.sh"
    )
}

# Reports 1 when the output carries the line, 0 otherwise. grep -c counts lines, which would
# report 2 where a message is split across substeps, so the answer is normalised here.
has() {
    if printf '%s\n' "$1" | grep -qF "$2"; then echo 1; else echo 0; fi
}

echo "an RDNA 1 / Polaris host is told no ROCm wheels exist, not to fix its groups"
for _card in navi10 polaris; do
    case "$_card" in
        navi10)  _pci="$_LSPCI_NAVI10" ;;
        polaris) _pci="$_LSPCI_POLARIS" ;;
    esac
    _out=$(run_case "$_pci" 1 "" "" false "$_card")
    check "$_card: says no ROCm wheels are installed" \
        "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 1
    check "$_card: no render/video group advice" \
        "$(has "$_out" 'render and video groups')" 0
    check "$_card: not reported as a broken ROCm install" \
        "$(has "$_out" 'PyTorch cannot use it')" 0
    check "$_card: names the CPU-only outcome" \
        "$(has "$_out" 'torch stays CPU-only')" 1
done

echo "the guard is targeted: a covered card with dead torch keeps its diagnosis"
# An RX 7900 that torch cannot use IS a driver or group problem, and #8577 never claimed
# otherwise. Losing this line would trade one wrong report for another.
_out=$(run_case "$_LSPCI_NAVI31" 1 "" "" false navi31)
check "gfx1100-class: keeps the group advice" \
    "$(has "$_out" 'render and video groups')" 1
check "gfx1100-class: claims no unsupported arch" \
    "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 0

echo "a healthy host still reports normally"
_out=$(run_case "$_LSPCI_NAVI31" 0 gfx1100 "AMD Radeon RX 7900 XTX" false healthy)
check "healthy gfx1100: plain ROCm report" "$(has "$_out" 'AMD ROCm (gfx1100)')" 1
check "healthy gfx1100: no unsupported claim" \
    "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 0
check "healthy gfx1100: no group advice" "$(has "$_out" 'render and video groups')" 0

# A working torch with no gfx token from the tools: amd-smi reports the market name only.
# Kept as its own case because it is the one healthy shape the unsupported arm can reach --
# with $_setup_gfx set the arm above wins first, so that case cannot see this arm over-fire.
_out=$(run_case "$_LSPCI_NAVI31" 0 "" "AMD Radeon RX 7900 XTX" false healthy_nogfx)
check "healthy, no gfx token: plain ROCm report" \
    "$(printf '%s\n' "$_out" | grep -cx 'STEP: AMD ROCm')" 1
check "healthy, no gfx token: no unsupported claim" \
    "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 0

echo "the explicit-backend and hidden-mask arms are outranked too"
# Neither a different backend pin nor a changed visibility mask can reach a wheel that is
# not published, so both would send the user after a fix that cannot land.
_out=$(UNSLOTH_TORCH_BACKEND=cpu run_case "$_LSPCI_NAVI10" 1 "" "" false backend)
check "explicit cpu backend: no ROCm wheels wins" \
    "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 1
check "explicit cpu backend: not blamed on the selection" \
    "$(has "$_out" 'disabled by the explicit')" 0

_out=$(_setup_vis_name=HIP_VISIBLE_DEVICES run_case "$_LSPCI_NAVI10" 1 "" "" true hidden)
check "hidden mask: no ROCm wheels wins" \
    "$(has "$_out" 'no ROCm PyTorch wheels Unsloth installs')" 1
check "hidden mask: not blamed on the mask" \
    "$(has "$_out" 'intentionally hides every AMD device')" 0

echo "an explicit index pin keeps its own wording through the arm that now wins"
# The unsupported arm carries the pin case itself, so outranking the others loses nothing.
_out=$(UNSLOTH_TORCH_INDEX_URL=https://example.invalid/simple \
    run_case "$_LSPCI_NAVI10" 1 "" "" false pinned)
check "pinned: pin wording is reported" \
    "$(has "$_out" 'index you pinned is used as given')" 1
check "pinned: no CPU-only claim over the pin" \
    "$(has "$_out" 'torch stays CPU-only')" 0

echo "the guard is present at every arm it has to cover"
_guarded=$(grep -c '_setup_rocm_torch_ok" = false \] && \[ -z "\$_setup_unsup_gfx" \]' "$SETUP_SH")
check "all three torch arms are guarded" "$_guarded" 3

printf '\n%s\n' "passed: $PASS  failed: $FAIL"
[ "$FAIL" -eq 0 ]
