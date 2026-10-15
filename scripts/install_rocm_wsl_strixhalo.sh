#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Enable ROCm-on-WSL for AMD GPUs on Ubuntu 24.04 WSL2: installs ROCm userspace and builds
# librocdxg (the /dev/dxg bridge, loaded with HSA_ENABLE_DXG_DETECTION=1). Invoked by install.sh.
# Needs Adrenalin 26.2.2+ on Windows. librocdxg may cap VRAM at the WSL VM RAM (ROCm/ROCm#6022).
set -euo pipefail

# install.sh refuses a helper that does not declare the contract it requires; bump both on change.
# 2 = verify the clone against LIBROCDXG_SHA, and fail a checkout that has no SHA to check.
UNSLOTH_ROCM_WSL_HELPER_CONTRACT=2

ROCM_VER="${UNSLOTH_WSL_ROCM_VER:-7.2.1}"            # ROCm release to install
# Empty = auto-detect from rocminfo; only verification and the smoke test need the arch.
GFX="${UNSLOTH_WSL_GFX:-}"
# Built and installed as root, so pinned to a commit (v1.2.2) and verified after clone.
LIBROCDXG_REF="${UNSLOTH_LIBROCDXG_REF:-4955d12888a3ec57057f1cf8660c2485e415e74c}"
LIBROCDXG_SHA="${UNSLOTH_LIBROCDXG_SHA:-4955d12888a3ec57057f1cf8660c2485e415e74c}"
# An explicit ref with no SHA is a deliberate operator choice: skip the pin check.
if [ -n "${UNSLOTH_LIBROCDXG_REF:-}" ] && [ -z "${UNSLOTH_LIBROCDXG_SHA:-}" ]; then
    LIBROCDXG_SHA=""
fi
TORCH_INDEX=""
# Off by default: install.sh installs torch into the real venv right after.
SMOKE_TEST="${UNSLOTH_WSL_SMOKE_TEST:-0}"
# Required: without it pip prefers PyPI's newer CUDA torch over the ROCm wheel.
TORCH_CONSTRAINT="${UNSLOTH_WSL_TORCH_CONSTRAINT:-torch>=2.11.0,<2.12.0}"
ROCM_DIR=""                                          # resolved after install

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[1;31m[BLOCKED] %s\033[0m\n' "$*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "Need root or sudo to install ROCm."
    SUDO="sudo"
fi

# librocdxg's build needs the Windows SDK 'shared' headers from the Windows host.
_WIN_SDK_INC_BASE="/mnt/c/Program Files (x86)/Windows Kits/10/Include"

# find + read loop because the base path contains a space.
_find_win_sdk() {
    [ -d "$_WIN_SDK_INC_BASE" ] || return 0
    while IFS= read -r _inc; do
        [ -n "$_inc" ] || continue
        if [ -d "$_inc/shared" ]; then printf '%s' "$_inc"; return 0; fi
    done < <(find "$_WIN_SDK_INC_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -Vr)
    return 0
}

# Best effort: install the Windows 11 SDK on the host via winget (one UAC prompt).
# Opt out with UNSLOTH_SKIP_WIN_SDK_INSTALL=1.
_install_windows_sdk_via_winget() {
    [ "${UNSLOTH_SKIP_WIN_SDK_INSTALL:-0}" = "1" ] && { note "Skipping Windows SDK auto-install (UNSLOTH_SKIP_WIN_SDK_INSTALL=1)."; return 0; }
    command -v powershell.exe >/dev/null 2>&1 || return 0
    # command -v succeeds even with WSL interop off, so check that it actually executes.
    powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1 || return 0
    if ! powershell.exe -NoProfile -Command "if (Get-Command winget -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
        note "winget not available on the Windows host -- cannot auto-install the Windows SDK."
        return 0
    fi
    say "Installing the Windows 11 SDK on the Windows host via winget"
    note "librocdxg needs its headers. Approve the UAC prompt on the Windows desktop."
    note "One-time (~1-3 GB download); opt out with UNSLOTH_SKIP_WIN_SDK_INSTALL=1."
    # Header presence is the source of truth, not winget's exit code.
    # </dev/null so winget never consumes the stdin of a piped install.
    for _sdk_id in Microsoft.WindowsSDK.10.0.26100 Microsoft.WindowsSDK.10.0.22621; do
        note "winget install ${_sdk_id} ..."
        # Pin the winget source so a broken msstore source cannot abort resolution.
        powershell.exe -NoProfile -Command "winget install --id ${_sdk_id} -e --source winget --accept-source-agreements --accept-package-agreements --disable-interactivity" </dev/null || true
        if [ -n "$(_find_win_sdk)" ]; then
            note "Windows SDK headers present after install."
            return 0
        fi
    done
    note "Automatic Windows SDK install did not complete."
    return 0
}

say "Preflight checks"

# shellcheck disable=SC1091
. /etc/os-release 2>/dev/null || true
if [ "${VERSION_ID:-}" != "24.04" ]; then
    die "This targets Ubuntu 24.04 (found '${VERSION_ID:-unknown}'). AMD's ROCm-on-WSL supports 24.04; create a dedicated distro:  wsl --install Ubuntu-24.04  (do not run on 26.04 -- ROCm 7.2 does not target it yet)."
fi

if [ ! -e /dev/dxg ]; then
    die "/dev/dxg missing -- WSL GPU paravirtualization not present. Ensure this is WSL2 (not WSL1) on a recent Windows build, and that an AMD GPU + ROCDXG-capable Adrenalin driver is installed on the Windows host (then reboot)."
fi
note "Ubuntu 24.04 + /dev/dxg present."
# A working ROCDXG setup does not need hsa/rocm libs in /usr/lib/wsl/lib; rocminfo is the check.

say "Installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
# Ubuntu only recommends make, so minimal images lack it.
$SUDO apt-get install -y cmake make gcc g++ git wget gpg ca-certificates python3-venv python3-pip

say "Installing ROCm ${ROCM_VER} userspace"
if ! command -v rocminfo >/dev/null 2>&1 && [ ! -x /opt/rocm/bin/rocminfo ]; then
    $SUDO mkdir -p /etc/apt/keyrings
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor | $SUDO tee /etc/apt/keyrings/rocm.gpg >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} noble main" \
        | $SUDO tee /etc/apt/sources.list.d/rocm.list >/dev/null
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        | $SUDO tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
    $SUDO apt-get update -y
    # Large: about 5 GB download, 23 GB installed.
    $SUDO apt-get install -y rocm-libs rocminfo hip-runtime-amd
else
    note "ROCm already present -- skipping apt install."
fi

_real="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_real" ] && [ ! -L /opt/rocm ] && [ -d /opt/rocm ]; then
    # Treat /opt/rocm as a stray stub only if it is not a real ROCm install, and then move
    # it aside, never rm.
    if [ -e /opt/rocm/bin/rocminfo ] || [ -e /opt/rocm/bin/hipcc ] || [ -e /opt/rocm/.info/version ]; then
        note "/opt/rocm is a real ROCm install -- leaving it untouched (will install librocdxg into it)."
    else
        note "Moving stray /opt/rocm stub aside -> $_real (not deleting it)"
        $SUDO cp -an /opt/rocm/. "$_real"/ 2>/dev/null || true
        $SUDO mv /opt/rocm "/opt/rocm.unsloth-stub-bak.$(date +%s)" 2>/dev/null || true
        [ -e /opt/rocm ] || $SUDO ln -s "$_real" /opt/rocm
    fi
elif [ -n "$_real" ] && [ ! -e /opt/rocm ]; then
    $SUDO ln -s "$_real" /opt/rocm
fi
if [ -L /opt/rocm ] || [ -d /opt/rocm ]; then ROCM_DIR="/opt/rocm"; else ROCM_DIR="$_real"; fi
{ [ -n "$ROCM_DIR" ] && [ -d "$ROCM_DIR" ]; } || die "ROCm not found under /opt after install."
note "ROCm at ${ROCM_DIR}"

say "Building librocdxg (${LIBROCDXG_REF})"
if [ -e "${ROCM_DIR}/lib/librocdxg.so" ]; then
    note "librocdxg already installed -- skipping build."
else
    _win_sdk="$(_find_win_sdk)"
    if [ -z "$_win_sdk" ]; then
        note "Windows 11 SDK headers not found -- attempting automatic install..."
        _install_windows_sdk_via_winget
        _win_sdk="$(_find_win_sdk)"
    fi
    [ -n "$_win_sdk" ] || die "Windows 11 SDK headers not found under 'C:\\Program Files (x86)\\Windows Kits\\10\\Include\\*\\shared', and the automatic winget install did not complete. Install it on the Windows host (e.g. 'winget install Microsoft.WindowsSDK.10.0.26100') and re-run."
    note "Windows SDK: ${_win_sdk}"
    _src="${HOME}/.unsloth/librocdxg"
    rm -rf "$_src"
    # --branch cannot take a commit, so a 40-char ref does a full clone and checks it out.
    if [ "${#LIBROCDXG_REF}" = "40" ]; then
        git clone "https://github.com/ROCm/librocdxg.git" "$_src"
    else
        git clone --depth 1 --branch "$LIBROCDXG_REF" https://github.com/ROCm/librocdxg.git "$_src" \
            || git clone "https://github.com/ROCm/librocdxg.git" "$_src"
    fi
    (
        cd "$_src"
        _co_failed=0
        git checkout "$LIBROCDXG_REF" 2>/dev/null || _co_failed=1
        # A failed checkout leaves the default branch; without a SHA nothing else would catch that.
        if [ "$_co_failed" = "1" ] && [ -z "$LIBROCDXG_SHA" ]; then
            die "could not check out librocdxg ref '${LIBROCDXG_REF}'. Refusing to build the repository's default branch instead."
        fi
        # Verify the pin before cmake or `sudo make install` run anything from this tree.
        if [ -n "$LIBROCDXG_SHA" ]; then
            _got_sha="$(git rev-parse HEAD 2>/dev/null || true)"
            [ "$_got_sha" = "$LIBROCDXG_SHA" ] || die "librocdxg ${LIBROCDXG_REF} resolved to ${_got_sha:-unknown}, expected ${LIBROCDXG_SHA}. Refusing to build and install unverified source as root. Set UNSLOTH_LIBROCDXG_REF (and optionally UNSLOTH_LIBROCDXG_SHA) to build a different revision on purpose."
        fi
        mkdir -p build && cd build
        cmake .. -DWIN_SDK="${_win_sdk}/shared"
        make -j"$(nproc)"
        $SUDO make install
    )
fi
_dxg_real="$(ls -1 "${ROCM_DIR}"/lib/librocdxg.so.*.* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_dxg_real" ]; then
    _dxg_base="$(basename "$_dxg_real")"
    _dxg_major="$(printf '%s' "$_dxg_base" | sed -E 's/librocdxg\.so\.([0-9]+).*/\1/')"
    $SUDO ln -sf "$_dxg_base" "${ROCM_DIR}/lib/librocdxg.so.${_dxg_major}"
    $SUDO ln -sf "librocdxg.so.${_dxg_major}" "${ROCM_DIR}/lib/librocdxg.so"
fi
echo "${ROCM_DIR}/lib" | $SUDO tee /etc/ld.so.conf.d/rocm.conf >/dev/null
$SUDO ldconfig

say "Persisting ROCm-on-WSL environment"
_envfile="/etc/profile.d/unsloth-rocm-wsl.sh"
$SUDO tee "$_envfile" >/dev/null <<EOF
# >>> Unsloth ROCm-on-WSL >>>
export HSA_ENABLE_DXG_DETECTION=1
export PATH="${ROCM_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:\${LD_LIBRARY_PATH:-}"
# <<< Unsloth ROCm-on-WSL <<<
EOF
if [ -n "${HOME:-}" ] && ! grep -q "Unsloth ROCm-on-WSL" "${HOME}/.bashrc" 2>/dev/null; then
    cat "$_envfile" >> "${HOME}/.bashrc"
fi
export HSA_ENABLE_DXG_DETECTION=1
export PATH="${ROCM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:${LD_LIBRARY_PATH:-}"

say "Verifying rocminfo enumerates the GPU over DXG"
# Capture first: `grep -q` SIGPIPEs rocminfo, which pipefail reports as a failure.
_rocminfo_out="$(rocminfo 2>/dev/null || true)"
# gfx[1-9] skips the gfx000 CPU agent; drop the generic fallback ISA.
_detected_gfx="$(printf '%s\n' "$_rocminfo_out" | grep -E 'Name:[[:space:]]*gfx[1-9]' | grep -v 'generic' | grep -oE 'gfx[1-9][0-9a-z]*' | head -1 || true)"
if [ -z "$_detected_gfx" ]; then
    printf '%s\n' "$_rocminfo_out" | head -25 >&2 || true
    die "rocminfo did not enumerate any GPU agent. Most common cause: the Windows AMD driver predates production ROCDXG -- update Adrenalin (install.ps1 offers this), reboot, and re-run."
fi
# Consuming grep, not -q: under pipefail -q would SIGPIPE printf and misreport the arch.
if [ -n "$GFX" ] && ! printf '%s\n' "$_rocminfo_out" | grep -E "Name:[[:space:]]*${GFX}([^0-9]|$)" >/dev/null; then
    die "rocminfo enumerated '${_detected_gfx}' but not the requested UNSLOTH_WSL_GFX='${GFX}'."
fi
GFX="${GFX:-$_detected_gfx}"
# Best effort: head's early pipe close must not fail the run under pipefail.
printf '%s\n' "$_rocminfo_out" | grep -E 'Marketing Name|Device Type|Compute Unit' | grep -iE "Radeon|GPU|Compute" | head -3 || true
note "ROCm-on-WSL runtime is live for ${GFX}."

if [ "$SMOKE_TEST" = "1" ]; then
    say "Smoke-testing PyTorch on ${GFX} (throwaway venv)"
    case "$GFX" in
        gfx1200|gfx1201)                 _fam="gfx120X-all" ;;
        gfx1100|gfx1101|gfx1102|gfx1103) _fam="gfx110X-all" ;;
        *)                               _fam="$GFX" ;;   # gfx1150/gfx1151/gfx90a: own index
    esac
    TORCH_INDEX="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}/${_fam}/"
    _venv="${HOME}/.unsloth/rocm-smoketest"
    rm -rf "$_venv"; python3 -m venv "$_venv"
    "$_venv/bin/pip" install --quiet --upgrade pip
    # The constraint keeps pip on the ROCm wheel rather than a newer PyPI CUDA torch.
    "$_venv/bin/pip" install --index-url "$TORCH_INDEX" \
        --extra-index-url https://pypi.org/simple "$TORCH_CONSTRAINT" || \
        die "torch install from ${TORCH_INDEX} failed."
    # torch's bundled ROCr must load the DXG bridge, so copy librocdxg into torch/lib.
    _tlib="$("$_venv/bin/python" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null || true)"
    [ -d "$_tlib" ] && cp -f "${ROCM_DIR}"/lib/librocdxg.so* "$_tlib"/ 2>/dev/null || true
    "$_venv/bin/python" - <<'PY'
import torch
ok = torch.cuda.is_available()
print("torch:", torch.__version__, "| cuda(rocm) available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"mem: free={free/1e9:.1f} GB total={total/1e9:.1f} GB")
    import time
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(10): c = a @ b
    torch.cuda.synchronize()
    print(f"matmul ok ({(time.time()-t0)/10*1e3:.1f} ms/iter)")
raise SystemExit(0 if ok else 1)
PY
    rm -rf "$_venv"
fi

say "Done."
note "ROCm-on-WSL is ready for ${GFX}. If you ran this standalone, install Unsloth"
note "in THIS distro and it will detect the GPU automatically:"
note "  curl -fsSL https://unsloth.ai/install.sh | sh"
