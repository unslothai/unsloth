#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# ──────────────────────────────────────────────────────────────────────────────
# Enable ROCm-on-WSL for AMD GPUs (Strix Halo/Point APUs AND discrete Radeon RX
# 7000/9000). Verified on gfx1151 (Radeon 8060S) and gfx1200 (Radeon RX 9060 XT).
# ──────────────────────────────────────────────────────────────────────────────
# install.sh routes the detected arch to the right ROCm wheels once a runtime exists;
# what it does NOT do is install AMD's ROCm userspace + the WSL DXG bridge (librocdxg).
# This helper does that Linux-side prerequisite on Ubuntu 24.04 WSL2, invoked by
# install.sh when it sees an AMD GPU via /dev/dxg but no ROCm yet. Arch-agnostic: the
# arch is auto-detected from rocminfo (override UNSLOTH_WSL_GFX=gfx1200). Idempotent.
#
# Manual, admin-gated Windows prerequisite: an AMD Adrenalin driver with
# production ROCDXG/WSL support (26.2.2+). install.ps1 offers to update it. Once
# installed + rebooted, /dev/dxg is exposed to WSL and this script builds the rest.
#
# HOW ROCDXG WORKS (and why older /usr/lib/wsl/lib notes are wrong): librocdxg.so
# is AMD's user-mode bridge between the Linux HSA runtime and the Windows driver
# over /dev/dxg. The STANDARD hsa-rocr runtime (NOT the gone "roc4wsl" package)
# loads it when HSA_ENABLE_DXG_DETECTION=1. No hsa/rocm libs need injecting into
# /usr/lib/wsl/lib (it holds only d3d12/dxcore), yet rocminfo enumerates gfx1151
# fine -- so we gate on /dev/dxg, not on WSL lib injection.
#
# KNOWN CAVEAT (ROCm/ROCm#6022): librocdxg can cap usable ROCm VRAM at the WSL
# VM's RAM (.wslconfig [wsl2] memory=) on some BIOS UMA layouts, and amd-smi
# doesn't work in WSL. On OOM below capacity, raise memory= (then wsl --shutdown)
# and watch GPU use from Windows. Large-UMA BIOS exposes the full pool regardless.
#
# Verified on Ryzen AI Max+ PRO 395 / Radeon 8060S (gfx1151) with ROCm 7.2.1 +
# Ubuntu 24.04 + WSL2 + Adrenalin. These pins MOVE; bump + re-verify on newer ROCm.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Tunables (override via env) ──────────────────────────────────────────────
ROCM_VER="${UNSLOTH_WSL_ROCM_VER:-7.2.1}"            # ROCm release to install
# GPU arch: empty = auto-detect from rocminfo after install (override UNSLOTH_WSL_GFX=gfx1200).
# The ROCm + librocdxg setup is arch-agnostic; only verify + the smoke test need the arch.
GFX="${UNSLOTH_WSL_GFX:-}"
LIBROCDXG_REF="${UNSLOTH_LIBROCDXG_REF:-develop}"    # ROCm/librocdxg git ref to build
# AMD's wheel index for the (optional) smoke test; resolved after arch detection.
TORCH_INDEX=""
# Optional torch smoke test (throwaway venv). OFF by default: install.sh installs
# torch itself into the real venv right after, so a duplicate download is wasteful.
SMOKE_TEST="${UNSLOTH_WSL_SMOKE_TEST:-0}"
# REQUIRED constraint -- without it pip prefers PyPI's newer CUDA torch over the
# gfx1151 ROCm wheel. 2.11 carries AMD's real gfx1151 fix (matches install.sh).
TORCH_CONSTRAINT="${UNSLOTH_WSL_TORCH_CONSTRAINT:-torch>=2.11.0,<2.12.0}"
ROCM_DIR=""                                          # resolved after install

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
note() { printf '   %s\n' "$*"; }
die()  { printf '\n\033[1;31m[BLOCKED] %s\033[0m\n' "$*" >&2; exit 1; }

# sudo only if not already root (WSL distros often run as root)
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    command -v sudo >/dev/null 2>&1 || die "Need root or sudo to install ROCm."
    SUDO="sudo"
fi

# ── Windows 11 SDK (headers for the librocdxg build) ─────────────────────────
# librocdxg's cmake build needs the Windows SDK 'shared' headers, which live on
# the Windows HOST under C:\Program Files (x86)\Windows Kits\10\Include\<ver>\.
_WIN_SDK_INC_BASE="/mnt/c/Program Files (x86)/Windows Kits/10/Include"

# Print the newest installed SDK include dir with 'shared' headers, or nothing.
# find + read loop (not `for ... in $(ls)`) since the base path has a space.
_find_win_sdk() {
    [ -d "$_WIN_SDK_INC_BASE" ] || return 0
    while IFS= read -r _inc; do
        [ -n "$_inc" ] || continue
        if [ -d "$_inc/shared" ]; then printf '%s' "$_inc"; return 0; fi
    done < <(find "$_WIN_SDK_INC_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -Vr)
    return 0
}

# Best-effort: install the Windows 11 SDK on the Windows HOST via winget so the
# build has its headers with no manual step. Elevates -> ONE UAC prompt; headers
# appear under /mnt/c immediately (no reboot). Never fatal -- failure falls
# through to a manual-install message. Opt out: UNSLOTH_SKIP_WIN_SDK_INSTALL=1.
_install_windows_sdk_via_winget() {
    [ "${UNSLOTH_SKIP_WIN_SDK_INSTALL:-0}" = "1" ] && { note "Skipping Windows SDK auto-install (UNSLOTH_SKIP_WIN_SDK_INSTALL=1)."; return 0; }
    command -v powershell.exe >/dev/null 2>&1 || return 0
    # `command -v` succeeds even with WSL interop OFF (.exe on PATH but fails
    # with "Exec format error"); verify it actually executes.
    powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1 || return 0
    if ! powershell.exe -NoProfile -Command "if (Get-Command winget -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
        note "winget not available on the Windows host -- cannot auto-install the Windows SDK."
        return 0
    fi
    say "Installing the Windows 11 SDK on the Windows host via winget"
    note "librocdxg needs its headers. Approve the UAC prompt on the Windows desktop."
    note "One-time (~1-3 GB download); opt out with UNSLOTH_SKIP_WIN_SDK_INSTALL=1."
    # Newest SDK first, then a fallback. Header presence is the source of truth
    # (re-check each attempt), not winget's exit code. </dev/null so winget never
    # consumes a piped `curl | sh` stdin.
    for _sdk_id in Microsoft.WindowsSDK.10.0.26100 Microsoft.WindowsSDK.10.0.22621; do
        note "winget install ${_sdk_id} ..."
        # --source winget: pin the community source so a broken default msstore
        # source (the cert failure this PR fixes) can't abort SDK resolution.
        powershell.exe -NoProfile -Command "winget install --id ${_sdk_id} -e --source winget --accept-source-agreements --accept-package-agreements --disable-interactivity" </dev/null || true
        if [ -n "$(_find_win_sdk)" ]; then
            note "Windows SDK headers present after install."
            return 0
        fi
    done
    note "Automatic Windows SDK install did not complete."
    return 0
}

# ── PREFLIGHT ────────────────────────────────────────────────────────────────
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
# Don't block on hsa/rocm libs in /usr/lib/wsl/lib: a working ROCDXG setup
# doesn't need them (only d3d12/dxcore). Real readiness is checked via rocminfo.

# ── Step 1: build/runtime prerequisites ──────────────────────────────────────
say "Installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
# `make` is explicit: cmake shells out to it but Ubuntu only *recommends* it, so
# minimal images lack it and the librocdxg `make -j` build would fail.
$SUDO apt-get install -y cmake make gcc g++ git wget gpg ca-certificates python3-venv python3-pip

# ── Step 2: ROCm ${ROCM_VER} userspace (no DKMS -- WSL uses the Windows driver) ─
say "Installing ROCm ${ROCM_VER} userspace"
if ! command -v rocminfo >/dev/null 2>&1 && [ ! -x /opt/rocm/bin/rocminfo ]; then
    # Direct apt-repo install (leaner than amdgpu-install; repo is indexed by
    # ROCm version, e.g. .../apt/7.2.1).
    $SUDO mkdir -p /etc/apt/keyrings
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor | $SUDO tee /etc/apt/keyrings/rocm.gpg >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} noble main" \
        | $SUDO tee /etc/apt/sources.list.d/rocm.list >/dev/null
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        | $SUDO tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
    $SUDO apt-get update -y
    # rocm-libs pulls everything torch links at runtime (rocblas, hipblas,
    # miopen-hip, rccl, ...); hsa-rocr + rocminfo come as deps. Large (~5 GB
    # download / ~23 GB installed).
    $SUDO apt-get install -y rocm-libs rocminfo hip-runtime-amd
else
    note "ROCm already present -- skipping apt install."
fi

# Resolve the real ROCm dir and ensure the canonical /opt/rocm symlink. apt lays
# ROCm under /opt/rocm-<ver> and rocm-core symlinks /opt/rocm -> that; repair if
# an earlier partial run left /opt/rocm as a real dir blocking the symlink.
_real="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_real" ] && [ ! -L /opt/rocm ] && [ -d /opt/rocm ]; then
    # /opt/rocm is a real dir blocking the symlink. Only treat it as a removable
    # stray stub if it's NOT a real ROCm install (a real one has bin/rocminfo /
    # bin/hipcc / .info/version) -- this protects a user's pre-existing ROCm. Even
    # then we MOVE IT ASIDE, never rm -rf, so a wrong guess can't lose data.
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

# ── Step 3: build librocdxg (DXG <-> HSA bridge; not yet an apt package) ──────
say "Building librocdxg (${LIBROCDXG_REF})"
if [ -e "${ROCM_DIR}/lib/librocdxg.so" ]; then
    note "librocdxg already installed -- skipping build."
else
    # Discover the newest installed Win11 SDK (version differs per machine). If
    # absent, auto-install via winget (one UAC prompt) and re-discover; only if
    # that ALSO fails do we stop with manual instructions.
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
    git clone --depth 1 --branch "$LIBROCDXG_REF" https://github.com/ROCm/librocdxg.git "$_src" \
        || git clone "https://github.com/ROCm/librocdxg.git" "$_src"
    (
        cd "$_src"
        git checkout "$LIBROCDXG_REF" 2>/dev/null || true
        mkdir -p build && cd build
        cmake .. -DWIN_SDK="${_win_sdk}/shared"
        make -j"$(nproc)"
        $SUDO make install
    )
fi
# Ensure soname symlinks resolve to whatever version was built (e.g. 1.2.0).
_dxg_real="$(ls -1 "${ROCM_DIR}"/lib/librocdxg.so.*.* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_dxg_real" ]; then
    _dxg_base="$(basename "$_dxg_real")"                       # librocdxg.so.1.2.0
    _dxg_major="$(printf '%s' "$_dxg_base" | sed -E 's/librocdxg\.so\.([0-9]+).*/\1/')"
    $SUDO ln -sf "$_dxg_base" "${ROCM_DIR}/lib/librocdxg.so.${_dxg_major}"
    $SUDO ln -sf "librocdxg.so.${_dxg_major}" "${ROCM_DIR}/lib/librocdxg.so"
fi
echo "${ROCM_DIR}/lib" | $SUDO tee /etc/ld.so.conf.d/rocm.conf >/dev/null
$SUDO ldconfig

# ── Step 4: persist environment (system-wide so Studio's worker inherits it) ──
say "Persisting ROCm-on-WSL environment"
_envfile="/etc/profile.d/unsloth-rocm-wsl.sh"
$SUDO tee "$_envfile" >/dev/null <<EOF
# >>> Unsloth ROCm-on-WSL >>>
export HSA_ENABLE_DXG_DETECTION=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PATH="${ROCM_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:\${LD_LIBRARY_PATH:-}"
# <<< Unsloth ROCm-on-WSL <<<
EOF
# also drop into ~/.bashrc for interactive shells
if [ -n "${HOME:-}" ] && ! grep -q "Unsloth ROCm-on-WSL" "${HOME}/.bashrc" 2>/dev/null; then
    cat "$_envfile" >> "${HOME}/.bashrc"
fi
# export into the current process so verification below works immediately
export HSA_ENABLE_DXG_DETECTION=1
export PATH="${ROCM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:${LD_LIBRARY_PATH:-}"

# ── Step 5: verify the runtime enumerates the GPU ────────────────────────────
say "Verifying rocminfo enumerates the GPU over DXG"
# Capture rocminfo into a var BEFORE grepping: piping into `grep -q` SIGPIPEs
# rocminfo on first match, which under `set -o pipefail` turns a successful match
# into a pipeline failure.
_rocminfo_out="$(rocminfo 2>/dev/null || true)"
# GPU agents advertise an ISA "Name: gfxNNNN". Match gfx[1-9] (excludes gfx000, the CPU
# agent), drop the "gfx*-generic" fallback ISA, and take the first real GPU arch.
_detected_gfx="$(printf '%s\n' "$_rocminfo_out" | grep -E 'Name:[[:space:]]*gfx[1-9]' | grep -v 'generic' | grep -oE 'gfx[1-9][0-9a-z]*' | head -1 || true)"
if [ -z "$_detected_gfx" ]; then
    printf '%s\n' "$_rocminfo_out" | head -25 >&2 || true
    die "rocminfo did not enumerate any GPU agent. Most common cause: the Windows AMD driver predates production ROCDXG -- update Adrenalin (install.ps1 offers this), reboot, and re-run."
fi
# Honour a caller-pinned arch (sanity-check via a consuming grep, not grep -q: under
# pipefail -q would SIGPIPE printf on large output and misreport the arch); else adopt.
if [ -n "$GFX" ] && ! printf '%s\n' "$_rocminfo_out" | grep -E "Name:[[:space:]]*${GFX}([^0-9]|$)" >/dev/null; then
    die "rocminfo enumerated '${_detected_gfx}' but not the requested UNSLOTH_WSL_GFX='${GFX}'."
fi
GFX="${GFX:-$_detected_gfx}"
# Display-only summary: best-effort (|| true) so head's early pipe-close under
# `set -o pipefail` can't fail the bootstrap after verification already passed.
printf '%s\n' "$_rocminfo_out" | grep -E 'Marketing Name|Device Type|Compute Unit' | grep -iE "Radeon|GPU|Compute" | head -3 || true
note "ROCm-on-WSL runtime is live for ${GFX}."

# ── Step 6 (optional): torch smoke test from AMD's per-arch wheel index ───────
if [ "$SMOKE_TEST" = "1" ]; then
    say "Smoke-testing PyTorch on ${GFX} (throwaway venv)"
    # Map the detected arch to AMD's repo.amd.com wheel family index.
    case "$GFX" in
        gfx1200|gfx1201)                 _fam="gfx120X-all" ;;
        gfx1100|gfx1101|gfx1102|gfx1103) _fam="gfx110X-all" ;;
        *)                               _fam="$GFX" ;;   # gfx1150/gfx1151/gfx90a: own index
    esac
    TORCH_INDEX="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}/${_fam}/"
    _venv="${HOME}/.unsloth/rocm-smoketest"
    rm -rf "$_venv"; python3 -m venv "$_venv"
    "$_venv/bin/pip" install --quiet --upgrade pip
    # AMD arch index is primary (torch + triton); PyPI only an extra for pure-py
    # deps. The constraint keeps pip on the ROCm wheel, not a newer PyPI CUDA torch.
    "$_venv/bin/pip" install --index-url "$TORCH_INDEX" \
        --extra-index-url https://pypi.org/simple "$TORCH_CONSTRAINT" || \
        die "torch install from ${TORCH_INDEX} failed."
    # WSL: torch's bundled ROCr must load the DXG bridge -- drop librocdxg into torch/lib.
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
