#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# ──────────────────────────────────────────────────────────────────────────────
# Enable ROCm-on-WSL for AMD Strix Halo (Radeon 8060S / gfx1151)
# ──────────────────────────────────────────────────────────────────────────────
# Unsloth's installer already routes gfx1151 to the right ROCm wheels once a
# ROCm runtime is present (see install.sh: 8060S -> gfx1151, repo.amd.com/rocm/
# whl/gfx1151). What it does NOT do is install AMD's ROCm userspace + the WSL
# DXG bridge -- a large, bleeding-edge prerequisite. This helper automates the
# Linux side of that prerequisite on an Ubuntu 24.04 WSL2 distro and is invoked
# automatically by install.sh when it detects a Strix Halo APU plumbed into WSL
# (via /dev/dxg) but no ROCm runtime yet. It is fully idempotent -- re-running
# after a successful setup is a no-op that just re-verifies.
#
# The ONE manual, admin-gated prerequisite (Windows side) is the AMD driver:
#   AMD Adrenalin Edition with production ROCDXG/WSL support (shipped 26.2.2+,
#   covering Strix / Strix Halo / Ryzen AI Max+). The Windows-side installer
#   (install.ps1) detects an out-of-date driver and offers to update it. Once a
#   ROCDXG-capable driver is installed + rebooted, /dev/dxg is exposed to WSL
#   and this script can build the rest.
#
# HOW ROCDXG WORKS (and why older notes about /usr/lib/wsl/lib are wrong):
#   ROCDXG (librocdxg.so) is AMD's open-source user-mode bridge between the
#   Linux ROCm HSA runtime and the Windows GPU driver, talking over /dev/dxg.
#   The STANDARD hsa-rocr runtime (NOT the legacy "roc4wsl" package, which no
#   longer exists in the ROCm apt repo) loads librocdxg when
#   HSA_ENABLE_DXG_DETECTION=1 is set. The driver does NOT need to inject
#   hsa/rocm libs into /usr/lib/wsl/lib -- on a working Strix Halo box that
#   directory holds only the base d3d12/dxcore libs, yet rocminfo enumerates
#   gfx1151 fine. So we gate on /dev/dxg, not on WSL lib injection.
#
# KNOWN CAVEAT (ROCm/ROCm#6022): librocdxg can cap the usable ROCm VRAM pool at
# the WSL VM's RAM (.wslconfig [wsl2] memory=...) on some BIOS UMA layouts, and
# amd-smi does not work inside WSL. If you hit OOM well below the APU's capacity,
# raise .wslconfig memory= (then `wsl --shutdown`) and watch GPU usage from
# Windows (Task Manager / Adrenalin). On large-UMA BIOS settings the full pool
# is exposed (e.g. ~80 GB) regardless.
#
# Procedure verified on this hardware (Ryzen AI Max+ PRO 395 / Radeon 8060S,
# gfx1151) against AMD's documented combo: ROCm 7.2.1 + Ubuntu 24.04 + WSL2 +
# Adrenalin (Apr 2026 build). These pins are MOVING targets; bump + re-verify
# when AMD ships a newer ROCm-on-WSL.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Tunables (override via env) ──────────────────────────────────────────────
ROCM_VER="${UNSLOTH_WSL_ROCM_VER:-7.2.1}"            # ROCm release to install
GFX="gfx1151"
LIBROCDXG_REF="${UNSLOTH_LIBROCDXG_REF:-develop}"    # ROCm/librocdxg git ref to build
# AMD's gfx1151 wheel index -- the SAME one install.sh routes gfx1151 to. Used
# only by the optional final torch smoke test (UNSLOTH_WSL_SMOKE_TEST=1).
TORCH_INDEX="${UNSLOTH_AMD_ROCM_MIRROR:-https://repo.amd.com/rocm/whl}/${GFX}/"
# Optional final torch smoke test (throwaway venv) to prove torch.cuda
# end-to-end. OFF by default: install.sh installs torch itself (into the real
# Studio venv, with the correct version constraint) right after this helper
# returns, so the heavy duplicate download is wasteful. Set
# UNSLOTH_WSL_SMOKE_TEST=1 for a standalone confidence check.
SMOKE_TEST="${UNSLOTH_WSL_SMOKE_TEST:-0}"
# Match install.sh's Strix routing: torch 2.11+rocm7.13 carries AMD's real
# gfx1151 fix. The constraint is REQUIRED -- without it pip prefers PyPI's
# newer CUDA torch (e.g. 2.12.0) over the gfx1151 ROCm wheel.
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

# Print the newest installed SDK include dir that ships the 'shared' headers (no
# trailing slash), or nothing. NB: the base path contains a space, so we use
# find + a read loop -- a `for ... in $(ls)` form would word-split it.
_find_win_sdk() {
    [ -d "$_WIN_SDK_INC_BASE" ] || return 0
    while IFS= read -r _inc; do
        [ -n "$_inc" ] || continue
        if [ -d "$_inc/shared" ]; then printf '%s' "$_inc"; return 0; fi
    done < <(find "$_WIN_SDK_INC_BASE" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort -Vr)
    return 0
}

# Best-effort: install the Windows 11 SDK on the Windows HOST from inside WSL via
# winget, so the build has its headers with no manual download. The SDK installer
# elevates -> ONE UAC prompt on the Windows desktop; the headers then appear
# under /mnt/c immediately (drvfs is live, no reboot). Never fatal: any failure
# falls through to a clear manual-install message. Opt out with
# UNSLOTH_SKIP_WIN_SDK_INSTALL=1. The user already consented to the ROCm-on-WSL
# setup, so this needs no extra prompt beyond the OS UAC gate.
_install_windows_sdk_via_winget() {
    [ "${UNSLOTH_SKIP_WIN_SDK_INSTALL:-0}" = "1" ] && { note "Skipping Windows SDK auto-install (UNSLOTH_SKIP_WIN_SDK_INSTALL=1)."; return 0; }
    command -v powershell.exe >/dev/null 2>&1 || return 0
    # `command -v` succeeds even when WSL interop is OFF (the .exe is on PATH via
    # drvfs but fails with "Exec format error"); verify it actually executes.
    powershell.exe -NoProfile -Command "exit 0" >/dev/null 2>&1 || return 0
    if ! powershell.exe -NoProfile -Command "if (Get-Command winget -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >/dev/null 2>&1; then
        note "winget not available on the Windows host -- cannot auto-install the Windows SDK."
        return 0
    fi
    say "Installing the Windows 11 SDK on the Windows host via winget"
    note "librocdxg needs its headers. Approve the UAC prompt on the Windows desktop."
    note "One-time (~1-3 GB download); opt out with UNSLOTH_SKIP_WIN_SDK_INSTALL=1."
    # Newest stable Win11 SDK first, then a fallback. The actual presence of the
    # headers is the source of truth (re-check after each attempt), not winget's
    # exit code. </dev/null so winget never consumes a piped `curl | sh` stdin.
    for _sdk_id in Microsoft.WindowsSDK.10.0.26100 Microsoft.WindowsSDK.10.0.22621; do
        note "winget install ${_sdk_id} ..."
        powershell.exe -NoProfile -Command "winget install --id ${_sdk_id} -e --accept-source-agreements --accept-package-agreements --disable-interactivity" </dev/null || true
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
# Non-fatal: the older heuristic looked for hsa/rocm libs in /usr/lib/wsl/lib,
# but a working ROCDXG setup does NOT require them (only d3d12/dxcore live
# there). We therefore do not block on it; we detect real readiness via
# rocminfo at the end instead.

# ── Step 1: build/runtime prerequisites ──────────────────────────────────────
say "Installing build prerequisites"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
# `make` is listed explicitly: cmake's Unix Makefiles generator shells out to it,
# but Ubuntu only *recommends* make, so minimal/--no-install-recommends images can
# lack it and the librocdxg `make -j` build below would fail with "make: not found".
$SUDO apt-get install -y cmake make gcc g++ git wget gpg ca-certificates python3-venv python3-pip

# ── Step 2: ROCm ${ROCM_VER} userspace (no DKMS -- WSL uses the Windows driver) ─
say "Installing ROCm ${ROCM_VER} userspace"
if ! command -v rocminfo >/dev/null 2>&1 && [ ! -x /opt/rocm/bin/rocminfo ]; then
    # Direct apt-repo install (leaner + more deterministic than the amdgpu-install
    # .deb dance; the rocm apt repo is indexed by ROCm version, e.g. .../apt/7.2.1).
    $SUDO mkdir -p /etc/apt/keyrings
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor | $SUDO tee /etc/apt/keyrings/rocm.gpg >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} noble main" \
        | $SUDO tee /etc/apt/sources.list.d/rocm.list >/dev/null
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        | $SUDO tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
    $SUDO apt-get update -y
    # rocm-libs pulls everything torch links at runtime (rocblas, hipblas,
    # hipblaslt, rocsolver, rocfft, rocrand, hipsparse, miopen-hip, rccl,
    # roctracer -> libroctx64, ...). hsa-rocr (standard HSA runtime) + rocminfo
    # come in as deps. NB: this is large (~5 GB download / ~23 GB installed).
    $SUDO apt-get install -y rocm-libs rocminfo hip-runtime-amd
else
    note "ROCm already present -- skipping apt install."
fi

# Resolve the real ROCm dir and ensure the canonical /opt/rocm symlink. apt lays
# ROCm down under /opt/rocm-<ver>; rocm-core normally creates /opt/rocm -> that.
# If an earlier partial run left /opt/rocm as a *real* directory (e.g. a stray
# librocdxg make-install), it blocks the symlink -- repair that here.
_real="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1 || true)"
if [ -n "$_real" ] && [ ! -L /opt/rocm ] && [ -d /opt/rocm ]; then
    # /opt/rocm is a real directory blocking the canonical symlink. ONLY treat it
    # as a removable stray stub (e.g. a librocdxg make-install that dropped just
    # lib/ + share/) if it is NOT a real ROCm install -- a real one has
    # bin/rocminfo / bin/hipcc / .info/version. This guards against ever deleting
    # a user's pre-existing ROCm (amdgpu-install / manual install that lays down a
    # real /opt/rocm). And even for a stub we MOVE IT ASIDE, never rm -rf, so a
    # wrong guess can never lose data.
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
    # Discover the newest installed Windows 11 SDK (do not hardcode a version --
    # it differs per machine). If absent, auto-install it on the Windows host via
    # winget (one UAC prompt) so the user has nothing manual to do, then
    # re-discover. Only if that ALSO fails do we stop with manual instructions.
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
# >>> Unsloth ROCm-on-WSL (gfx1151) >>>
export HSA_ENABLE_DXG_DETECTION=1
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export PATH="${ROCM_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:\${LD_LIBRARY_PATH:-}"
# <<< Unsloth ROCm-on-WSL (gfx1151) <<<
EOF
# also drop into the invoking user's ~/.bashrc for interactive shells
if [ -n "${HOME:-}" ] && ! grep -q "Unsloth ROCm-on-WSL" "${HOME}/.bashrc" 2>/dev/null; then
    cat "$_envfile" >> "${HOME}/.bashrc"
fi
# export into the current process so the verification below works immediately
export HSA_ENABLE_DXG_DETECTION=1
export PATH="${ROCM_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:${LD_LIBRARY_PATH:-}"

# ── Step 5: verify the runtime enumerates the GPU ────────────────────────────
say "Verifying rocminfo sees ${GFX}"
# NB: capture rocminfo output into a var BEFORE grepping. Piping straight into
# `grep -q` makes grep close the pipe on first match, which SIGPIPEs rocminfo
# (exit 141); under `set -o pipefail` that turns a *successful* match into a
# pipeline failure. The GPU's marketing name also contains "Radeon" on the CPU
# agent line, so we look specifically for the gfx1151 ISA "Name:" agent. Require
# gfx1151 exactly (not a broad gfx1[0-9]) so a generic fallback ISA
# (e.g. gfx11-generic) or an unrelated RDNA GPU does not pass this Strix-Halo
# verification while the real GPU is absent.
_rocminfo_out="$(rocminfo 2>/dev/null || true)"
if ! printf '%s\n' "$_rocminfo_out" | grep -qE "Name:[[:space:]]*${GFX}([^0-9]|$)"; then
    printf '%s\n' "$_rocminfo_out" | head -25 >&2 || true
    die "rocminfo did not enumerate a ${GFX} GPU agent. Most common cause: the Windows AMD driver predates production ROCDXG -- update Adrenalin (install.ps1 offers this), reboot, and re-run."
fi
printf '%s\n' "$_rocminfo_out" | grep -E 'Marketing Name|Device Type|Compute Unit' | grep -iE "Radeon|GPU|Compute" | head -3
note "ROCm-on-WSL runtime is live for ${GFX}."

# ── Step 6 (optional): torch smoke test from the gfx1151 index ───────────────
if [ "$SMOKE_TEST" = "1" ]; then
    say "Smoke-testing PyTorch on ${GFX} (throwaway venv)"
    _venv="${HOME}/.unsloth/rocm-smoketest"
    rm -rf "$_venv"; python3 -m venv "$_venv"
    "$_venv/bin/pip" install --quiet --upgrade pip
    # gfx1151 index is the primary (torch + triton); PyPI is only an extra for
    # pure-python deps. The version constraint keeps pip on the gfx1151 ROCm
    # wheel instead of a newer PyPI CUDA torch.
    "$_venv/bin/pip" install --index-url "$TORCH_INDEX" \
        --extra-index-url https://pypi.org/simple "$TORCH_CONSTRAINT" || \
        die "torch install from ${TORCH_INDEX} failed."
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
