#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# ──────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL: enable ROCm-on-WSL for AMD Strix Halo (Radeon 8060S / gfx1151)
# ──────────────────────────────────────────────────────────────────────────────
# Unsloth's installer already routes gfx1151 to the right ROCm wheels once a
# ROCm runtime is present (see install.sh: 8060S -> gfx1151, repo.amd.com/rocm/
# whl/gfx1151). What it does NOT do is install AMD's driver/ROCm stack -- that is
# a large, admin-gated, bleeding-edge prerequisite. This helper automates the
# Linux side of that prerequisite on a *dedicated* Ubuntu 24.04 WSL2 distro.
#
# RUN THIS *AFTER* the two manual prerequisites are done (the preflight enforces
# them and refuses to half-install otherwise):
#   1. Windows: AMD Adrenalin driver >= 26.3.1 installed (admin) + reboot.
#   2. A dedicated Ubuntu-24.04 WSL distro:  wsl --install Ubuntu-24.04
#      (do NOT run this in an Ubuntu 26.04 distro -- ROCm 7.2 targets 24.04).
#
# Procedure adapted from AMD's ROCm-on-WSL docs and the community
# andweng/wsl-rocm gfx1151 notes. This is bleeding edge; read each step before
# trusting it on a work machine.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# Verified against: ROCm 7.2 + AMD Adrenalin 26.3.1 + Ubuntu 24.04.4 + WSL2,
# circa 2026-06. These are MOVING targets -- the amdgpu-install .deb is scraped
# from repo.radeon.com, librocdxg is built from a Git ref, and the version pins
# below WILL rot. Bump them (and re-verify) when AMD ships a newer ROCm-on-WSL.
ROCM_VER="7.2.0"           # ROCm release to install; bump as AMD ships newer
ROCM_BIN_VER="7.2.1"       # rocr4wsl userspace tools ship under 7.2.1
GFX="gfx1151"
LIBROCDXG_REF="develop"    # ROCm/librocdxg ref to build; pin to a known-good commit if develop breaks
WIN_SDK_INC='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0'
TORCH_INDEX="https://repo.amd.com/rocm/whl/${GFX}/"   # what install.sh routes gfx1151 to
ROCM_DIR=""                # resolved to the real /opt/rocm-* dir after install (may differ from ROCM_VER)

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die()  { printf '\n\033[1;31m[BLOCKED] %s\033[0m\n' "$*" >&2; exit 1; }

# ── PREFLIGHT: fail fast with a clear reason rather than half-installing ──
say "Preflight checks"

# shellcheck disable=SC1091
. /etc/os-release 2>/dev/null || true
[ "${VERSION_ID:-}" = "24.04" ] || die "This script targets Ubuntu 24.04 (found '${VERSION_ID:-unknown}'). Create a dedicated distro:  wsl --install Ubuntu-24.04  -- do not run on 26.04."

[ -e /dev/dxg ] || die "/dev/dxg missing -- WSL GPU paravirtualization not present. Ensure WSL2 (not WSL1) and a recent Windows build."

# The Adrenalin >=26.3.1 driver injects ROCm/DXG runtime libs into
# /usr/lib/wsl/lib. If only the base D3D12/dxcore libs are present, the driver
# is too old -- stop before touching apt.
if ! ls /usr/lib/wsl/lib/ 2>/dev/null | grep -qiE 'hsa|rocm|amdhip|dxg'; then
    cat >&2 <<'EOF'

[BLOCKED] No AMD ROCm/DXG runtime libraries found in /usr/lib/wsl/lib
          (only the base D3D12/dxcore libs are present).

The Windows AMD Adrenalin driver does NOT yet provide ROCm-on-WSL.
Fix (needs Windows admin), then reboot and re-run this script:
  1. Update to AMD Adrenalin Edition >= 26.3.1 for Ryzen AI Max+ / Radeon 8060S:
       https://www.amd.com/en/support/download/drivers.html
  2. Reboot Windows.
  3. Verify in WSL:  ls /usr/lib/wsl/lib/   (should now list hsa/rocm libs)
EOF
    exit 1
fi
echo "Preflight OK: Ubuntu 24.04, /dev/dxg present, driver injects WSL ROCm libs."

# ── Step 0: build deps ──
say "Installing build tools"
sudo apt update
sudo apt install -y cmake gcc g++ git wget python3-venv python3-pip

# ── Step 1: ROCm 7.2 (WSL usecase, no kernel DKMS -- WSL uses the Windows driver) ──
say "Installing ROCm ${ROCM_VER} (WSL usecase)"
if [ ! -d "/opt/rocm-${ROCM_VER}" ]; then
    # Grab the current amdgpu-install for ROCm 7.2 / noble. If this URL 404s,
    # pick the matching .deb from
    # https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/noble/
    base="https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/noble"
    deb_name="$(wget -qO- "${base}/" | grep -oE 'amdgpu-install_[0-9][^"]*_all\.deb' | sort -u | tail -1)"
    [ -n "${deb_name}" ] || die "Could not find amdgpu-install .deb under ${base}/ -- check ROCM_VER."
    wget -O /tmp/amdgpu-install.deb "${base}/${deb_name}"
    sudo apt install -y /tmp/amdgpu-install.deb
    sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
fi
# Resolve the actual install dir -- amdgpu-install may lay ROCm down under a
# patch-version dir (e.g. /opt/rocm-7.2.1) that differs from $ROCM_VER. Prefer
# the generic /opt/rocm symlink, else the newest /opt/rocm-* dir.
if [ -d /opt/rocm ]; then ROCM_DIR="/opt/rocm"; else ROCM_DIR="$(ls -d /opt/rocm-* 2>/dev/null | sort -V | tail -1)"; fi
{ [ -n "${ROCM_DIR}" ] && [ -d "${ROCM_DIR}" ]; } || die "ROCm not installed under /opt (expected /opt/rocm or /opt/rocm-*)."
echo "Using ROCm at ${ROCM_DIR}"

# ── Step 2: swap in the WSL/DXG-aware HSA runtime ──
say "Installing WSL-specific HSA runtime (rocr4wsl)"
sudo apt remove -y libhsa-runtime64-1 libhsakmt1 || true
sudo apt update
sudo apt install -y hsa-runtime-rocr4wsl-amdgpu
dpkg -l | grep -q rocr4wsl || die "hsa-runtime-rocr4wsl-amdgpu did not install."

# ── Step 3: build librocdxg (DXG <-> HSA bridge) ──
say "Building librocdxg (${LIBROCDXG_REF})"
[ -d "${WIN_SDK_INC}/shared" ] || die "Windows SDK headers not found at ${WIN_SDK_INC}/shared. Install the Windows 11 SDK on Windows (or adjust WIN_SDK_INC)."
rm -rf "${HOME}/librocdxg"
git clone https://github.com/ROCm/librocdxg.git "${HOME}/librocdxg"
( cd "${HOME}/librocdxg" && git checkout "${LIBROCDXG_REF}" \
    && mkdir -p build && cd build \
    && cmake .. -DWIN_SDK="${WIN_SDK_INC}/shared" \
    && make -j"$(nproc)" \
    && sudo make install )
# Fix the soname symlink ROCm's loader expects (under the resolved ROCm dir)
if [ -f "${ROCM_DIR}/lib/librocdxg.so.1.1.0" ]; then
    sudo ln -sf "librocdxg.so.1.1.0" "${ROCM_DIR}/lib/librocdxg.so.1"
fi
sudo ldconfig

# ── Step 4: persist environment ──
say "Writing ROCm env to ~/.bashrc"
if ! grep -q "ROCm-on-WSL (gfx1151)" "${HOME}/.bashrc" 2>/dev/null; then
    {
        echo "# >>> ROCm-on-WSL (gfx1151) >>>"
        echo "export HSA_ENABLE_DXG_DETECTION=1"
        echo "export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
        echo "export PATH=${ROCM_DIR}/bin:/opt/rocm/bin:\$PATH"
        echo "export LD_LIBRARY_PATH=${ROCM_DIR}/lib:\${LD_LIBRARY_PATH:-}"
        echo "# <<< ROCm-on-WSL (gfx1151) <<<"
    } >> "${HOME}/.bashrc"
fi
export HSA_ENABLE_DXG_DETECTION=1
export PATH="${ROCM_DIR}/bin:/opt/rocm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_DIR}/lib:${LD_LIBRARY_PATH:-}"

# ── Step 5: verify the runtime sees the GPU ──
say "Verifying rocminfo sees ${GFX}"
rocminfo | grep -E 'Load librocdxg|gfx1151|Marketing Name' || die "rocminfo did not enumerate ${GFX}. Re-check the driver version and reboot."

# ── Step 6: PyTorch (the same index install.sh routes gfx1151 to) ──
say "Installing PyTorch for ${GFX} and testing GPU"
python3 -m venv "${HOME}/rocm-test"
# shellcheck disable=SC1091
. "${HOME}/rocm-test/bin/activate"
pip install --upgrade pip
pip install --index-url "${TORCH_INDEX}" torch torchvision torchaudio
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda(=rocm) available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok:", (x @ x).sum().item() is not None)
PY

say "Done. If torch.cuda.is_available() is True, re-run the unsloth installer in THIS distro:"
echo "  curl -fsSL https://unsloth.ai/install.sh | sh"
echo "  (it will detect ${GFX} via rocminfo and pull the matching ROCm wheels automatically)"
