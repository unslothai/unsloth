#!/usr/bin/env bash
# Container startup checks for Unsloth (AMD ROCm build).
#
# Fails fast with actionable messages when the host GPU is not reachable, in
# the order the failures tend to bite:
#   1. /dev/kfd missing or unreadable   (no --device flags, no --group-add,
#                                        Docker Desktop, amdgpu not loaded)
#   2. rocm-smi sees no GPU             (no --device /dev/dri, driver too old)
#   3. torch is a HIP build and torch.cuda.is_available() is True
#   4. the card's gfx arch against what this image's wheels carry
#
# Bypass for offline tooling / CI:
#   docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    exec "$@"
fi

# UNSLOTH_DEV_ROOT prefixes the /dev probes (DESTDIR idiom) so the regression
# tests can stage a device tree; leave it unset in normal use.
DEV_ROOT="${UNSLOTH_DEV_ROOT:-}"
# What the image was built with; written by Dockerfile.rocm.
BUILD_INFO="${UNSLOTH_ROCM_BUILD_INFO:-/etc/unsloth-rocm-build}"
IMAGE_ROCM=""
IMAGE_GFX=""
if [[ -r "$BUILD_INFO" ]]; then
    IMAGE_ROCM="$(sed -n 's/^ROCM_VERSION=//p' "$BUILD_INFO")"
    IMAGE_GFX="$(sed -n 's/^ROCM_GFX=//p' "$BUILD_INFO")"
fi

# --- Check 1: /dev/kfd is accessible ----------------------------------------
# /dev/kfd is the AMD Kernel Fusion Driver node. It must exist AND be readable
# by the container user before any HIP / torch.cuda call can succeed.
if [[ ! -e "$DEV_ROOT/dev/kfd" ]]; then
    err "/dev/kfd not found inside the container."
    cat >&2 <<'MSG'

The AMD GPU device node is missing. Likely causes:

  1. The container was started without --device /dev/kfd --device /dev/dri.
     Re-launch with the bundled wrapper, which adds them and the group ids:
       bash docker/run.sh --rocm <cmd>
     Or by hand:
       docker run --device /dev/kfd --device /dev/dri \
         --group-add $(getent group video | cut -d: -f3) \
         --group-add $(getent group render | cut -d: -f3) \
         <other-flags> unsloth/unsloth-rocm:latest <cmd>

  2. The host has no /dev/kfd to pass through. Docker Desktop on Windows and
     macOS never has one (ROCm needs the Linux amdgpu driver; WSL exposes
     /dev/dxg instead), so this image cannot reach a GPU there. On Linux, check
     the driver:  lsmod | grep amdgpu

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

if [[ ! -r "$DEV_ROOT/dev/kfd" ]]; then
    err "/dev/kfd exists but is not readable by the container user."
    cat >&2 <<'MSG'
Add the host's video and render group ids to the container (NUMERIC: the
names do not exist inside the image):
  --group-add $(getent group video | cut -d: -f3) --group-add $(getent group render | cut -d: -f3)
bash docker/run.sh --rocm does this for you.
MSG
    exit 1
fi

# --- Check 2: rocm-smi present and can see at least one GPU -----------------
if ! command -v rocm-smi >/dev/null 2>&1; then
    err "rocm-smi not found inside the container."
    err "The ROCm runtime in this image is broken. Re-pull the image."
    exit 1
fi

if ! rocm-smi --showid 2>/dev/null | grep -q 'GPU\['; then
    err "No GPU visible to rocm-smi from inside the container."
    cat >&2 <<'MSG'

Likely causes (in order of frequency):

  1. The container was started without --device /dev/dri (both /dev/kfd and
     /dev/dri are needed). Re-launch with:
       bash docker/run.sh --rocm <cmd>

  2. The container user cannot open the render nodes: pass the host's video
     and render group ids with --group-add <gid> (run.sh --rocm does this).

  3. Host amdgpu driver too old for the ROCm version baked into this image.
     Check the host: rocm-smi --version  (and  rocminfo  to list the card)

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

# --- Check 3: a HIP torch that can see the device ---------------------------
# ROCm maps the CUDA Python API, so torch.cuda.is_available() is the runtime
# test; torch.version.hip first, so a CUDA or CPU torch that crept in is named
# as such rather than blamed on the host driver.
IMAGE_ROCM="$IMAGE_ROCM" python - >&2 <<'PY' || exit 1
import os
import sys

import torch

hip_ver = getattr(torch.version, "hip", None)
print(f"torch {torch.__version__}  HIP={hip_ver}")
if hip_ver is None:
    print("ERROR: this torch is not a ROCm build (torch.version.hip is None), so no")
    print("AMD GPU can be used. The image is broken: re-pull unsloth/unsloth-rocm, or")
    print("rebuild it with bash docker/build.sh --rocm.")
    sys.exit(1)

if torch.cuda.is_available():
    sys.exit(0)
image_rocm = os.environ.get("IMAGE_ROCM") or ".".join(hip_ver.split(".")[:2])
print("ERROR: torch.version.hip is set but torch.cuda.is_available() is False, despite")
print("rocm-smi seeing the card.")
print()
print(f"This image was built against ROCm {image_rocm} (HIP {hip_ver}). The host's")
print("amdgpu driver has to be at least as new. Check the host (NOT the container):")
print("  rocm-smi --version   or   cat /opt/rocm/.info/version   or   dkms status")
print()
print("If the host is older, upgrade its driver, or rebuild with the matching base:")
print("  ROCM_VERSION=<host version> TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm<X.Y> \\")
print("    bash docker/build.sh --rocm")
print("If HSA_OVERRIDE_GFX_VERSION is set, a wrong value also produces this.")
sys.exit(1)
PY

# --- Check 4: the card's gfx arch against this image's wheels ---------------
# PyTorch ROCm surfaces the gfx code in gcnArchName (e.g. "gfx1100:sramecc+").
# Not a gate (ROCm can often run an unlisted arch, and HSA_OVERRIDE_GFX_VERSION
# exists for the rest), but the one place the user is told which build to use.
IMAGE_ROCM="$IMAGE_ROCM" IMAGE_GFX="$IMAGE_GFX" python - >&2 <<'PY' || exit 1
import os
import sys

import torch

image_gfx = os.environ.get("IMAGE_GFX", "")
image_rocm = os.environ.get("IMAGE_ROCM") or ".".join(torch.version.hip.split(".")[:2])


def rocm_tuple(v):
    try:
        return tuple(int(x) for x in v.split(".")[:2])
    except ValueError:
        return (0, 0)


here = rocm_tuple(image_rocm)

# gfx -> (family, note). Families only: the marketing-name tables live in
# install.sh and studio/, and are kept in parity by their own test.
STRIX = {"gfx1150", "gfx1151", "gfx1152"}
RDNA4 = {"gfx1200", "gfx1201"}
FAMILY = {
    "gfx906": "GCN 5.1 / Vega 20",
    "gfx908": "CDNA 1",
    "gfx90a": "CDNA 2",
    "gfx940": "CDNA 3",
    "gfx941": "CDNA 3",
    "gfx942": "CDNA 3",
    "gfx1030": "RDNA 2",
    "gfx1031": "RDNA 2",
    "gfx1100": "RDNA 3",
    "gfx1101": "RDNA 3",
    "gfx1102": "RDNA 3",
    "gfx1150": "RDNA 3.5 APU",
    "gfx1151": "RDNA 3.5 APU",
    "gfx1152": "RDNA 3.5 APU",
    "gfx1200": "RDNA 4",
    "gfx1201": "RDNA 4",
}

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    arch = getattr(props, "gcnArchName", "").split(":")[0]  # strip e.g. :sramecc+
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}  arch={arch}  bf16={torch.cuda.is_bf16_supported()}")
    if not arch:
        continue
    fam = FAMILY.get(arch)
    if image_gfx:
        if arch == image_gfx or (arch in RDNA4 and image_gfx in RDNA4):
            print(f"  -> {fam}: this image was built for {image_gfx}")
        else:
            print(f"  NOTE: this image carries per-arch wheels for {image_gfx}, not {arch}.")
            print("        Use the generic image (unsloth/unsloth-rocm:latest), or rebuild:")
            print(f"          ROCM_GFX={arch} bash docker/build.sh --rocm")
    elif arch == "gfx906":
        if here >= (6, 4):
            print(f"  NOTE: {arch} ({fam}) has no kernels in ROCm {image_rocm}: AMD dropped it after 6.3,")
            print("        so rocBLAS will fail on the first matmul. Rebuild on the last ROCm that carries it:")
            print("          ROCM_VERSION=6.3.4 TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3 \\")
            print("            bash docker/build.sh --rocm")
        else:
            print(f"  -> {fam} (no prebuilt bitsandbytes kernels for gfx906: 4-bit needs a source build)")
    elif arch in STRIX or arch in RDNA4:
        print(f"  NOTE: {arch} ({fam}) runs best on AMD's per-arch wheels, which carry Strix/RDNA4")
        print("        fixes the generic rocm index lacks. Build that image with:")
        print(f"          ROCM_GFX={arch} bash docker/build.sh --rocm")
        print("        This generic image may still work; if training crashes, that is the fix.")
    elif fam:
        print(f"  -> {fam}")
    else:
        print(f"  NOTE: {arch} is not in the list of arches these wheels are built for.")
        print("        Training may still work if ROCm can JIT-compile for it. If not, set")
        print("        HSA_OVERRIDE_GFX_VERSION to the nearest built arch (RDNA2 cards other")
        print("        than gfx1030: 10.3.0; RDNA3 other than gfx1100: 11.0.0).")

sys.exit(0)
PY

exec "$@"
