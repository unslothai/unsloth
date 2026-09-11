#!/usr/bin/env bash
# Container startup checks for Unsloth (AMD ROCm build).
#
# Fails fast with actionable messages when the host GPU is not reachable.
# Covers the most common failure modes, in the order they tend to bite:
#
#   1. /dev/kfd missing or unreadable
#        - User forgot --device /dev/kfd (or --group-add video for permission)
#        - Host amdgpu / amdkfd kernel module not loaded
#   2. rocm-smi can't see any GPU from inside the container
#        - User forgot --device /dev/dri, or host driver too old
#   3. rocm-smi works but torch.cuda.is_available() is False
#        - ROCm runtime version mismatch between host and image
#   4. The detected gfx arch is not in the precompiled list
#        - Hint to set HSA_OVERRIDE_GFX_VERSION to the nearest supported arch
#
# Bypass for offline tooling / CI:
#   docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    exec "$@"
fi

# --- Check 1: /dev/kfd is accessible ----------------------------------------
# /dev/kfd is the AMD Kernel Fusion Driver node. It must exist AND be readable
# by the container user before any HIP / torch.cuda call can succeed.
if [[ ! -e /dev/kfd ]]; then
    err "/dev/kfd not found inside the container."
    cat >&2 <<'MSG'

The AMD GPU device node is missing. Likely causes:

  1. You started the container without --device /dev/kfd --device /dev/dri.
     Re-launch with:
       docker run --device /dev/kfd --device /dev/dri --group-add video \
         <other-flags> unsloth/unsloth-rocm:latest <cmd>
     Or use the bundled wrapper:
       bash docker/run.sh --rocm <cmd>

  2. The host is missing the amdgpu / amdkfd kernel module.
     Check with:  lsmod | grep amdgpu
     Load with:   sudo modprobe amdgpu

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

if [[ ! -r /dev/kfd ]]; then
    err "/dev/kfd exists but is not readable by the container user."
    cat >&2 <<'MSG'
Add --group-add video (or --group-add render on newer kernels) to your
docker run command so the container user has permission to open /dev/kfd.
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

  1. You started the container without the AMD GPU device flags.
     Re-launch with:
       docker run --device /dev/kfd --device /dev/dri \
                  --group-add video <other-flags> \
                  unsloth/unsloth-rocm:latest <cmd>
     Or use the bundled wrapper:
       bash docker/run.sh --rocm <cmd>

  2. Your user is not in the 'video' group on the host.
     Fix:  sudo usermod -aG video,render $USER
     Then: log out and back in (or newgrp video)

  3. Host amdgpu driver too old for the ROCm version baked into this image.
     Check the host: rocm-smi --version  (and  rocminfo  to list the card)

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

# --- Check 3: torch.cuda.is_available() (ROCm maps the CUDA Python API) ----
python - >&2 <<'PY' || exit 1
import sys
import torch

hip_ver = getattr(torch.version, "hip", None)
print(f"torch {torch.__version__}  HIP={hip_ver}")

if torch.cuda.is_available():
    sys.exit(0)
print("ERROR: torch.cuda.is_available() is False despite rocm-smi working.")
print()
print("This image was built against ROCm 6.2.  The host ROCm stack must be")
print("6.2 or newer.  Check the host (NOT the container) with:")
print("  rocm-smi --version")
print()
print("If the host ROCm version is older than 6.2, either upgrade the host")
print("drivers or use an image built against an older ROCm version.")
print("If HSA_OVERRIDE_GFX_VERSION is set, an incorrect value can also cause this.")
sys.exit(1)
PY

# --- Check 4: detected gfx arch is in the supported list ---------------------
# PyTorch ROCm surfaces the gfx code in gcnArchName (e.g. "gfx1100:sramecc+").
# We don't gate on it -- ROCm can often JIT-compile for unlisted archs -- but
# we print a clear HSA_OVERRIDE hint when the card isn't precompiled.
python - >&2 <<'PY' || exit 1
import sys
import torch

# SUPPORTED maps gfx arch string -> (family, example cards).
SUPPORTED = {
    "gfx906":  ("Vega20 / CDNA0", "Radeon VII"),
    "gfx908":  ("CDNA1",          "Instinct MI100"),
    "gfx90a":  ("CDNA2",          "Instinct MI200 / MI210 / MI250"),
    "gfx940":  ("CDNA3",          "Instinct MI300A"),
    "gfx942":  ("CDNA3",          "Instinct MI300X"),
    "gfx1030": ("RDNA2",          "RX 6800 / 6800 XT / 6900 XT"),
    "gfx1031": ("RDNA2",          "RX 6700 XT"),
    "gfx1100": ("RDNA3",          "RX 7900 XTX / XT"),
    "gfx1101": ("RDNA3",          "RX 7800 / 7700 XT"),
    "gfx1102": ("RDNA3",          "RX 7600"),
    "gfx1150": ("RDNA3.5 APU",    "Strix Point (needs rocm7.2 image)"),
    "gfx1151": ("RDNA3.5 APU",    "Strix Halo / 8060S (needs rocm7.2 image)"),
    "gfx1200": ("RDNA4",          "RX 9060 (needs rocm7.2 image)"),
    "gfx1201": ("RDNA4",          "RX 9070 / 9070 XT (needs rocm7.2 image)"),
}

n = torch.cuda.device_count()
for i in range(n):
    name  = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    arch  = getattr(props, "gcnArchName", "").split(":")[0]  # strip e.g. :sramecc+
    bf16  = torch.cuda.is_bf16_supported()
    print(f"GPU {i}: {name}  arch={arch}  bf16={bf16}")
    if arch and arch in SUPPORTED:
        fam, ex = SUPPORTED[arch]
        print(f"  -> Supported: {fam} ({ex})")
    elif arch:
        print(f"  NOTE: {arch} is not in the precompiled arch list.")
        print( "        Training may still work if ROCm can JIT-compile for it.")
        print( "        If it fails, set HSA_OVERRIDE_GFX_VERSION to the nearest")
        print( "        supported arch (e.g. RX 6800 -> HSA_OVERRIDE_GFX_VERSION=10.3.0),")
        print( "        or rebuild for RDNA3.5/RDNA4 with a rocm7.2 image:")
        print( "          ROCM_VERSION=7.x.x TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 \\")
        print( "            bash docker/build.sh --rocm")
        print( "        Tip: UNSLOTH_ROCM_GFX_ARCH=gfx1151 (or your arch) pins the target.")

sys.exit(0)
PY

exec "$@"
