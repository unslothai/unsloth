#!/usr/bin/env bash
# Container startup checks for Unsloth (AMD ROCm build).
#
# Fails fast with actionable messages when the host GPU is not reachable, in
# the order the failures tend to bite:
#   1. /dev/kfd missing or unreadable   (no --device flags, no --group-add,
#                                        Docker Desktop, amdgpu not loaded)
#   2. what rocm-smi sees, as a note    (it does not list every APU)
#   3. torch is a HIP build and torch.cuda.is_available() is True: the gate
#   4. the card's gfx arch against what this image's wheels carry
#
# Bypass for offline tooling / CI:
#   docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

# Studio image: relink its code into the home (maybe an earlier image's volume) before
# anything reads the venv, as entrypoint.sh does on the CUDA image. Fatal on failure
# (a half-linked home); no-op on the base image, which has no linker.
if [[ -x /usr/local/bin/unsloth-studio-home ]]; then
    /usr/local/bin/unsloth-studio-home || {
        echo "ERROR: could not link Unsloth Studio's code into ${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}; see the messages above" >&2
        exit 1
    }
fi

# best-effort, gated by UNSLOTH_SKIP_NOTEBOOK_SYNC, never blocks the container;
# no-op on the base image, which ships no notebooks
sync_notebooks() {
    if [[ -x /usr/local/bin/unsloth-sync-notebooks ]]; then
        /usr/local/bin/unsloth-sync-notebooks || true
    fi
}

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

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
# A per-arch image has native kernels for its gfx; a leftover
# HSA_OVERRIDE_GFX_VERSION (the generic-wheel workaround) would make ROCr present
# an arch this image has no kernels for. install.sh clears it the same way.
case "$IMAGE_GFX" in
    gfx1150|gfx1151|gfx1152|gfx1200|gfx1201)
        if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
            warn "ignoring HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}: this image carries native ${IMAGE_GFX} kernels"
            unset HSA_OVERRIDE_GFX_VERSION
        fi ;;
esac

# The skip flag bypasses the diagnostics only; the override cleanup above
# still applies, since it changes what the command sees.
if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    sync_notebooks
    exec "$@"
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

# --- Check 2: what rocm-smi sees (advisory) --------------------------------
# rocm-smi does not list every APU (measured on gfx1151: no GPU[..] line inside
# the container), so it cannot be the gate; check 3 asks torch itself.
if ! command -v rocm-smi >/dev/null 2>&1; then
    warn "rocm-smi not found inside the container; skipping its listing."
elif ! rocm-smi --showid 2>/dev/null | grep -q 'GPU\['; then
    warn "rocm-smi lists no GPU from inside the container. That is normal for some APUs;"
    warn "if the torch check below fails too, the usual causes are a missing --device /dev/dri,"
    warn "the host's video/render group ids not passed with --group-add (run.sh --rocm does"
    warn "both), or a host amdgpu driver older than the ROCm in this image (rocm-smi --version)."
else
    rocm-smi --showid 2>/dev/null | grep 'GPU\[' | head -4 >&2
fi

# --- Check 3: a HIP torch that can see the device ---------------------------
# ROCm maps the CUDA Python API, so torch.cuda.is_available() is the test;
# torch.version.hip first, so a CUDA or CPU torch is named, not the host driver.
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
print("ERROR: torch is a ROCm build but torch.cuda.is_available() is False: HIP could")
print("not open the device the container was given.")
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
# gcnArchName carries the gfx code (e.g. "gfx1100:sramecc+"). Not a gate (ROCm
# often runs unlisted arches), but where the user learns which build to use.
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


def kfd_physical_archs():
    """gfx arches the KERNEL sees, from KFD topology sysfs. amdkfd writes
    gfx_target_version itself, so HSA_OVERRIDE_GFX_VERSION (a ROCr userland
    spoof, which run.sh forwards) cannot hide the silicon here. Encoding is
    major*10000 + minor*100 + stepping: 100303 is gfx1033, 110501 is gfx1151.
    Same reader as install.sh's _kfd_gfx_targets; empty when sysfs is not there."""
    out = []
    # UNSLOTH_KFD_TOPOLOGY: the regression tests stage a topology; unset in normal use
    root = os.environ.get("UNSLOTH_KFD_TOPOLOGY") or "/sys/class/kfd/kfd/topology/nodes"
    if not os.path.isdir(root):
        return out
    for node in sorted(os.listdir(root)):
        try:
            with open(os.path.join(root, node, "properties"), encoding = "utf-8") as fh:
                props = dict(ln.split(None, 1) for ln in fh if len(ln.split(None, 1)) == 2)
        except OSError:
            continue
        gtv = int(props.get("gfx_target_version", "0").strip() or 0)
        if props.get("vendor_id", "").strip() != "4098" or gtv <= 0:
            continue
        maj, mn, step = gtv // 10000 % 100, gtv // 100 % 100, gtv % 100
        if maj <= 0 or mn > 9 or step > 15:
            continue
        out.append(f"gfx{maj}{mn}{step:x}")
    return out


physical = kfd_physical_archs()
if physical:
    print(f"KFD reports: {' '.join(physical)}")
if "gfx1033" in physical and os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
    print("ERROR: the kernel reports a gfx1033 (Van Gogh, Steam Deck) and HSA_OVERRIDE_GFX_VERSION is")
    print(f"       set ({os.environ['HSA_OVERRIDE_GFX_VERSION']}), which makes ROCm present it as another arch.")
    print("       gfx1033 computes incorrect results under ROCm whatever it is called (training")
    print("       diverges to NaN, studio/ROCM_RDNA2_APU.md), so this image refuses to train on it.")
    print("       UNSLOTH_SKIP_GPU_CHECK=1 bypasses this check; the results stay wrong.")
    sys.exit(1)

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    arch = getattr(props, "gcnArchName", "").split(":")[0]  # strip e.g. :sramecc+
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}  arch={arch}  bf16={torch.cuda.is_bf16_supported()}")
    if not arch:
        continue
    fam = FAMILY.get(arch)
    if arch == "gfx1033":
        # Van Gogh (Steam Deck): forward math looks fine but training diverges to
        # NaN and fails gradcheck (studio/ROCM_RDNA2_APU.md); install.sh routes it
        # to CPU torch, a HIP-only image can only refuse. Spoofing gfx1030 hides
        # the silicon from this check, not the arithmetic.
        print(f"ERROR: {arch} (Van Gogh, Steam Deck) computes incorrect results under ROCm:")
        print("       training diverges to NaN even though forward passes look valid, so this")
        print("       image refuses to train on it. Use the CPU image path (unsloth/unsloth")
        print("       with UNSLOTH_ALLOW_CPU=1) or llama.cpp over Vulkan for inference.")
        print("       UNSLOTH_SKIP_GPU_CHECK=1 bypasses this check; the results stay wrong.")
        sys.exit(1)
    if image_gfx:
        if arch == image_gfx or (arch in RDNA4 and image_gfx in RDNA4):
            print(f"  -> {fam}: this image was built for {image_gfx}")
            if image_gfx == "gfx906":
                print("     (no bitsandbytes: no prebuilt gfx906 kernels, so load_in_4bit is unavailable)")
        elif image_gfx == "gfx906":
            print(f"  NOTE: this image was built for gfx906 (ROCm 6.3, no bitsandbytes), not {arch}.")
            print("        Use the generic image: unsloth/unsloth-rocm:latest (bash docker/build.sh --rocm).")
        elif arch in STRIX or arch in RDNA4:
            print(f"  NOTE: this image carries per-arch wheels for {image_gfx}, not {arch}. Rebuild:")
            print(f"          ROCM_GFX={arch} bash docker/build.sh --rocm")
        else:
            print(f"  NOTE: this image carries per-arch wheels for {image_gfx}, not {arch}, which has")
            print("        no per-arch index. Use the generic image: unsloth/unsloth-rocm:latest")
            print("        (bash docker/build.sh --rocm).")
    elif arch == "gfx906":
        if here >= (6, 4):
            print(f"  NOTE: {arch} ({fam}) has no kernels in ROCm {image_rocm}: AMD dropped it after 6.3,")
            print("        so rocBLAS will fail on the first matmul. Rebuild on the last ROCm that carries it")
            print("        (ROCM_GFX=gfx906 leaves out bitsandbytes, which has no gfx906 kernels):")
            print("          ROCM_GFX=gfx906 ROCM_VERSION=6.3.4 TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm6.3 \\")
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

sync_notebooks
exec "$@"
