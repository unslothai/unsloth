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

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }
note() { printf "\033[1;36mNOTE:\033[0m %s\n"  "$*" >&2; }

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
    exec "$@"
fi

# --- Check 1: a device node the HSA runtime can open ------------------------
# Two ways in, and only one of them is /dev/kfd:
#   KFD  the AMD Kernel Fusion Driver node, from the Linux amdgpu driver. Must
#        exist AND be readable before any HIP call can succeed.
#   DXG  WSL2, where amdgpu is not loaded and there is no /dev/kfd at all. The
#        standard hsa-rocr runtime reaches the card through librocdxg over
#        /dev/dxg when HSA_ENABLE_DXG_DETECTION=1. Same evidence install.sh
#        gates a WSL host on (_infer_linux_amd_gfx_arch).
# The DXG path takes the rest of the checks unchanged: check 3 asks torch, which
# is the only gate that has ever mattered.
UNSLOTH_ROCM_DEV_PATH=kfd
if [[ ! -e "$DEV_ROOT/dev/kfd" && -e "$DEV_ROOT/dev/dxg" ]]; then
    _dxg_lib=""
    # UNSLOTH_ROCM_DXG_LIBDIRS pins the search for the regression tests, which must
    # not answer for whatever ROCm the machine running them happens to have.
    for _d in ${UNSLOTH_ROCM_DXG_LIBDIRS:-/opt/rocm/lib /opt/rocm/lib64 /opt/rocm-*/lib \
              /opt/rocm-*/lib64 /usr/lib/x86_64-linux-gnu}; do
        if [[ -e "$_d/librocdxg.so" || -e "$_d/librocdxg.so.1" ]]; then
            _dxg_lib="$_d"
            break
        fi
    done
    if [[ -n "$_dxg_lib" ]]; then
        UNSLOTH_ROCM_DEV_PATH=dxg
        # Only the runtime reads this, and only when asked; exporting it here means
        # a plain `docker run --device /dev/dxg` works without the caller knowing.
        export HSA_ENABLE_DXG_DETECTION="${HSA_ENABLE_DXG_DETECTION:-1}"
        note "no /dev/kfd, but /dev/dxg and librocdxg are present: using the WSL2 DXG bridge."
    else
        err "/dev/dxg is present but librocdxg is not, so the DXG bridge cannot load."
        cat >&2 <<'MSG'

This looks like WSL2, where there is no /dev/kfd and the card is reached through
librocdxg instead. The image cannot ship that library (its build needs Windows
SDK headers), so it comes off the host, along with the libdxcore it dlopens from
WSL's own lib directory. The bundled wrapper mounts both:
  bash docker/run.sh --rocm <cmd>
Or by hand:
  docker run --device /dev/dxg \
    -v /opt/rocm/lib/librocdxg.so.1:/usr/lib/x86_64-linux-gnu/librocdxg.so:ro \
    -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
    <other-flags> unsloth/unsloth-rocm:latest <cmd>

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
        exit 1
    fi
fi

if [[ "$UNSLOTH_ROCM_DEV_PATH" == "kfd" && ! -e "$DEV_ROOT/dev/kfd" ]]; then
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

  2. The host has no /dev/kfd to pass through. On Linux, check the driver:
       lsmod | grep amdgpu
     On Windows there is never one: the card is reached over WSL2's /dev/dxg,
     which only a docker engine running INSIDE your WSL distribution can pass
     through (`bash docker/run.sh --rocm` does it). Docker Desktop's own engine
     exposes neither node.

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

if [[ "$UNSLOTH_ROCM_DEV_PATH" == "kfd" && ! -r "$DEV_ROOT/dev/kfd" ]]; then
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
IMAGE_ROCM="$IMAGE_ROCM" UNSLOTH_ROCM_DEV_PATH="$UNSLOTH_ROCM_DEV_PATH" python - >&2 <<'PY' || exit 1
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

# BEFORE the first torch.cuda call, which is where this would otherwise abort with a
# glog FATAL and no explanation. librocprofiler-sdk enumerates GPUs from the KFD sysfs
# topology; on the DXG bridge there is none, so it finds zero agents, disagrees with the
# HSA agents that are there, and calls abort(). Match that library alone: the
# librocprofiler-register.so beside it is harmless, and torch 2.11+rocm7.2 ships it and
# runs here, so matching "rocprof" refuses a build that works.
if os.environ.get("UNSLOTH_ROCM_DEV_PATH") == "dxg":
    _lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    _prof = sorted(f for f in os.listdir(_lib) if f.startswith("librocprofiler-sdk")) if os.path.isdir(_lib) else []
    if _prof:
        print("ERROR: this image cannot use the WSL2 DXG bridge.")
        print()
        print(f"Its torch bundles {', '.join(_prof)}, which enumerates GPUs through the")
        print("KFD sysfs topology that WSL does not have; it aborts rather than falling back.")
        print("Two builds do work here: AMD's per-arch wheels, which bundle no rocprofiler")
        print("(Strix APUs and RDNA4: gfx1150/1151/1152/1200/1201, no RDNA3 yet),")
        print("  ROCM_GFX=<your gfx, e.g. gfx1201> bash docker/build.sh --rocm")
        print("or a torch before 2.12 from the same index, which ships only -register.so.")
        print("UNSLOTH_SKIP_GPU_CHECK=1 reaches the abort itself rather than avoiding it.")
        sys.exit(1)

if torch.cuda.is_available():
    sys.exit(0)
image_rocm = os.environ.get("IMAGE_ROCM") or ".".join(hip_ver.split(".")[:2])
print("ERROR: torch is a ROCm build but torch.cuda.is_available() is False: HIP could")
print("not open the device the container was given.")
print()


if os.environ.get("UNSLOTH_ROCM_DEV_PATH") == "dxg":
    print("This container is on the WSL2 DXG bridge, so the host amdgpu driver advice")
    print("below does not apply; check the Windows AMD driver instead (a current Adrenalin")
    print("with ROCDXG support, as scripts/install_rocm_wsl_strixhalo.sh documents).")
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

exec "$@"
