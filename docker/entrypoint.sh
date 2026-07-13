#!/usr/bin/env bash
# Container startup checks for Unsloth.
#
# Fails fast with actionable error messages when the host GPU isn't reachable,
# instead of letting torch crash deep with cryptic CUDA errors. Catches the
# three failure modes that cover ~95% of "it doesn't work" tickets:
#
#   1. nvidia-smi inside the container can't see any GPU
#        - User forgot --gpus all
#        - Host missing nvidia-container-toolkit
#   2. nvidia-smi works but torch.cuda.is_available() is False
#        - Host driver too old for CUDA 12.8
#   3. GPU attaches but is older than Ampere (sm < 80)
#        - Unsloth requires sm_80+
#
# Bypass for offline tooling / docs / CI:
#   docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

# --- CUDA JIT toolchain selection (device-gated) ----------------------------
# The image bakes CUDA 13 ptxas + NVRTC ONLY so the two Blackwell datacenter
# arches the cu12.8 tools cannot target -- sm_103 (B300 / GB300) and sm_121
# (GB10 / DGX Spark) -- can JIT Triton and torch/NVRTC kernels. Both launched
# AFTER cu12.8, so any host carrying them runs a >= 580 driver, which is exactly
# what a cu13-produced cubin needs to LOAD.
#
# Every OTHER supported arch (Turing..sm_120) works with the bundled cu12.8
# tools and is allowed on a 570-579 driver (the documented floor). A cu13 cubin
# CANNOT load on a 570-579 driver even when it targets an old arch like sm_80
# (CUDA has forward, not backward, driver compatibility across major versions),
# so routing those hosts' JIT through the cu13 tools would break ordinary
# training. ptxas/NVRTC are host-side compilers (they never link libcuda), so
# they RUN under any driver -- it is only their OUTPUT the older driver rejects.
#
# Pick per DEVICE at boot (the compute capability is unknown at build time):
# cu12.8 is the immutable baked default (loadable on every supported 570+
# driver), and only sm_103 / sm_121 -- which ship on >= 580 drivers -- switch
# Triton to cu13 ptxas and retarget the venv NVRTC symlink to the staged cu13
# alias. Runs before every early-exit below so the selection always applies.
# Best-effort: because the safe default needs no write, a non-root / read-only
# rootfs is always fine; only the rare non-root datacenter host cannot switch.
select_cuda_jit_tools() {
    local caps="" cc nvrtc_dir need_cu13=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        caps="$( { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true; } )"
    fi
    # Scan EVERY visible GPU, not just the first: a Blackwell datacenter part
    # (sm_103 B300/GB300 or sm_121 GB10/DGX Spark) can sit behind an H100/B200 in
    # the nvidia-smi ordering, so keying off only the first compute_cap would miss
    # it. If ANY visible GPU needs cu13, switch to it for the whole process --
    # those parts only ship on >= 580 drivers, so the host tolerates cu13 cubins
    # for every arch present.
    while IFS= read -r cc || [[ -n "${cc}" ]]; do
        cc="$(printf '%s' "${cc}" | tr -d '[:space:]')"
        case "${cc}" in
            10.3|12.1) need_cu13=1 ;;
        esac
    done <<< "${caps}"
    # Non-datacenter / undetectable / CPU host: cu12.8 is the immutable baked
    # default (libnvrtc.so.12 -> .cu128.orig, Triton on its bundled cu12.8
    # ptxas), loadable on every supported 570+ driver, and needs NO write -- so
    # a non-root `docker run --user` container is never left on a cu13 NVRTC a
    # 570-579 driver cannot load. One exception needs a write: an earlier boot
    # of this SAME container on sm_103/sm_121 left libnvrtc.so.12 -> .cu13 in
    # the writable layer, and the container now runs on a GPU whose 570-579
    # driver cannot load cu13 output -- deterministically reverse exactly that
    # selection (best-effort, same non-root caveat as the forward switch).
    if [[ "${need_cu13}" -ne 1 ]]; then
        for nvrtc_dir in \
            /opt/unsloth-venv/lib/python*/site-packages/nvidia/cuda_nvrtc/lib \
            "${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"/unsloth_studio/lib/python*/site-packages/nvidia/cuda_nvrtc/lib; do
            [[ -e "${nvrtc_dir}/libnvrtc.so.12.cu128.orig" ]] || continue
            [[ "$(readlink "${nvrtc_dir}/libnvrtc.so.12" 2>/dev/null)" == "libnvrtc.so.12.cu13" ]] || continue
            ln -sf libnvrtc.so.12.cu128.orig "${nvrtc_dir}/libnvrtc.so.12" 2>/dev/null || true
        done
        return 0
    fi
    # Blackwell datacenter present: cu12.8 cannot emit compute_103/121, so point
    # Triton at cu13 ptxas and retarget each venv's libnvrtc.so.12 -> the staged
    # cu13 alias. -z guard leaves an explicit `docker run -e TRITON_PTXAS_PATH`
    # win. Best-effort: a read-only / --user rootfs that cannot rewrite the
    # symlink simply keeps cu12.8 (a rare non-root datacenter case). Covers the
    # base venv and, on the Studio image, the Studio venv.
    if [[ -x /usr/local/cuda-13.0/bin/ptxas && -z "${TRITON_PTXAS_PATH:-}" ]]; then
        export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
    fi
    for nvrtc_dir in \
        /opt/unsloth-venv/lib/python*/site-packages/nvidia/cuda_nvrtc/lib \
        "${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}"/unsloth_studio/lib/python*/site-packages/nvidia/cuda_nvrtc/lib; do
        [[ -e "${nvrtc_dir}/libnvrtc.so.12.cu13" ]] || continue
        ln -sf libnvrtc.so.12.cu13 "${nvrtc_dir}/libnvrtc.so.12" 2>/dev/null || true
    done
}
# Best-effort: never let JIT-tool selection block container startup.
select_cuda_jit_tools || true

# Make the unslothai/notebooks collection available under /workspace before the
# user command runs (JupyterLab, unsloth-run, or a shell). Best-effort: it is
# fully gated by UNSLOTH_SKIP_NOTEBOOK_SYNC and never blocks or fails the
# container (see unsloth_sync_notebooks.sh).
sync_notebooks() {
    if [[ -x /usr/local/bin/unsloth-sync-notebooks ]]; then
        /usr/local/bin/unsloth-sync-notebooks || true
    fi
}

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    sync_notebooks
    exec "$@"
fi

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

# CPU mode for hosts that cannot pass a GPU into a Linux container at all:
# Docker Desktop on macOS (no Metal passthrough), Docker Desktop on Windows
# without WSL2 GPU support, plain CPU Linux boxes, and CI runners. CPU mode
# covers Jupyter, the GGUF tooling and llama.cpp-backed Studio chat (llama.cpp
# runs on CPU), and Data Recipes. It does NOT cover training or loading an
# Unsloth model for chat (FastLanguageModel.from_pretrained runs CUDA probes
# like torch.cuda.get_device_properties and raises without a GPU). With
# UNSLOTH_ALLOW_CPU=1 a missing GPU degrades to a warning instead of the hard
# pre-flight failure; when a GPU IS visible the normal checks below still run so
# a broken GPU setup is not silently ignored.
if [[ "${UNSLOTH_ALLOW_CPU:-0}" == "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        warn "UNSLOTH_ALLOW_CPU=1 and no GPU visible -- continuing on CPU."
        warn "CPU mode covers Jupyter, GGUF tooling and llama.cpp (GGUF) Studio chat."
        warn "Training and loading Unsloth models (FastLanguageModel) still require an NVIDIA GPU."
        sync_notebooks
        exec "$@"
    fi
fi

# --- Check 1: nvidia-smi present and can enumerate at least one GPU ---------
# nvidia-smi is injected by nvidia-container-toolkit when the container is
# started with a GPU request; it is NOT baked into the image. A missing
# binary therefore means "no GPU was attached", the same failure class as
# an empty -L listing, not a broken image.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    err "No GPU visible inside the container."
    cat >&2 <<'MSG'

Likely causes (in order of frequency):

  1. You started the container without --gpus all.
     Re-launch with:
       docker run --gpus all <other-flags> unsloth/unsloth:latest <cmd>
     Or use the bundled wrapper:
       bash docker/run.sh <cmd>

  2. Host is missing nvidia-container-toolkit.
     Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
     Then:    sudo systemctl restart docker

  3. nvidia-container-toolkit is installed but the Docker daemon was not
     restarted after install. Run:
       sudo systemctl restart docker

  4. You are using Podman / Kubernetes / a managed container service that
     needs a different GPU flag than --gpus all. See the relevant docs:
       podman:     --device nvidia.com/gpu=all
       k8s:        nvidia.com/gpu resource request + GPU operator

  5. This host has no NVIDIA GPU at all (Docker Desktop on macOS, Windows
     without WSL2 GPU support, CPU-only Linux). Training and loading Unsloth
     models need a GPU, but Jupyter, GGUF tooling and llama.cpp (GGUF) Studio
     chat work on CPU:
       docker run -e UNSLOTH_ALLOW_CPU=1 ...

To bypass this check entirely (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

# --- Check 2: torch can actually use the GPU --------------------------------
# This catches host-driver-too-old (the GPU enumerates via nvidia-smi but
# the kernel module rejects CUDA contexts).
python - >&2 <<'PY' || exit 1
import sys
import torch
if torch.cuda.is_available():
    sys.exit(0)
print("ERROR: torch.cuda.is_available() is False despite nvidia-smi working.")
print()
print("This image bakes in CUDA 12.8, so the host driver MUST be:")
print("  >= 570.26   (toolkit floor for cu128, applies to every GPU)")
print()
print("Two GPUs need an even newer driver because their launch driver was")
print("released after cu128's:")
print("  >= 580      B300 / GB300                              (sm_103)")
print("  >= 580      GB10 / DGX Spark                          (sm_121)")
print()
print("Check the host (NOT the container) with:  nvidia-smi")
print("Then upgrade the driver to match.")
sys.exit(1)
PY

# --- Check 3: compute capability is supported -------------------------------
python - >&2 <<'PY' || exit 1
import sys
import torch
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
n = torch.cuda.device_count()
print(f"Unsloth container: {n} GPU(s). Primary: {name}  sm_{major}{minor}  bf16={torch.cuda.is_bf16_supported()}")

# Image targets every current x86_64 NVIDIA arch from Turing onward, per
# https://developer.nvidia.com/cuda/gpus.
SUPPORTED = (
    ("sm_75",  "Turing",       "T4, RTX 20-series, Quadro RTX"),
    ("sm_80",  "Ampere DC",    "A100, A30"),
    ("sm_86",  "Ampere",       "A40, RTX A6000, RTX 30-series"),
    ("sm_89",  "Ada",          "L4, L40, L40S, RTX 40-series"),
    ("sm_90",  "Hopper",       "H100, H200, GH200"),
    ("sm_100", "Blackwell DC", "B100, B200, GB200"),
    ("sm_103", "Blackwell DC", "B300, GB300"),
    ("sm_120", "Blackwell",    "RTX 50-series, RTX PRO 6000 Blackwell"),
    ("sm_121", "Blackwell",    "GB10 (DGX Spark)"),
)
if major < 7 or (major == 7 and minor < 5):
    print()
    print(f"ERROR: Unsloth image requires Turing or newer (sm_75+). Got {name} sm_{major}{minor}.")
    print()
    print("Supported architectures in this image:")
    for arch, fam, ex in SUPPORTED:
        print(f"  {arch:7s} {fam:13s} ({ex})")
    sys.exit(1)
if major < 8:
    print(f"NOTE: {name} is Turing (sm_{major}{minor}) -- bfloat16 is not supported.")
    print("      Unsloth will fall back to fp16. Training works but is slightly slower.")
PY

sync_notebooks
exec "$@"
