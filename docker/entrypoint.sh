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

# DGX Spark fix, arm64 image only: prefer the cu13 ptxas we baked into the
# image at /usr/local/cuda-13.0/bin/ptxas over Triton's bundled tools. The
# file only exists on the arm64 variant; amd64 images skip this and use
# Triton's own ptxas (cu13 in triton>=3.6.0).
if [[ -x /usr/local/cuda-13.0/bin/ptxas ]] && [[ -z "${TRITON_PTXAS_PATH:-}" ]]; then
    export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
fi

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    exec "$@"
fi

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

# --- Check 1: nvidia-smi present and can enumerate at least one GPU ---------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi not found inside the container."
    err "The CUDA runtime in this image is broken. Re-pull the image."
    exit 1
fi

if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    err "No GPU visible to nvidia-smi from inside the container."
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

To bypass this check (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
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

exec "$@"
