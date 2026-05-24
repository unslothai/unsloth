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
print("Most likely the host NVIDIA driver is too old for CUDA 12.8.")
print("Required host driver versions for this image:")
print("  >= 570   RTX 50-series, RTX 6000 Pro Blackwell  (sm_120)")
print("  >= 555   B100 / B200                            (sm_100)")
print("  >= 535   H100 / H200                            (sm_90)")
print("  >= 525   Ada / Ampere                           (sm_80 / sm_86 / sm_89)")
print()
print("Check the host (NOT the container) with:  nvidia-smi")
print("Then upgrade the driver to match your GPU.")
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
if major < 8:
    print()
    print(f"ERROR: Unsloth requires Ampere or newer (sm_80+). Got {name} sm_{major}{minor}.")
    print()
    print("Supported architectures baked into this image:")
    print("  sm_80   Ampere       (A100, A40, A30)")
    print("  sm_86   Ampere       (RTX 30-series, A10)")
    print("  sm_89   Ada          (RTX 40-series, L40)")
    print("  sm_90   Hopper       (H100, H200)")
    print("  sm_100  Blackwell DC (B100, B200)")
    print("  sm_120  Blackwell    (RTX 50-series, RTX 6000 Pro Blackwell)")
    sys.exit(1)
PY

exec "$@"
