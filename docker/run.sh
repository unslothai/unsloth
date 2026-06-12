#!/usr/bin/env bash
# Convenience wrapper for `docker run unsloth/unsloth[-rocm]`.
# Sets the GPU passthrough flags and mounts that people most often forget.
#
# NVIDIA (default):
#   --gpus all           Without this, no GPU is attached and the container's
#                        entrypoint will refuse to start.
#
# AMD (--rocm):
#   --device /dev/kfd    AMD GPU compute node -- required for any torch.cuda call.
#   --device /dev/dri    AMD render nodes -- required for ROCm libs.
#   --group-add video    Grants render group access inside the container.
#
# Both backends:
#   --ipc=host           PyTorch DataLoader workers need ample /dev/shm. The
#                        default 64MB causes "DataLoader worker (pid X) exited
#                        unexpectedly" on any non-trivial dataset.
#   --ulimit memlock=-1  Unlimited pinned memory for NCCL / ROCm pinned buffers.
#   --ulimit stack=64MB  Larger thread stack for libtorch.
#
# Usage:
#   bash docker/run.sh [--rocm] [cmd...]
#
#   bash docker/run.sh                                   # NVIDIA, python REPL
#   bash docker/run.sh python /workspace/smoke_test.py   # NVIDIA smoke test
#   bash docker/run.sh --rocm                            # AMD, python REPL
#   bash docker/run.sh --rocm python /workspace/smoke_test_rocm.py
#   bash docker/run.sh --rocm bash                       # AMD shell
#
# The full NVIDIA image (unsloth/unsloth:latest) starts Studio (8000) +
# JupyterLab (8888) by default; publish the ports when you want them:
#   UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
# CPU-only hosts (Docker Desktop on macOS, Windows without WSL2 GPU, plain
# CPU Linux): no --gpus and set UNSLOTH_ALLOW_CPU=1. Training is unavailable
# but Studio chat / Data Recipes, Jupyter and GGUF tooling work:
#   UNSLOTH_GPUS=none UNSLOTH_ALLOW_CPU=1 \
#       UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
#
# Overridable env (both backends):
#   UNSLOTH_IMAGE          image:tag to pull/run
#   UNSLOTH_PORTS=         extra -p publish flags, e.g. "-p 8000:8000 -p 8888:8888"
#   HF_HOME                host HF cache dir (default $HOME/.cache/huggingface)
#   TRITON_CACHE_DIR       host Triton cache dir
#   UNSLOTH_WORKDIR        host dir mounted at /workspace/host (default $PWD)
#
# NVIDIA-only env:
#   UNSLOTH_GPUS=all       GPUs to expose ("all" | "0" | "0,1" | "none" for CPU)
#   UNSLOTH_ALLOW_CPU=     set to 1 to allow GPU-less runs
#
# AMD-only env:
#   HSA_OVERRIDE_GFX_VERSION   force gfx version (e.g. 10.3.0 for RX 6800)
#   UNSLOTH_ROCM_GFX_ARCH      gfx target override (e.g. gfx1151 for Strix Halo)
set -euo pipefail

# --- Parse --rocm; everything else is passed through to the container --------
ROCM=0
PASSTHROUGH=()
for arg in "$@"; do
    if [[ "$arg" == "--rocm" ]]; then
        ROCM=1
    else
        PASSTHROUGH+=("$arg")
    fi
done
set -- "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.cache/unsloth-triton}"
WORK_DIR="${UNSLOTH_WORKDIR:-$PWD}"

mkdir -p "$HF_CACHE" "$TRITON_CACHE"

declare -a GPU_FLAGS=()
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)

if [[ $ROCM -eq 1 ]]; then
    IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth-rocm:latest}"

    if [[ ! -e /dev/kfd ]]; then
        printf "\033[1;33mWARN:\033[0m /dev/kfd not found -- ROCm drivers may not be installed.\n" >&2
        printf "      sudo usermod -aG video,render \$USER  then log out/in.\n\n" >&2
    fi

    GPU_FLAGS+=(--device /dev/kfd --device /dev/dri --group-add video)
    [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]] && ENV_FORWARD+=(-e HSA_OVERRIDE_GFX_VERSION)
    [[ -n "${UNSLOTH_ROCM_GFX_ARCH:-}"   ]] && ENV_FORWARD+=(-e UNSLOTH_ROCM_GFX_ARCH)
else
    IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth:latest}"
    GPUS="${UNSLOTH_GPUS:-all}"

    # Translate index selectors to Docker's `device=` form. Docker reads a bare
    # integer for --gpus as a COUNT, not an INDEX, so `UNSLOTH_GPUS=0` would
    # expose zero GPUs and the entrypoint would refuse to start. `all` and
    # already-quoted `device=...` selectors pass through. "none" omits --gpus
    # entirely (CPU mode; pair with UNSLOTH_ALLOW_CPU=1).
    GPU_FLAGS=(--gpus "$GPUS")
    case "$GPUS" in
        none)     GPU_FLAGS=()                            ;;
        all|"")                                           ;;
        \"device=*|device=*)                              ;;
        *[!0-9]*) GPU_FLAGS=(--gpus "\"device=${GPUS}\"") ;;  # comma list / UUID
        *)        GPU_FLAGS=(--gpus "\"device=${GPUS}\"") ;;  # bare integer index
    esac

    if [[ "$GPUS" != "none" ]] && ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
        printf "\033[1;33mWARN:\033[0m 'docker info' does not list 'nvidia' as a runtime.\n" >&2
        printf "      Install nvidia-container-toolkit if --gpus all fails:\n" >&2
        printf "      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n\n" >&2
    fi
fi

# Forward common secrets only if set in the host environment. Use the dash-only
# form `-e VAR` (no `=VALUE`): Docker reads the value from the parent shell, so
# the literal secret never lands in argv where `ps auxe` / /proc/<pid>/cmdline
# would expose it. An empty string would shadow whatever is baked in the image.
[[ -n "${HF_TOKEN:-}"          ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"     ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}"   ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)
[[ -n "${UNSLOTH_ALLOW_CPU:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_ALLOW_CPU)

# Extra publish flags for the service ports (Studio 8000, Jupyter 8888).
declare -a PORT_FLAGS=()
if [[ -n "${UNSLOTH_PORTS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of "-p X -p Y"
    PORT_FLAGS=(${UNSLOTH_PORTS})
fi

TTY_FLAG=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAG=(-it)
fi

exec docker run --rm "${TTY_FLAG[@]}" \
    "${GPU_FLAGS[@]}" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    "${PORT_FLAGS[@]}" \
    "$IMAGE" "$@"
