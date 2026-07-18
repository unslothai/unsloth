#!/usr/bin/env bash
# Convenience wrapper for `docker run unsloth/unsloth`. Sets the flags that
# people most often forget and that cause the most confusing failures:
#
#   --gpus all           Without this, no GPU is attached and the container's
#                        entrypoint will refuse to start.
#   --ipc=host           PyTorch DataLoader workers need ample /dev/shm. The
#                        default 64MB causes "DataLoader worker (pid X) exited
#                        unexpectedly" on any non-trivial dataset.
#   --ulimit memlock=-1  Unlimited pinned memory for NCCL / CUDA pinned host
#                        buffers. Without this, multi-GPU training stalls.
#   --ulimit stack=64MB  Larger thread stack for libtorch (some kernels OOM
#                        the default 8MB stack).
#
# Plus mounts the host Hugging Face cache and Triton JIT cache so model
# downloads and compiled kernels persist across container runs.
#
# Usage:
#   bash docker/run.sh                                  # interactive python REPL
#   bash docker/run.sh bash                             # shell in the container
#   bash docker/run.sh python /workspace/smoke_test.py  # run the smoke test
#   bash docker/run.sh python /workspace/host/train.py  # run your training script
#                                                         ($PWD is mounted at
#                                                          /workspace/host)
#
# The full image (unsloth/unsloth:latest) starts Studio (8000) + JupyterLab
# (8888) by default; publish the ports when you want them:
#   UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
# JupyterLab on the lean core image (unsloth/unsloth:core):
#   UNSLOTH_PORTS="-p 8888:8888" UNSLOTH_IMAGE=unsloth/unsloth:core \
#       bash docker/run.sh jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
# CPU-only hosts (Docker Desktop on macOS, Windows without WSL2 GPU, plain
# CPU Linux): no --gpus and set UNSLOTH_ALLOW_CPU=1. Training is unavailable
# but Studio chat / Data Recipes, Jupyter and GGUF tooling work:
#   UNSLOTH_GPUS=none UNSLOTH_ALLOW_CPU=1 \
#       UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
#
# Overridable env:
#   UNSLOTH_IMAGE=unsloth/unsloth:latest    image and tag to pull/run
#   UNSLOTH_GPUS=all                        GPUs to expose ("all" | "0" | "0,1"
#                                           | "none" to run without GPU)
#   UNSLOTH_ALLOW_CPU=                      set to 1 to allow GPU-less runs
#   UNSLOTH_PORTS=                          extra -p publish flags, e.g.
#                                           "-p 8000:8000 -p 8888:8888"
#   HF_HOME=$HOME/.cache/huggingface        host HF cache dir to mount
#   TRITON_CACHE_DIR=$HOME/.cache/unsloth-triton
#                                           host Triton cache dir to mount
#   UNSLOTH_WORKDIR=$PWD                    host dir mounted at /workspace/host
set -euo pipefail

IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth:latest}"
GPUS="${UNSLOTH_GPUS:-all}"
# Translate index selectors to Docker's `device=` form. The header docstring
# advertises UNSLOTH_GPUS values like "0" and "0,1" but Docker reads a bare
# integer for --gpus as a COUNT, not an INDEX, so `UNSLOTH_GPUS=0` would
# expose zero GPUs and the entrypoint would refuse to start. `all` and
# already-quoted `device=...` / `"device=..."` selectors pass through.
# "none" omits --gpus entirely (CPU mode; pair with UNSLOTH_ALLOW_CPU=1).
GPU_FLAG=(--gpus "$GPUS")
case "$GPUS" in
    none)       GPU_FLAG=()              ;;
    all|"")                              ;;
    \"device=*)                          ;;
    device=*,*) GPU_FLAG=(--gpus "\"${GPUS}\"") ;;  # native comma list: docker needs the quotes
    device=*)                            ;;  # single device, fine unquoted
    *[!0-9]*) GPU_FLAG=(--gpus "\"device=${GPUS}\"") ;;  # comma list / UUID
    *)        GPU_FLAG=(--gpus "\"device=${GPUS}\"") ;;  # bare integer index
esac
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.cache/unsloth-triton}"
WORK_DIR="${UNSLOTH_WORKDIR:-$PWD}"

mkdir -p "$HF_CACHE" "$TRITON_CACHE"

# Warn early if the host doesn't have the nvidia runtime registered.
# We let `docker run` fail loudly rather than abort here -- some setups
# (rootless docker, custom runtimes) report runtimes differently.
if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    printf "\033[1;33mWARN:\033[0m 'docker info' does not list 'nvidia' as a runtime.\n" >&2
    printf "      If --gpus all fails below, install nvidia-container-toolkit:\n" >&2
    printf "      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n\n" >&2
fi

# Forward common secrets only if they're set in the host environment.
# Empty strings would shadow whatever is already inside the image.
# IMPORTANT: use the dash-only form `-e VAR` (no `=VALUE`). Docker reads
# the value from the parent shell, so the literal secret never lands in
# argv where it would be visible to any user on the host via
# `ps auxe` / `/proc/<pid>/cmdline` for the lifetime of the docker CLI
# process.
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"          ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"     ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}"   ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)
[[ -n "${UNSLOTH_ALLOW_CPU:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_ALLOW_CPU)
# Studio/Jupyter service config read by studio_launch.sh. Same dash-only -e VAR
# form as the secrets above: the value comes from the parent env, so even
# JUPYTER_PASSWORD never lands in argv (ps auxe / /proc/<pid>/cmdline). Without
# these, `JUPYTER_PASSWORD=... bash docker/run.sh` silently got a random
# password, PUBLIC_KEY/SSH_KEY never enabled sshd, and UNSLOTH_JUPYTER_CLOUDFLARE
# never started the tunnel when using the bundled launcher.
[[ -n "${JUPYTER_PASSWORD:-}"           ]] && ENV_FORWARD+=(-e JUPYTER_PASSWORD)
[[ -n "${PUBLIC_KEY:-}"                 ]] && ENV_FORWARD+=(-e PUBLIC_KEY)
[[ -n "${SSH_KEY:-}"                    ]] && ENV_FORWARD+=(-e SSH_KEY)
[[ -n "${UNSLOTH_JUPYTER_CLOUDFLARE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_JUPYTER_CLOUDFLARE)

# Extra publish flags for the service ports (Studio 8000, Jupyter 8888).
declare -a PORT_FLAGS=()
if [[ -n "${UNSLOTH_PORTS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of "-p X -p Y"
    PORT_FLAGS=(${UNSLOTH_PORTS})
fi

# Only attach -t when our own stdin/stdout are a TTY; CI / piped invocations
# otherwise hit `the input device is not a TTY` and never reach the entrypoint.
TTY_FLAG=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAG=(-it)
fi

# Avoid `set -x` here so the literal HF_TOKEN / WANDB_API_KEY / UNSLOTH_LICENSE
# values do not get echoed to stdout/CI logs. The forwarded env vars are
# already in ENV_FORWARD; printing them again was a secret leak.
# The ${arr[@]+"${arr[@]}"} form keeps empty arrays nounset-safe on
# bash 3.2 (macOS /bin/bash), where a bare "${empty[@]}" trips set -u.
exec docker run --rm ${TTY_FLAG[@]+"${TTY_FLAG[@]}"} \
    ${GPU_FLAG[@]+"${GPU_FLAG[@]}"} \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$HF_CACHE":/workspace/.cache/huggingface \
    -v "$TRITON_CACHE":/workspace/.cache/triton \
    -v "$WORK_DIR":/workspace/host \
    "${ENV_FORWARD[@]}" \
    ${PORT_FLAGS[@]+"${PORT_FLAGS[@]}"} \
    "$IMAGE" "$@"
