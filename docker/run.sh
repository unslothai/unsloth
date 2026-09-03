#!/usr/bin/env bash
# Convenience wrapper for `docker run unsloth/unsloth`. Sets the easily-forgotten
# flags behind the most confusing failures:
#   --gpus all           attach a GPU (entrypoint refuses to start without one)
#   --ipc=host           ample /dev/shm; the default 64MB crashes DataLoader workers
#   --ulimit memlock=-1  unlimited pinned memory (else multi-GPU training stalls)
#   --ulimit stack=64MB  larger libtorch thread stack (some kernels OOM the 8MB default)
# Plus mounts the host HF + Triton caches so downloads and kernels persist.
#
# With no command the image's own CMD runs, which on unsloth/unsloth:latest is the
# Studio (8000) + JupyterLab (8888) launcher, not a REPL. $PWD is at /workspace/host.
#   bash docker/run.sh                                  # start Studio + JupyterLab
#   bash docker/run.sh bash                             # shell in the container
#   bash docker/run.sh python /workspace/host/train.py  # run your training script
#   UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh   # publish the ports
#
# JupyterLab on the lean core image (unsloth/unsloth:core):
#   UNSLOTH_PORTS="-p 8888:8888" UNSLOTH_IMAGE=unsloth/unsloth:core \
#       bash docker/run.sh jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
# CPU-only hosts: no --gpus and UNSLOTH_ALLOW_CPU=1. No training, but Studio chat,
# Jupyter and GGUF tooling work:
#   UNSLOTH_GPUS=none UNSLOTH_ALLOW_CPU=1 \
#       UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" bash docker/run.sh
#
# Overridable env:
#   UNSLOTH_IMAGE=unsloth/unsloth:latest    image and tag to pull/run
#   UNSLOTH_GPUS=all                        "all" | "0" | "0,1" | "none"
#   UNSLOTH_ALLOW_CPU=                      set to 1 to allow GPU-less runs
#   UNSLOTH_PORTS=                          extra -p publish flags
#   HF_HOME=$HOME/.cache/huggingface        host HF cache dir to mount
#   TRITON_CACHE_DIR=...unsloth-triton      host Triton cache dir to mount
#   UNSLOTH_WORKDIR=$PWD                    host dir mounted at /workspace/host
set -euo pipefail

IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth:latest}"
GPUS="${UNSLOTH_GPUS:-all}"
# Translate index selectors to Docker's `device=` form: a bare integer is a COUNT,
# not an INDEX, so `UNSLOTH_GPUS=0` would expose zero GPUs.
GPU_FLAG=(--gpus "$GPUS")
case "$GPUS" in
    none)       GPU_FLAG=()              ;;
    all|"")                              ;;
    \"device=*)                          ;;
    device=*,*) GPU_FLAG=(--gpus "\"${GPUS}\"") ;;  # docker needs the quotes
    device=*)                            ;;  # single device, fine unquoted
    *[!0-9]*) GPU_FLAG=(--gpus "\"device=${GPUS}\"") ;;  # comma list / UUID
    *)        GPU_FLAG=(--gpus "\"device=${GPUS}\"") ;;  # bare integer index
esac
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.cache/unsloth-triton}"
WORK_DIR="${UNSLOTH_WORKDIR:-$PWD}"

mkdir -p "$HF_CACHE" "$TRITON_CACHE"

# warn, do not abort: some setups report runtimes differently
if ! docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'; then
    printf "\033[1;33mWARN:\033[0m 'docker info' does not list 'nvidia' as a runtime.\n" >&2
    printf "      If --gpus all fails below, install nvidia-container-toolkit:\n" >&2
    printf "      https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html\n\n" >&2
fi

# Only if set, else an empty string shadows the image's. The dash-only `-e VAR` form
# makes Docker read the value from the parent shell, so it never lands in argv.
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"          ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"     ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}"   ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)
[[ -n "${UNSLOTH_ALLOW_CPU:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_ALLOW_CPU)
# read by studio_launch.sh; without these it uses a random password and no sshd
[[ -n "${JUPYTER_PASSWORD:-}"           ]] && ENV_FORWARD+=(-e JUPYTER_PASSWORD)
[[ -n "${PUBLIC_KEY:-}"                 ]] && ENV_FORWARD+=(-e PUBLIC_KEY)
[[ -n "${SSH_KEY:-}"                    ]] && ENV_FORWARD+=(-e SSH_KEY)
[[ -n "${UNSLOTH_JUPYTER_CLOUDFLARE:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_JUPYTER_CLOUDFLARE)

declare -a PORT_FLAGS=()
if [[ -n "${UNSLOTH_PORTS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of "-p X -p Y"
    PORT_FLAGS=(${UNSLOTH_PORTS})
fi

# CI / piped invocations otherwise hit "the input device is not a TTY"
TTY_FLAG=()
if [ -t 0 ] && [ -t 1 ]; then
    TTY_FLAG=(-it)
fi

# No `set -x`: it would echo HF_TOKEN / WANDB_API_KEY to CI logs. The
# ${arr[@]+"${arr[@]}"} form keeps empty arrays nounset-safe on bash 3.2 (macOS).
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
