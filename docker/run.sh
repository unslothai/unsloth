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
# AMD hosts: --rocm runs the ROCm image and passes the AMD device nodes instead
# of --gpus, which is NVIDIA-only.
#   bash docker/run.sh --rocm python /workspace/smoke_test_rocm.py
#
# Overridable env:
#   UNSLOTH_IMAGE=unsloth/unsloth:latest    image and tag to pull/run
#   UNSLOTH_GPUS=all                        "all" | "0" | "0,1" | "none"
#   UNSLOTH_ALLOW_CPU=                      set to 1 to allow GPU-less runs
#   UNSLOTH_PORTS=                          extra -p publish flags
#   HF_HOME=$HOME/.cache/huggingface        host HF cache dir to mount
#   TRITON_CACHE_DIR=...unsloth-triton      host Triton cache dir to mount
#   UNSLOTH_WORKDIR=$PWD                    host dir mounted at /workspace/host
# --rocm only:
#   HSA_OVERRIDE_GFX_VERSION                force a gfx target (e.g. 10.3.0)
#   UNSLOTH_ROCM_GFX_ARCH                   gfx arch override (e.g. gfx1151)
set -euo pipefail

# --rocm is ours; everything else is the container's command line.
ROCM=0
PASSTHROUGH=()
for arg in "$@"; do
    if [[ "$arg" == "--rocm" ]]; then ROCM=1; else PASSTHROUGH+=("$arg"); fi
done
set -- ${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}

# UNSLOTH_DEV_ROOT prefixes the /dev probes (DESTDIR idiom). It exists so the
# regression tests can stage a fake device tree; leave it unset in normal use.
DEV_ROOT="${UNSLOTH_DEV_ROOT:-}"

# --group-add needs NUMERIC gids: a name is resolved INSIDE the container, where
# the host's video/render groups do not exist.
amd_device_flags() {
    printf '%s\n' --device /dev/kfd --device /dev/dri
    command -v getent >/dev/null 2>&1 || return 0
    local _grp _gid
    for _grp in video render; do
        # getent exits nonzero when the name is not in NSS, and under pipefail
        # that would take the assignment down with set -e. Minimal hosts really
        # do ship without a render group.
        _gid="$(getent group "$_grp" | cut -d: -f3)" || _gid=""
        [[ -n "$_gid" ]] && printf '%s\n' --group-add "$_gid"
    done
    return 0
}

if [[ $ROCM -eq 1 ]]; then
    IMAGE="${UNSLOTH_IMAGE:-unsloth/unsloth-rocm:latest}"
    GPUS=none
    if [[ -e "$DEV_ROOT/dev/kfd" ]]; then
        mapfile -t GPU_FLAG < <(amd_device_flags)
    else
        GPU_FLAG=()
        printf "\033[1;33mWARN:\033[0m /dev/kfd is not present, so no AMD GPU can be passed through.\n" >&2
        printf "      On Linux install the amdgpu driver and add yourself to the video/render\n" >&2
        printf "      groups. Docker Desktop on Windows and macOS has no /dev/kfd at all: the\n" >&2
        printf "      ROCm image cannot reach a GPU there, whatever the host card is.\n\n" >&2
    fi
else
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
fi
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.cache/unsloth-triton}"
WORK_DIR="${UNSLOTH_WORKDIR:-$PWD}"

mkdir -p "$HF_CACHE" "$TRITON_CACHE"

# Docker resolves --gpus in the DAEMON, before the container exists: on a host with
# no NVIDIA GPU it dies with "failed to discover GPU vendor from CDI: no known GPU
# vendor found" and exit 125, so entrypoint.sh never runs and its diagnostics never
# print. Drop the flag instead and let the container start, so the user gets the
# entrypoint's explanation (or, on :latest, Studio in CPU mode).
host_has_nvidia() {
    [[ -e "$DEV_ROOT/dev/nvidiactl" ]] && return 0
    command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L 2>/dev/null | grep -q '^GPU' && return 0
    return 1
}

if [[ $ROCM -eq 0 && ${#GPU_FLAG[@]} -gt 0 ]] && ! host_has_nvidia; then
    printf "\033[1;33mWARN:\033[0m no NVIDIA GPU on this host; dropping --gpus %s.\n" "$GPUS" >&2
    printf "      'docker run --gpus' would fail at the daemon (exit 125) before the\n" >&2
    printf "      container starts. Set UNSLOTH_GPUS=none to silence this.\n" >&2
    GPU_FLAG=()
    # AMD host: hand llama.cpp/GGUF the render nodes. This is NOT torch acceleration
    # -- torch in the image is cu128 and torch.cuda.is_available() stays False here.
    # Run --rocm instead for a torch that can use the card.
    if [[ -e "$DEV_ROOT/dev/kfd" && -d "$DEV_ROOT/dev/dri" ]]; then
        mapfile -t GPU_FLAG < <(amd_device_flags)
        printf "      AMD devices found: passing /dev/kfd and /dev/dri through.\n" >&2
    fi
    printf "\n" >&2
fi

# A GPU host with no nvidia runtime would fail --gpus at the daemon with exit 125, so offer the installer.
# UNSLOTH_INSTALL_TOOLKIT=1 says yes without a prompt, =0 never asks; with neither and no terminal, print the one-liner and continue.
DOCKER_INFO=""
DOCKER_ERR=""
# A mixed NVIDIA + AMD host under --rocm is not missing anything: it runs the ROCm
# image through the AMD device nodes, so the toolkit is irrelevant there.
if [[ $ROCM -eq 0 && ${#GPU_FLAG[@]} -gt 0 ]] && host_has_nvidia; then
    # stderr folded in: on failure the captured text IS the daemon's error
    DOCKER_INFO="$(docker info 2>&1)" || { DOCKER_ERR="$DOCKER_INFO"; DOCKER_INFO=""; }
fi
if [[ -n "$DOCKER_ERR" ]]; then
    printf "\033[1;33mWARN:\033[0m 'docker info' failed, so the GPU runtime could not be checked:\n      %s\n" "${DOCKER_ERR##*$'\n'}" >&2
    printf "      Start the Docker daemon, or add yourself to the docker group (newgrp docker).\n\n" >&2
elif [[ $ROCM -eq 0 && ${#GPU_FLAG[@]} -gt 0 ]] && host_has_nvidia \
        && ! grep -qi 'Runtimes:.*nvidia' <<<"$DOCKER_INFO"; then
    INSTALLER="$(dirname "${BASH_SOURCE[0]}")/install_nvidia_toolkit.sh"
    printf "\033[1;33mWARN:\033[0m 'docker info' does not list 'nvidia' as a runtime: the NVIDIA\n" >&2
    printf "      Container Toolkit is not set up, so --gpus %s would fail at the daemon.\n" "$GPUS" >&2
    answer="${UNSLOTH_INSTALL_TOOLKIT:-}"
    if [[ -z "$answer" && -t 0 && -t 1 ]]; then
        read -r -p "      Install it now with sudo (bash $INSTALLER)? [Y/n] " answer </dev/tty || answer=n
        answer="${answer:-y}"
    fi
    case "$answer" in
        1|[Yy]*)
            # -E keeps UNSLOTH_TOOLKIT_VERIFY and the proxy settings through env_reset; a failed, cancelled or driver-too-old install (exit 3) must not stop the docker run below.
            if [[ "$(id -u)" = 0 ]]; then bash "$INSTALLER" || true; else sudo -E bash "$INSTALLER" || true; fi
            ;;
        *)
            printf "      Install it with one command (Linux, needs sudo):\n" >&2
            printf "      curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh\n\n" >&2
            ;;
    esac
fi

# Only if set, else an empty string shadows the image's. The dash-only `-e VAR` form
# makes Docker read the value from the parent shell, so it never lands in argv.
declare -a ENV_FORWARD=(-e HF_HUB_ENABLE_HF_TRANSFER=1)
[[ -n "${HF_TOKEN:-}"          ]] && ENV_FORWARD+=(-e HF_TOKEN)
[[ -n "${WANDB_API_KEY:-}"     ]] && ENV_FORWARD+=(-e WANDB_API_KEY)
[[ -n "${UNSLOTH_LICENSE:-}"   ]] && ENV_FORWARD+=(-e UNSLOTH_LICENSE)
[[ -n "${UNSLOTH_ALLOW_CPU:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_ALLOW_CPU)
# gfx overrides for cards the installed ROCm build has no kernels for
[[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]] && ENV_FORWARD+=(-e HSA_OVERRIDE_GFX_VERSION)
[[ -n "${UNSLOTH_ROCM_GFX_ARCH:-}"    ]] && ENV_FORWARD+=(-e UNSLOTH_ROCM_GFX_ARCH)
# read by studio_launch.sh; without these it uses a random password and no sshd
[[ -n "${JUPYTER_PASSWORD:-}"           ]] && ENV_FORWARD+=(-e JUPYTER_PASSWORD)
[[ -n "${UNSLOTH_STUDIO_PASSWORD:-}"    ]] && ENV_FORWARD+=(-e UNSLOTH_STUDIO_PASSWORD)
[[ -n "${UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT:-}" ]] && ENV_FORWARD+=(-e UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT)
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
