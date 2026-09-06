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

# Docker resolves --gpus in the DAEMON, before the container exists: on a host with
# no NVIDIA GPU it dies with "failed to discover GPU vendor from CDI: no known GPU
# vendor found" and exit 125, so entrypoint.sh never runs and its diagnostics never
# print. Drop the flag instead and let the container start, so the user gets the
# entrypoint's explanation (or, on :latest, Studio in CPU mode).
# UNSLOTH_DEV_ROOT prefixes the /dev probes below (DESTDIR idiom). It exists so the
# regression tests can stage a fake device tree; leave it unset in normal use.
DEV_ROOT="${UNSLOTH_DEV_ROOT:-}"
host_has_nvidia() {
    [[ -e "$DEV_ROOT/dev/nvidiactl" ]] && return 0
    command -v nvidia-smi >/dev/null 2>&1 \
        && nvidia-smi -L 2>/dev/null | grep -q '^GPU' && return 0
    return 1
}

if [[ ${#GPU_FLAG[@]} -gt 0 ]] && ! host_has_nvidia; then
    printf "\033[1;33mWARN:\033[0m no NVIDIA GPU on this host; dropping --gpus %s.\n" "$GPUS" >&2
    printf "      'docker run --gpus' would fail at the daemon (exit 125) before the\n" >&2
    printf "      container starts. Set UNSLOTH_GPUS=none to silence this.\n" >&2
    GPU_FLAG=()
    # AMD host: hand llama.cpp/GGUF the render nodes. This is NOT torch acceleration
    # -- torch in the image is cu128 and torch.cuda.is_available() stays False here.
    # --group-add needs NUMERIC gids: a name is resolved INSIDE the container, where
    # the host's video/render groups do not exist.
    if [[ -e "$DEV_ROOT/dev/kfd" && -d "$DEV_ROOT/dev/dri" ]]; then
        GPU_FLAG=(--device /dev/kfd --device /dev/dri)
        # A missing group is not fatal: getent exits nonzero when the name is not in
        # NSS, and under `set -o pipefail` that would take the whole assignment down
        # with `set -e` before docker run is ever reached. Trailing `|| _gid=` puts the
        # assignment in an OR-list, which suppresses that and leaves the gid empty.
        # Minimal hosts really do ship without a render group.
        if command -v getent >/dev/null 2>&1; then
            for _grp in video render; do
                _gid="$(getent group "$_grp" | cut -d: -f3)" || _gid=""
                [[ -n "$_gid" ]] && GPU_FLAG+=(--group-add "$_gid")
            done
        fi
        printf "      AMD devices found: passing /dev/kfd and /dev/dri through.\n" >&2
    fi
    printf "\n" >&2
fi

# A GPU host whose Docker has no nvidia runtime is missing the NVIDIA Container
# Toolkit, which is the one thing the image cannot bring along. Offer to install it
# (install_nvidia_toolkit.sh, NVIDIA's own recipe) rather than let --gpus fail at
# the daemon with exit 125. UNSLOTH_INSTALL_TOOLKIT=1 says yes without a prompt,
# =0 never asks; with neither and no terminal, print the one-liner and continue.
DOCKER_INFO=""
DOCKER_ERR=""
if [[ ${#GPU_FLAG[@]} -gt 0 ]] && host_has_nvidia; then
    # stderr folded in: on failure the captured text IS the daemon's error
    DOCKER_INFO="$(docker info 2>&1)" || { DOCKER_ERR="$DOCKER_INFO"; DOCKER_INFO=""; }
fi
if [[ -n "$DOCKER_ERR" ]]; then
    # no daemon, or no permission on its socket: an install offer would be the wrong
    # answer, and the docker run below fails with the same message
    printf "\033[1;33mWARN:\033[0m 'docker info' failed, so the GPU runtime could not be checked:\n      %s\n" "${DOCKER_ERR##*$'\n'}" >&2
    printf "      Start the Docker daemon, or add yourself to the docker group (newgrp docker).\n\n" >&2
elif [[ ${#GPU_FLAG[@]} -gt 0 ]] && host_has_nvidia \
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
            # -E keeps UNSLOTH_TOOLKIT_VERIFY and proxy settings through sudo's env_reset.
            # An offer, not a precondition: a failed, cancelled or driver-too-old install
            # (exit 3, toolkit working) must not take run.sh down before docker run.
            if [[ "$(id -u)" = 0 ]]; then bash "$INSTALLER" || true; else sudo -E bash "$INSTALLER" || true; fi
            ;;
        *)
            printf "      Install it with one command (Linux, needs sudo):\n" >&2
            printf "      curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh | sudo -E bash\n\n" >&2
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
