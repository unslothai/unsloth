#!/usr/bin/env bash
# Installs the NVIDIA Container Toolkit into Docker. Host side only: Docker resolves --gpus in the daemon, before a container exists, so no image can do this itself.
#   curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh -o install_nvidia_toolkit.sh && sudo -E bash install_nvidia_toolkit.sh
# UNSLOTH_TOOLKIT_VERIFY=0 skips the final `docker run --gpus all` check. UNSLOTH_DESTDIR, UNSLOTH_OS_RELEASE,
# UNSLOTH_PROC_VERSION, UNSLOTH_RUN_USER_DIR and UNSLOTH_WSL_LIB_DIR redirect the host paths they name (tests only).
set -euo pipefail

DESTDIR="${UNSLOTH_DESTDIR:-}"
OS_RELEASE="${UNSLOTH_OS_RELEASE:-/etc/os-release}"
PROC_VERSION="${UNSLOTH_PROC_VERSION:-/proc/version}"
RUN_USER_DIR="${UNSLOTH_RUN_USER_DIR:-/run/user}"
WSL_LIB_DIR="${UNSLOTH_WSL_LIB_DIR:-/usr/lib/wsl/lib}"
BASE="https://nvidia.github.io/libnvidia-container"

say()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

command -v docker >/dev/null 2>&1 \
    || fail "docker is not installed. Install Docker Engine first (https://docs.docker.com/engine/install/), then run this again." 2

# Every grep of a command's output below reads a variable, not a pipe. `grep -q` exits on the first
# match and closes the pipe; `set -o pipefail` above then promotes the producer's SIGPIPE (141) to the
# status of the whole pipeline, so a match reads as a miss whenever the producer was still writing.
# Measured on `docker info` with a 50ms pause mid-output: status 141, the Docker Desktop guard skipped,
# and the script elevating to install on the one daemon it exists to leave alone.

# Docker Desktop ships its own GPU integration; installing here would configure a daemon it does not use. Checked before elevating.
docker_info="$(docker info 2>/dev/null || true)"
if grep -qi 'Operating System: Docker Desktop' <<<"$docker_info"; then
    if [[ "$(uname -s)" == Darwin ]]; then
        say "Docker Desktop on macOS: no NVIDIA GPU can be attached on a Mac, so there is nothing to install."
        say "The image runs CPU-only there: drop --gpus and set UNSLOTH_ALLOW_CPU=1."
        exit 0
    elif grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then
        say "Docker Desktop with the WSL 2 backend: GPU support comes with it, nothing to install here."
        say "Keep a current NVIDIA Windows driver installed (from nvidia.com; wsl --update updates WSL itself, not the driver)."
        exit 0
    fi
    fail "Docker Desktop for Linux has no NVIDIA GPU support. Install Docker Engine instead
       (https://docs.docker.com/engine/install/) and run this again." 2
fi

# Everything below edits THIS machine; DOCKER_CONTEXT overrides DOCKER_HOST, which overrides the selected context.
if [[ -n "${DOCKER_CONTEXT:-}" || -z "${DOCKER_HOST:-}" ]]; then
    endpoint="$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null)" \
        || fail "cannot inspect the Docker context '${DOCKER_CONTEXT:-current}' (it may exist only in the invoking user's Docker config); refusing to guess which daemon to configure." 2
else
    endpoint="$DOCKER_HOST"
fi
case "$endpoint" in
    ""|unix:///var/run/docker.sock|unix:///run/docker.sock|npipe://*) ;;
    unix://*) fail "the Docker CLI talks to a daemon on ${endpoint}, not the system daemon this script configures (/etc/docker/daemon.json, service docker). Point it at the default socket or configure that daemon by hand." 2 ;;
    *) fail "the Docker CLI talks to a remote daemon (${endpoint}); run this script on that host, it configures the local Docker only." 2 ;;
esac

# Rootless Docker keeps its own daemon, which the steps below would miss; root sees a different one, and `curl | sudo bash` arrives root with DOCKER_HOST stripped, so probe SUDO_UID's socket too.
rootless_daemon() {
    local opts
    opts="$(docker info --format '{{join .SecurityOptions ","}}' 2>/dev/null || true)"
    grep -q rootless <<<"$opts" && return 0
    [[ -n "${SUDO_UID:-}" && -S "${RUN_USER_DIR}/${SUDO_UID}/docker.sock" ]] || return 1
    opts="$(DOCKER_HOST="unix://${RUN_USER_DIR}/${SUDO_UID}/docker.sock" \
            docker info --format '{{join .SecurityOptions ","}}' 2>/dev/null || true)"
    grep -q rootless <<<"$opts"
}
if rootless_daemon; then
    fail "rootless Docker detected. Follow NVIDIA's rootless procedure instead:
       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#rootless-mode" 2
fi

if [[ "$(id -u)" != 0 ]]; then
    # piped through `curl | bash` there is no file to re-run: $0 is just "bash"
    if [[ -r "$0" ]] && command -v sudo >/dev/null 2>&1; then
        say "Re-running with sudo."
        exec sudo -E bash "$0" "$@"
    fi
    fail "run this as root: sudo -E bash install_nvidia_toolkit.sh" 2
fi

# On WSL 2 the Windows driver puts nvidia-smi under /usr/lib/wsl/lib, which sudo's secure_path drops from PATH.
NVSMI="$(command -v nvidia-smi 2>/dev/null || true)"
[[ -z "$NVSMI" && -x "${WSL_LIB_DIR}/nvidia-smi" ]] && NVSMI="${WSL_LIB_DIR}/nvidia-smi"
gpu_list=""
[[ -n "$NVSMI" ]] && gpu_list="$("$NVSMI" -L 2>/dev/null || true)"
if [[ -z "$NVSMI" ]] || ! grep -q '^GPU' <<<"$gpu_list"; then
    if grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then
        fail "no NVIDIA GPU is visible in this WSL 2 distro. Install a current NVIDIA Windows driver
       from nvidia.com (never a Linux driver inside WSL), restart WSL (wsl --shutdown), then run
       this again. Unsloth images need driver 570.26 or newer." 2
    fi
    fail "no NVIDIA driver found (nvidia-smi lists no GPU). Install the driver first, with your
       distribution's packages (Ubuntu: 'sudo ubuntu-drivers install'; RHEL/Fedora: the
       nvidia-driver module from the CUDA repository), reboot, then run this again.
       Unsloth images need driver 570.26 or newer." 2
fi
MIN_DRIVER=570.26
# `sed -n 1p`, not `head -1`: head closes the pipe after one line, and nvidia-smi prints one per GPU,
# so on a multi-GPU host the SIGPIPE becomes the substitution's status and `set -e` kills the script
# here with nothing printed. sed reads to the end. (Measured: exit 141, no output, 8 GPUs.)
DRIVER="$("$NVSMI" --query-gpu=driver_version --format=csv,noheader 2>/dev/null | sed -n 1p | tr -d '[:space:]')"
driver_ok() {
    [[ -n "$DRIVER" ]] && [[ "$(printf '%s\n' "$MIN_DRIVER" "$DRIVER" | sort -V | head -1)" == "$MIN_DRIVER" ]]
}

# Explicit platform: a cached ubuntu image of the other arch would be picked and fail on "exec nvidia-smi: no such file".
case "$(uname -m)" in
    aarch64|arm64) PLATFORM=linux/arm64 ;;
    x86_64|amd64)  PLATFORM=linux/amd64 ;;
    *) fail "unsupported architecture $(uname -m): the Unsloth images are built for linux/amd64 and linux/arm64 only." 2 ;;
esac

configured() {
    local info; info="$(docker info 2>/dev/null || true)"
    grep -qi 'Runtimes:.*nvidia' <<<"$info"
}

if configured; then
    say "Docker already lists the nvidia runtime; nothing to install."
elif command -v nvidia-ctk >/dev/null 2>&1 && command -v nvidia-container-runtime >/dev/null 2>&1; then
    # nvidia-container-toolkit-base ships nvidia-ctk without the runtime, so check both before skipping the install.
    say "nvidia-container-toolkit is installed but Docker does not list the nvidia runtime."
else
    [[ -r "$OS_RELEASE" ]] || fail "cannot read $OS_RELEASE to pick a package manager." 2
    # shellcheck disable=SC1090
    ID_LIKE="$( . "$OS_RELEASE"; printf '%s %s' "${ID:-}" "${ID_LIKE:-}" )"
    case " $ID_LIKE " in
        *" ubuntu "*|*" debian "*)
            say "Adding NVIDIA's apt repository and installing nvidia-container-toolkit."
            apt-get update -qq
            apt-get install -y -qq --no-install-recommends ca-certificates curl gnupg2 >/dev/null
            mkdir -p "${DESTDIR}/usr/share/keyrings" "${DESTDIR}/etc/apt/sources.list.d"
            # Staged then renamed: a failed download must not leave a truncated keyring or list behind.
            keyring="${DESTDIR}/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
            list="${DESTDIR}/etc/apt/sources.list.d/nvidia-container-toolkit.list"
            curl -fsSL "${BASE}/gpgkey" | gpg --dearmor --yes -o "${keyring}.tmp"
            chmod 0644 "${keyring}.tmp"  # apt reads Signed-By keys as _apt, so not umask 077
            mv -f "${keyring}.tmp" "$keyring"
            curl -fsSL "${BASE}/stable/deb/nvidia-container-toolkit.list" \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                > "${list}.tmp"
            mv -f "${list}.tmp" "$list"
            apt-get update -qq
            apt-get install -y -qq nvidia-container-toolkit
            ;;
        *" rhel "*|*" fedora "*|*" centos "*|*" amzn "*|*" rocky "*|*" almalinux "*)
            command -v curl >/dev/null 2>&1 || { command -v dnf >/dev/null 2>&1 && dnf install -y curl || yum install -y curl; }
            mkdir -p "${DESTDIR}/etc/yum.repos.d"
            repo="${DESTDIR}/etc/yum.repos.d/nvidia-container-toolkit.repo"
            curl -fsSL "${BASE}/stable/rpm/nvidia-container-toolkit.repo" > "${repo}.tmp"
            mv -f "${repo}.tmp" "$repo"
            if command -v dnf >/dev/null 2>&1; then
                say "Installing nvidia-container-toolkit with dnf."
                dnf install -y nvidia-container-toolkit
            else
                say "Installing nvidia-container-toolkit with yum."
                yum install -y nvidia-container-toolkit
            fi
            ;;
        *" suse "*|*" opensuse "*|*" sles "*|*" opensuse-leap "*|*" opensuse-tumbleweed "*)
            say "Installing nvidia-container-toolkit with zypper."
            zypper --non-interactive ar "${BASE}/stable/rpm/nvidia-container-toolkit.repo" || true
            zypper --non-interactive --gpg-auto-import-keys install nvidia-container-toolkit
            ;;
        *)
            fail "unrecognised distribution ($ID_LIKE). Install the toolkit by hand:
       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" 2
            ;;
    esac
fi

if ! configured; then
    say "Registering the nvidia runtime with Docker."
    nvidia-ctk runtime configure --runtime=docker
    if command -v systemctl >/dev/null 2>&1 && systemctl restart docker 2>/dev/null; then
        say "Docker restarted (systemctl)."
    elif command -v service >/dev/null 2>&1 && service docker restart 2>/dev/null; then
        say "Docker restarted (service)."
    else
        fail "could not restart Docker; restart it yourself, then run: docker run --rm --gpus all --platform ${PLATFORM} ubuntu:24.04 nvidia-smi -L"
    fi
    configured || fail "Docker still does not list the nvidia runtime after the install; see 'docker info'."
fi

if [[ "${UNSLOTH_TOOLKIT_VERIFY:-1}" != 0 ]]; then
    say "Verifying: docker run --rm --gpus all --platform ${PLATFORM} ubuntu:24.04 nvidia-smi -L"
    if ! docker run --rm --gpus all --platform "${PLATFORM}" ubuntu:24.04 nvidia-smi -L; then
        fail "a container could not see the GPU. On WSL 2 install a current NVIDIA Windows driver
       from nvidia.com (wsl --update updates WSL itself, not the driver); otherwise see
       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html"
    fi
fi
if grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then
    say "WSL 2 host: the GPU comes from the Windows NVIDIA driver; update it with NVIDIA's Windows installer."
fi
if ! driver_ok; then
    fail "the toolkit works, but NVIDIA driver ${DRIVER:-unknown} is below ${MIN_DRIVER}, the minimum for
       the Unsloth images. Update the driver, reboot, then run: docker run --rm --gpus all --platform ${PLATFORM} ubuntu:24.04 nvidia-smi -L" 3
fi
say "Driver ${DRIVER} meets the ${MIN_DRIVER} minimum."
say "Done. Run: docker run -d --gpus all -p 8000:8000 -p 8888:8888 unsloth/unsloth"
