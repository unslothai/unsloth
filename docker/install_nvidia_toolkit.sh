#!/usr/bin/env bash
# Installs the NVIDIA Container Toolkit on a Linux host and wires it into Docker, so
# `docker run --gpus all unsloth/unsloth` works. This is the host side that no image
# can do for itself: Docker resolves --gpus in the daemon, before a container exists.
#
#   curl -fsSL https://raw.githubusercontent.com/unslothai/unsloth/main/docker/install_nvidia_toolkit.sh | sudo bash
#
# Follows NVIDIA's install guide (apt, dnf/yum, zypper), then
# `nvidia-ctk runtime configure --runtime=docker`, restarts Docker and verifies with a
# real `docker run --gpus all`. Idempotent: a host that is already set up only runs
# the verification. The NVIDIA driver itself is a kernel module and is not installed
# here; without one this stops and says how to get it.
#
# Environment:
#   UNSLOTH_TOOLKIT_VERIFY=0   skip the final `docker run --gpus all` check
#   UNSLOTH_DESTDIR            prefix for /etc and /usr/share paths (tests only)
#   UNSLOTH_OS_RELEASE         alternative os-release file (tests only)
#   UNSLOTH_PROC_VERSION       alternative /proc/version file (tests only)
set -euo pipefail

DESTDIR="${UNSLOTH_DESTDIR:-}"
OS_RELEASE="${UNSLOTH_OS_RELEASE:-/etc/os-release}"
PROC_VERSION="${UNSLOTH_PROC_VERSION:-/proc/version}"
BASE="https://nvidia.github.io/libnvidia-container"

say()  { printf '%s\n' "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

if [[ "$(id -u)" != 0 ]]; then
    # piped through `curl | bash` there is no file to re-run: $0 is just "bash"
    if [[ -r "$0" ]] && command -v sudo >/dev/null 2>&1; then
        say "Re-running with sudo."
        exec sudo -E bash "$0" "$@"
    fi
    fail "run this as root: pipe it into 'sudo bash', or 'sudo bash docker/install_nvidia_toolkit.sh'." 2
fi

# Docker Desktop (macOS, or Windows with the WSL 2 backend) ships its own GPU
# integration: nothing to install on that side, and apt inside the WSL distro would
# configure a daemon Docker Desktop does not use.
if docker info 2>/dev/null | grep -qi 'Operating System: Docker Desktop'; then
    say "Docker Desktop detected: GPU support comes with it, nothing to install."
    say "On Windows keep the WSL 2 backend on and the NVIDIA Windows driver current (wsl --update)."
    exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
    fail "no NVIDIA driver found (nvidia-smi lists no GPU). Install the driver first, with your
       distribution's packages (Ubuntu: 'sudo ubuntu-drivers install'; RHEL/Fedora: the
       nvidia-driver module from the CUDA repository), reboot, then run this again.
       Unsloth images need driver 570.26 or newer." 2
fi

configured() {
    command -v nvidia-ctk >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi 'Runtimes:.*nvidia'
}

if configured; then
    say "NVIDIA Container Toolkit is already installed and Docker lists the nvidia runtime."
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
            curl -fsSL "${BASE}/gpgkey" \
                | gpg --dearmor --yes -o "${DESTDIR}/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
            curl -fsSL "${BASE}/stable/deb/nvidia-container-toolkit.list" \
                | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
                > "${DESTDIR}/etc/apt/sources.list.d/nvidia-container-toolkit.list"
            apt-get update -qq
            apt-get install -y -qq nvidia-container-toolkit
            ;;
        *" rhel "*|*" fedora "*|*" centos "*|*" amzn "*|*" rocky "*|*" almalinux "*)
            mkdir -p "${DESTDIR}/etc/yum.repos.d"
            curl -fsSL "${BASE}/stable/rpm/nvidia-container-toolkit.repo" \
                > "${DESTDIR}/etc/yum.repos.d/nvidia-container-toolkit.repo"
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

    say "Registering the nvidia runtime with Docker."
    nvidia-ctk runtime configure --runtime=docker
    # WSL 2 distros and containers often run without systemd; `service` covers them.
    if command -v systemctl >/dev/null 2>&1 && systemctl restart docker 2>/dev/null; then
        say "Docker restarted (systemctl)."
    elif command -v service >/dev/null 2>&1 && service docker restart 2>/dev/null; then
        say "Docker restarted (service)."
    else
        fail "could not restart Docker; restart it yourself, then run: docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L"
    fi
    configured || fail "Docker still does not list the nvidia runtime after the install; see 'docker info'."
fi

if [[ "${UNSLOTH_TOOLKIT_VERIFY:-1}" != 0 ]]; then
    # explicit platform: a cached ubuntu image of the other arch would otherwise be
    # picked and fail on "exec nvidia-smi: no such file" for the wrong reason
    case "$(uname -m)" in aarch64|arm64) PLATFORM=linux/arm64 ;; *) PLATFORM=linux/amd64 ;; esac
    say "Verifying: docker run --rm --gpus all --platform ${PLATFORM} ubuntu:24.04 nvidia-smi -L"
    if ! docker run --rm --gpus all --platform "${PLATFORM}" ubuntu:24.04 nvidia-smi -L; then
        fail "a container could not see the GPU. If this host is WSL 2 without Docker Desktop, make
       sure the Windows NVIDIA driver is current (wsl --update); otherwise see
       https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html"
    fi
fi
if grep -qi microsoft "$PROC_VERSION" 2>/dev/null; then
    say "WSL 2 host: the GPU is the Windows driver's; keep it current with the NVIDIA installer."
fi
say "Done. Run: docker run -d --gpus all -p 8000:8000 -p 8888:8888 unsloth/unsloth"
