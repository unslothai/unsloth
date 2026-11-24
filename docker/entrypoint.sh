#!/usr/bin/env bash
# Container startup checks for Unsloth. Fails fast with actionable errors when the
# host GPU isn't reachable, catching the three modes behind ~95% of tickets:
#   1. nvidia-smi sees no GPU (missing --gpus all or nvidia-container-toolkit)
#   2. nvidia-smi works but torch.cuda.is_available() is False (driver too old)
#   3. GPU older than Ampere (sm < 80; Unsloth requires sm_80+)
# Bypass for offline tooling/docs/CI: docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

# Relink the Studio code into the home before the CUDA tool selection reads the venv.
if [[ -x /usr/local/bin/unsloth-studio-home ]]; then
    /usr/local/bin/unsloth-studio-home || {
        echo "ERROR: could not link Unsloth Studio's code into ${UNSLOTH_STUDIO_HOME:-/opt/unsloth-studio}; see the messages above" >&2
        exit 1
    }
fi

# CUDA 13 ptxas/NVRTC are only for sm_103/sm_121 (need >=580 drivers); everything else uses
# cu12.8 on the 570 floor. Chosen per device at boot.
select_cuda_jit_tools() {
    local caps="" cc nvrtc_dir need_cu13=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        caps="$( { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true; } )"
    fi
    # scan EVERY visible GPU: an sm_103/sm_121 part can sit behind an H100
    while IFS= read -r cc || [[ -n "${cc}" ]]; do
        cc="$(printf '%s' "${cc}" | tr -d '[:space:]')"
        case "${cc}" in
            10.3|12.1) need_cu13=1 ;;
        esac
    done <<< "${caps}"
    # reverse a .cu13 link an earlier sm_103/sm_121 boot left in the writable layer
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
select_cuda_jit_tools || true

# Hand root:root downloads in a host-owned HF cache mount back to its owner, keeping group root rwX for Studio's
# capability-less uid 0 tool sandbox; that sandbox writes this cache, hence fd-relative no-follow updates only.
hf_cache_uid="$(stat -c '%u' "${HF_HOME:-}" 2>/dev/null || true)"
if [[ "$(id -u):$(id -g)" == 0:0 && -n "${hf_cache_uid}" && "${hf_cache_uid}" != 0 ]] && command -v python3 >/dev/null 2>&1; then
    while :; do
        python3 -I -S - "${HF_HOME}" "${hf_cache_uid}" <<'PY' || true
import os, stat, sys

top, uid = sys.argv[1], int(sys.argv[2])
src, src_gid = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 0)


def has_acl(fd):
    try:
        return bool(os.getxattr(fd, "system.posix_acl_access"))
    except OSError:
        return False


def still_here(name, dfd, st, is_dir):
    """The name is still this inode, its only link: an unlinked cache name could leave it living outside the cache."""
    try:
        now = os.stat(name, dir_fd = dfd, follow_symlinks = False)
    except OSError:
        return False
    return (now.st_dev, now.st_ino) == (st.st_dev, st.st_ino) and (is_dir or now.st_nlink == 1)


def hand_over(name, dfd):
    st = os.stat(name, dir_fd = dfd, follow_symlinks = False)
    # Symlinks stay root's: removing one only needs the parent directory, and a path-based lchown can be redirected.
    if (st.st_uid, st.st_gid) != (src, src_gid) or stat.S_ISLNK(st.st_mode) or (st.st_nlink > 1 and not stat.S_ISDIR(st.st_mode)):
        return
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd = dfd)
    try:
        st = os.fstat(fd)
        is_dir = stat.S_ISDIR(st.st_mode)
        if (st.st_uid, st.st_gid) != (src, src_gid) or not (is_dir or (stat.S_ISREG(st.st_mode) and st.st_nlink == 1)):
            return
        if not still_here(name, dfd, st, is_dir):
            return
        # Under an ACL the group bits set the mask, not group root's entry, so the grant below would not hold.
        if has_acl(fd):
            return
        mode = stat.S_IMODE(st.st_mode) | (0o070 if is_dir or st.st_mode & 0o111 else 0o060)
        os.fchmod(fd, mode if is_dir else mode & ~0o6000)
        os.fchown(fd, uid, -1)
        if not still_here(name, dfd, st, is_dir):
            os.fchown(fd, src, -1)
            os.fchmod(fd, stat.S_IMODE(st.st_mode))
    finally:
        os.close(fd)


for _, dirs, files, dfd in os.fwalk(top, follow_symlinks = False):
    # Studio resumes only partials it owns; they are handed over once published as blobs.
    for name in dirs + [f for f in files if not f.endswith(".incomplete")]:
        try:
            hand_over(name, dfd)
        except OSError:
            pass
PY
        sleep 30
    done </dev/null >/dev/null 2>&1 &
fi

# best-effort, gated by UNSLOTH_SKIP_NOTEBOOK_SYNC, never blocks the container
sync_notebooks() {
    if [[ -x /usr/local/bin/unsloth-sync-notebooks ]]; then
        /usr/local/bin/unsloth-sync-notebooks || true
    fi
}

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

# nvidia-smi is injected on a GPU request, so a missing binary means "no GPU attached"
gpu_visible() {
    local listing
    # nvidia-smi -L ignores CUDA_VISIBLE_DEVICES but torch honours it: "" and -1 leave device_count() == 0, so they are no GPU
    case "${CUDA_VISIBLE_DEVICES-unset}" in
        ""|-1) return 1 ;;
    esac
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    listing="$(nvidia-smi -L 2>/dev/null || true)"
    grep -q '^GPU' <<< "${listing}"
}

# The library reads UNSLOTH_ALLOW_CPU=1 as CPU-only CI and skips its TRL trainer patches, so it may only reach processes with no GPU.
# Images opt in via UNSLOTH_IMAGE_ALLOW_CPU; `-`, not `:-`, so an explicitly empty UNSLOTH_ALLOW_CPU means off.
allow_cpu="${UNSLOTH_ALLOW_CPU-${UNSLOTH_IMAGE_ALLOW_CPU:-0}}"
if gpu_visible; then
    has_gpu=1
    if [[ "${UNSLOTH_ALLOW_CPU:-}" == "1" ]]; then
        warn "Ignoring UNSLOTH_ALLOW_CPU=1: a GPU is visible, and the variable would turn off Unsloth's TRL trainer patches."
        warn "Drop it from docker run; shells opened with docker exec still see the value you passed."
    fi
    unset UNSLOTH_ALLOW_CPU
else
    has_gpu=0
    if [[ "${allow_cpu}" == "1" ]]; then
        export UNSLOTH_ALLOW_CPU=1
    fi
fi

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    sync_notebooks
    exec "$@"
fi

if [[ "${has_gpu}" == "0" && "${allow_cpu}" == "1" ]]; then
    warn "UNSLOTH_ALLOW_CPU=1 and no GPU visible -- continuing on CPU."
    warn "CPU mode covers Jupyter, GGUF tooling and llama.cpp (GGUF) Unsloth Studio chat."
    warn "Training and loading Unsloth models (FastLanguageModel) still require an NVIDIA GPU."
    sync_notebooks
    exec "$@"
fi

if [[ "${has_gpu}" == "0" ]]; then
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
     models need a GPU, but Jupyter, GGUF tooling and llama.cpp (GGUF) Unsloth Studio
     chat work on CPU:
       docker run -e UNSLOTH_ALLOW_CPU=1 ...

To bypass this check entirely (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

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

python - >&2 <<'PY' || exit 1
import sys
import torch
major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
n = torch.cuda.device_count()
print(f"Unsloth container: {n} GPU(s). Primary: {name}  sm_{major}{minor}  bf16={torch.cuda.is_bf16_supported()}")

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

# an unsupported secondary only surfaces when a job pins to it, so warn now
for d in range(1, n):
    dmaj, dmin = torch.cuda.get_device_capability(d)
    if dmaj < 7 or (dmaj == 7 and dmin < 5):
        dname = torch.cuda.get_device_name(d)
        print(f"WARNING: GPU {d} ({dname}, sm_{dmaj}{dmin}) is below this image's sm_75 floor.")
        print("         Multi-GPU runs that include it, or jobs pinned to it, will fail;")
        print("         exclude it with CUDA_VISIBLE_DEVICES or --gpus device=<supported>.")
PY

# Upstream ships no CUDA 12 arm64 llama.cpp, so the arm64 image bakes cu13 while torch
# runs on 570+: below 580, GGUF export and Studio chat fail even though training works.
if [ "$(uname -m)" = "aarch64" ]; then
    _drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    _drv_major="${_drv%%.*}"
    case "$_drv_major" in
        *[!0-9]* | "") ;;  # unreadable driver version -> no claim to make
        *)
            if [ "$_drv_major" -lt 580 ]; then
                echo "WARNING: this arm64 image bakes a CUDA 13 llama.cpp (upstream ships no CUDA 12 arm64 build)." >&2
                echo "         Host driver $_drv is < 580, which cannot load CUDA 13 binaries:" >&2
                echo "         training (torch cu128) works, but GGUF export / Unsloth Studio chat will fail" >&2
                echo "         until the host driver is upgraded to >= 580." >&2
            fi
            ;;
    esac
fi

sync_notebooks
exec "$@"
