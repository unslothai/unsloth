#!/usr/bin/env bash
# Container startup checks for Unsloth. Fails fast with actionable errors when the
# host GPU isn't reachable, catching the three modes behind ~95% of tickets:
#   1. nvidia-smi sees no GPU (missing --gpus all or nvidia-container-toolkit)
#   2. nvidia-smi works but torch.cuda.is_available() is False (driver too old)
#   3. GPU older than Ampere (sm < 80; Unsloth requires sm_80+)
# Bypass for offline tooling/docs/CI: docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
set -euo pipefail

# --- CUDA JIT toolchain selection (device-gated) ----------------------------
# The image bakes CUDA 13 ptxas + NVRTC only for the two Blackwell datacenter
# arches cu12.8 can't target -- sm_103 (B300/GB300) and sm_121 (GB10/DGX Spark).
# Both launched after cu12.8, so their hosts run a >=580 driver, exactly what a
# cu13 cubin needs to load. Every other arch (Turing..sm_120) uses the cu12.8
# tools on the documented 570-579 floor; a cu13 cubin can't load there (CUDA
# driver compat is forward-only), so routing their JIT through cu13 would break
# training. ptxas/NVRTC are host-side compilers, so they RUN under any driver --
# only their output the old driver rejects.
# Pick per DEVICE at boot (cap unknown at build time): cu12.8 is the immutable
# default (loadable on 570+), only sm_103/sm_121 switch Triton to cu13 ptxas and
# retarget the NVRTC symlink. Runs before every early-exit. Best-effort: the safe
# default needs no write (non-root/read-only fine); only a non-root datacenter
# host can't switch.
select_cuda_jit_tools() {
    local caps="" cc nvrtc_dir need_cu13=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        caps="$( { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true; } )"
    fi
    # Scan EVERY visible GPU: a sm_103/sm_121 part can sit behind an H100/B200 in
    # nvidia-smi ordering. If ANY needs cu13, switch for the whole process -- those
    # parts ship on >=580 drivers, so the host tolerates cu13 cubins for all archs.
    while IFS= read -r cc || [[ -n "${cc}" ]]; do
        cc="$(printf '%s' "${cc}" | tr -d '[:space:]')"
        case "${cc}" in
            10.3|12.1) need_cu13=1 ;;
        esac
    done <<< "${caps}"
    # Non-datacenter / undetectable / CPU host: keep cu12.8 (libnvrtc.so.12 ->
    # .cu128.orig, Triton on bundled cu12.8 ptxas), loadable on 570+ and needs no
    # write. One exception needs a write: an earlier boot on sm_103/sm_121 left
    # libnvrtc.so.12 -> .cu13 and this GPU's 570-579 driver can't load it --
    # reverse that selection (best-effort, same non-root caveat).
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
    # Blackwell datacenter present: point Triton at cu13 ptxas and retarget each
    # venv's libnvrtc.so.12 -> the staged cu13 alias. -z guard lets an explicit
    # TRITON_PTXAS_PATH win. Best-effort: a read-only/--user rootfs keeps cu12.8.
    # Covers the base venv and the Studio venv.
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
# Best-effort: never let JIT-tool selection block container startup.
select_cuda_jit_tools || true

# Make unslothai/notebooks available under /workspace before the user command.
# Best-effort, gated by UNSLOTH_SKIP_NOTEBOOK_SYNC, never blocks the container
# (see unsloth_sync_notebooks.sh).
sync_notebooks() {
    if [[ -x /usr/local/bin/unsloth-sync-notebooks ]]; then
        /usr/local/bin/unsloth-sync-notebooks || true
    fi
}

if [[ "${UNSLOTH_SKIP_GPU_CHECK:-0}" == "1" ]]; then
    sync_notebooks
    exec "$@"
fi

err()  { printf "\033[1;31mERROR:\033[0m %s\n" "$*" >&2; }
warn() { printf "\033[1;33mWARN:\033[0m %s\n"  "$*" >&2; }

# CPU mode for hosts that can't pass a GPU into a container (Docker Desktop on
# macOS/Windows-without-WSL2, CPU Linux, CI). Covers Jupyter, GGUF tooling,
# llama.cpp Studio chat and Data Recipes; NOT training or loading an Unsloth
# model (FastLanguageModel runs CUDA probes and raises without a GPU). With
# UNSLOTH_ALLOW_CPU=1 a missing GPU warns instead of failing pre-flight; a
# visible GPU still runs the checks below.
if [[ "${UNSLOTH_ALLOW_CPU:-0}" == "1" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
        warn "UNSLOTH_ALLOW_CPU=1 and no GPU visible -- continuing on CPU."
        warn "CPU mode covers Jupyter, GGUF tooling and llama.cpp (GGUF) Studio chat."
        warn "Training and loading Unsloth models (FastLanguageModel) still require an NVIDIA GPU."
        sync_notebooks
        exec "$@"
    fi
fi

# --- Check 1: nvidia-smi present and enumerates at least one GPU ------------
# nvidia-smi is injected by nvidia-container-toolkit on a GPU request, not baked
# in; a missing binary means "no GPU attached", same class as an empty -L.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L 2>/dev/null | grep -q '^GPU'; then
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
     models need a GPU, but Jupyter, GGUF tooling and llama.cpp (GGUF) Studio
     chat work on CPU:
       docker run -e UNSLOTH_ALLOW_CPU=1 ...

To bypass this check entirely (e.g. offline tooling), set UNSLOTH_SKIP_GPU_CHECK=1.
MSG
    exit 1
fi

# --- Check 2: torch can actually use the GPU --------------------------------
# Catches host-driver-too-old (nvidia-smi enumerates but CUDA contexts fail).
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

# Image targets every current NVIDIA arch from Turing onward.
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

# Secondary devices: all GPUs are exposed by default, so an unsupported later
# device only surfaces when a job pins to it. Device 0 is fatal above;
# secondaries warn now while excluding them is still cheap.
for d in range(1, n):
    dmaj, dmin = torch.cuda.get_device_capability(d)
    if dmaj < 7 or (dmaj == 7 and dmin < 5):
        dname = torch.cuda.get_device_name(d)
        print(f"WARNING: GPU {d} ({dname}, sm_{dmaj}{dmin}) is below this image's sm_75 floor.")
        print("         Multi-GPU runs that include it, or jobs pinned to it, will fail;")
        print("         exclude it with CUDA_VISIBLE_DEVICES or --gpus device=<supported>.")
PY

# --- arm64 note: baked llama.cpp is a CUDA 13 build -------------------------
# Upstream ships no CUDA 12 arm64 llama.cpp (only arm64-cpu/arm64-cuda13), so the
# arm64 image bakes cu13 while the torch stack (cu128) runs on 570+. A cu13 cubin
# can't load on 570-579, so below 580 GGUF export / Studio chat fail even though
# training works -- say so up front instead of failing mysteriously later.
if [ "$(uname -m)" = "aarch64" ]; then
    _drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    _drv_major="${_drv%%.*}"
    case "$_drv_major" in
        *[!0-9]* | "") ;;  # unreadable driver version -> no claim to make
        *)
            if [ "$_drv_major" -lt 580 ]; then
                echo "WARNING: this arm64 image bakes a CUDA 13 llama.cpp (upstream ships no CUDA 12 arm64 build)." >&2
                echo "         Host driver $_drv is < 580, which cannot load CUDA 13 binaries:" >&2
                echo "         training (torch cu128) works, but GGUF export / Studio chat will fail" >&2
                echo "         until the host driver is upgraded to >= 580." >&2
            fi
            ;;
    esac
fi

sync_notebooks
exec "$@"
