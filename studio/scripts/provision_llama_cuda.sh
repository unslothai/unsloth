#!/usr/bin/env bash
# Provision a CUDA-enabled llama.cpp for Unsloth Studio GGUF *inference*.
#
# Builds into ~/.unsloth/llama.cpp (the dir Unsloth Studio's llama-server
# resolver checks: <dir>/build/bin/llama-server). Best-effort and idempotent:
# safe to re-run, never hard-fails the caller (always exits 0).
#
# Why this exists: torch ships its own bundled CUDA runtime, so training +
# GGUF *export* work without a system CUDA toolkit. But GGUF *inference* needs
# a CUDA-linked llama-server, and on NVIDIA ARM machines (NVIDIA DGX Spark /
# GB10, N1X "RTX" laptops) there is no published aarch64+CUDA prebuilt, so we
# build one. Handles the known gotchas on these platforms:
#   * nvcc rejects gcc-15            -> force gcc-14 / g++-14 as the host compiler
#   * glibc >= 2.41 vs CUDA < 13.3   -> install CUDA 13.3 (rsqrt header clash)
#   * sm_121 (Blackwell) GPUs        -> derive arch from the GPU's compute_cap
#
# Opt out entirely with UNSLOTH_NO_LLAMA_CUDA=1 (handled by the caller).
set -uo pipefail

LLAMA_DIR="${UNSLOTH_LLAMA_CPP_PATH:-$HOME/.unsloth/llama.cpp}"
SERVER="$LLAMA_DIR/build/bin/llama-server"
log() { printf '  - %s\n' "$*"; }

is_cuda_server() { [ -x "$1" ] && ldd "$1" 2>/dev/null | grep -qi 'libggml-cuda'; }

# 0. Already provisioned?
if is_cuda_server "$SERVER"; then
    log "CUDA llama-server already present: $SERVER"
    exit 0
fi

# 1. Require an NVIDIA GPU (this script is only meaningful with one).
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "no nvidia-smi found; skipping CUDA llama.cpp build"
    exit 0
fi

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
HAVE_APT=0; command -v apt-get >/dev/null 2>&1 && HAVE_APT=1

# 2. Base toolchain. gcc-14 is required because nvcc rejects gcc-15.
if [ "$HAVE_APT" -eq 1 ]; then
    $SUDO apt-get update -y >/dev/null 2>&1 || true
    $SUDO apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates gcc-14 g++-14 >/dev/null 2>&1 || true
fi

# 3. Locate nvcc; install the CUDA toolkit if missing.
find_nvcc() { command -v nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1; }
NVCC="$(find_nvcc)"
if [ -z "$NVCC" ] && [ "$HAVE_APT" -eq 1 ]; then
    log "CUDA toolkit (nvcc) not found - installing CUDA 13.3 (matches torch cu13x; avoids glibc>=2.41 rsqrt clash)"
    # shellcheck disable=SC1091
    . /etc/os-release 2>/dev/null || true
    case "$(uname -m)" in
        aarch64) NV_ARCH=sbsa ;;
        x86_64)  NV_ARCH=x86_64 ;;
        *)       NV_ARCH="" ;;
    esac
    case "${ID:-}${VERSION_ID:-}" in
        ubuntu24.04) NV_DISTRO=ubuntu2404 ;;
        ubuntu22.04) NV_DISTRO=ubuntu2204 ;;
        debian12)    NV_DISTRO=debian12 ;;
        *)           NV_DISTRO="" ;;
    esac
    if [ -n "$NV_ARCH" ] && [ -n "$NV_DISTRO" ]; then
        KR=/tmp/cuda-keyring.deb
        if curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/$NV_DISTRO/$NV_ARCH/cuda-keyring_1.1-1_all.deb" -o "$KR" 2>/dev/null; then
            $SUDO dpkg -i "$KR" >/dev/null 2>&1 || true
            $SUDO apt-get update -y >/dev/null 2>&1 || true
            $SUDO apt-get install -y cuda-toolkit-13-3 >/dev/null 2>&1 \
                || $SUDO apt-get install -y cuda-toolkit >/dev/null 2>&1 || true
        fi
    fi
    NVCC="$(find_nvcc)"
fi

if [ -z "$NVCC" ]; then
    log "could not provision a CUDA toolkit. Training + GGUF export still work;"
    log "GGUF *inference* in Studio will be unavailable until a CUDA toolkit exists."
    log "Re-run this script after installing one to enable GGUF inference."
    exit 0
fi

CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDAToolkit_ROOT="$CUDA_HOME"

# 4. Host compiler: prefer gcc-14 / g++-14 (nvcc rejects 15).
HCC=gcc;  command -v gcc-14 >/dev/null 2>&1 && HCC=gcc-14
HCXX=g++; command -v g++-14 >/dev/null 2>&1 && HCXX=g++-14
export CC="$HCC" CXX="$HCXX" CUDAHOSTCXX="$HCXX"

# 5. CUDA arch from the GPU's compute capability (e.g. "12.1" -> 121). Fallback: native.
CC_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
if [ -n "$CC_CAP" ]; then CUDA_ARCH="$CC_CAP"; else CUDA_ARCH="native"; fi

# 6. Clone + build into ~/.unsloth/llama.cpp.
mkdir -p "$(dirname "$LLAMA_DIR")"
if [ ! -d "$LLAMA_DIR/.git" ]; then
    rm -rf "$LLAMA_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" >/dev/null 2>&1 \
        || { log "git clone failed"; exit 0; }
fi
cd "$LLAMA_DIR" || exit 0

log "building CUDA llama.cpp (arch=$CUDA_ARCH, host=$HCXX) - this takes a few minutes..."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DCMAKE_CUDA_HOST_COMPILER="$HCXX" \
    -DLLAMA_CURL=ON >/dev/null 2>&1 || { log "cmake configure failed"; exit 0; }
# Build the full set unsloth-zoo's GGUF exporter expects too (llama-mtmd-cli,
# llama-gguf-split), so a pre-provisioned build satisfies both Studio inference
# AND save_pretrained_gguf without triggering a --clean-first rebuild later.
cmake --build build -j"$(nproc)" --target \
    llama-server llama-cli llama-quantize llama-mtmd-cli llama-gguf-split >/dev/null 2>&1 \
    || { log "cmake build failed"; exit 0; }

if is_cuda_server "$SERVER"; then
    log "CUDA llama-server ready: $SERVER"
else
    log "build finished but CUDA llama-server could not be confirmed"
fi
exit 0
