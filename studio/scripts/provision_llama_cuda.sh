#!/usr/bin/env bash
# Build a CUDA llama.cpp for Unsloth Studio GGUF *inference* into
# ~/.unsloth/llama.cpp (resolver checks <dir>/build/bin/llama-server).
# Idempotent, best-effort: safe to re-run, always exits 0.
#
# Needed because no aarch64+CUDA llama.cpp prebuilt exists for NVIDIA ARM hosts
# (DGX Spark / GB10, N1X "RTX" laptops). Handles the platform gotchas:
#   * nvcc rejects gcc-15            -> force gcc-14 / g++-14 as the host compiler
#   * glibc >= 2.41 vs CUDA < 13.3   -> install CUDA 13.3 (rsqrt header clash)
#   * sm_121 (Blackwell) GPUs        -> derive arch from the GPU's compute_cap
#
# Opt out with UNSLOTH_NO_LLAMA_CUDA=1 (handled by the caller).
set -uo pipefail

LLAMA_DIR="${UNSLOTH_LLAMA_CPP_PATH:-$HOME/.unsloth/llama.cpp}"
SERVER="$LLAMA_DIR/build/bin/llama-server"
log() { printf '  - %s\n' "$*"; }

# CUDA-capable in two layouts: old monolithic (libggml-cuda is a direct ldd dep)
# or current split build (CUDA is a dlopen-ed backend libggml-cuda.so* beside the
# binary, not shown by ldd). ldd alone false-negatives on current llama.cpp; a
# CPU-only build has no libggml-cuda.so, so its presence is the reliable signal.
is_cuda_server() {
    [ -x "$1" ] || return 1
    ldd "$1" 2>/dev/null | grep -qi 'libggml-cuda' && return 0
    for _so in "$(dirname "$1")"/libggml-cuda.so*; do [ -e "$_so" ] && return 0; done
    return 1
}

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
# CUDA toolkit + Linux dirs FIRST so the build uses Linux cmake/gcc/git, not a
# Windows tool leaked into PATH via WSL interop (/mnt/c, also has spaces). Keep
# the original PATH after so nvidia-smi etc. still resolve.
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
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
_cmake_configure() {
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
        -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DCMAKE_CUDA_HOST_COMPILER="$HCXX" \
        -DLLAMA_CURL=ON >/dev/null 2>&1
}
# A pre-existing build/ may carry an incompatible CMake cache (e.g. the installer
# relocates a versioned build dir here, leaving stale absolute paths + GGML_CUDA=OFF),
# making CUDA configure fail. Try to reuse build/ first (fast incremental resume);
# only wipe and configure clean if that fails.
if ! _cmake_configure; then
    log "stale/incompatible CMake cache detected; wiping build dir for a clean CUDA configure"
    rm -rf build
    _cmake_configure || { log "cmake configure failed"; exit 0; }
fi
# Build the full target set unsloth-zoo's GGUF exporter also needs (llama-mtmd-cli,
# llama-gguf-split) so one build serves both Studio inference and save_pretrained_gguf.
# Parallelism default = ~half the cores: much faster than a tiny -j4, but leaves
# thermal/power headroom -- a full -j(nproc) CUDA build trips shutdowns on
# thermally constrained NVIDIA-ARM laptops (e.g. N1X "RTX Spark"). Also cap by RAM
# (~1.5 GB per nvcc job) to avoid OOM. Tune with UNSLOTH_LLAMA_BUILD_JOBS=N (raise
# on a well-cooled box, lower if it still trips). Incremental: a re-run resumes.
_ncpu="$(nproc 2>/dev/null || echo 4)"
# Honor a valid positive-int override; ignore junk/0 (cmake reads -j0 as "all cores").
if [ -n "${UNSLOTH_LLAMA_BUILD_JOBS:-}" ] && [ "${UNSLOTH_LLAMA_BUILD_JOBS}" -ge 1 ] 2>/dev/null; then
    JOBS="$UNSLOTH_LLAMA_BUILD_JOBS"
else
    _half=$(( (_ncpu + 1) / 2 ))            # ~half the cores for thermal headroom
    if [ "$_ncpu" -le 4 ]; then _half="$_ncpu"; fi   # tiny boxes: use all cores
    _memkb="$(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0)"
    _memjobs=$(( _memkb / 1572864 ))        # 1.5 GB per nvcc job
    if [ "$_memjobs" -lt 1 ]; then _memjobs=1; fi
    JOBS="$_half"
    if [ "$_memjobs" -lt "$JOBS" ]; then JOBS="$_memjobs"; fi
fi
log "building with -j${JOBS} (cores=${_ncpu})"
# Lowest CPU + idle I/O priority so this background build keeps full speed when the
# box is idle but instantly yields to a foreground `unsloth studio` / training run.
_NICE=""
command -v nice   >/dev/null 2>&1 && _NICE="nice -n 19"
command -v ionice >/dev/null 2>&1 && _NICE="$_NICE ionice -c 3"
_cmake_build() {
    $_NICE cmake --build build -j"$JOBS" --target \
        llama-server llama-cli llama-quantize llama-mtmd-cli llama-gguf-split >/dev/null 2>&1
}
if ! _cmake_build; then
    # An interrupted build (e.g. a thermal/power shutdown mid-compile, which this
    # machine class is prone to) can leave a partially-linked libggml-cuda.so that
    # then fails to link llama-server on resume (undefined ggml_cuda_op_* refs).
    # Wipe build/ and rebuild clean once before giving up.
    log "build failed (likely interrupted/partial); wiping build dir and rebuilding clean"
    rm -rf build
    _cmake_configure || { log "cmake configure failed"; exit 0; }
    _cmake_build || { log "cmake build failed"; exit 0; }
fi

if is_cuda_server "$SERVER"; then
    log "CUDA llama-server ready: $SERVER"
else
    log "build finished but CUDA llama-server could not be confirmed"
fi
exit 0
