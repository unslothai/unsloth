#!/usr/bin/env bash
# Build CUDA llama.cpp for Studio GGUF *inference* into ~/.unsloth/llama.cpp
# (resolver checks <dir>/build/bin/llama-server). Idempotent, best-effort, exits
# 0. Exists because no aarch64+CUDA prebuilt covers NVIDIA ARM hosts (DGX Spark /
# GB10, N1X "RTX" laptops). Platform gotchas handled:
#   * nvcc rejects gcc-15          -> force gcc-14 / g++-14 host compiler
#   * glibc >= 2.41 vs CUDA < 13.3 -> install CUDA 13.3 (rsqrt header clash)
#   * sm_121 (Blackwell)           -> derive arch from the GPU's compute_cap
# Opt out with UNSLOTH_NO_LLAMA_CUDA=1 (handled by the caller).
set -uo pipefail

LLAMA_DIR="${UNSLOTH_LLAMA_CPP_PATH:-$HOME/.unsloth/llama.cpp}"
SERVER="$LLAMA_DIR/build/bin/llama-server"
log() { printf '  - %s\n' "$*"; }

# Serialize against install_llama_prebuilt.py (same lock file as its
# install_lock_path: <parent>/.<name>.install.lock; its filelock backend is
# flock(2), so this interoperates) and against a second copy of this script: the
# detached background builder can otherwise race an installer rerun or `unsloth
# studio update`, both of which mv/rm -rf inside $LLAMA_DIR. Append-mode open so
# the Python O_EXCL fallback's PID file is never truncated. 2h cap matches a
# worst-case source build; losing the wait means another provisioner is already
# doing this exact job, so exiting 0 is correct.
_LOCK_DIR="$(dirname "$LLAMA_DIR")"
mkdir -p "$_LOCK_DIR" 2>/dev/null
if command -v flock >/dev/null 2>&1; then
    if exec 9>>"$_LOCK_DIR/.$(basename "$LLAMA_DIR").install.lock" 2>/dev/null; then
        flock -w 7200 9 || { log "another llama.cpp install holds the lock; skipping"; exit 0; }
    fi
fi

# Detect CUDA two ways: monolithic (libggml-cuda in ldd) or split (dlopen-ed
# libggml-cuda.so* beside the binary, missed by ldd). CPU-only builds ship no
# libggml-cuda.so, so its presence is the reliable signal.
is_cuda_server() {
    [ -x "$1" ] || return 1
    ldd "$1" 2>/dev/null | grep -qi 'libggml-cuda' && return 0
    for _so in "$(dirname "$1")"/libggml-cuda.so*; do [ -e "$_so" ] && return 0; done
    return 1
}

# 0. Already provisioned? Skip when the server links libggml-cuda directly (ldd)
# or when a co-located libggml-cuda.so* is paired with the completion stamp this
# script writes after its own final CUDA check. The stamp closes the one gap in
# the structural check: an in-place rebuild interrupted after libggml-cuda.so is
# linked but before llama-server relinks leaves new .so + old CPU server, which
# the bare .so test would wrongly skip. We deliberately do NOT run a functional
# `--list-devices` probe here: this script runs in a stripped-down detached shell whose
# loader path can miss /usr/lib/wsl/lib, so the CUDA backend may fail to enumerate even
# on a perfectly good server -- and a false negative would wipe a validated build and
# trigger a needless, thermally-dangerous source rebuild on the NVIDIA-ARM laptops this
# targets. Trust the .so; never gamble the machine's thermals on an env-fragile probe.
_CUDA_STAMP="$LLAMA_DIR/build/bin/.unsloth-cuda-ok"
if is_cuda_server "$SERVER"; then
    if ldd "$SERVER" 2>/dev/null | grep -qi 'libggml-cuda' || [ -e "$_CUDA_STAMP" ]; then
        log "CUDA llama-server already present: $SERVER"
        exit 0
    fi
    log "CUDA .so present but the build never stamped complete (interrupted relink?); rebuilding"
fi

# 1. Require an NVIDIA GPU (this script is only meaningful with one).
if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "no nvidia-smi found; skipping CUDA llama.cpp build"
    exit 0
fi

SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
HAVE_APT=0; command -v apt-get >/dev/null 2>&1 && HAVE_APT=1

# 2. Base toolchain first, then gcc-14 (nvcc rejects gcc-15) in a SEPARATE apt
# transaction: gcc-14 is absent from default Ubuntu 22.04 / Debian 12 sources, so
# a combined transaction would abort and lose the base build tools too.
if [ "$HAVE_APT" -eq 1 ]; then
    $SUDO apt-get update -y >/dev/null 2>&1 || true
    # libcurl4-openssl-dev: -DLLAMA_CURL=ON needs it, and the WSL deferred path
    # skips setup.sh's GGUF dep install that would otherwise provide libcurl.
    $SUDO apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates libcurl4-openssl-dev >/dev/null 2>&1 || true
    $SUDO apt-get install -y --no-install-recommends gcc-14 g++-14 >/dev/null 2>&1 || true
fi

# 3. Locate nvcc; install the CUDA toolkit if missing. Prefer the highest
# /usr/local/cuda-<ver> toolkit: a stale unversioned `cuda` symlink or an older
# nvcc earlier on PATH could otherwise win and rebuild with CUDA 12.x, re-hitting
# the glibc>=2.41 / Blackwell clash this script exists to avoid. Fall back to a
# PATH nvcc (e.g. conda) only when no versioned system toolkit is present.
find_nvcc() {
    local _v
    _v="$(ls -d /usr/local/cuda-*/bin/nvcc 2>/dev/null | sort -V | tail -1)"
    if [ -n "$_v" ]; then printf '%s\n' "$_v"; return 0; fi
    command -v nvcc 2>/dev/null || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1
}
NVCC="$(find_nvcc)"
# A CUDA < 13 toolkit cannot build for the sm_121 Spark class (and CUDA < 13.3
# hits the glibc >= 2.41 rsqrt clash from the header) -- keeping it made every
# rerun fail configure/build and exit with the CPU server forever. When apt can
# provide 13.3, upgrade past a stale toolkit; find_nvcc's sort -V then prefers
# the new install, and if the install fails the old toolkit remains the last
# resort (previous behavior, still fine on non-Spark hosts like GH200 + cu12x).
_nvcc_stale=0
if [ -n "$NVCC" ]; then
    _nvcc_major="$("$NVCC" --version 2>/dev/null | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)"
    if [ -n "$_nvcc_major" ] && [ "$_nvcc_major" -lt 13 ] 2>/dev/null; then
        log "existing CUDA $_nvcc_major toolkit ($NVCC) predates this machine class; provisioning CUDA 13.3 alongside it"
        _nvcc_stale=1
    fi
fi
if { [ -z "$NVCC" ] || [ "$_nvcc_stale" -eq 1 ]; } && [ "$HAVE_APT" -eq 1 ]; then
    [ -z "$NVCC" ] && log "CUDA toolkit (nvcc) not found - installing CUDA 13.3 (matches torch cu13x; avoids glibc>=2.41 rsqrt clash)"
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
# CUDA + Linux dirs FIRST so the build uses Linux cmake/gcc/git, not Windows tools
# leaked in via WSL interop (/mnt/c). Keep original PATH so nvidia-smi resolves.
export PATH="$CUDA_HOME/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
export CUDAToolkit_ROOT="$CUDA_HOME"

# 4. Host compiler: prefer gcc-14 / g++-14 (nvcc rejects 15).
HCC=gcc;  command -v gcc-14 >/dev/null 2>&1 && HCC=gcc-14
HCXX=g++; command -v g++-14 >/dev/null 2>&1 && HCXX=g++-14
export CC="$HCC" CXX="$HCXX" CUDAHOSTCXX="$HCXX"

# 5. CUDA arch from the GPU's compute capability (e.g. "12.1" -> 121). Fallback: native.
# Only a purely-numeric capability is a valid CMAKE_CUDA_ARCHITECTURES; some WSL
# GPU-PV / driver combos report "N/A". "native" needs CMake >= 3.24 (Ubuntu
# 22.04 apt ships 3.22), so the fallback omits the flag entirely and lets
# ggml's version-guarded CMake defaults pick the arches instead.
CC_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
case "$CC_CAP" in
    ''|*[!0-9]*) CUDA_ARCH="" ;;
    *)           CUDA_ARCH="$CC_CAP" ;;
esac

# 6. Clone + build into ~/.unsloth/llama.cpp, honoring a UNSLOTH_LLAMA_TAG pin
# (same var setup.sh uses) instead of always tracking ggml-org main.
mkdir -p "$(dirname "$LLAMA_DIR")"
_LLAMA_REF="${UNSLOTH_LLAMA_TAG:-}"
# setup.sh's install policy pins source builds to the newest RELEASE ("latest"
# resolved to a tag; master bypasses the pin). Mirror it: unset or literal
# "latest" resolves via the GitHub API; on API failure the empty ref keeps the
# existing default-branch clone fallback (best effort, as before).
if [ -z "$_LLAMA_REF" ] || [ "$_LLAMA_REF" = "latest" ]; then
    _LLAMA_REF="$(curl -fsSL --max-time 15 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest 2>/dev/null \
        | grep -om1 '"tag_name": *"[^"]*"' | cut -d'"' -f4)"
    [ -n "$_LLAMA_REF" ] && log "pinning llama.cpp to release $_LLAMA_REF"
fi
# Back up any existing (e.g. CPU-only) llama.cpp: restored on any failure exit,
# dropped only once the fresh build yields a server -- never leave NO server.
_LLAMA_BAK=""
_restore_prev() {
    if [ -n "$_LLAMA_BAK" ] && [ -e "$_LLAMA_BAK" ]; then
        rm -rf "$LLAMA_DIR" 2>/dev/null
        mv "$_LLAMA_BAK" "$LLAMA_DIR" 2>/dev/null && log "restored previous llama.cpp install"
    fi
}
if [ ! -d "$LLAMA_DIR/.git" ]; then
    if [ -e "$LLAMA_DIR" ]; then
        _LLAMA_BAK="${LLAMA_DIR}.prev.$$"
        rm -rf "$_LLAMA_BAK" 2>/dev/null
        mv "$LLAMA_DIR" "$_LLAMA_BAK" 2>/dev/null || { rm -rf "$LLAMA_DIR" 2>/dev/null; _LLAMA_BAK=""; }
    fi
    _clone_ok=0
    if [ -n "$_LLAMA_REF" ]; then
        git clone --depth 1 --branch "$_LLAMA_REF" https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" >/dev/null 2>&1 && _clone_ok=1
    fi
    if [ "$_clone_ok" -ne 1 ]; then
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR" >/dev/null 2>&1 && _clone_ok=1
    fi
    if [ "$_clone_ok" -ne 1 ]; then
        log "git clone failed"
        _restore_prev
        exit 0
    fi
else
    # Existing checkout: honor the pin on reruns too (previously all ref
    # handling lived in the fresh-clone branch, so an existing tree rebuilt
    # whatever commit it had regardless of the pin). Best-effort -- an
    # unreachable ref keeps the current commit, matching the clone fallback.
    if [ -n "$_LLAMA_REF" ]; then
        _cur_head="$(git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null)"
        if git -C "$LLAMA_DIR" fetch --depth 1 origin "$_LLAMA_REF" >/dev/null 2>&1; then
            _ref_head="$(git -C "$LLAMA_DIR" rev-parse FETCH_HEAD 2>/dev/null)"
            if [ -n "$_ref_head" ] && [ "$_ref_head" != "$_cur_head" ]; then
                git -C "$LLAMA_DIR" checkout -q FETCH_HEAD >/dev/null 2>&1 \
                    && log "updated existing llama.cpp checkout to $_LLAMA_REF" \
                    || log "could not check out $_LLAMA_REF; keeping the current commit"
            fi
        else
            log "could not fetch $_LLAMA_REF; keeping the current commit"
        fi
    fi
fi
# Honor a UNSLOTH_LLAMA_PR pin (same var setup.sh supports) on fresh clones and
# existing checkouts alike; best-effort -- a failed fetch keeps what's there.
case "${UNSLOTH_LLAMA_PR:-}" in
    ''|*[!0-9]*) ;;
    *)
        if git -C "$LLAMA_DIR" fetch --depth 1 origin "pull/${UNSLOTH_LLAMA_PR}/head:_unsloth_pr_${UNSLOTH_LLAMA_PR}" >/dev/null 2>&1 \
                && git -C "$LLAMA_DIR" checkout "_unsloth_pr_${UNSLOTH_LLAMA_PR}" >/dev/null 2>&1; then
            log "checked out llama.cpp PR #${UNSLOTH_LLAMA_PR} (UNSLOTH_LLAMA_PR)"
        else
            log "could not fetch llama.cpp PR #${UNSLOTH_LLAMA_PR}; building the default branch"
        fi
        ;;
esac
cd "$LLAMA_DIR" || { _restore_prev; exit 0; }

# When rebuilding in-place over an existing git checkout, the whole-dir backup above
# was skipped (_LLAMA_BAK empty) -- but build/ may already hold a working (e.g. CPU)
# llama-server from a prior setup.sh source build. The wipe-on-failure paths below
# would destroy it with nothing to restore, leaving NO server despite the "keeps the
# existing server" promise (a thermal shutdown mid-CUDA-build is a real failure mode
# here). Back up the existing binaries so a failed rebuild can put them back. Only
# bin/ (server + dlopen-ed backends) is needed; cheap since any pre-existing server
# here is the non-CUDA fallback (a CUDA one would have exited at step 0).
_BUILD_BAK=""
if [ -z "$_LLAMA_BAK" ] && [ -x "$SERVER" ]; then
    _BUILD_BAK="${LLAMA_DIR}.binbak.$$"
    rm -rf "$_BUILD_BAK" 2>/dev/null
    cp -a "$LLAMA_DIR/build/bin" "$_BUILD_BAK" 2>/dev/null || _BUILD_BAK=""
fi
_restore_build() {
    if [ -n "$_BUILD_BAK" ] && [ -e "$_BUILD_BAK" ] && [ ! -x "$SERVER" ]; then
        mkdir -p "$LLAMA_DIR/build" 2>/dev/null
        rm -rf "$LLAMA_DIR/build/bin" 2>/dev/null
        mv "$_BUILD_BAK" "$LLAMA_DIR/build/bin" 2>/dev/null && log "restored previous llama-server (rebuild failed)"
    fi
    [ -n "$_BUILD_BAK" ] && rm -rf "$_BUILD_BAK" 2>/dev/null
    _BUILD_BAK=""
}

log "building CUDA llama.cpp (arch=${CUDA_ARCH:-cmake-default}, host=$HCXX) - this takes a few minutes..."
_cmake_configure() {
    # Empty CUDA_ARCH (unreadable compute_cap): omit the flag so ggml's own
    # CMake defaults apply -- "native" would need CMake >= 3.24.
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON -DGGML_CUDA_F16=ON \
        ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"} \
        -DCMAKE_CUDA_HOST_COMPILER="$HCXX" \
        -DLLAMA_CURL=ON >/dev/null 2>&1
}
# A pre-existing build/ may carry a stale CMake cache (relocated dir: bad absolute
# paths + GGML_CUDA=OFF). Reuse first (fast incremental); wipe only on failure.
if ! _cmake_configure; then
    log "stale/incompatible CMake cache detected; wiping build dir for a clean CUDA configure"
    rm -rf build
    _cmake_configure || { log "cmake configure failed"; cd /; _restore_build; _restore_prev; exit 0; }
fi
# Also builds the targets unsloth-zoo's GGUF exporter needs (llama-mtmd-cli,
# llama-gguf-split). Jobs default to ~half the cores (full -j(nproc) CUDA builds
# trip thermal shutdowns on NVIDIA-ARM laptops like the N1X "RTX Spark"), capped
# at ~1.5 GB/nvcc job. Tune: UNSLOTH_LLAMA_BUILD_JOBS=N; re-runs resume.
_ncpu="$(nproc 2>/dev/null || echo 4)"
# Honor a valid positive-int override; ignore junk/0 (cmake treats -j0 as all cores).
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
# nice/ionice: full speed when idle, yields to foreground Studio/training runs.
_NICE=""
command -v nice   >/dev/null 2>&1 && _NICE="nice -n 19"
command -v ionice >/dev/null 2>&1 && _NICE="$_NICE ionice -c 3"
_cmake_build() {
    # Only llama-server is REQUIRED: an old UNSLOTH_LLAMA_TAG pin may predate the
    # helper targets, whose absence must not fail the whole provision.
    $_NICE cmake --build build -j"$JOBS" --target llama-server >/dev/null 2>&1
}
_cmake_build_extras() {
    # Helper targets unsloth-zoo's GGUF exporter also uses -- best-effort each.
    for _t in llama-cli llama-quantize llama-mtmd-cli llama-gguf-split; do
        $_NICE cmake --build build -j"$JOBS" --target "$_t" >/dev/null 2>&1 || true
    done
}
if ! _cmake_build; then
    # An interrupted build (thermal/power shutdown, common on this machine class)
    # can leave a half-linked libggml-cuda.so that breaks the resume link
    # (undefined ggml_cuda_op_* refs); wipe and rebuild clean.
    log "build failed (likely interrupted/partial); wiping build dir and rebuilding clean"
    rm -rf build
    _cmake_configure || { log "cmake configure failed"; cd /; _restore_build; _restore_prev; exit 0; }
    _cmake_build || { log "cmake build failed"; cd /; _restore_build; _restore_prev; exit 0; }
fi
_cmake_build_extras
# Drop the backup on a successful build, or restore the prior server if the rebuild
# yielded none (idempotent; only restores when $SERVER is missing).
_restore_build

if is_cuda_server "$SERVER"; then
    : > "$_CUDA_STAMP" 2>/dev/null || true
    # unsloth_zoo's check_llama_cpp only searches the repo root, so mirror
    # setup.sh's root shim for the GGUF exporter's quantize binary.
    if [ -x "$LLAMA_DIR/build/bin/llama-quantize" ] && [ ! -e "$LLAMA_DIR/llama-quantize" ]; then
        ln -sf build/bin/llama-quantize "$LLAMA_DIR/llama-quantize" 2>/dev/null || true
    fi
    log "CUDA llama-server ready: $SERVER"
    [ -n "$_LLAMA_BAK" ] && rm -rf "$_LLAMA_BAK" 2>/dev/null
elif [ -x "$SERVER" ]; then
    # A server exists but isn't CUDA-confirmed; still better than the old backup.
    log "build finished but CUDA llama-server could not be confirmed"
    [ -n "$_LLAMA_BAK" ] && rm -rf "$_LLAMA_BAK" 2>/dev/null
else
    log "build finished but no llama-server was produced"
    cd /; _restore_prev
fi
exit 0
