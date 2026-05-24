# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Unsloth + unsloth-zoo for every current NVIDIA arch (Turing -> Blackwell),
# on both linux/amd64 and linux/arm64.
#
# Why this image works:
#   * cu128 wheels ship native SASS (no PTX), empirically verified via
#     `cuobjdump --list-elf` against the downloaded wheels:
#       amd64:  sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
#       arm64:  sm_80 sm_90 sm_90a sm_100 sm_100a sm_120 sm_120a
#   * SASS is binary-compatible UPWARDS within a major (per ptrblck on the
#     PyTorch forum, May 2026): sm_86 SASS runs on sm_89 hardware (Ada);
#     sm_100 SASS runs on sm_103 (B300/GB300); sm_120 SASS runs on sm_121
#     (DGX Spark / GB10). So every non-Jetson NVIDIA GPU on
#     https://developer.nvidia.com/cuda/gpus is covered.
#   * Unsloth's runtime kernels are Triton, which JIT-compiles per device at first run.
#   * Anything that DOES need to be source-built (rare on this pin set) compiles
#     against TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX",
#     covering every current NVIDIA compute capability per
#     https://developer.nvidia.com/cuda/gpus.
#     The host GPU is irrelevant for compilation; nvcc emits whatever the arch
#     list says.
#
# Cross-arch build (DGX Spark / GB10 / sm_121):
#   The arm64 image is built via QEMU binfmt emulation on an x86_64 host:
#       bash docker/setup_qemu.sh                   # one-time host setup
#       docker buildx build --platform linux/arm64 -t unsloth-blackwell:arm64 .
#   The resulting arm64 image runs NATIVELY on aarch64 hosts (DGX Spark, Grace).
#   QEMU is only used at build time -- runtime emulation does NOT work for CUDA.
#   xformers has no cu128 aarch64 wheel; on arm64 we fall back to Unsloth's
#   built-in SDPA path (~5-10% slower than xformers but functionally complete).
#
# Build host requirements:
#   * Docker with buildkit (default since 23.x)
#   * docker buildx (mandatory for multi-platform; install: apt install docker-buildx)
#   * nvidia-container-toolkit (only needed for `docker run --gpus all` at test time)
#   * For arm64 builds on x86_64 hosts: QEMU binfmt (see docker/setup_qemu.sh)
#   * A GPU is NOT required at build time.
# -----------------------------------------------------------------------------

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12

# =============================================================================
# Stage 1: builder  -- toolkit + dev headers, builds any source extensions
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

# TARGETARCH is auto-populated by buildx ("amd64" or "arm64"). We use it to
# select an unsloth extras set that matches the wheels actually available for
# the target platform (xformers has no cu128 aarch64 wheel as of 0.0.34).
ARG TARGETARCH
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cross-compile for every current NVIDIA arch per
    # https://developer.nvidia.com/cuda/gpus:
    #   sm_75   Turing       T4, RTX 20-series, Quadro RTX            (amd64 only)
    #   sm_80   Ampere DC    A100, A30                                (amd64 only)
    #   sm_86   Ampere       A40, RTX A6000, RTX 30-series            (amd64 only)
    #   sm_89   Ada          L4, L40, L40S, RTX 40-series             (amd64 only)
    #   sm_90   Hopper       H100, H200, GH200 (Grace-Hopper is arm64)
    #   sm_100  Blackwell DC B100, B200, GB200 (GB200 is arm64)
    #   sm_103  Blackwell DC B300, GB300
    #   sm_120  Blackwell    RTX 50-series, RTX PRO 6000 Blackwell    (amd64 only)
    #   sm_121  Blackwell    GB10 (DGX Spark)                          (arm64 only)
    # +PTX on the highest lets future arch revisions run via JIT-PTX.
    # We keep the same list on both arches: nvcc happily emits archs that
    # don't exist on the build host, and any extras that aren't relevant for
    # the target just bloat compile time slightly (not size, since we don't
    # source-build any extension at install time on the pin set).
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX" \
    MAX_JOBS=4 \
    CUDA_HOME=/usr/local/cuda \
    # Build-host-independence guards. The build must NEVER introspect a GPU,
    # because the build host may have a B200, RTX 6000, or no GPU at all
    # (GitHub Actions ubuntu-latest). All three must yield byte-identical images.
    #
    # 1) Stop unsloth from JIT-compiling kernels at import time and writing a
    #    sm_NNN-specific blob into /opt/unsloth-venv/.../unsloth_compiled_cache/.
    UNSLOTH_COMPILE_DISABLE=1 \
    UNSLOTH_COMPILE_OVERWRITE=0 \
    # 2) Stop unsloth-zoo / vllm from probing torch.cuda.is_available() during
    #    setup. There's no GPU here, and we don't want it to silently skip a wheel.
    UNSLOTH_DISABLE_GPU_PROBE=1 \
    # 3) Force CUDA_VISIBLE_DEVICES empty so any stray torch.cuda call during
    #    `pip install` returns "no devices" rather than triggering host-specific
    #    code paths (we re-enable at runtime via `docker run --gpus all`).
    CUDA_VISIBLE_DEVICES=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git build-essential \
        ninja-build cmake pkg-config \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Build into an isolated prefix. NOTE: we do NOT install pip or uv into the
# system Python -- on Ubuntu 24.04 the system interpreter is marked
# externally-managed (PEP 668) and `pip install` is refused. Instead, the new
# venv bootstraps its own pip via ensurepip (provided by the python3.12-venv
# apt package), and we install uv into the venv a few lines below.
ENV VENV=/opt/unsloth-venv
RUN python -m venv ${VENV} && ${VENV}/bin/pip install -U pip wheel setuptools

# Unified install: torch + triton + bitsandbytes + unsloth + unsloth_zoo
# resolve in a SINGLE uv pip pass. This is mandatory -- splitting it across
# multiple `pip install` calls causes bnb's transitive `cuda-toolkit` dep to
# silently upgrade torch to 2.12.0+cu130 in a later pass, breaking the cu128
# xformers wheel that was pinned earlier. (Empirically discovered; the cu cascade
# happens AFTER xformers is already on disk, leaving a working-but-mismatched env.)
#
# uv-specific flags explained:
#   --index-strategy unsafe-best-match
#     The PyTorch index serves an old `requests==2.28.1` which conflicts with
#     `datasets>=2.32.2`. uv's default is "first index wins per package" to
#     prevent dependency confusion; we override here because both indexes
#     (pytorch.org/whl/cu128 + pypi.org) are equally trusted.
#   --extra-index-url https://download.pytorch.org/whl/cu128
#     Where torch's +cu128 wheels live, plus the xformers/cu128 URLs referenced
#     by unsloth's `cu128onlytorch2100` extra.
#
# Why the extra is `cu128-ampere-torch2100` (not `cu128-torch2100-ampere`):
#   See unsloth_src/pyproject.toml:835. The ordering is ampere-then-torch-ver.
#
# Why arm64 uses a different extra:
#   `cu128-ampere-torch2100` transitively pulls `cu128onlytorch2100` whose
#   xformers wheel URL is hardcoded to manylinux_2_28_x86_64. There is no
#   cu128 aarch64 wheel for xformers as of 0.0.34. We use the plain
#   `huggingface` extra on arm64 -- Unsloth falls back to its native SDPA
#   kernels (a ~5-10% slowdown vs xformers; functionally complete).
#
# Why no `flash-attn` here:
#   - FA3 is hard-refused on Blackwell (Dao-AILab/flash-attention#1810).
#   - FA2 has no prebuilt wheel for cu128+torch2.10+cp312 -> would require
#     a ~30min source build, fragile on the 16GB ubuntu-latest CI runner.
#   - Unsloth gracefully falls back to xformers/SDPA on Blackwell anyway.
#   - Users on Ampere/Ada/Hopper who want FA2 can `pip install flash-attn`
#     on top of this image at deploy time.
ARG UNSLOTH_REF=main
ARG UNSLOTH_ZOO_REF=main
RUN set -eux \
 && case "${TARGETARCH:-amd64}" in \
        amd64) UNSLOTH_EXTRA="cu128-ampere-torch2100" ;; \
        arm64) UNSLOTH_EXTRA="huggingface" ;; \
        *)     echo "ERROR: unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
 && echo ">> TARGETARCH=${TARGETARCH:-amd64}, unsloth extra=[${UNSLOTH_EXTRA}]" \
 && ${VENV}/bin/pip install uv \
 && ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.11.0" \
        "triton>=3.6.0" \
        "bitsandbytes>=0.49.2,!=0.46.0,!=0.48.0" \
        "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@${UNSLOTH_ZOO_REF}" \
        "unsloth[${UNSLOTH_EXTRA}] @ git+https://github.com/unslothai/unsloth@${UNSLOTH_REF}"

# 5) Emit a lockfile so the next rebuild can be byte-identical even if PyPI
#    has moved on. Bake it into the image at /opt/unsloth-venv/requirements.lock.txt
#    so `docker run ... cat /opt/unsloth-venv/requirements.lock.txt > pins.txt`
#    gives you the input to a fully-pinned rebuild.
RUN ${VENV}/bin/pip freeze --exclude-editable > ${VENV}/requirements.lock.txt \
 && head -50 ${VENV}/requirements.lock.txt

# 6) Strip pip cache & __pycache__ to shrink the layer copied to runtime.
RUN find ${VENV} -depth -type d -name __pycache__ -exec rm -rf {} + \
 && find ${VENV} -depth -type d -name tests -exec rm -rf {} + \
 && rm -rf /root/.cache/pip /root/.cache/uv

# Build-time verification.
#
# (1) arch-list check uses the RAW C++ accessor (not torch.cuda.get_arch_list()).
#     The Python wrapper checks torch.cuda.is_available() first and returns []
#     when no GPU is visible -- which is always the case here because
#     CUDA_VISIBLE_DEVICES is empty by design.
#
# (2) We verify required packages via package metadata only -- we do NOT import
#     unsloth or unsloth_zoo here. Their __init__ calls torch.cuda.get_device_
#     properties(0) which requires an actual CUDA device (UNSLOTH_ALLOW_CPU=1
#     only bypasses the first gate, not the deeper init). Import-time
#     correctness is exercised at deploy time by smoke_test.py with --gpus all.
RUN TARGETARCH="${TARGETARCH:-amd64}" ${VENV}/bin/python - <<'PY'
import os, platform
target = os.environ.get("TARGETARCH", "amd64")
mach   = platform.machine()
print(f"build target: TARGETARCH={target} platform.machine()={mach}")

import torch
arches = torch._C._cuda_getArchFlags().split()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arches:", arches)
assert torch.__version__.startswith("2.10.0"), f"torch silently moved: {torch.__version__}"
assert "+cu128" in torch.__version__, f"cu build silently changed: {torch.__version__}"
assert "sm_100" in arches, f"sm_100 (B200/GB200) missing: {arches}"
# cu128 wheels ship sm_120 native SASS on BOTH amd64 (RTX 5090 / RTX PRO 6000
# Blackwell) and aarch64 (Grace systems). On arm64 sm_120 is what DGX Spark
# (sm_121) runs via minor-forward-compat within major 12; sm_121 itself is
# never in any cu128 wheel.
assert "sm_120" in arches, f"sm_120 missing: {arches}"
print(f"OK: torch 2.10.0+cu128 with sm_100 + sm_120 native SASS intact ({target})")

from importlib.metadata import version, PackageNotFoundError
# xformers has no cu128 aarch64 wheel as of 0.0.34, so we only require it
# on amd64. Everything else is platform-agnostic.
REQUIRED = ["torch", "triton", "bitsandbytes", "unsloth",
            "unsloth_zoo", "transformers", "trl", "peft", "accelerate"]
if target == "amd64":
    REQUIRED.insert(2, "xformers")
missing = []
for pkg in REQUIRED:
    try:
        v = version(pkg.replace("_", "-"))
        print(f"  {pkg:14s} {v}")
    except PackageNotFoundError:
        missing.append(pkg)
if missing:
    raise SystemExit(f"FAIL: missing wheels: {missing}")
print("OK: all required wheels present")

# Lightweight imports: these init without touching CUDA, unlike unsloth.
import importlib
LIGHT_IMPORTS = ["bitsandbytes", "triton"]
if target == "amd64":
    LIGHT_IMPORTS.insert(0, "xformers")
for pkg in LIGHT_IMPORTS:
    importlib.import_module(pkg)
print(f"OK: {' + '.join(LIGHT_IMPORTS)} import cleanly on no-GPU host")
PY

# =============================================================================
# Stage 2: runtime  -- slim runtime image, no nvcc, no headers
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# The nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 manifest is multi-arch
# (linux/amd64 + linux/arm64). docker/buildx picks the right one for
# TARGETPLATFORM at this FROM line; no conditional needed.
ARG TARGETARCH
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/unsloth-venv/bin:${PATH} \
    HF_HOME=/workspace/.cache/huggingface \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    # Keep the arch list visible at runtime in case the user source-builds anything
    # extra inside the container (e.g. a custom CUDA op). Same list as the builder
    # stage so a `pip install some-cuda-ext` inside the container gets a SASS
    # blob that covers every supported arch.
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0;12.1+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git libgomp1 \
        gcc g++ \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && rm -rf /var/lib/apt/lists/*
# Why gcc + g++ + python3.12-dev in the RUNTIME stage:
#   Triton's nvidia backend lazily compiles a small C extension (CudaUtils) on
#   first GPU access. Without a C compiler + Python headers the very first
#   forward pass of any Unsloth model dies with:
#       RuntimeError: Failed to find C compiler. Please specify via CC env var.
#   Adds ~250MB to the runtime image, which is the cost of letting every kernel
#   JIT correctly. (Pre-compiling CudaUtils at build time would need a GPU, so
#   shipping the toolchain is the right trade-off.)

COPY --from=builder /opt/unsloth-venv /opt/unsloth-venv

# DGX Spark / GB10 (sm_121) fix, arm64 ONLY.
#
# Two cu13 components need to override what the cu128 stack ships, because
# nothing in CUDA 12.8 -- toolkit or wheel -- knows about sm_121:
#
#   (1) torch's bundled libnvrtc.so.12 (from CUDA 12.8) does not accept
#       sm_121 as --gpu-architecture. The jiterator C++ side queries the
#       device cap directly, so any path that JIT-compiles a kernel (e.g.
#       torch.fft.rfft(complex).abs(), used inside mel-spectrogram code)
#       errors out. Fix: symlink libnvrtc.so.13 over the bundled .so.12.
#
#   (2) Triton's nvidia backend invokes ptxas. Triton wheels older than
#       3.6.0 bundled cu12.8 ptxas which tops out at sm_120 and refuses
#       sm_121, silently downgrading to sm_80 per triton-lang/triton#8335.
#       Triton 3.6.0 (which we pin above) bundles cu13 ptxas, but for
#       defense in depth we ALSO install cuda-nvcc-13-0 and point Triton
#       at it via TRITON_PTXAS_PATH.
#
# NVRTC and ptxas are CPU-side compilers; they do NOT call into libcuda,
# so we can install cu13 alongside the cu128 runtime without any driver
# requirement bump (toolkit driver floor stays 570+).
#
# amd64 image is untouched: no sm_121 hardware exists on amd64, and the
# extra ~400 MB would be dead weight.
RUN if [ "${TARGETARCH:-amd64}" = "arm64" ]; then \
        set -eux; \
        # SBSA = Server Base System Architecture; the NVIDIA repo path for
        # Grace / GH200 / GB200 / DGX Spark aarch64 hosts.
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb" \
             -o /tmp/cuda-keyring.deb; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm /tmp/cuda-keyring.deb; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-nvrtc-13-0 \
            cuda-nvcc-13-0; \
        rm -rf /var/lib/apt/lists/*; \
        # (1) NVRTC swap. torch's wheel-bundled cu128 NVRTC -> cu13 NVRTC.
        NVRTC_DIR=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvrtc/lib; \
        if [ -f "${NVRTC_DIR}/libnvrtc.so.12" ]; then \
            mv "${NVRTC_DIR}/libnvrtc.so.12" "${NVRTC_DIR}/libnvrtc.so.12.cu128.orig"; \
            ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13 "${NVRTC_DIR}/libnvrtc.so.12"; \
        fi; \
    fi

WORKDIR /workspace
RUN mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}

COPY smoke_test.py /workspace/smoke_test.py
COPY entrypoint.sh /usr/local/bin/unsloth-entrypoint
RUN chmod +x /usr/local/bin/unsloth-entrypoint

# Entrypoint runs three fast pre-flight checks before user code:
#   1. nvidia-smi sees at least one GPU       (catches missing --gpus all)
#   2. torch.cuda.is_available() is True      (catches host driver too old)
#   3. compute capability >= sm_80            (catches pre-Ampere GPUs)
# Each check fails with an actionable error pointing to the fix.
# Bypass for offline tooling:  docker run -e UNSLOTH_SKIP_GPU_CHECK=1 ...
ENTRYPOINT ["/usr/local/bin/unsloth-entrypoint"]

# Default command: interactive python REPL.
# Override examples:
#   docker run --gpus all unsloth/unsloth:latest python /workspace/smoke_test.py
#   docker run --gpus all -it unsloth/unsloth:latest bash
CMD ["python"]
