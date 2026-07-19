# syntax=docker/dockerfile:1.7
# -----------------------------------------------------------------------------
# Unsloth + unsloth-zoo for every current NVIDIA arch (Turing -> Blackwell),
# on linux/amd64 and linux/arm64.
#
# Why it works:
#   * cu128 wheels ship native SASS (no PTX), verified via `cuobjdump --list-elf`:
#       amd64:  sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
#       arm64:  sm_80 sm_90 sm_90a sm_100 sm_100a sm_120 sm_120a
#   * SASS is forward-compatible within a major: sm_86->sm_89 (Ada),
#     sm_100->sm_103 (B300/GB300), sm_120->sm_121 (DGX Spark/GB10), so every
#     non-Jetson GPU on https://developer.nvidia.com/cuda/gpus runs precompiled
#     SASS (torch, llama.cpp, source-built ops).
#   * Triton kernels JIT per-device at first run; the bundled cu12.8 ptxas/NVRTC
#     cannot emit compute_103/compute_121, so the cu13 override below handles
#     amd64 sm_103 and arm64 sm_121 (SASS still runs there via forward-compat, so
#     only JIT-heavy paths need it).
#   * Rare source builds compile against
#     TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"; the host GPU is
#     irrelevant, nvcc emits whatever the arch list says.
#
# Cross-arch build (arm64 / sm_121): built via QEMU binfmt on an x86_64 host
#   (`docker run --privileged --rm tonistiigi/binfmt --install all` once, then
#   `docker buildx build --platform linux/arm64 ...`). QEMU is build-time only;
#   the image runs natively on aarch64. xformers has no cu128 aarch64 wheel, so
#   arm64 falls back to Unsloth's SDPA (~5-10% slower, functionally complete).
#
# Build host needs Docker buildkit + buildx, and QEMU binfmt for arm64-on-x86_64;
# nvidia-container-toolkit only for test-time `--gpus all`. No GPU at build time.
# -----------------------------------------------------------------------------

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04
ARG PYTHON_VERSION=3.12

# =============================================================================
# Stage 1: builder  -- toolkit + dev headers, builds any source extensions
# =============================================================================
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS builder

# TARGETARCH (buildx: amd64/arm64) selects the unsloth extras matching the
# wheels available for the platform (xformers aarch64 gap -- see header).
ARG TARGETARCH
ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Cross-compile for every current NVIDIA arch (developer.nvidia.com/cuda/gpus):
    #   sm_75 Turing (T4, RTX 20xx) | sm_80 A100/A30 | sm_86 A40/RTX 30xx
    #   sm_89 Ada (L4/L40/RTX 40xx) | sm_90 Hopper (H100/H200/GH200)
    #   sm_100 Blackwell DC (B100/B200/GB200) | sm_120 Blackwell (RTX 50xx, RTX PRO 6000)
    # sm_103 (B300/GB300) and sm_121 (GB10) are omitted: CUDA 12.8 nvcc cannot
    # compile compute_103/121 and sm_100/sm_120 SASS cover them via forward-compat.
    # +PTX on the highest lets future revisions JIT. Same list on both arches:
    # nvcc emits archs absent from the host, irrelevant extras only cost a little
    # compile time (no source builds on this pin set).
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
    MAX_JOBS=4 \
    CUDA_HOME=/usr/local/cuda \
    # Build-host-independence guards: the build must NEVER introspect a GPU (host
    # may be a B200, RTX 6000, or GPU-less CI) so all yield byte-identical images.
    # 1) no JIT-compiled sm_NNN blob into unsloth_compiled_cache/ at import.
    UNSLOTH_COMPILE_DISABLE=1 \
    UNSLOTH_COMPILE_OVERWRITE=0 \
    # 2) don't probe torch.cuda.is_available() at setup (would silently skip wheels).
    UNSLOTH_DISABLE_GPU_PROBE=1 \
    # 3) empty CUDA_VISIBLE_DEVICES so stray torch.cuda calls see no devices, not
    #    host-specific paths (re-enabled at runtime via `docker run --gpus all`).
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

# Isolated prefix. We do NOT touch the system Python (Ubuntu 24.04 marks it
# externally-managed, PEP 668); the venv bootstraps pip via ensurepip (from the
# python3.12-venv apt package) and gets uv a few lines below.
ENV VENV=/opt/unsloth-venv
RUN python -m venv ${VENV} && ${VENV}/bin/pip install -U pip wheel setuptools

# Unified install: torch + triton + bitsandbytes + unsloth + unsloth_zoo in a
# SINGLE uv pass. Mandatory -- splitting it lets bnb's transitive `cuda-toolkit`
# silently upgrade torch to 2.12.0+cu130 in a later pass, breaking the pinned
# cu128 xformers wheel (the cu cascade hits after xformers is on disk).
#
# Flags:
#   --index-strategy unsafe-best-match: the PyTorch index serves an old
#     requests==2.28.1 that conflicts with datasets>=2.32.2; both indexes
#     (pytorch cu128 + pypi) are equally trusted, so override uv's first-wins.
#   --extra-index-url .../cu128: torch +cu128 wheels + the xformers/cu128 URLs
#     from unsloth's cu128onlytorch2110 extra.
#
# Plain `huggingface` extra + explicit xformers pin (amd64): the cu128 extras on
# main stop at torch2100 (xformers 0.0.34 -> torch 2.10.0), conflicting with the
# torch 2.11.0 held below; a missing extra name would only WARN and drop xformers
# until the check below failed. Pinning xformers==0.0.35 (untied to torch) keeps
# this self-contained; arm64 stays xformers-less (no cu128 aarch64 wheel).
#
# No flash-attn: FA3 is hard-refused on Blackwell (Dao-AILab/flash-attention#1810);
# FA2 has no cu128+torch2.11+cp312 wheel (~30min fragile source build on CI) and
# Unsloth falls back to xformers/SDPA anyway. Users on Ampere/Ada/Hopper can
# `pip install flash-attn` on top at deploy time.
ARG UNSLOTH_REF=main
ARG UNSLOTH_ZOO_REF=main
RUN set -eux \
 && case "${TARGETARCH:-amd64}" in \
        amd64) UNSLOTH_EXTRA="huggingface"; XFORMERS_PIN="xformers==0.0.35" ;; \
        arm64) UNSLOTH_EXTRA="huggingface"; XFORMERS_PIN="" ;; \
        *)     echo "ERROR: unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac \
 && echo ">> TARGETARCH=${TARGETARCH:-amd64}, unsloth extra=[${UNSLOTH_EXTRA}], xformers=[${XFORMERS_PIN}]" \
 && ${VENV}/bin/pip install uv \
 && ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
        ${XFORMERS_PIN} \
        "triton>=3.6.0" \
        "bitsandbytes>=0.49.2,!=0.46.0,!=0.48.0" \
        "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@${UNSLOTH_ZOO_REF}" \
        "unsloth[${UNSLOTH_EXTRA}] @ git+https://github.com/unslothai/unsloth@${UNSLOTH_REF}" \
        "timm>=1.0.11" "addict"

# vLLM: required by Unsloth's GRPO path (fast_inference=True). Installed as a
# SECOND uv pass so the unified pass settles on torch 2.11.0 first, then vLLM
# bolts on top: with torch held, uv picks the newest compatible vLLM (0.20+ pins
# torch 2.11.0) and tracks our pin. PyPI ships x86_64 + aarch64 wheels since 0.17,
# so this runs on arm64 too; amd64 failures abort, arm64 is fail-soft (aarch64
# kernels are validated on Spark hardware, not CI).
# https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
# https://wheels.vllm.ai/nightly
ARG INSTALL_VLLM=auto
RUN set -eux \
 && WANT_VLLM=0 \
 && case "${INSTALL_VLLM}" in \
        auto|1|true|yes) WANT_VLLM=1 ;; \
        0|false|no) WANT_VLLM=0 ;; \
        *) echo "ERROR: invalid INSTALL_VLLM=${INSTALL_VLLM}" >&2; exit 1 ;; \
    esac \
 && if [ "${WANT_VLLM}" = "1" ]; then \
        echo ">> installing vLLM (TARGETARCH=${TARGETARCH:-amd64})"; \
        # Explicit && chain, not `set -e` -- POSIX shells disable errexit inside a
        # condition context (verified on dash), masking install failures.
        # Step 1: uv resolves vLLM's deps with torch==2.11.0 held, landing on the
        # newest matching vLLM (fails loudly if none); unsafe-best-match picks the
        # best wheel across the three indexes.
        # Step 2: vLLM pulls numpy down to 2.2.6 whose wheel ships a broken
        # numpy.testing (tests/ stripped), crashing `from numpy import *` and thus
        # `import unsloth`; upgrade numpy back to a self-consistent release.
        # Step 3: vLLM pins numba 0.61.2 which hard-refuses numpy>=2.3, but the
        # stack needs numpy>=2.3; lift numba to one supporting numpy 2.4 (0.65
        # imports cleanly, vllm still imports).
        { ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --pre \
            --index-strategy unsafe-best-match \
            --extra-index-url https://wheels.vllm.ai/nightly \
            --extra-index-url https://download.pytorch.org/whl/cu128 \
            "torch==2.11.0" \
            vllm \
       && ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --upgrade "numpy>=2.4" \
       && ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --upgrade "numba>=0.62" \
       && ${VENV}/bin/python -c "import vllm; print('vllm', vllm.__version__)" \
       && ${VENV}/bin/python -c "import numpy.testing, numpy; print('numpy', numpy.__version__, 'testing ok')" \
       && ${VENV}/bin/python -c "import numba; print('numba', numba.__version__, 'imports ok')" \
        # flashinfer-jit-cache: precompiled cubins so flashinfer ops don't hit the
        # JIT path (standalone `vllm serve` dies there for fmha_gen on sm_100a;
        # in-process GRPO survives since zoo blocks the FlashInfer JIT). Removes
        # the runtime-compile failure class for ~1.5 GB.
       && { ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --index-url https://flashinfer.ai/whl/cu128 \
            "flashinfer-jit-cache==0.6.6" \
         || echo ">> flashinfer-jit-cache unavailable for ${TARGETARCH:-amd64}; vllm serve may require nvcc for uncached ops"; } \
       && echo ">> vLLM installed (numpy + numba re-upgraded post-vllm)"; \
        } || { \
            if [ "${TARGETARCH:-amd64}" != "amd64" ]; then \
                echo ">> vLLM skipped on ${TARGETARCH}: install or import check failed (fail-soft on non-amd64)"; \
                # A partial install must not poison the base stack: drop vllm and
                # restore the numpy/numba floor it may have moved (numpy 2.2.6's
                # broken numpy.testing breaks `import unsloth`). arm64 staging CI
                # re-verifies `import unsloth` after this.
                ${VENV}/bin/uv pip uninstall --python ${VENV}/bin/python vllm || true; \
                ${VENV}/bin/uv pip install --python ${VENV}/bin/python \
                    --upgrade "numpy>=2.4" "numba>=0.62"; \
                ${VENV}/bin/python -c "import numpy.testing, numba; print('numpy/numba restored')"; \
            else \
                echo "ERROR: vLLM install failed on amd64" >&2; exit 1; \
            fi; \
        }; \
    else \
        echo ">> vLLM skipped (INSTALL_VLLM=${INSTALL_VLLM}, TARGETARCH=${TARGETARCH:-amd64})"; \
    fi

# JupyterLab so the image runs unslothai/notebooks out of the box:
#   docker run --gpus all -p 8888:8888 unsloth/unsloth \
#       jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
# Separate pass AFTER the torch-pinned resolves: pure-Python, never names torch,
# so uv can't disturb the cu128 pin set. These are declared by notebook install
# cells (neutralised by the in-image runner), so bake them here:
#   matplotlib   plotting; some trust_remote_code files import it (DeepSeek-OCR)
#   soundfile    TTS audio read/write (bundles libsndfile)
#   evaluate+jiwer  Whisper WER metric
#   tensorboard  default TrainingArguments report_to backend
#   langid       DeepSeek-R1 GRPO reward language-id check
#   easydict     some vision trust_remote_code modeling files
#   protobuf     slow->fast tokenizer conversion for sentencepiece
#   omegaconf    TTS + NeMo-Gym RL notebook configs
#   einx         TTS codec tensor-rearrange (Llasa/Oute/Spark)
#   librosa      Whisper audio features (pulls numba, already pinned >=0.65)
#   ftfy         Oute TTS text normalisation
# decord is separate below (no aarch64 wheel). Pinned (==) for reproducible
# rebuilds; bump deliberately. The resolve must NOT move torch/numpy/numba --
# the assertion below fails the build if it did. Transitive deps land in
# /opt/unsloth-venv/requirements.lock.txt.
RUN ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        "jupyterlab==4.6.0" "notebook==7.6.0" "ipywidgets==8.1.8" "matplotlib==3.11.0" \
        "soundfile==0.14.0" "evaluate==0.4.6" "jiwer==4.0.0" "tensorboard==2.20.0" \
        "langid==1.1.6" "easydict==1.13" "protobuf==6.33.6" \
        "omegaconf==2.3.1" "einx==0.4.3" "librosa==0.11.0" "ftfy==6.3.1" \
 && ${VENV}/bin/python -c "import torch, numpy, numba; from packaging.version import Version; assert torch.__version__.startswith('2.11.0'), torch.__version__; assert Version(numpy.__version__) >= Version('2.3'), numpy.__version__; assert Version(numba.__version__) >= Version('0.65'), numba.__version__; print('notebook-deps pins OK:', torch.__version__, numpy.__version__, numba.__version__)"

# decord (ERNIE-VL video decode) has wheels only for x86_64. Installed alone:
# HARD on amd64 (a missing wheel is a real regression), fail-soft elsewhere.
RUN if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        ${VENV}/bin/uv pip install --python ${VENV}/bin/python "decord==0.6.0"; \
    else \
        ${VENV}/bin/uv pip install --python ${VENV}/bin/python "decord==0.6.0" \
        || echo ">> decord skipped (no matching wheel for ${TARGETARCH:-}); ERNIE-VL video decode unavailable"; \
    fi

# Audio decode out of the box: TTS/STT notebooks decode datasets' Audio features
# through torchcodec. Three traps: (1) torchcodec 0.11 must pair with torch 2.11;
# (2) the wheel must come from the cu128 channel, not the PyPI default (cu13,
# dlopens libnvrtc.so.13); (3) its libs dlopen venv torch/NVIDIA libs that ld.so
# can't see, registered via ld.so.conf.d in the runtime stage (this layer only
# installs the wheels; the import check needs ffmpeg from the runtime stage).
# Fail-soft on arches without a matching wheel.
RUN set -eux \
 && { ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torchcodec==0.11.0" \
   && ${VENV}/bin/uv pip install --python ${VENV}/bin/python nvidia-npp-cu12; } \
 || echo ">> torchcodec bake skipped (no matching wheel for ${TARGETARCH:-amd64})"

# transformers SIDECARS for per-notebook version activation (see
# unsloth_nb_compat.py). unslothai/notebooks pin many transformers versions; the
# base venv ships the newest 5.x. Each sidecar is transformers==X + matched
# huggingface_hub/tokenizers/safetensors, installed --no-deps into its own
# --target under ${VENV}/tf-sidecars. Activating one (prepend to sys.path before
# any ML import) swaps transformers WITHOUT touching the cu128 torch/vLLM/unsloth
# base -- verified: base unsloth loads + generates under 4.57.6 and 5.5.0 on B200.
# Versions mirror Studio's tiers (4.57.6 + 5.3.0/5.5.0/5.10.2); companions are
# resolved at build time. ~300MB after the strip below. Fail-soft per arch/wheel.
RUN set -eux \
 && for TFV in 4.57.6 5.3.0 5.5.0 5.10.2; do \
        SCRATCH="$(mktemp -d)"; \
        if ! ${VENV}/bin/uv pip install --python ${VENV}/bin/python \
                --target "$SCRATCH" "transformers==${TFV}" >/dev/null 2>&1; then \
            echo ">> sidecar resolve failed for ${TFV}; skipping"; rm -rf "$SCRATCH"; continue; \
        fi; \
        pin() { ls -d "$SCRATCH/$1"-*.dist-info 2>/dev/null \
            | sed -E "s@.*/$1-([0-9][0-9A-Za-z.]*)\.dist-info@\1@" | head -1; }; \
        HFV="$(pin huggingface_hub)"; TKV="$(pin tokenizers)"; SFV="$(pin safetensors)"; \
        rm -rf "$SCRATCH"; \
        DEST="${VENV}/tf-sidecars/t_$(echo "${TFV}" | tr . _)"; \
        ${VENV}/bin/uv pip install --python ${VENV}/bin/python --target "$DEST" --no-deps \
            "transformers==${TFV}" \
            ${HFV:+"huggingface_hub==${HFV}"} \
            ${TKV:+"tokenizers==${TKV}"} \
            ${SFV:+"safetensors==${SFV}"}; \
        echo ">> sidecar transformers==${TFV} (hf_hub=${HFV} tokenizers=${TKV} safetensors=${SFV})"; \
    done \
 && { du -sh ${VENV}/tf-sidecars || true; }

# 5) Informational pin record (NOT byte-reproducible: pip freeze omits wheel
#    hashes and unsloth/unsloth_zoo/vllm --pre float from VCS/nightly). Read via
#    `docker run --rm <image> cat /opt/unsloth-venv/requirements.lock.txt`.
RUN ${VENV}/bin/pip freeze --exclude-editable > ${VENV}/requirements.lock.txt \
 && head -50 ${VENV}/requirements.lock.txt

# 6) Strip pip cache & __pycache__ to shrink the layer copied to runtime.
# The `-name tests` strip excludes numpy's tests dirs: numpy 2.4 ships
# numpy/_core/tests/ back, and removing it re-triggers the
# `from numpy._core.tests._natype import pd_NA` ImportError the earlier upgrade
# fixed. Other verified-safe size cuts:
#   * npp: torchcodec dlopens only libnppicc + libnppc; drop the other ~10 libs (~388MB).
#   * static .a archives (~143MB): link-time only, nothing links venv archives.
#   * nvshmem device .bc (~30MB): device-relink only; host .so kept.
# Do NOT strip headers (torch/include): causal-conv1d / mamba-ssm build against
# them at notebook time with --no-build-isolation.
RUN set -eux \
 && find ${VENV} -depth -type d -name __pycache__ -exec rm -rf {} + \
 && find ${VENV} -depth -type d -name tests \
        ! -path "*numpy/_core/tests*" \
        ! -path "*numpy/tests*" \
        ! -path "*numpy/ma/tests*" \
        -exec rm -rf {} + \
 && rm -rf /root/.cache/pip /root/.cache/uv \
 && SP=${VENV}/lib/python${PYTHON_VERSION}/site-packages \
 && if [ -d "$SP/nvidia/npp/lib" ]; then \
        find "$SP/nvidia/npp/lib" -maxdepth 1 -name 'libnpp*.so.*' \
            ! -name 'libnppicc.so.*' ! -name 'libnppc.so.*' -delete; \
    fi \
 && find ${VENV} -name '*.a' -delete \
 && rm -f "$SP"/nvidia/nvshmem/lib/libnvshmem_device.bc \
 && echo "venv size after prune:" && du -sh ${VENV}

# Build-time verification.
# (1) arch-list check uses the RAW C++ accessor: torch.cuda.get_arch_list()
#     returns [] with no GPU visible (CUDA_VISIBLE_DEVICES is empty here).
# (2) required packages verified via metadata only -- we do NOT import unsloth/
#     unsloth_zoo (their __init__ needs a real CUDA device). Import correctness is
#     exercised at deploy time by smoke_test.py with --gpus all.
RUN TARGETARCH="${TARGETARCH:-amd64}" ${VENV}/bin/python - <<'PY'
import os, platform
target = os.environ.get("TARGETARCH", "amd64")
mach   = platform.machine()
print(f"build target: TARGETARCH={target} platform.machine()={mach}")

import torch
arches = torch._C._cuda_getArchFlags().split()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("arches:", arches)
assert torch.__version__.startswith("2.11.0"), f"torch silently moved: {torch.__version__}"
assert "+cu128" in torch.__version__, f"cu build silently changed: {torch.__version__}"
assert "sm_100" in arches, f"sm_100 (B200/GB200) missing: {arches}"
# cu128 wheels ship sm_120 native SASS on both amd64 and aarch64. On arm64 DGX
# Spark (sm_121) runs it via forward-compat; sm_121 is never in a cu128 wheel.
assert "sm_120" in arches, f"sm_120 missing: {arches}"
print(f"OK: torch 2.11.0+cu128 with sm_100 + sm_120 native SASS intact ({target})")

from importlib.metadata import version, PackageNotFoundError
# xformers is amd64-only (aarch64 wheel gap -- see header).
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
# Stage 2: runtime  -- slim runtime image, no nvcc, no cuDNN/cuBLAS layers
# =============================================================================
# The "-base-" variant (vs "-cudnn-runtime-") drops ~2.7 GB of system CUDA libs
# we never load: torch wheels bake their own cuDNN/cuBLAS/etc into torch/lib/ and
# libtorch_cuda.so's RPATH resolves through the wheel, not the system (verified
# via `readelf -d torch/lib/libtorch_cuda.so`). The base still provides
# nvidia-smi + libcuda stubs + libnvidia-ml, all the entrypoint + torch.cuda need.
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

# The base manifest is multi-arch; buildx picks the right one for
# TARGETPLATFORM at this FROM line, no conditional needed.
ARG TARGETARCH
ARG PYTHON_VERSION
ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/unsloth-venv/bin:${PATH} \
    HF_HOME=/workspace/.cache/huggingface \
    TRITON_CACHE_DIR=/workspace/.cache/triton \
    # Keep the arch list at runtime so an in-container source build gets the same
    # SASS coverage as the builder; 10.3 omitted (cu12.8 can't emit compute_103;
    # sm_103 runs sm_100 SASS via forward-compat).
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"

# System packages needed by the notebooks:
#   zstd    Ollama installer (`curl ollama.com/install.sh | sh`) extracts a zstd tarball
#   ffmpeg  torchcodec dlopens system FFmpeg libs (not bundled in the wheel)
#   wget    notebooks fetch assets with `!wget URL`
#   ninja-build  flashinfer cpp_ext JIT shells out to ninja; subprocesses may not
#                inherit the venv pip ninja on PATH
#   cuda-nvcc + cudart-dev  flash-linear-attention's TileLang backend (Qwen3.5 in
#                Studio) JIT-compiles CUDA kernels via /usr/local/cuda/bin/nvcc,
#                absent from the -base image
RUN CUDA_PKG="$(echo "${CUDA_VERSION}" | awk -F. '{print $1"-"$2}')" \
    && apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git libgomp1 \
        gcc g++ zstd ffmpeg ninja-build \
        "cuda-nvcc-${CUDA_PKG}" "cuda-cudart-dev-${CUDA_PKG}" \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/local/bin/python3 \
    && test -x /usr/local/cuda/bin/nvcc \
    && rm -rf /var/lib/apt/lists/*
# gcc + g++ + python3.12-dev in runtime: Triton's nvidia backend lazily compiles
# a C extension (CudaUtils) on first GPU access; without a C compiler + Python
# headers the first forward pass dies with "Failed to find C compiler". ~250MB,
# the cost of correct JIT (pre-compiling would need a GPU).

COPY --from=builder /opt/unsloth-venv /opt/unsloth-venv

# Blackwell JIT fix for sm_103 (amd64) and sm_121 (arm64) -- the cu12.8 JIT gap
# from the header. Two JIT paths need the cu13 override:
#   (1) torch's bundled libnvrtc.so.12 (CUDA 12.8): any NVRTC JIT path (e.g.
#       torch.fft.rfft(complex).abs() in mel-spectrogram code) errors on
#       sm_103/sm_121. Fix: stage a cu13 NVRTC alias beside the cu12.8 default.
#   (2) Triton's bundled ptxas (still 12.8 in triton 3.6.0) tops out at sm_120,
#       rejects sm_103, silently downgrades sm_121 to sm_80 (triton-lang/triton#8335).
#       Fix: install cu13 ptxas, point Triton at it via TRITON_PTXAS_PATH.
# Both cu13 tools are CPU-side compilers (no install-time driver bump), but their
# output cubin needs a >=580 driver to LOAD, so neither is a global default (would
# break Ampere/Ada/Hopper/Turing on 570-579 drivers). select_cuda_jit_tools in
# entrypoint.sh activates them per device, only for sm_103/sm_121 (>=580 drivers,
# always safe). Both arches carry the ~400 MB.
RUN set -eux; \
        # The nvidia/cuda base already configures the CUDA apt repo with its own
        # Signed-By keyring; adding cuda-keyring_1.1-1_all.deb would add a second
        # sources file with a different Signed-By and make `apt-get update` refuse
        # the repo ("Conflicting values set for option Signed-By"). The base repo
        # is monolithic and serves 13.x too, so install cu13 packages directly
        # without touching the keyring. Verified on x86_64 and arm runners.
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-nvrtc-13-0 \
            cuda-nvcc-13-0; \
        # cu13's config-common postinst flips the /usr/local/cuda alternative to
        # cuda-13.0 (priority 130 beats 12.8's 128). Pin it back: TileLang JIT
        # and torch.utils.cpp_extension resolve /usr/local/cuda/bin/nvcc, and
        # cu13-emitted cubins need driver >= 580 while this image supports 570+.
        # The cu13 tools stay reachable by absolute path (how the entrypoint
        # activates them on sm_103/sm_121), and --set switches the alternative
        # to manual mode so later apt operations cannot flip it again.
        update-alternatives --set cuda /usr/local/cuda-12.8; \
        rm -rf /var/lib/apt/lists/*; \
        # (1) NVRTC staging: keep the wheel's cu12.8 lib as .cu128.orig, point
        #     libnvrtc.so.12 at it (relative symlink), stage .cu13 -> the cu13 lib;
        #     select_cuda_jit_tools retargets the symlink only on sm_103/sm_121.
        #     The default needs no runtime write, so a non-root --user container
        #     keeps cu12.8, loadable on every supported 570+ driver.
        NVRTC_DIR=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvrtc/lib; \
        if [ -f "${NVRTC_DIR}/libnvrtc.so.12" ] && [ ! -L "${NVRTC_DIR}/libnvrtc.so.12" ]; then \
            mv "${NVRTC_DIR}/libnvrtc.so.12" "${NVRTC_DIR}/libnvrtc.so.12.cu128.orig"; \
            ln -s libnvrtc.so.12.cu128.orig "${NVRTC_DIR}/libnvrtc.so.12"; \
            ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13 "${NVRTC_DIR}/libnvrtc.so.12.cu13"; \
        fi
# (2) ptxas: the cu13 nvcc package above provides it; TRITON_PTXAS_PATH is set
# per device at boot (select_cuda_jit_tools) for the same driver-floor reason.

# Register the venv's torch + NVIDIA lib dirs with the loader so torchcodec can
# dlopen them. ld.so.conf.d, NOT LD_LIBRARY_PATH: the cache is consulted after
# DT_RUNPATH, so llama.cpp keeps resolving its own $ORIGIN libs first. Import
# check runs here (ffmpeg lives in this stage); fail-soft if the wheel is absent.
RUN set -eux \
 && SP=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages \
 && printf "%s\n" "$SP/torch/lib" "$SP/nvidia/cuda_nvrtc/lib" \
                  "$SP/nvidia/cuda_runtime/lib" "$SP/nvidia/npp/lib" \
        > /etc/ld.so.conf.d/zz-unsloth-venv.conf \
 && ldconfig \
 && { /opt/unsloth-venv/bin/python -c \
        "import torchcodec; print('torchcodec', torchcodec.__version__)" \
   || echo ">> torchcodec unavailable on this arch (audio decode falls back)"; }

# Prebuilt llama.cpp so GGUF export works out of the box. unsloth_zoo's
# save_pretrained_gguf() -> check_llama_cpp() looks for llama-quantize +
# convert_hf_to_gguf.py + gguf-py/ in $UNSLOTH_LLAMA_CPP_PATH; without a baked
# install the first export hits install_llama_cpp()'s prompt + slow source build.
#
# NOT studio/install_llama_prebuilt.py: it selects a bundle for the CURRENT host
# (nvidia-smi etc., which leak through a GPU build host). The build must never
# introspect the host, so pin release + asset by build target (see
# fetch_llama_prebuilt.py):
#   * amd64 -> app-<tag>-linux-x64-cuda12-portable.tar.gz   (sm_70..sm_120)
#     arm64 -> app-<tag>-linux-arm64-cuda13-portable.tar.gz (sm_90..sm_121)
#   * portable bundles carry their own CUDA libs, so they also run CPU-only
#   * sha256-verified against the release's llama-prebuilt-sha256.json
#   * converter + gguf-py from the SAME release's source tarball (mappings match)
# /opt (not /root) so it survives `docker run --user`; UNSLOTH_LLAMA_CPP_PATH lets
# zoo find it regardless of $HOME. Default "latest" resolves the newest
# unslothai/llama.cpp release at build time; build.sh pins it to a concrete tag so
# the cache busts only on new upstream releases. --build-arg LLAMA_PREBUILT_TAG=<tag>
# for a frozen build.
ARG LLAMA_PREBUILT_TAG=latest
COPY fetch_llama_prebuilt.py /tmp/fetch_llama_prebuilt.py
RUN /opt/unsloth-venv/bin/python /tmp/fetch_llama_prebuilt.py \
        "${LLAMA_PREBUILT_TAG}" "${TARGETARCH:-amd64}" /opt/unsloth/llama.cpp \
 && rm -f /tmp/fetch_llama_prebuilt.py \
 && cat /opt/unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json
ENV UNSLOTH_LLAMA_CPP_PATH=/opt/unsloth/llama.cpp

WORKDIR /workspace
# World-writable so `docker run --user <uid>` (documented non-root use) can
# create notebooks and populate the default caches without a bind mount.
RUN mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR} \
 && chmod -R a+rwX /workspace

# ---------------------------------------------------------------------------
# Per-notebook transformers version activation -- run unslothai/notebooks
# UNCHANGED (see unsloth_nb_compat.py). Pieces:
#   * unsloth_nb_compat.py -> site-packages: tier detection + sidecar resolution
#     + activation + IPython hook.
#   * pip/uv shim on a PATH dir AHEAD of the venv bin: makes `!pip install` /
#     `!uv pip install` cells safe + idempotent (keeps the baked stack, records
#     the requested transformers to activate its sidecar).
#   * unsloth_nb_pip_magic.py -> site-packages: re-points `%pip`/`%uv` magics and
#     `!python -m pip` at the same shim so in-process installs can't bypass PATH.
#   * IPython startup hook: activates the right sidecar before the first model cell.
#   * unsloth-run: headless `unsloth-run <notebook|url>`, the robust driven path.
# ---------------------------------------------------------------------------
COPY unsloth_nb_compat.py unsloth_pip_shim.py unsloth_nb_pip_magic.py unsloth_ipython_startup.py unsloth_run.py unsloth_sync_notebooks.sh unsloth_nb_content_sig.py unsloth_nb_view.py unsloth_nb_strip_colab.py unsloth_colab_compat.py /opt/unsloth-nb/
RUN set -eux \
 && SP=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages \
 && cp /opt/unsloth-nb/unsloth_nb_compat.py "$SP/unsloth_nb_compat.py" \
 && cp /opt/unsloth-nb/unsloth_nb_pip_magic.py "$SP/unsloth_nb_pip_magic.py" \
 && cp /opt/unsloth-nb/unsloth_colab_compat.py "$SP/unsloth_colab_compat.py" \
 && chmod +x /opt/unsloth-nb/unsloth_pip_shim.py /opt/unsloth-nb/unsloth_run.py /opt/unsloth-nb/unsloth_sync_notebooks.sh /opt/unsloth-nb/unsloth_nb_content_sig.py /opt/unsloth-nb/unsloth_nb_view.py /opt/unsloth-nb/unsloth_nb_strip_colab.py \
 && mkdir -p /opt/unsloth-nb/bin \
 && for t in pip pip3 uv; do ln -sf /opt/unsloth-nb/unsloth_pip_shim.py /opt/unsloth-nb/bin/$t; done \
 && ln -sf /opt/unsloth-nb/unsloth_run.py /usr/local/bin/unsloth-run \
 && ln -sf /opt/unsloth-nb/unsloth_sync_notebooks.sh /usr/local/bin/unsloth-sync-notebooks \
 && ln -sf /opt/unsloth-nb/unsloth_nb_content_sig.py /usr/local/bin/unsloth-nb-content-sig \
 && ln -sf /opt/unsloth-nb/unsloth_nb_view.py /usr/local/bin/unsloth-nb-view \
 && ln -sf /opt/unsloth-nb/unsloth_nb_strip_colab.py /usr/local/bin/unsloth-nb-strip-colab \
 && mkdir -p /opt/unsloth-nb/ipython/profile_default/startup \
 && cp /opt/unsloth-nb/unsloth_ipython_startup.py /opt/unsloth-nb/ipython/profile_default/startup/00-unsloth-nb.py \
 && chmod -R a+rX /opt/unsloth-nb/ipython \
 && /opt/unsloth-venv/bin/python -c "import sys, glob; sys.path.insert(0, '$SP'); import unsloth_nb_compat, unsloth_colab_compat; print('nb-compat OK; baked sidecars:', sorted(glob.glob('/opt/unsloth-venv/tf-sidecars/t_*')))" \
 && /opt/unsloth-venv/bin/python /opt/unsloth-nb/unsloth_pip_shim.py --unsloth-selfcheck-value-flags
# Shim dir AHEAD of the venv bin so `!pip`/`!uv` resolve to the shim, not the real tool.
ENV PATH=/opt/unsloth-nb/bin:${PATH}
# Load the notebook startup hook (sidecar activation + %pip/%uv magic re-point)
# for EVERY kernel, any uid: IPYTHONDIR (via ENV) points IPython at this shared
# profile, so it loads under `--user <uid>` too -- unlike /root/.ipython, which
# only a root kernel reads. Writable state (history.sqlite) still lands per-user.
ENV IPYTHONDIR=/opt/unsloth-nb/ipython

# Pre-clone unslothai/notebooks so JupyterLab opens with them present. Baked as a
# READ-ONLY template (~206MB, .git stripped); on boot the entrypoint copies it to
# /workspace/unsloth-notebooks and best-effort refreshes from GitHub, never
# overwriting a user-touched notebook and skipping rewrites when only the install
# header/announcements/footer moved (see unsloth_sync_notebooks.sh +
# unsloth_nb_content_sig.py). Inherited as-is by the studio image.
#
# UNSLOTH_NOTEBOOKS_REF pins ONE commit/branch/tag so a multi-arch publish bakes
# identical templates into both legs. The publish workflow resolves the live HEAD
# sha once (like LLAMA_PREBUILT_TAG); default "main" tracks the tip. Fetched at
# depth 1 (sha by object, branch/tag by name).
ARG UNSLOTH_NOTEBOOKS_REF=main
RUN set -eux \
 && git init -q /opt/unsloth-notebooks \
 && git -C /opt/unsloth-notebooks remote add origin https://github.com/unslothai/notebooks \
 && git -C /opt/unsloth-notebooks fetch -q --depth 1 origin "${UNSLOTH_NOTEBOOKS_REF}" \
 && git -C /opt/unsloth-notebooks checkout -q FETCH_HEAD \
 && git -C /opt/unsloth-notebooks rev-parse HEAD > /opt/unsloth-notebooks/.unsloth_template_commit \
 && rm -rf /opt/unsloth-notebooks/.git \
 && du -sh /opt/unsloth-notebooks

# JupyterLab lives in the venv. notebooks are pre-populated into
# /workspace/unsloth-notebooks on boot; mount a volume on /workspace to persist.
EXPOSE 8888

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
