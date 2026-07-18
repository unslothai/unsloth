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
#     https://developer.nvidia.com/cuda/gpus runs the PRECOMPILED SASS (torch,
#     llama.cpp, source-built ops per the arch list below).
#   * Unsloth's runtime kernels are Triton, which JIT-compiles per device at
#     first run. JIT targets the ACTUAL device cap, and the bundled cu12.8
#     ptxas/NVRTC cannot emit compute_103 (sm_103) or compute_121 (sm_121).
#     Both are handled by the cu13 NVRTC/ptxas override below: amd64 sm_103
#     (B300/GB300) and arm64 sm_121 (DGX Spark / GB10). Precompiled SASS also
#     runs on sm_103 via sm_100 forward-compat and on sm_121 via sm_120
#     forward-compat, so only JIT-heavy paths depend on the override.
#   * Anything that DOES need to be source-built (rare on this pin set) compiles
#     against TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX",
#     covering every current NVIDIA compute capability per
#     https://developer.nvidia.com/cuda/gpus.
#     The host GPU is irrelevant for compilation; nvcc emits whatever the arch
#     list says.
#
# Cross-arch build (DGX Spark / GB10 / sm_121):
#   The arm64 image is built via QEMU binfmt emulation on an x86_64 host:
#       docker run --privileged --rm tonistiigi/binfmt --install all   # one-time host setup
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
#   * For arm64 builds on x86_64 hosts: QEMU binfmt (the one-time setup above)
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
# the target platform (the xformers aarch64 gap -- see header).
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
    #   sm_103  Blackwell DC B300, GB300 -- covered by sm_100 SASS (see above),
    #           NOT a separate target here: the bundled CUDA 12.8 nvcc cannot
    #           compile compute_103 (added in CUDA 12.9), so listing 10.3 would
    #           break any source / JIT build that honors TORCH_CUDA_ARCH_LIST.
    #   sm_120  Blackwell    RTX 50-series, RTX PRO 6000 Blackwell    (amd64 only)
    #   sm_121  Blackwell    GB10 (DGX Spark)                          (arm64 only)
    # +PTX on the highest lets future arch revisions run via JIT-PTX.
    # We keep the same list on both arches: nvcc happily emits archs that
    # don't exist on the build host, and any extras that aren't relevant for
    # the target just bloat compile time slightly (not size, since we don't
    # source-build any extension at install time on the pin set).
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX" \
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
#     by unsloth's `cu128onlytorch2110` extra.
#
# Why the plain `huggingface` extra plus an EXPLICIT xformers pin (amd64):
#   The cu128 CUDA extras on main stop at the torch2100 family, which pins
#   xformers 0.0.34 -> torch==2.10.0 and would conflict with the torch 2.11.0
#   line held below; a not-yet-existing extra name would only WARN (pip/uv
#   install plain unsloth) and silently drop xformers until the required-
#   package check below failed the build. Pinning `xformers==0.0.35` directly
#   (it does not pin torch, pairing with 2.11.0) keeps this build
#   self-contained on today's main; arm64 stays xformers-less (no cu128
#   aarch64 wheel -- see header).
#
# Why no `flash-attn` here:
#   - FA3 is hard-refused on Blackwell (Dao-AILab/flash-attention#1810).
#   - FA2 has no prebuilt wheel for cu128+torch2.11+cp312 -> would require
#     a ~30min source build, fragile on the 16GB ubuntu-latest CI runner.
#   - Unsloth gracefully falls back to xformers/SDPA on Blackwell anyway.
#   - Users on Ampere/Ada/Hopper who want FA2 can `pip install flash-attn`
#     on top of this image at deploy time.
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

# vLLM. Required by Unsloth's GRPO path when the notebook sets
# fast_inference=True. We install it as a SECOND uv pass rather than
# appending to the unified one because:
#   * vLLM releases pin an exact torch (0.19.1 -> torch==2.10.0; 0.20+
#     moved to torch==2.11.0). Adding it to the unified resolve forces uv
#     to consider whether to swap our pinned torch 2.11.0 for vLLM's
#     choice. Splitting the install lets the unified pass settle on torch
#     2.11.0 first, then vLLM bolts on top: with torch held at 2.11.0 the
#     resolver lands on the newest compatible vLLM (0.20+, which pins torch
#     2.11.0) by itself, and tracks our torch pin when it moves.
#   * PyPI ships both x86_64 AND aarch64 abi3 wheels for every release
#     since 0.17, so the SAME pass now runs on the arm64 leg (DGX Spark /
#     GB10 class). amd64 failures abort the build; arm64 is fail-soft
#     because aarch64 wheels are newer and the GPU-side kernels there are
#     validated manually on Spark hardware, not in CI.
#
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
        # NOTE: an explicit && chain, not `set -e` -- POSIX shells disable
        # errexit inside any condition context (verified on dash), so a
        # (set -e; ...) subshell here would mask install failures.
        # Step 1: let uv resolve vLLM's transitive deps. We pin
        # torch==2.11.0 so uv MUST hold our torch fixed; with torch held,
        # the resolver lands on the newest vLLM whose pin matches and the
        # build fails loudly if none does. `unsafe-best-match` lets uv
        # pull from whichever of the three indexes has a better wheel.
        # Step 2: vLLM pulls numpy down to 2.2.6 whose wheel ships a
        # broken numpy.testing (`from numpy._core.tests._natype import
        # pd_NA` -- tests/ is stripped from the wheel). Any path that hits
        # `from numpy import *` (e.g. scipy.optimize) then crashes, taking
        # `import unsloth` with it. Upgrade numpy back to a release with a
        # self-consistent testing module.
        # Step 3: vLLM pins numba==0.61.2, which hard-refuses numpy >= 2.3
        # at import time -- and the stack needs numpy >= 2.3, so the numpy
        # ceiling cannot move down. Lift numba to a release that supports
        # numpy 2.4 (verified: numba 0.65 imports cleanly, vllm still
        # imports). Same intentional-override class as the numpy bump.
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
        # flashinfer-jit-cache: the runtime image ships no nvcc, so any
        # flashinfer op missing from the cubin package would hit the JIT
        # path and die (standalone `vllm serve` does exactly this for
        # fmha_gen on sm_100a; in-process GRPO survives because zoo blocks
        # the FlashInfer JIT). The precompiled cache removes the entire
        # runtime-compile failure class for ~1.5 GB.
       && { ${VENV}/bin/uv pip install \
            --python ${VENV}/bin/python \
            --index-url https://flashinfer.ai/whl/cu128 \
            "flashinfer-jit-cache==0.6.6" \
         || echo ">> flashinfer-jit-cache unavailable for ${TARGETARCH:-amd64}; vllm serve may require nvcc for uncached ops"; } \
       && echo ">> vLLM installed (numpy + numba re-upgraded post-vllm)"; \
        } || { \
            if [ "${TARGETARCH:-amd64}" != "amd64" ]; then \
                echo ">> vLLM skipped on ${TARGETARCH}: install or import check failed (fail-soft on non-amd64)"; \
                # A partial install must not poison the base stack: drop
                # vllm and restore the numpy/numba floor it may have moved
                # (numpy 2.2.6's broken numpy.testing breaks `import
                # unsloth` outright). The arm64 staging CI contract probe
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

# JupyterLab so the published image runs unslothai/notebooks out of the box:
#   docker run --gpus all -p 8888:8888 unsloth/unsloth \
#       jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
# Separate pass AFTER the torch-pinned resolves: this closure is pure-Python
# and never names torch, so uv cannot disturb the cu128 pin set (naming torch
# without the cu128 index could swap in the PyPI CPU wheel).
# matplotlib rides along for the notebook crowd: plotting is table stakes in
# a Jupyter image, and several model repos' trust_remote_code modeling files
# (e.g. DeepSeek-OCR) import it unconditionally.
# These are declared by notebook install cells that the in-image runner
# neutralises (deps are meant to be prebaked), so bake them here. All
# pure-Python or self-contained wheels; none names torch, so the cu128 pin
# is undisturbed:
#   soundfile  TTS notebooks read/write audio (bundles libsndfile in its wheel)
#   evaluate + jiwer  Whisper notebook's WER metric (evaluate.load("wer") -> jiwer)
#   tensorboard  default TrainingArguments report_to backend
#   langid  DeepSeek-R1 GRPO reward's language-id check
#   easydict  some vision trust_remote_code modeling files
#   protobuf  slow->fast tokenizer conversion for sentencepiece models
#   omegaconf  TTS families + both NeMo-Gym RL notebooks' config objects
#   einx  TTS codec tensor-rearrange (Llasa / Oute / Spark TTS)
#   librosa  Whisper audio feature extraction (pairs with soundfile + torchcodec)
#   ftfy  Oute TTS text normalisation
# decord is installed separately below (no aarch64 wheel; see that block).
# librosa pulls numba/soxr/audioread; numba is already pinned >=0.65 (numpy 2.4
# compatible) by the vLLM pass, so the resolve must NOT move torch/numpy/numba --
# the assertion below fails the build loudly if it did.
# Pinned (==) to the resolved, tested versions for reproducible rebuilds -- the
# same convention as the cu128 core (torch/torchvision/torchaudio). Bump these
# deliberately, not silently on the next build. Transitive deps of these are
# captured by the in-image pin record (/opt/unsloth-venv/requirements.lock.txt).
RUN ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        "jupyterlab==4.6.0" "notebook==7.6.0" "ipywidgets==8.1.8" "matplotlib==3.11.0" \
        "soundfile==0.14.0" "evaluate==0.4.6" "jiwer==4.0.0" "tensorboard==2.20.0" \
        "langid==1.1.6" "easydict==1.13" "protobuf==6.33.6" \
        "omegaconf==2.3.1" "einx==0.4.3" "librosa==0.11.0" "ftfy==6.3.1" \
 && ${VENV}/bin/python -c "import torch, numpy, numba; from packaging.version import Version; assert torch.__version__.startswith('2.11.0'), torch.__version__; assert Version(numpy.__version__) >= Version('2.3'), numpy.__version__; assert Version(numba.__version__) >= Version('0.65'), numba.__version__; print('notebook-deps pins OK:', torch.__version__, numpy.__version__, numba.__version__)"

# decord (ERNIE-VL video decode) publishes wheels only for x86_64 / win_amd64.
# Install it on its own: HARD on amd64 (a missing/incompatible wheel is a real
# regression there and must fail the build), fail-soft on arm64/other (no wheel
# exists, so drop the ERNIE-VL video path rather than break the image build).
RUN if [ "${TARGETARCH:-amd64}" = "amd64" ]; then \
        ${VENV}/bin/uv pip install --python ${VENV}/bin/python "decord==0.6.0"; \
    else \
        ${VENV}/bin/uv pip install --python ${VENV}/bin/python "decord==0.6.0" \
        || echo ">> decord skipped (no matching wheel for ${TARGETARCH:-}); ERNIE-VL video decode unavailable"; \
    fi

# Audio decode out of the box: the TTS/STT notebooks feed datasets' Audio
# features, which decode through torchcodec. Three traps, all defended:
#   * version pairing: torchcodec 0.11 pairs with torch 2.11 (a mismatched
#     build references other torch symbols and fails to dlopen);
#   * CUDA line: the PyPI default wheel pairs with the cu13 torch line and
#     dlopens libnvrtc.so.13 -- it must come from the cu128 channel;
#   * its libraries dlopen torch + NVIDIA runtime libs that live inside the
#     venv where ld.so cannot see them. Registered via ld.so.conf.d in the
#     RUNTIME stage (this builder layer only installs the wheels, which ride
#     along in the venv copy; the import check needs ffmpeg, which only the
#     runtime stage installs).
# Fail-soft on arches without a matching wheel.
RUN set -eux \
 && { ${VENV}/bin/uv pip install \
        --python ${VENV}/bin/python \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torchcodec==0.11.0" \
   && ${VENV}/bin/uv pip install --python ${VENV}/bin/python nvidia-npp-cu12; } \
 || echo ">> torchcodec bake skipped (no matching wheel for ${TARGETARCH:-amd64})"

# Coherent transformers SIDECARS for per-notebook version activation (see
# docker/unsloth_nb_compat.py). unslothai/notebooks pin many transformers
# versions in their install cells; the base venv ships one (newest 5.x). Each
# sidecar is `transformers==X` + its matched huggingface_hub/tokenizers/
# safetensors, installed --no-deps into its own --target dir under
# ${VENV}/tf-sidecars (rides along in the COPY to runtime). Activating one
# (prepend to sys.path before any ML import) swaps transformers for a model
# WITHOUT touching the cu128 torch/vLLM/unsloth base stack -- verified: base
# unsloth loads + generates under both a 4.57.6 and a 5.5.0 sidecar on B200.
# Versions mirror Unsloth Studio's tiers (4.57.6 default + 5.3.0/5.5.0/5.10.2).
# The companion versions are RESOLVED at build time (not hardcoded) so they
# always satisfy each transformers' hard requirements. ~300MB total after the
# __pycache__/tests strip in the cleanup RUN below. Fail-soft per arch/wheel.
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

# 5) Emit an informational pin record so downstream consumers can see exactly
#    what was resolved. This is NOT a byte-reproducible lockfile -- `pip freeze`
#    captures version strings but not wheel hashes, and several deps (unsloth,
#    unsloth_zoo, vllm --pre) resolve from VCS / nightly indexes that float.
#    Read it with `docker run --rm <image> cat /opt/unsloth-venv/requirements.lock.txt`
#    if you need to forensic-diff two images.
RUN ${VENV}/bin/pip freeze --exclude-editable > ${VENV}/requirements.lock.txt \
 && head -50 ${VENV}/requirements.lock.txt

# 6) Strip pip cache & __pycache__ to shrink the layer copied to runtime.
#
# Note on the `-name tests` strip: numpy 2.4 ships `numpy/_core/tests/`
# back into the wheel (numpy 2.2.6 had stripped it, which is what the
# explicit upgrade earlier in this Dockerfile was meant to fix). Blowing
# away every `tests/` directory under the venv would re-introduce the
# same `from numpy._core.tests._natype import pd_NA` ImportError on the
# deployed image. Exclude numpy's tests directories explicitly so the
# upgrade fix stays in effect; keep stripping the rest.
# Also (size reductions, all verified runtime-safe):
#   * npp: torchcodec dlopens only libnppicc + libnppc; the other ~10 npp libs
#     (libnppif 163M, libnppist, libnppig, ...) are dead weight (~388MB). Nothing
#     else in the venv links them.
#   * static .a archives (~143MB): xgrammar/triton-cupti/nvperf/nvshmem ship .a
#     alongside the .so they actually load at runtime; .a are link-time only and
#     nothing in the image links venv archives (nvcc JIT links /usr/local/cuda).
#   * nvshmem device-side bitcode (~30MB): host .so kept; .bc is device-relink only.
# We deliberately do NOT strip headers (torch/include etc.): the leave-to-pip
# kernels causal-conv1d / mamba-ssm build against torch headers at notebook time
# with --no-build-isolation, so torch/include must survive.
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
assert torch.__version__.startswith("2.11.0"), f"torch silently moved: {torch.__version__}"
assert "+cu128" in torch.__version__, f"cu build silently changed: {torch.__version__}"
assert "sm_100" in arches, f"sm_100 (B200/GB200) missing: {arches}"
# cu128 wheels ship sm_120 native SASS on BOTH amd64 (RTX 5090 / RTX PRO 6000
# Blackwell) and aarch64 (Grace systems). On arm64 sm_120 is what DGX Spark
# (sm_121) runs via minor-forward-compat within major 12; sm_121 itself is
# never in any cu128 wheel.
assert "sm_120" in arches, f"sm_120 missing: {arches}"
print(f"OK: torch 2.11.0+cu128 with sm_100 + sm_120 native SASS intact ({target})")

from importlib.metadata import version, PackageNotFoundError
# xformers is amd64-only (the aarch64 wheel gap -- see Dockerfile header).
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
# The "-base-" variant (vs "-cudnn-runtime-") drops ~2.7 GB of system CUDA
# libraries we never load. torch wheels bake their OWN cuDNN/cuBLAS/cuSPARSE/
# cuRAND/cuSOLVER/cuFFT/NCCL/cuSparseLt inside `torch/lib/`, and libtorch_cuda.so
# has RPATH `$ORIGIN/../../nvidia/cudnn/lib:$ORIGIN/../../nvidia/cublas/lib:...`
# so the dynamic loader resolves through the wheel, never the system. Empirical
# verification via `readelf -d torch/lib/libtorch_cuda.so` (Fork 5 audit). The
# base image still provides nvidia-smi, libcuda stubs, libnvidia-ml -- which
# is everything our entrypoint pre-flight + torch.cuda need.
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

# The nvidia/cuda:12.8.1-base-ubuntu24.04 manifest is multi-arch
# (linux/amd64 + linux/arm64). docker/buildx picks the right one for
# TARGETPLATFORM at this FROM line; no conditional needed.
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
    # Keep the arch list visible at runtime so an in-container source build of a
    # custom CUDA op gets the same SASS coverage as the builder stage. 10.3 is
    # omitted for the same cu12.8-cannot-emit-compute_103 reason as the builder
    # list + header (sm_103 runs sm_100 SASS via forward-compat).
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0+PTX"

# zstd: the official Ollama notebooks run `curl ollama.com/install.sh | sh`
# inside the container, and that installer extracts a zstd tarball -- without
# it the Llama3 Ollama-export notebook dies at the install cell.
# ffmpeg: torchcodec (datasets' audio decode path, pip-installed by the
# TTS/STT notebooks) dlopens the system FFmpeg libraries; the wheel does not
# bundle them, so without ffmpeg every audio notebook dies at load_dataset.
# wget: notebooks fetch sample assets with `!wget URL`; without it the cell
# "succeeds" with sh's not-found on stderr and the next cell crashes on the
# missing file (the Whisper notebook died exactly this way).
# ninja-build: flashinfer's cpp_ext JIT shells out to ninja; subprocesses
# (e.g. `vllm serve` launched by unsloth.dataprep) do not always inherit the
# venv bin on PATH, so the pip ninja alone is not reachable there.
# cuda-nvcc + cudart-dev: flash-linear-attention's TileLang backend (Qwen3.5
# gated-delta-rule models in Studio) JIT-compiles CUDA kernels at runtime via
# nvcc; the -base image ships only runtime libs, so without the compiler any
# TileLang-allowlisted model dies with "[Errno 2] No such file or directory:
# '/usr/local/cuda/bin/nvcc'" on its first backward pass.
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
# Why gcc + g++ + python3.12-dev in the RUNTIME stage:
#   Triton's nvidia backend lazily compiles a small C extension (CudaUtils) on
#   first GPU access. Without a C compiler + Python headers the very first
#   forward pass of any Unsloth model dies with:
#       RuntimeError: Failed to find C compiler. Please specify via CC env var.
#   Adds ~250MB to the runtime image, which is the cost of letting every kernel
#   JIT correctly. (Pre-compiling CudaUtils at build time would need a GPU, so
#   shipping the toolchain is the right trade-off.)

COPY --from=builder /opt/unsloth-venv /opt/unsloth-venv

# Blackwell JIT fix for sm_103 (amd64) and sm_121 (arm64) -- the cu12.8 JIT
# gap described in the header. Two JIT paths need the cu13 override:
#
#   (1) torch's bundled libnvrtc.so.12 is CUDA 12.8. The jiterator C++ side
#       queries the device cap directly, so any NVRTC JIT path (e.g.
#       torch.fft.rfft(complex).abs(), used inside mel-spectrogram code)
#       errors out on sm_103/sm_121. Fix: stage a cu13 NVRTC alias beside the
#       immutable cu12.8 default (mechanics at the staging step below).
#
#   (2) Triton's nvidia backend invokes its OWN bundled ptxas, which in the
#       triton 3.6.0 we pin is still CUDA 12.8 (V12.8.93): it tops out at
#       sm_120, rejects sm_103, and silently downgrades sm_121 to sm_80 per
#       triton-lang/triton#8335. Fix: install cu13 ptxas and point Triton at
#       it with TRITON_PTXAS_PATH.
#
# Both cu13 tools are CPU-side compilers (no driver-floor bump at INSTALL
# time), but their OUTPUT cubin needs a >= 580 driver to LOAD, so neither is
# baked as a global ENV/symlink default -- that would break the Ampere/Ada/
# Hopper/Turing GPUs this image still supports on 570-579 drivers. Instead
# select_cuda_jit_tools in entrypoint.sh activates them per device, only for
# sm_103/sm_121 (which only ship on >= 580 drivers, so the gate is always
# safe). Both arches carry the ~400 MB: amd64 for sm_103, arm64 for sm_121.
RUN set -eux; \
        # The nvidia/cuda base already configures the CUDA apt repo (x86_64 or
        # sbsa) with its own Signed-By keyring at
        # /usr/share/keyrings/cuda-archive-keyring.gpg. Installing
        # cuda-keyring_1.1-1_all.deb on top adds a second sources file with a
        # different Signed-By, which makes `apt-get update` refuse the entire
        # repo ("Conflicting values set for option Signed-By"). The base's repo
        # URL is monolithic and serves every CUDA version including 13.x, so we
        # install cu13 packages directly without touching the keyring at all.
        # Verified on the ubuntu-24.04 (x86_64) and ubuntu-24.04-arm runners.
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
        # (1) NVRTC staging: keep the wheel's real cu12.8 lib as .cu128.orig,
        #     point libnvrtc.so.12 at it (relative symlink), and stage
        #     .cu13 -> the cu13 lib; select_cuda_jit_tools retargets the
        #     symlink ONLY on sm_103/sm_121 hosts. The default needs no
        #     runtime write, so a non-root `docker run --user` container
        #     (which cannot rewrite the symlink) keeps cu12.8, loadable on
        #     every supported 570+ driver.
        NVRTC_DIR=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages/nvidia/cuda_nvrtc/lib; \
        if [ -f "${NVRTC_DIR}/libnvrtc.so.12" ] && [ ! -L "${NVRTC_DIR}/libnvrtc.so.12" ]; then \
            mv "${NVRTC_DIR}/libnvrtc.so.12" "${NVRTC_DIR}/libnvrtc.so.12.cu128.orig"; \
            ln -s libnvrtc.so.12.cu128.orig "${NVRTC_DIR}/libnvrtc.so.12"; \
            ln -s /usr/local/cuda-13.0/lib64/libnvrtc.so.13 "${NVRTC_DIR}/libnvrtc.so.12.cu13"; \
        fi
# (2) ptxas: the cu13 nvcc package above provides it; TRITON_PTXAS_PATH is set
# per device at boot (select_cuda_jit_tools) for the same driver-floor reason.

# Register the venv's torch + NVIDIA lib dirs with the loader so torchcodec
# (installed in the builder, see the bake comment there) can dlopen them.
# ld.so.conf.d, NOT LD_LIBRARY_PATH: the cache is consulted only after
# DT_RUNPATH, so the llama.cpp bundle keeps resolving its own $ORIGIN
# libraries first. The import check runs here because ffmpeg lives in this
# stage; fail-soft where the torchcodec wheel was unavailable.
RUN set -eux \
 && SP=/opt/unsloth-venv/lib/python${PYTHON_VERSION}/site-packages \
 && printf "%s\n" "$SP/torch/lib" "$SP/nvidia/cuda_nvrtc/lib" \
                  "$SP/nvidia/cuda_runtime/lib" "$SP/nvidia/npp/lib" \
        > /etc/ld.so.conf.d/zz-unsloth-venv.conf \
 && ldconfig \
 && { /opt/unsloth-venv/bin/python -c \
        "import torchcodec; print('torchcodec', torchcodec.__version__)" \
   || echo ">> torchcodec unavailable on this arch (audio decode falls back)"; }

# Prebuilt llama.cpp so GGUF export works out of the box.
#
# unsloth_zoo's save_pretrained_gguf() calls check_llama_cpp(), which looks
# for llama-quantize + convert_hf_to_gguf.py + gguf-py/ in
# $UNSLOTH_LLAMA_CPP_PATH (default ~/.unsloth/llama.cpp). Without a baked
# install the first GGUF export inside the container would hit
# install_llama_cpp()'s interactive prompt and then a slow source build.
#
# Why NOT studio/install_llama_prebuilt.py here: that resolver selects a
# bundle for the CURRENT host (nvidia-smi, /proc/driver/nvidia -- which
# leaks through from a GPU build host even inside docker build -- and the
# installed CUDA runtime). The build must never introspect the host, so we
# pin release + asset by build target instead (see fetch_llama_prebuilt.py):
#   * amd64 -> app-<tag>-linux-x64-cuda12-portable.tar.gz   (sm_70..sm_120)
#     arm64 -> app-<tag>-linux-arm64-cuda13-portable.tar.gz (sm_90..sm_121,
#     DGX Spark / Grace)
#   * portable bundles carry their own CUDA runtime libs and dlopen the
#     CUDA backend, so the same binaries also run CPU-only
#   * sha256-verified against the release's llama-prebuilt-sha256.json
#   * converter + gguf-py hydrated from the SAME release's source tarball
#     so the python-side tensor mappings match the binaries
# /opt (not /root) so the install survives a `docker run --user` override;
# UNSLOTH_LLAMA_CPP_PATH makes zoo find it regardless of $HOME.
# Default "latest" -> fetch_llama_prebuilt.py resolves the newest
# unslothai/llama.cpp release at build time (follows the /releases/latest
# redirect, no API token). build.sh resolves this to a concrete tag before
# invoking docker build so the layer cache busts only when upstream publishes;
# pass --build-arg LLAMA_PREBUILT_TAG=<tag> for a frozen, reproducible build.
ARG LLAMA_PREBUILT_TAG=latest
COPY fetch_llama_prebuilt.py /tmp/fetch_llama_prebuilt.py
RUN /opt/unsloth-venv/bin/python /tmp/fetch_llama_prebuilt.py \
        "${LLAMA_PREBUILT_TAG}" "${TARGETARCH:-amd64}" /opt/unsloth/llama.cpp \
 && rm -f /tmp/fetch_llama_prebuilt.py \
 && cat /opt/unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json
ENV UNSLOTH_LLAMA_CPP_PATH=/opt/unsloth/llama.cpp

WORKDIR /workspace
RUN mkdir -p ${HF_HOME} ${TRITON_CACHE_DIR}

# ---------------------------------------------------------------------------
# Per-notebook transformers version activation -- run unslothai/notebooks
# UNCHANGED. See docker/unsloth_nb_compat.py for the full rationale. Pieces:
#   * unsloth_nb_compat.py -> site-packages (importable everywhere): tier
#     detection + sidecar resolution + activation + the IPython hook.
#   * pip/uv shim on a PATH dir AHEAD of the venv bin: a notebook's
#     `!pip install ...` / `!uv pip install ...` cell becomes SAFE + idempotent
#     (keeps the baked torch/vLLM stack; records the requested transformers so
#     its sidecar is activated for the model cells).
#   * unsloth_nb_pip_magic.py -> site-packages: re-points the IPython `%pip` /
#     `%uv` line magics and the `!python -m pip` form at the same shim, so the
#     in-process / module install paths cannot bypass PATH and clobber the stack.
#   * IPython startup hook: activates the right sidecar before the first model
#     cell in manual JupyterLab.
#   * unsloth-run: headless `unsloth-run <notebook|url>` that auto-picks the
#     sidecar and executes every cell -- the robust driven path.
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
# for EVERY kernel, whatever uid runs it. IPYTHONDIR (inherited by any user via
# ENV) points IPython at this shared profile, so the hook still loads when the
# container is started with `--user <uid>` and $HOME is not /root -- unlike a
# /root/.ipython startup dir, which only a root kernel reads. Kernel-writable
# state (history.sqlite) still lands under each user's own path, so a read-only
# profile dir is fine.
ENV IPYTHONDIR=/opt/unsloth-nb/ipython

# Pre-clone unslothai/notebooks so JupyterLab opens with the notebooks already
# present (no git clone or wget needed). Baked here as a READ-ONLY template
# (~206MB, .git stripped); on boot the entrypoint copies it to
# /workspace/unsloth-notebooks and best-effort refreshes from GitHub when
# upstream has advanced. It never overwrites a notebook the user has touched,
# and for untouched notebooks it skips the rewrite when only the install header
# / announcements / footer moved upstream (the tutorial body is unchanged) --
# see unsloth_sync_notebooks.sh + unsloth_nb_content_sig.py. Inherited as-is by
# the studio image (FROM base).
#
# UNSLOTH_NOTEBOOKS_REF pins ONE commit/branch/tag so a multi-arch publish bakes
# identical templates into both arch legs even if unslothai/notebooks advances
# mid-build. The publish workflow resolves the live HEAD sha once (like
# LLAMA_PREBUILT_TAG) and passes it here; the default "main" keeps a plain
# `docker build` tracking the tip. We fetch the single resolved ref (a full
# 40-char sha fetches by object; a branch/tag fetches by name) at depth 1, so
# the bake stays a shallow one-commit pull.
ARG UNSLOTH_NOTEBOOKS_REF=main
RUN set -eux \
 && git init -q /opt/unsloth-notebooks \
 && git -C /opt/unsloth-notebooks remote add origin https://github.com/unslothai/notebooks \
 && git -C /opt/unsloth-notebooks fetch -q --depth 1 origin "${UNSLOTH_NOTEBOOKS_REF}" \
 && git -C /opt/unsloth-notebooks checkout -q FETCH_HEAD \
 && git -C /opt/unsloth-notebooks rev-parse HEAD > /opt/unsloth-notebooks/.unsloth_template_commit \
 && rm -rf /opt/unsloth-notebooks/.git \
 && du -sh /opt/unsloth-notebooks

# JupyterLab lives in the venv (see builder stage). The unslothai/notebooks
# collection is pre-populated into /workspace/unsloth-notebooks on boot; mount a
# volume on /workspace to persist your own notebooks and outputs across runs.
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
