# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for the llama.cpp CUDA backend inside the Docker image.

The portable llama.cpp bundle ships libggml-cuda.so and loads it with dlopen
(ggml_backend_dl), but the bundle does NOT carry the CUDA math libraries it
links against, and the CUDA runtime base image only carries libcudart. With no
libcublas on the loader path the backend fails to load SILENTLY: llama.cpp
prints nothing, `--list-devices` comes back empty and every GGUF request runs on
the CPU. Measured on a B200 with gemma-4-E2B UD-Q4_K_XL: 1.6 tok/s instead of
224 tok/s, a 140x regression that no functional test would have caught.

The Dockerfile therefore has to do two things, and these tests pin both:
  * put torch's bundled libcublas on the loader path (ld.so.conf.d, not
    LD_LIBRARY_PATH, so llama.cpp's own $ORIGIN libs keep winning);
  * fail the build when any non-driver dependency of libggml-cuda.so is still
    unresolved, so a CPU-only image can never be published again.

Static: parses the Dockerfile only. No docker, no GPU, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


@pytest.fixture(scope = "module")
def dockerfile() -> str:
    assert DOCKERFILE.is_file(), f"missing {DOCKERFILE}"
    return DOCKERFILE.read_text()


def test_cublas_dir_is_registered_with_the_loader(dockerfile: str):
    conf = re.search(
        r"ld\.so\.conf\.d/zz-unsloth-venv\.conf",
        dockerfile,
    )
    assert conf, "the venv loader-config layer disappeared"
    block = dockerfile[: conf.end()]
    assert "$SP/nvidia/cublas/lib" in block, (
        "libggml-cuda.so links against libcublas, which only exists in the venv's "
        "wheel copy; without this entry the CUDA backend fails to dlopen and GGUF "
        "silently runs on the CPU"
    )


def test_loader_config_is_not_ld_library_path(dockerfile: str):
    # LD_LIBRARY_PATH is consulted BEFORE DT_RUNPATH, so it would let the venv's
    # copies shadow llama.cpp's own $ORIGIN libs. ld.so.conf.d is consulted after.
    assert "ld.so.conf.d/zz-unsloth-venv.conf" in dockerfile
    assert not re.search(
        r"ENV\s+LD_LIBRARY_PATH=.*site-packages/nvidia",
        dockerfile,
    ), "the venv nvidia libs must not go on LD_LIBRARY_PATH"


def test_build_fails_on_an_unresolved_cuda_backend(dockerfile: str):
    assert "libggml-cuda.so" in dockerfile, "the CUDA backend guard disappeared"
    guard = dockerfile[dockerfile.index("CUDA_SO=") :]
    assert "ldd" in guard, "the guard must inspect the backend's dependencies"
    assert "not found" in guard
    assert "exit 1" in guard, "an unresolved backend must fail the build"
    # The driver stub is injected by nvidia-container-toolkit at `docker run
    # --gpus`, so it is never resolvable inside the build and must be exempt.
    assert re.search(
        r"grep -v .libcuda\\?\.so\\?\.1", guard
    ), "libcuda.so.1 must be exempt from the guard or every build fails"


def test_guard_installs_the_matching_cublas_major(dockerfile: str):
    # The amd64 bundle is CUDA 12 and torch already ships libcublas.so.12, but
    # the arm64 bundle is CUDA 13. Deriving the major from ldd keeps the two
    # legs correct without hardcoding either.
    guard = dockerfile[dockerfile.index("CUDA_SO=") :]
    assert (
        "nvidia-cublas-cu${major}" in guard
    ), "the guard must install the cublas major the bundle actually asks for"
    assert "libcublas" in guard


def test_guard_runs_after_the_prebuilt_is_fetched(dockerfile: str):
    fetch = dockerfile.index("fetch_llama_prebuilt.py")
    guard = dockerfile.index("CUDA_SO=")
    assert fetch < guard, "the guard can only inspect a bundle that already exists"


def test_flashinfer_jit_cache_tracks_flashinfer(dockerfile: str):
    # flashinfer raises at import when flashinfer-jit-cache and flashinfer-python
    # disagree, and that exception kills the vLLM EngineCore, which is what
    # Unsloth's GRPO fast_inference path runs on. A literal pin drifts the moment
    # vLLM bumps its flashinfer requirement, so the version has to be derived.
    assert (
        "flashinfer-jit-cache==${FI_VER}" in dockerfile
    ), "flashinfer-jit-cache must be pinned to the resolved flashinfer-python version"
    assert not re.search(
        r"flashinfer-jit-cache==[0-9]", dockerfile
    ), "a literal flashinfer-jit-cache version will drift away from flashinfer-python"
    assert "import flashinfer" in dockerfile, (
        "the build must prove flashinfer imports, or a mismatch stays silent "
        "until the first vLLM engine start"
    )


def test_cli_can_reach_the_studio_backend(dockerfile: str):
    # unsloth_cli's train / export / chat / list-checkpoints import
    # studio.backend.core.*, which needs structlog. It is a studio backend
    # requirement rather than an unsloth[huggingface] one, so the base venv has
    # to ask for it explicitly or the whole CLI dies on ModuleNotFoundError.
    assert '"structlog"' in dockerfile, "the base venv must install structlog for unsloth_cli"
    assert (
        "from studio.backend.core.export import ExportBackend" in dockerfile
    ), "a build-time import guard must prove the CLI can reach the studio backend"
