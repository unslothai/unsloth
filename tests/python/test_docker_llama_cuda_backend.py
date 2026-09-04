# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Regression guard for the llama.cpp CUDA backend inside the Docker image.

The portable bundle dlopens libggml-cuda.so but carries none of the CUDA math
libraries it links against, and with no libcublas on the loader path the backend
fails to load SILENTLY: nothing printed, `--list-devices` empty, every GGUF request
on the CPU. These pin the two Dockerfile halves that prevent it.
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
    return DOCKERFILE.read_text(encoding = "utf-8")


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
    # LD_LIBRARY_PATH is consulted BEFORE DT_RUNPATH and would let the venv's copies
    # shadow llama.cpp's own $ORIGIN libs; ld.so.conf.d is consulted after
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
    assert re.search(
        r"grep -v .libcuda\\?\.so\\?\.1", guard
    ), "libcuda.so.1 must be exempt from the guard or every build fails"


def test_guard_installs_the_matching_cublas_major(dockerfile: str):
    # the two arch bundles differ in CUDA major, so it must come from ldd
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
    # flashinfer raises at import when jit-cache and python disagree, killing the vLLM
    # EngineCore GRPO fast_inference runs on. A literal pin would drift.
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
    # unsloth_cli imports studio.backend.core.*, which needs structlog: a studio
    # requirement, not an unsloth[huggingface] one, so the base venv must ask for it
    assert '"structlog"' in dockerfile, "the base venv must install structlog for unsloth_cli"
    assert (
        "from studio.backend.core.export import ExportBackend" in dockerfile
    ), "a build-time import guard must prove the CLI can reach the studio backend"
