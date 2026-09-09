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
import shlex
import subprocess
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
    # Both layouts, because the wheel moved with the name: nvidia-cublas-cu12
    # unpacks to nvidia/cublas/lib, nvidia-cublas (13+) to nvidia/cu13/lib
    assert "$SP/nvidia/cu13/lib" in block, (
        "the CUDA 13 cublas wheel unpacks to nvidia/cu13/lib, so dropping this entry "
        "puts libcublas.so.13 off the loader path even after the guard installs it"
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


def _cublas_pkg_for(dockerfile: str, soname: str) -> str:
    """The wheel the guard installs for a bundle asking for ``soname``.

    Runs the Dockerfile's own selection lines instead of matching them. A name
    is only correct if it resolves on PyPI, which no string comparison can tell
    you, so pinning one spelling is how this went wrong the first time.
    """
    guard = dockerfile[dockerfile.index("CUDA_SO=") :]
    start = guard.index('major="${want##*.}"')
    snippet = guard[start : guard.index('echo ">> $want missing', start)]
    lines = snippet.replace("\\\n", "\n")
    script = "\n".join([f"want={shlex.quote(soname)}", lines, 'printf %s "$pkg"'])
    return subprocess.run(
        ["sh", "-eu", "-c", script], capture_output = True, text = True, check = True
    ).stdout


@pytest.mark.parametrize(
    "soname, expected",
    [
        ("libcublas.so.12", "nvidia-cublas-cu12"),
        ("libcublas.so.13", "nvidia-cublas==13.*"),
        ("libcublas.so.14", "nvidia-cublas==14.*"),
    ],
)
def test_guard_installs_the_matching_cublas_major(dockerfile: str, soname: str, expected: str):
    # NVIDIA dropped the -cuXX suffix at CUDA 13; neither project spans both majors:
    # nvidia-cublas publishes 13.x only, nvidia-cublas-cu12 has no unsuffixed twin
    assert _cublas_pkg_for(dockerfile, soname) == expected


def test_the_cublas_major_comes_from_the_bundle(dockerfile: str):
    # The two arch bundles differ in CUDA major, so the name must follow ldd; a
    # hardcoded major would satisfy any single case above on its own
    assert _cublas_pkg_for(dockerfile, "libcublas.so.12") != _cublas_pkg_for(
        dockerfile, "libcublas.so.13"
    )


def test_the_guard_never_asks_for_a_retired_suffixed_wheel(dockerfile: str):
    # The regression: the arm64 bundle links libcublas.so.13, the guard derived
    # nvidia-cublas-cu13, a 0.0.1 sdist whose build backend exits 1 saying to use
    # nvidia-cublas. That fails the build, not the install, so the fail-soft paths
    # elsewhere in this file could not catch it.
    for major in range(13, 20):
        assert f"-cu{major}" not in _cublas_pkg_for(dockerfile, f"libcublas.so.{major}")


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
