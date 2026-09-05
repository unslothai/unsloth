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


def _guard_shell(dockerfile: str) -> str:
    """The guard's RUN body as one shell script: strip the Dockerfile's `RUN set -eux`
    prefix, drop comment lines, and join the backslash continuations."""
    start = dockerfile.index("CUDA_SO=")
    body = dockerfile[dockerfile.rindex("RUN set -eux", 0, start) + len("RUN set -eux") :]
    # the Dockerfile parser drops comment lines and joins backslash continuations into
    # one logical line; the instruction ends at the first line without a backslash
    out = []
    for line in body.splitlines():
        if line.strip().startswith("#"):
            continue
        if line.endswith("\\"):
            out.append(line[:-1])
            continue
        out.append(line)
        break
    return " ".join(out)


@pytest.mark.parametrize(
    "soname, expected",
    [
        # arm64: upstream ships a CUDA 13 arm64 llama.cpp; the -cu13 name on PyPI is a
        # deprecated stub whose sdist fails on purpose (the first publish died there)
        ("libcublas.so.13", "nvidia-cublas==13.*"),
        # amd64: cu12, where the suffixed name is still the real package
        ("libcublas.so.12", "nvidia-cublas-cu12"),
    ],
)
def test_guard_uses_the_package_name_pypi_actually_serves(
    dockerfile: str, soname: str, expected: str, tmp_path: Path
):
    """Drive the guard's own shell with ldd and uv stubbed, and check the spec it hands
    to uv. Only the resolution is faked: the branch logic is the Dockerfile's."""
    import os
    import subprocess

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ldd").write_text(
        "#!/bin/sh\n"
        f"printf '\\t%s => not found\\n' {soname}\n"
        "printf '\\tlibcuda.so.1 => not found\\n'\n",
        encoding = "utf-8",
    )
    (bin_dir / "ldconfig").write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    venv_bin = tmp_path / "opt" / "unsloth-venv" / "bin"
    venv_bin.mkdir(parents = True)
    (venv_bin / "uv").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" >> {tmp_path / 'uv_args'}\n"
        # pretend the install resolved the library: rewrite what ldd reports
        f"printf '#!/bin/sh\\nprintf \"\\\\tlibcuda.so.1 => not found\\\\n\"\\n' > {bin_dir / 'ldd'}\n",
        encoding = "utf-8",
    )
    (venv_bin / "python").write_text("#!/bin/sh\n", encoding = "utf-8")
    for f in (bin_dir / "ldd", bin_dir / "ldconfig", venv_bin / "uv", venv_bin / "python"):
        f.chmod(0o755)
    cuda_so = tmp_path / "opt" / "unsloth" / "llama.cpp" / "libggml-cuda.so"
    cuda_so.parent.mkdir(parents = True)
    cuda_so.write_bytes(b"")

    script = _guard_shell(dockerfile).replace("/opt/", f"{tmp_path}/opt/")
    res = subprocess.run(
        ["sh", "-c", "set -eux " + script],
        capture_output = True,
        text = True,
        env = dict(os.environ, PATH = f"{bin_dir}:{os.environ['PATH']}"),
    )
    assert res.returncode == 0, res.stderr
    args = (tmp_path / "uv_args").read_text(encoding = "utf-8").splitlines()
    assert args[-1] == expected, (
        f"for {soname} the guard asked uv for {args[-1]!r}, expected {expected!r}; "
        f"stderr={res.stderr!r}"
    )
    assert "OK: llama.cpp CUDA backend dependencies all resolve" in res.stdout


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
