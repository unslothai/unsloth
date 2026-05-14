# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Windows pip-nvidia DLL dir resolver.

Studio installs torch with bundled CUDA wheels (nvidia-cuda-runtime-cu13,
nvidia-cublas-cu13, etc.) and the prebuilt llama-server.exe must find
those DLLs at runtime to load CUDA. Mirrors the Linux LD_LIBRARY_PATH
block. See unslothai/unsloth#5106.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy deps before importing the module under test.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


def _make_nvidia_layout(prefix: Path, pkgs_with_layout: dict[str, str]):
    """Build a fake <prefix>/Lib/site-packages/nvidia/<pkg>/{bin|Library/bin}
    tree with a stub DLL inside each leaf so isdir() picks them up."""
    nv = prefix / "Lib" / "site-packages" / "nvidia"
    for pkg, layout in pkgs_with_layout.items():
        if layout == "bin":
            d = nv / pkg / "bin"
        elif layout == "library_bin":
            d = nv / pkg / "Library" / "bin"
        else:
            raise ValueError(layout)
        d.mkdir(parents = True, exist_ok = True)
        (d / "stub.dll").write_bytes(b"")


class TestWindowsPipNvidiaDllDirs:
    def test_returns_empty_when_no_nvidia_wheels(self, tmp_path):
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert result == []

    def test_picks_up_bin_layout(self, tmp_path):
        _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": "bin",
                "cublas": "bin",
                "cudnn": "bin",
            },
        )
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert len(result) == 3
        assert all(Path(p).is_dir() for p in result)
        assert all(Path(p).name == "bin" for p in result)
        names = {Path(p).parent.name for p in result}
        assert names == {"cuda_runtime", "cublas", "cudnn"}

    def test_picks_up_library_bin_layout(self, tmp_path):
        _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": "library_bin",
                "cublas": "library_bin",
            },
        )
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert len(result) == 2
        for p in result:
            assert Path(p).is_dir()
            assert Path(p).parent.name == "Library"
            assert Path(p).parent.parent.name in {"cuda_runtime", "cublas"}

    def test_mixed_layouts_all_resolved(self, tmp_path):
        _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": "bin",
                "cublas": "library_bin",
                "cudnn": "bin",
                "nvjitlink": "library_bin",
            },
        )
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert len(result) == 4

    def test_does_not_walk_outside_known_paths(self, tmp_path):
        # Only nvidia/<pkg>/{bin,Library/bin} and torch/lib are picked
        # up. Unrelated site-packages contents (numpy, scipy, ...) must
        # be ignored.
        site = tmp_path / "Lib" / "site-packages"
        (site / "numpy").mkdir(parents = True)
        (site / "scipy" / "linalg").mkdir(parents = True)
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert result == []

    def test_picks_up_torch_lib(self, tmp_path):
        # PyTorch's Windows CUDA wheel bundles cudart64_X.dll /
        # cublas64_X.dll directly under Lib/site-packages/torch/lib/
        # instead of as separate nvidia-* wheels. Without this, users
        # on torch-bundled-CUDA installs still hit #5106.
        torch_lib = tmp_path / "Lib" / "site-packages" / "torch" / "lib"
        torch_lib.mkdir(parents = True)
        (torch_lib / "cudart64_12.dll").write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert len(result) == 1
        assert Path(result[0]) == torch_lib

    def test_torch_lib_combined_with_nvidia_wheels(self, tmp_path):
        # Both modular nvidia-* wheels and torch/lib are returned when
        # present together.
        _make_nvidia_layout(
            tmp_path,
            {
                "cuda_runtime": "bin",
                "cublas": "bin",
            },
        )
        torch_lib = tmp_path / "Lib" / "site-packages" / "torch" / "lib"
        torch_lib.mkdir(parents = True)
        (torch_lib / "cudart64_13.dll").write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert len(result) == 3
        names = {Path(p).name for p in result}
        assert names == {"bin", "lib"}
        assert any(Path(p) == torch_lib for p in result)

    def test_torch_lib_must_be_a_directory(self, tmp_path):
        # If torch/lib exists as a file (broken install), it is
        # ignored, not returned.
        site = tmp_path / "Lib" / "site-packages" / "torch"
        site.mkdir(parents = True)
        (site / "lib").write_bytes(b"not a dir")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert result == []

    def test_skips_non_directories(self, tmp_path):
        nv = tmp_path / "Lib" / "site-packages" / "nvidia"
        (nv / "cuda_runtime").mkdir(parents = True)
        # Create a regular file at the path where 'bin' would normally be a dir
        (nv / "cuda_runtime" / "bin").write_bytes(b"not a dir")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert result == []

    def test_missing_prefix_does_not_raise(self):
        # If sys.prefix points to a path that doesn't exist (unusual,
        # but possible during test setup), the resolver must just
        # return [] rather than raising.
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(
            "/this/path/does/not/exist/anywhere"
        )
        assert result == []

    def test_picks_up_cu13_bin_x86_64_layout(self, tmp_path):
        # Current ``nvidia-cuda-runtime`` 13.x and ``nvidia-cublas``
        # 13.x Windows wheels ship DLLs under
        # ``nvidia/cu13/bin/x86_64/`` instead of ``nvidia/<pkg>/bin/``.
        # Without this, users on the new CUDA 13 wheel generation hit
        # the original #5106 failure mode.
        dll_dir = (
            tmp_path / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
        )
        dll_dir.mkdir(parents = True)
        for name in ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"):
            (dll_dir / name).write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert str(dll_dir) in result, f"cu13 bin/x86_64 not in {result}"

    def test_picks_up_bin_x64_layout(self, tmp_path):
        # Some repackaged wheels use ``bin/x64`` (Windows-x64 convention)
        # instead of ``bin/x86_64`` (NVIDIA-internal convention).
        dll_dir = tmp_path / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x64"
        dll_dir.mkdir(parents = True)
        (dll_dir / "cudart64_13.dll").write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        assert str(dll_dir) in result

    def test_mixed_cu12_and_cu13_layouts(self, tmp_path):
        # A venv could have both the modular cu12 wheels (legacy) and
        # the unsuffixed cu13 wheel installed side by side. Both must
        # be reachable.
        site = tmp_path / "Lib" / "site-packages"
        cu12_bin = site / "nvidia" / "cuda_runtime" / "bin"
        cu13_arch = site / "nvidia" / "cu13" / "bin" / "x86_64"
        cu12_bin.mkdir(parents = True)
        cu13_arch.mkdir(parents = True)
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        result_set = {Path(p) for p in result}
        assert cu12_bin in result_set
        assert cu13_arch in result_set

    def test_glob_meta_in_prefix_is_safe(self, tmp_path):
        # Windows usernames / install paths can contain ``[`` or ``]``.
        # A glob-based resolver would interpret these as a character
        # class and silently return [] even when DLL dirs exist. The
        # iterdir-based implementation must work on such paths.
        prefix = tmp_path / "studio_[gpu]_install"
        dll_dir = prefix / "Lib" / "site-packages" / "nvidia" / "cuda_runtime" / "bin"
        dll_dir.mkdir(parents = True)
        (dll_dir / "cudart64_12.dll").write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
        assert str(dll_dir) in result, f"bracket-prefixed path returned empty: {result}"

    def test_arch_subdir_listed_before_parent_bin(self, tmp_path):
        # When both ``nvidia/<pkg>/bin/`` and
        # ``nvidia/<pkg>/bin/x86_64/`` exist, the arch-specific subdir
        # must be listed first so Windows DLL search picks up the
        # cudart64_X.dll location even if the parent ``bin`` is empty.
        site = tmp_path / "Lib" / "site-packages"
        outer_bin = site / "nvidia" / "cu13" / "bin"
        arch_bin = outer_bin / "x86_64"
        arch_bin.mkdir(parents = True)
        (arch_bin / "cudart64_13.dll").write_bytes(b"")
        result = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(tmp_path))
        # outer_bin exists as a directory (it contains arch_bin); the
        # arch-specific subdir should come first in the list.
        result_paths = [Path(p) for p in result]
        assert arch_bin in result_paths
        assert outer_bin in result_paths
        assert result_paths.index(arch_bin) < result_paths.index(outer_bin)
