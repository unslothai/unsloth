# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows GPU-detection regression test on a synthetic layout.

Bug (#5106): on Windows without a system CUDA toolkit, the prebuilt
llama-server.exe couldn't LoadLibrary cudart64_X / cublas64_X /
cublasLt64_X, so ggml-cuda.dll's static import on cublas64_X.dll failed
and the model fell back to CPU even when nvidia-smi reported the GPU.

Fix:
  * #5322 overlays upstream's paired cudart bundle into
    install_dir/build/bin/Release/ next to llama-server.exe.
  * #5324 prepends pip-installed nvidia/<pkg>/{bin,bin/x86_64,Library/
    bin} and torch/lib to PATH when launching llama-server.exe.

CI has no GPU so nvidia-smi is mocked; everything else (resolver, PATH
builder, install layout) runs against a real filesystem.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types as _types
import zipfile
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy deps only if they fail to import (unconditional stubs would shadow
# the real module for sibling tests). Use try-import, not find_spec: loggers
# imports fastapi at load, so find_spec succeeds but the import then raises.
import importlib as _importlib  # noqa: E402


def _maybe_stub(name: str, builder):
    try:
        _importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = _types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    return _types.ModuleType("structlog")


def _build_httpx_stub():
    m = _types.ModuleType("httpx")
    for _exc_name in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
    ):
        setattr(m, _exc_name, type(_exc_name, (Exception,), {}))
    m.Response = type("Response", (), {})

    class _FakeTimeout:
        def __init__(self, *a, **kw):
            pass

    m.Timeout = _FakeTimeout
    m.Client = type(
        "Client",
        (),
        {
            "__init__": lambda self, **kw: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *a: None,
        },
    )
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)
_maybe_stub("httpx", _build_httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


# Upstream b9103 cudart bundle: exactly these three DLLs per CUDA major,
# no executables or subdirectories. Verified by direct unzip.
REAL_UPSTREAM_CUDART_BUNDLE = {
    "12.4": ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"),
    "13.1": ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"),
}

# PyPI win_amd64 wheel layouts, verified via `pip download ... --platform
# win_amd64` + `unzip -l`. Resolver only cares about directory structure.
REAL_PIP_NVIDIA_WHEEL_LAYOUTS = {
    # Legacy cu-suffixed wheels
    "nvidia/cuda_runtime/bin": ["cudart64_12.dll"],
    "nvidia/cublas/bin": [
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "nvblas64_12.dll",
    ],
    "nvidia/cudnn/bin": [
        "cudnn64_9.dll",
        "cudnn_adv64_9.dll",
        "cudnn_ops64_9.dll",
    ],
    # Unsuffixed cu13 wheels
    "nvidia/cu13/bin/x86_64": [
        "cudart64_13.dll",
        "cublas64_13.dll",
        "cublasLt64_13.dll",
        "nvblas64_13.dll",
    ],
}


def _populate_studio_venv(prefix: Path) -> None:
    """Lay out fake nvidia + torch wheels matching real win_amd64 layouts (stub bytes)."""
    site = prefix / "Lib" / "site-packages"
    for rel, dlls in REAL_PIP_NVIDIA_WHEEL_LAYOUTS.items():
        d = site / Path(rel)
        d.mkdir(parents = True, exist_ok = True)
        for name in dlls:
            (d / name).write_bytes(b"PE-stub")
    # install_python_stack always installs torch beside nvidia.
    (site / "torch" / "lib").mkdir(parents = True, exist_ok = True)
    for fn in ("c10.dll", "torch.dll", "torch_cpu.dll", "torch_python.dll"):
        (site / "torch" / "lib" / fn).write_bytes(b"PE-stub")


def _populate_studio_install(install_dir: Path, runtime: str = "13.1") -> None:
    """Lay out install_dir/build/bin/Release/ as #5322 leaves it: payload + cudart overlay."""
    rel = install_dir / "build" / "bin" / "Release"
    rel.mkdir(parents = True, exist_ok = True)
    for fn in (
        "llama-server.exe",
        "llama-quantize.exe",
        "llama-cli.exe",
        "llama.dll",
        "ggml.dll",
        "ggml-base.dll",
        "ggml-cuda.dll",
        "mtmd.dll",
    ):
        (rel / fn).write_bytes(b"PE-stub")
    # The cudart overlay from #5322.
    for fn in REAL_UPSTREAM_CUDART_BUNDLE[runtime]:
        (rel / fn).write_bytes(b"PE-stub")


def _build_path_dirs_like_start_llama_server(
    binary_dir: Path,
    prefix: Path,
    cuda_path: str = "",
) -> list[str]:
    """Wrapper around the real _build_windows_path_dirs staticmethod."""
    return LlamaCppBackend._build_windows_path_dirs(
        str(binary_dir), str(prefix), cuda_path
    )


def _mock_nvidia_smi_run(fake_output: str, returncode: int = 0) -> "mock._patch":
    """Patch subprocess.run so the nvidia-smi probe returns fake_output;
    other calls pass through."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and cmd and "nvidia-smi" in cmd[0]:
            return subprocess.CompletedProcess(
                args = cmd, returncode = returncode, stdout = fake_output, stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


# --------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------- #
class TestWindowsGpuDetectionAfter5106Fix:
    """End-to-end #5106 fix on a synthetic Windows layout. nvidia-smi
    mocked; resolver, PATH builder, and install layout run live."""

    def test_nvidia_smi_probe_reports_synthetic_gpu(self, monkeypatch):
        """Probe parses CSV output and returns (index, free_mib)."""
        # Clear inherited masks so the synthetic CSV isn't filtered.
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising = False)
        # The #5106 reporter's exact reproducer: RTX 4090, 22805 MiB.
        fake_csv = "0, 22805\n"
        with _mock_nvidia_smi_run(fake_csv):
            gpus = LlamaCppBackend._get_gpu_free_memory()
        assert gpus == [
            (0, 22805)
        ], f"GPU probe failed to parse mocked nvidia-smi output: {gpus}"

    def test_nvidia_smi_probe_respects_cuda_visible_devices(self, monkeypatch):
        """CUDA_VISIBLE_DEVICES=1 -> only GPU 1 visible."""
        fake_csv = "0, 22805\n1, 24576\n2, 16384\n"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
        with _mock_nvidia_smi_run(fake_csv):
            gpus = LlamaCppBackend._get_gpu_free_memory()
        assert gpus == [(1, 24576)], gpus

    def test_get_gpu_memory_parses_three_and_two_column(self, monkeypatch):
        """Total is parsed when present; a legacy two-column line or a non-integer
        total ("N/A") yields total 0 (back-compat) rather than dropping the GPU,
        which would silently spill to CPU."""
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
        with _mock_nvidia_smi_run("0, 22805, 24576\n"):
            assert LlamaCppBackend._get_gpu_memory() == [(0, 22805, 24576)]
        with _mock_nvidia_smi_run("0, 22805\n"):
            assert LlamaCppBackend._get_gpu_memory() == [(0, 22805, 0)]
        # A non-integer total must keep the GPU (total 0), not drop it.
        with _mock_nvidia_smi_run("0, 22805, N/A\n"):
            assert LlamaCppBackend._get_gpu_memory() == [(0, 22805, 0)]
        # A bad free still skips that line (free is required).
        with _mock_nvidia_smi_run("0, N/A, 24576\n1, 22805, 24576\n"):
            assert LlamaCppBackend._get_gpu_memory() == [(1, 22805, 24576)]

    def test_windows_install_dir_has_all_three_cudart_dlls(self, tmp_path):
        """All three bundle DLLs must land in install_dir/build/bin/
        Release; any missing one breaks ggml-cuda.dll's PE import chain."""
        install = tmp_path / "studio_install"
        _populate_studio_install(install, runtime = "13.1")
        rel = install / "build" / "bin" / "Release"
        for fn in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            assert (rel / fn).exists(), f"missing {fn} in {rel}"
        assert (rel / "llama-server.exe").exists()
        assert (rel / "ggml-cuda.dll").exists()

    def test_resolver_finds_real_pypi_wheel_layouts(self, tmp_path):
        """Resolver must pick up every wheel layout: nvidia/<pkg>/bin,
        nvidia/<pkg>/bin/x86_64, torch/lib."""
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        out = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
        site = prefix / "Lib" / "site-packages"
        for expected in (
            site / "nvidia" / "cuda_runtime" / "bin",
            site / "nvidia" / "cublas" / "bin",
            site / "nvidia" / "cudnn" / "bin",
            site / "nvidia" / "cu13" / "bin" / "x86_64",
            site / "torch" / "lib",
        ):
            assert (
                str(expected) in out
            ), f"resolver missed {expected.relative_to(prefix)}: {out}"

    def test_path_assembly_makes_cudart_reachable_without_toolkit(self, tmp_path):
        """The #5106 scenario: GPU detected, pip nvidia wheels present,
        no system CUDA toolkit. cudart must be reachable from PATH via
        BOTH binary_dir (#5322) and a pip nvidia dir (#5324)."""
        prefix = tmp_path / "studio_venv"
        install = tmp_path / "studio_install"
        _populate_studio_venv(prefix)
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(
            binary_dir, prefix, cuda_path = ""
        )
        # binary_dir first -- Windows DLL search step 1.
        assert path_dirs[0] == str(
            binary_dir
        ), f"binary_dir must be first in PATH; got {path_dirs[0]}"
        cudart_locations = []
        for entry in path_dirs:
            for cudart_name in ("cudart64_12.dll", "cudart64_13.dll"):
                if (Path(entry) / cudart_name).exists():
                    cudart_locations.append((entry, cudart_name))
        assert cudart_locations, (
            f"cudart unreachable from any PATH entry -- #5106 not fixed.\n"
            f"PATH entries searched: {path_dirs}"
        )
        # Defence in depth: both fix paths contribute cudart.
        sources = {Path(e).relative_to(tmp_path).parts[0] for e, _ in cudart_locations}
        assert (
            "studio_install" in sources
        ), f"#5322's cudart drop not reachable: {cudart_locations}"
        assert (
            "studio_venv" in sources
        ), f"#5324's pip nvidia dir not contributing cudart: {cudart_locations}"

    def test_cublas_and_cublasLt_also_reachable(self, tmp_path):
        """ggml-cuda imports cublas64, which imports cublasLt64. All
        three must resolve or LoadLibrary returns NULL."""
        prefix = tmp_path / "studio_venv"
        install = tmp_path / "studio_install"
        _populate_studio_venv(prefix)
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(binary_dir, prefix)
        for required in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            reachable = any((Path(d) / required).exists() for d in path_dirs)
            assert reachable, (
                f"{required} unreachable from PATH; #5106 not fixed.\n"
                f"PATH entries: {path_dirs}"
            )

    def test_no_pip_nvidia_wheels_still_works_via_install_dir(self, tmp_path):
        """No pip nvidia wheels (CPU-only torch / standalone unsloth):
        cudart still resolves via #5322's binary_dir drop."""
        prefix = tmp_path / "bare_venv"
        prefix.mkdir()
        install = tmp_path / "studio_install"
        _populate_studio_install(install, runtime = "13.1")
        binary_dir = install / "build" / "bin" / "Release"
        path_dirs = _build_path_dirs_like_start_llama_server(binary_dir, prefix)
        assert path_dirs == [
            str(binary_dir)
        ], f"bare venv produced unexpected PATH: {path_dirs}"
        for required in REAL_UPSTREAM_CUDART_BUNDLE["13.1"]:
            assert (
                binary_dir / required
            ).exists(), f"{required} missing from binary_dir on bare venv install"

    def test_no_install_dir_still_works_via_pip_wheels(self, tmp_path):
        """Pre-#5322 install (binary_dir lacks cudart): #5324's pip
        wheel dirs on PATH still resolve cudart."""
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        install = tmp_path / "studio_install_pre5322"
        rel = install / "build" / "bin" / "Release"
        rel.mkdir(parents = True)
        # Main archive payload only; cudart bundle absent.
        for fn in (
            "llama-server.exe",
            "llama.dll",
            "ggml-cuda.dll",
            "ggml-base.dll",
        ):
            (rel / fn).write_bytes(b"PE-stub")
        path_dirs = _build_path_dirs_like_start_llama_server(rel, prefix)
        cudart_reachable = any(
            (Path(d) / "cudart64_12.dll").exists()
            or (Path(d) / "cudart64_13.dll").exists()
            for d in path_dirs
        )
        assert cudart_reachable, (
            "#5324 pip wheel fallback failed: cudart unreachable from PATH "
            f"on cudart-less install. PATH entries: {path_dirs}"
        )
        cublas_reachable = any(
            (Path(d) / "cublas64_12.dll").exists()
            or (Path(d) / "cublas64_13.dll").exists()
            for d in path_dirs
        )
        assert cublas_reachable, "cublas unreachable on cudart-less install"

    def test_pre_pr_scenario_would_have_failed(self, tmp_path):
        """Negative control: pre-#5322 + pre-#5324 leaves cudart
        unreachable -- the original failure mode. Confirms the test
        catches a regression."""
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        install = tmp_path / "pre_pr_install"
        rel = install / "build" / "bin" / "Release"
        rel.mkdir(parents = True)
        for fn in ("llama-server.exe", "llama.dll", "ggml-cuda.dll"):
            (rel / fn).write_bytes(b"PE-stub")
        # Pre-PR PATH: binary_dir only, no pip nvidia dirs, no toolkit.
        pre_pr_path_dirs = [str(rel)]
        cudart_reachable_pre = any(
            (Path(d) / "cudart64_12.dll").exists()
            or (Path(d) / "cudart64_13.dll").exists()
            for d in pre_pr_path_dirs
        )
        assert not cudart_reachable_pre, (
            "Test self-check failed: pre-PR scenario unexpectedly had "
            f"cudart reachable. {pre_pr_path_dirs}"
        )


class TestWindowsSysPlatformMocked:
    """Confirm we test the win32 branch in start_llama_server, not the
    linux fallback. Patches sys.platform and re-runs the branch-selecting
    helper."""

    def test_sys_platform_win32_uses_pip_nvidia_resolver(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sys, "platform", "win32")
        prefix = tmp_path / "studio_venv"
        _populate_studio_venv(prefix)
        out = LlamaCppBackend._windows_pip_nvidia_dll_dirs(str(prefix))
        assert out, f"resolver returned empty under sys.platform=win32: {out}"
        # cu13 arch dir must be in the output.
        cu13_arch = (
            prefix / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64"
        )
        assert str(cu13_arch) in out
