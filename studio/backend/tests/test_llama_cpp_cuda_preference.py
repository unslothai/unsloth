# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Windows CUDA-vs-CPU llama-server build preference.

On Windows a CPU-only cmake build (build/bin/Release/llama-server.exe) and a
CUDA cmake build (build-cuda/bin/Release/llama-server.exe) can both exist under
the same install root. _find_llama_server_binary() must prefer the CUDA build
when an NVIDIA GPU is present, otherwise fall back to the CPU build. Linux
behaviour must be unchanged and the nvidia probe must never run there.
See unslothai/unsloth#5941.
"""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# Stub heavy deps before importing the module under test (same set the sibling
# Windows DLL-resolver test stubs out).
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

_EXE = "llama-server.exe"


def _make_exe(path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"")
    return path


def _win_layout(root: Path, cpu: bool = True, cuda: bool = True):
    """Create the Windows cmake build layout under ``root``."""
    cpu_exe = root / "build" / "bin" / "Release" / _EXE
    cuda_exe = root / "build-cuda" / "bin" / "Release" / _EXE
    if cpu:
        _make_exe(cpu_exe)
    if cuda:
        _make_exe(cuda_exe)
    return cpu_exe, cuda_exe


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Isolate discovery so only the layout a test creates can be found.

    Empties HOME (so home-dir / legacy discovery finds nothing by default),
    clears the path env vars, forces legacy storage-root resolution so the
    home-dir branch points at the temp HOME, and removes any system
    llama-server from PATH.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)

    # Force the home-dir branch to use ~/.unsloth/llama.cpp (legacy layout).
    storage_stub = _types.ModuleType("utils.paths.storage_roots")
    storage_stub.studio_root = lambda: home / ".unsloth" / "studio"
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", storage_stub)

    import shutil

    monkeypatch.setattr(shutil, "which", lambda *a, **k: None)
    return home


def _set_nvidia(monkeypatch, present: bool):
    monkeypatch.setattr(
        LlamaCppBackend, "_nvidia_available", staticmethod(lambda: present)
    )


# ── Custom dir (UNSLOTH_LLAMA_CPP_PATH) ──────────────────────────────────────


def test_custom_dir_prefers_cuda_when_nvidia(tmp_path, monkeypatch, isolated):
    root = tmp_path / "custom"
    _cpu, cuda = _win_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    _set_nvidia(monkeypatch, True)
    assert LlamaCppBackend._find_llama_server_binary() == str(cuda)


def test_custom_dir_uses_cpu_when_no_nvidia(tmp_path, monkeypatch, isolated):
    root = tmp_path / "custom"
    cpu, _cuda = _win_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    _set_nvidia(monkeypatch, False)
    assert LlamaCppBackend._find_llama_server_binary() == str(cpu)


def test_custom_dir_cuda_only_not_used_without_nvidia(tmp_path, monkeypatch, isolated):
    # CUDA build present, no CPU build, no NVIDIA: the CUDA exe must NOT be
    # returned (it would fail to load without CUDA), discovery falls through.
    root = tmp_path / "custom"
    _cpu, cuda = _win_layout(root, cpu = False, cuda = True)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))
    _set_nvidia(monkeypatch, False)
    assert LlamaCppBackend._find_llama_server_binary() != str(cuda)


# ── Home dir (~/.unsloth/llama.cpp) ──────────────────────────────────────────


def test_home_dir_prefers_cuda_when_nvidia(monkeypatch, isolated):
    root = isolated / ".unsloth" / "llama.cpp"
    _cpu, cuda = _win_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    _set_nvidia(monkeypatch, True)
    assert LlamaCppBackend._find_llama_server_binary() == str(cuda)


def test_home_dir_uses_cpu_when_no_nvidia(monkeypatch, isolated):
    root = isolated / ".unsloth" / "llama.cpp"
    cpu, _cuda = _win_layout(root)
    monkeypatch.setattr(sys, "platform", "win32")
    _set_nvidia(monkeypatch, False)
    assert LlamaCppBackend._find_llama_server_binary() == str(cpu)


# ── Linux is unchanged and never probes nvidia ───────────────────────────────


def test_linux_unchanged_and_nvidia_never_probed(tmp_path, monkeypatch, isolated):
    root = tmp_path / "custom"
    # Linux cmake layout: build/bin/llama-server (no Release/, no .exe).
    linux_bin = _make_exe(root / "build" / "bin" / "llama-server")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(root))

    def _boom():
        raise AssertionError("_nvidia_available must not run on Linux")

    monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(_boom))
    assert LlamaCppBackend._find_llama_server_binary() == str(linux_bin)


# ── _nvidia_available unit behaviour ─────────────────────────────────────────


def _fake_run_factory(returncode = 0, stdout = ""):
    def _fake_run(*args, **kwargs):
        return _types.SimpleNamespace(returncode = returncode, stdout = stdout)

    return _fake_run


def test_nvidia_available_true_on_gpu_listing(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        _fake_run_factory(0, "GPU 0: NVIDIA GeForce RTX 5070 (UUID: GPU-...)"),
    )
    assert LlamaCppBackend._nvidia_available() is True


def test_nvidia_available_false_on_nonzero_return(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(1, ""))
    assert LlamaCppBackend._nvidia_available() is False


def test_nvidia_available_false_when_no_gpu_token(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run_factory(0, "No devices were found"))
    assert LlamaCppBackend._nvidia_available() is False


def test_nvidia_available_false_on_timeout(monkeypatch):
    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd = "nvidia-smi", timeout = 5)

    monkeypatch.setattr(subprocess, "run", _raise)
    assert LlamaCppBackend._nvidia_available() is False


def test_nvidia_available_false_when_smi_missing(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert LlamaCppBackend._nvidia_available() is False
