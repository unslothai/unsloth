# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for preferring a ``build-cuda`` llama-server binary over a CPU-only
``build`` one when an NVIDIA GPU is present.

Studio installs (manual source builds, ``UNSLOTH_LLAMA_CPP_PATH``-pointed
custom installs, or pre-prebuilt-binary installs) can leave both a CPU-only
``build/`` tree and a CUDA-enabled ``build-cuda/`` tree under the same
llama.cpp root. Before this fix, ``_layout_candidates`` had no concept of
``build-cuda/`` at all, so ``_find_llama_server_binary`` always resolved the
CPU-only binary and GGUF chat inference silently ran on CPU. See
unslothai/unsloth#5941.
"""

from __future__ import annotations

import re
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

# Reporter's captured paths and log output from unslothai/unsloth#5941,
# captured verbatim into .claude/pr-sweep/UNS-PR-TARGET-5941-REPRO.txt.
# Embedded here (rather than read from that bundle path at test time) so this
# regression test stays self-contained and runs the same way in this repo's
# own CI, which does not have the local PR-sweep bundle checked out.
_REPRO_FIXTURE_TEXT = r"""
Reporter's captured paths and log output from https://github.com/unslothai/unsloth/issues/5941:

Studio was resolving LLAMA_SERVER_PATH to:
C:\Users\artig\.unsloth\llama_cpp_fixed_b9016\build\bin\Release\llama-server.exe
which was the CPU-only build.

A separate CUDA build at:
C:\Users\artig\.unsloth\llama_cpp_fixed_b9016\build-cuda\bin\Release\llama-server.exe
worked correctly and reported:
CUDA0: NVIDIA GeForce RTX 5070

Studio inference logs showed:
warning: no usable GPU found, --gpu-layers option will be ignored
warning: one possible reason is that llama.cpp was compiled without GPU support
"""

_BINARY_NAME = "llama-server.exe" if sys.platform == "win32" else "llama-server"


def _parse_repro_build_dirs(text: str) -> list[str]:
    """Pull the ``build``/``build-cuda`` directory names out of the reporter's
    captured paths rather than hardcoding them, so the test genuinely derives
    its scenario from the fixture text."""
    pattern = re.compile(r"([\w-]+)\\bin\\Release\\llama-server\.exe")
    return pattern.findall(text)


def _make_binary(root: Path, *parts: str) -> Path:
    p = root.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


@pytest.fixture(autouse=True)
def _clean_llama_env(monkeypatch):
    # Both env vars can short-circuit discovery before it ever reaches
    # _layout_candidates; keep every test isolated from the host environment.
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising=False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising=False)


class TestBuildCudaLayoutPreference:
    def test_prefers_build_cuda_when_nvidia_available(self, tmp_path, monkeypatch):
        # Both a CPU-only build/ and a CUDA build-cuda/ tree exist side by
        # side; with an NVIDIA GPU present the CUDA one must win.
        monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(lambda: True))
        cuda_bin = _make_binary(tmp_path, "build-cuda", "bin", _BINARY_NAME)
        _make_binary(tmp_path, "build", "bin", _BINARY_NAME)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))

        result = LlamaCppBackend._find_llama_server_binary()

        assert result == str(cuda_bin)

    def test_find_llama_server_binary_prefers_cuda_build(self, tmp_path, monkeypatch):
        # Reproduction: base-fails/head-passes. Against the pre-fix
        # _layout_candidates (no build-cuda awareness), this resolves the
        # CPU-only build/ binary and reproduces the reporter's "no usable GPU
        # found" symptom; against the fix it resolves build-cuda/.
        build_dirs = _parse_repro_build_dirs(_REPRO_FIXTURE_TEXT)
        assert build_dirs == ["build", "build-cuda"], (
            f"fixture parse drifted from unslothai/unsloth#5941 repro text: {build_dirs}"
        )
        assert "no usable GPU found" in _REPRO_FIXTURE_TEXT

        # The reporter's paths carry a Windows-only build/bin/Release/ shape;
        # on other platforms _layout_candidates never probes a Release
        # subdir, so mirror the reporter's tree only where it is reachable.
        release_parts = ("Release",) if sys.platform == "win32" else ()

        monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(lambda: True))
        _make_binary(tmp_path, "build", "bin", *release_parts, _BINARY_NAME)
        cuda_bin = _make_binary(tmp_path, "build-cuda", "bin", *release_parts, _BINARY_NAME)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))

        result = LlamaCppBackend._find_llama_server_binary()

        assert result == str(cuda_bin), (
            "expected the CUDA build to win once an NVIDIA GPU is detected, "
            f"mirroring the fix for unslothai/unsloth#5941; got {result}"
        )

    def test_does_not_prefer_cuda_without_nvidia(self, tmp_path, monkeypatch):
        # Negative space: build-cuda/ exists (possibly stale from an older
        # driver-less build) but no NVIDIA GPU is detected; must still
        # resolve the CPU-only build/ binary, unchanged from today.
        monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(lambda: False))
        cpu_bin = _make_binary(tmp_path, "build", "bin", _BINARY_NAME)
        _make_binary(tmp_path, "build-cuda", "bin", _BINARY_NAME)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))

        result = LlamaCppBackend._find_llama_server_binary()

        assert result == str(cpu_bin)

    def test_single_build_tree_unaffected(self, tmp_path, monkeypatch):
        # Only a single build/ tree exists (the common case: no build-cuda/
        # dir at all). Preferring CUDA when an NVIDIA GPU is present must not
        # change this outcome -- the missing build-cuda candidates are simply
        # skipped and resolution falls through to build/ as before.
        monkeypatch.setattr(LlamaCppBackend, "_nvidia_available", staticmethod(lambda: True))
        cpu_bin = _make_binary(tmp_path, "build", "bin", _BINARY_NAME)
        monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(tmp_path))

        result = LlamaCppBackend._find_llama_server_binary()

        assert result == str(cpu_bin)

    def test_nvidia_available_calls_get_gpu_memory(self, monkeypatch):
        # _nvidia_available must delegate to the existing _get_gpu_memory
        # probe rather than reimplementing GPU detection.
        calls = []

        def _fake_get_gpu_memory():
            calls.append(1)
            return [(0, 1024, 2048)]

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_fake_get_gpu_memory))
        assert LlamaCppBackend._nvidia_available() is True
        assert len(calls) == 1

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(lambda: []))
        assert LlamaCppBackend._nvidia_available() is False
