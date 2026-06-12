# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vulkan free-VRAM reader regression tests on a synthetic probe output.

Covers the post-probe handling in
``LlamaCppBackend._get_gpu_free_memory_vulkan``:

  * integrated GPUs (probe reports is_igpu=1) leave a flat per-device host
    margin matching llama.cpp's --fit-target, so context auto-sizing can't
    over-commit shared RAM,
  * discrete GPUs (is_igpu=0) are left untouched,
  * an inherited ``GGML_VK_VISIBLE_DEVICES`` is stripped before probing so
    enumeration stays in ggml's canonical full-device space.

The ggml Vulkan library is never loaded: subprocess.run is mocked to emit
the tab-separated lines the real ``_vulkan_probe.py`` would print.
"""

from __future__ import annotations

import subprocess
import sys
import types as _types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

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


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", lambda: _types.ModuleType("structlog"))

from core.inference import llama_cpp as _llama_mod  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend, _vulkan_lib_filename  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB


def _make_vulkan_install(tmp_path: Path) -> str:
    """A binary whose sibling dir holds the Vulkan ggml lib, so the
    reader's ``is_vulkan_backend`` sibling-file check passes."""
    bindir = tmp_path / "build" / "bin"
    bindir.mkdir(parents = True)
    binary = bindir / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
    binary.write_bytes(b"stub")
    (bindir / _vulkan_lib_filename()).write_bytes(b"stub")
    return str(binary)


def _mock_probe(rows: list[str], captured_env: dict | None = None):
    """Patch subprocess.run so the _vulkan_probe.py call returns ``rows``
    (already tab-formatted), recording the env it was launched with."""
    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if isinstance(cmd, list) and any("_vulkan_probe" in str(c) for c in cmd):
            if captured_env is not None:
                captured_env.clear()
                captured_env.update(kwargs.get("env") or {})
            return subprocess.CompletedProcess(
                args = cmd, returncode = 0, stdout = "\n".join(rows), stderr = ""
            )
        return real_run(cmd, *args, **kwargs)

    return mock.patch("subprocess.run", side_effect = fake_run)


def _row(idx: int, free_bytes: int, is_igpu: int) -> str:
    return f"{idx}\t{free_bytes}\t{is_igpu}"


def test_integrated_gpu_leaves_host_margin(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # iGPU with 30 GiB free; reserve a flat 1024 MiB (llama.cpp --fit-target).
    rows = [_row(0, 30 * GIB, is_igpu = 1)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 30 * 1024 - 1024)], gpus


def test_discrete_gpu_free_is_untouched(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    rows = [_row(0, 23 * GIB, is_igpu = 0)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 23 * 1024)], gpus


def test_large_discrete_gpu_is_untouched(tmp_path):
    binary = _make_vulkan_install(tmp_path)
    # A 48 GiB discrete card stays untouched regardless of size; only the
    # iGPU flag triggers the host margin, never a VRAM/RAM ratio.
    rows = [_row(0, 47 * GIB, is_igpu = 0)]
    with _mock_probe(rows):
        gpus = LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert gpus == [(0, 47 * 1024)], gpus


def test_inherited_visible_devices_mask_is_stripped(tmp_path, monkeypatch):
    binary = _make_vulkan_install(tmp_path)
    monkeypatch.setenv("GGML_VK_VISIBLE_DEVICES", "1")
    captured: dict = {}
    rows = [_row(0, 23 * GIB, is_igpu = 0)]
    with _mock_probe(rows, captured_env = captured):
        LlamaCppBackend._get_gpu_free_memory_vulkan(binary)
    assert "GGML_VK_VISIBLE_DEVICES" not in captured, captured


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
