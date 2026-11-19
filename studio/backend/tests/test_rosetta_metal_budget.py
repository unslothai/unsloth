# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An x86_64 Python under Rosetta runs the native arm64 Metal llama.cpp, so it gets the Metal budget."""

from __future__ import annotations

import importlib.util
import platform
import sys
import types
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_placement_harness_rosetta", _TESTS_DIR / "test_llama_cpp_placement.py"
)
_placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placement)

from core.inference.llama_cpp import (  # noqa: E402
    _APPLE_UNIFIED_MEMORY_FRACTION,
    LlamaCppBackend,
    _metal_capable_host,
)
from utils.hardware import hardware as hw  # noqa: E402

GIB = 1024**3
MIB = 1024**2


def _mac(
    monkeypatch,
    machine,
    rosetta,
    working_set = int(10.67 * GIB),
):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: machine)
    monkeypatch.setattr(hw, "_ROSETTA", rosetta)
    monkeypatch.setattr(hw, "_metal_recommended_working_set_bytes", lambda: working_set)
    fake = types.ModuleType("psutil")
    vm = types.SimpleNamespace(total = 16 * GIB, available = 12 * GIB)
    fake.virtual_memory = lambda: vm
    monkeypatch.setitem(sys.modules, "psutil", fake)


@pytest.mark.usefixtures("real_apple_metal_budget")
@pytest.mark.parametrize(
    "machine, rosetta, metal",
    [("arm64", False, True), ("x86_64", True, True), ("x86_64", False, False)],
    ids = ["native", "rosetta", "intel"],
)
def test_the_metal_gate_counts_rosetta(monkeypatch, machine, rosetta, metal):
    _mac(monkeypatch, machine, rosetta)
    assert _metal_capable_host() is metal
    assert (LlamaCppBackend._apple_metal_memory_budget_bytes() > 0) is metal


@pytest.mark.usefixtures("real_apple_metal_budget")
def test_rosetta_without_a_working_set_probe_stays_inside_metal(monkeypatch):
    _mac(monkeypatch, "x86_64", True, working_set = 0)
    budget = LlamaCppBackend._apple_metal_memory_budget_bytes()
    assert budget == int(16 * GIB * 2 // 3 * _APPLE_UNIFIED_MEMORY_FRACTION)


def _auto_context(tmp_path, monkeypatch, machine, rosetta) -> int:
    tmp_path.mkdir()
    _mac(monkeypatch, machine, rosetta)
    monkeypatch.setattr(
        LlamaCppBackend, "_binary_ships_no_gpu_backend", staticmethod(lambda *_a, **_kw: False)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 12 * 1024)
    )
    backend, gguf = _placement._backend(tmp_path, vulkan = False, memory = [])
    del backend._can_estimate_kv

    def read(_path):
        backend._context_length = 262144
        backend._n_layers = 36
        backend._n_kv_heads = 8
        backend._n_heads = 32
        backend._embedding_length = 2560
        backend._kv_key_length = 128
        backend._kv_value_length = 128

    backend._read_gguf_metadata = read
    backend._get_gguf_size_bytes = lambda _path: int(2.5 * GIB)
    backend._estimate_compute_buffer_bytes = lambda **_kw: 512 * MIB
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    return int(_placement._launched(_placement._launch(backend, gguf, n_ctx = 0), "-c"))


@pytest.mark.usefixtures("real_apple_metal_budget")
def test_rosetta_auto_context_is_capped_like_native(tmp_path, monkeypatch):
    native = _auto_context(tmp_path / "native", monkeypatch, "arm64", False)
    rosetta = _auto_context(tmp_path / "rosetta", monkeypatch, "x86_64", True)
    assert native < 262144
    assert rosetta == native
