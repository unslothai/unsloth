# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto context on a machine with no GPU fits host RAM instead of the model's native window."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_placement_harness_cpu_auto_ctx", _TESTS_DIR / "test_llama_cpp_placement.py"
)
_placement = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_placement)

from core.inference.llama_cpp import _FIT_MIN_CTX, _HOST_RAM_HEADROOM_MIB, LlamaCppBackend  # noqa: E402

GIB = 1024**3
MIB = 1024**2
WEIGHTS = int(2.5 * GIB)
COMPUTE = 512 * MIB
NATIVE = 262144


def _qwen3_4b_header(backend, native):
    def read(_path):
        backend._context_length = native
        backend._n_layers = 36
        backend._n_kv_heads = 8
        backend._n_heads = 32
        backend._embedding_length = 2560
        backend._kv_key_length = 128
        backend._kv_value_length = 128

    return read


def _cpu_load(
    tmp_path,
    monkeypatch,
    *,
    available_mib,
    native = NATIVE,
    cpu_only_build = True,
    **load_kwargs,
):
    monkeypatch.setattr(
        LlamaCppBackend,
        "_binary_ships_no_gpu_backend",
        staticmethod(lambda *_a, **_kw: cpu_only_build),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: available_mib)
    )
    backend, gguf = _placement._backend(tmp_path, vulkan = False, memory = [])
    del backend._can_estimate_kv
    backend._read_gguf_metadata = _qwen3_4b_header(backend, native)
    backend._get_gguf_size_bytes = lambda _path: WEIGHTS
    backend._estimate_compute_buffer_bytes = lambda **_kw: COMPUTE
    backend._compute_buffer_ctx_bytes = lambda *_a, **_kw: 0
    result = _placement._launch(backend, gguf, **load_kwargs)
    return backend, int(_placement._launched(result, "-c"))


def test_auto_on_a_roomy_cpu_host_does_not_reserve_the_whole_native_window(tmp_path, monkeypatch):
    _backend, ctx = _cpu_load(tmp_path, monkeypatch, available_mib = 64 * 1024, n_ctx = 0)
    assert ctx == 65536


def test_auto_on_a_small_cpu_host_fits_the_kv_cache_in_free_ram(tmp_path, monkeypatch):
    available_mib = 8 * 1024
    backend, ctx = _cpu_load(tmp_path, monkeypatch, available_mib = available_mib, n_ctx = 0)
    assert _FIT_MIN_CTX <= ctx < 65536
    need = WEIGHTS + COMPUTE + backend._estimate_kv_cache_bytes(ctx)
    assert need <= (available_mib - _HOST_RAM_HEADROOM_MIB) * MIB


def test_a_shorter_native_window_keeps_its_own_length(tmp_path, monkeypatch):
    _backend, ctx = _cpu_load(tmp_path, monkeypatch, available_mib = 64 * 1024, native = 40960, n_ctx = 0)
    assert ctx == 40960


@pytest.mark.parametrize("available_mib", [8 * 1024, 64 * 1024])
def test_an_explicit_context_is_launched_as_asked(tmp_path, monkeypatch, available_mib):
    _backend, ctx = _cpu_load(tmp_path, monkeypatch, available_mib = available_mib, n_ctx = 131072)
    assert ctx == 131072


@pytest.mark.parametrize("via_env", [False, True])
def test_rpc_offload_is_not_sized_against_coordinator_ram(tmp_path, monkeypatch, via_env):
    if via_env:
        monkeypatch.setenv("LLAMA_ARG_RPC", "10.0.0.2:50052")
        extra = {}
    else:
        monkeypatch.delenv("LLAMA_ARG_RPC", raising = False)
        extra = {"extra_args": ["--rpc", "10.0.0.2:50052"]}
    _backend, ctx = _cpu_load(tmp_path, monkeypatch, available_mib = 8 * 1024, n_ctx = 0, **extra)
    assert ctx == NATIVE


def test_an_empty_gpu_probe_on_a_gpu_build_is_not_sized_against_host_ram(tmp_path, monkeypatch):
    _backend, ctx = _cpu_load(
        tmp_path, monkeypatch, available_mib = 8 * 1024, cpu_only_build = False, n_ctx = 0
    )
    assert ctx == NATIVE
