# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A Jetson whose nvidia-smi answers "[N/A]" for memory and whose torch cannot see the GPU is
priced from system RAM for llama-server, so the context is still capped (#4862)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from core.inference.llama_cpp import LlamaCppBackend, _apply_igpu_host_reserve_mib
from utils import tegra
from utils.hardware import gpu_query

_spec = importlib.util.spec_from_file_location(
    "_auto_offload_matrix_for_tegra",
    Path(__file__).resolve().parent / "test_auto_offload_ctx_platform_matrix.py",
)
M = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = M  # dataclasses resolves annotations through sys.modules
_spec.loader.exec_module(M)

TOTAL_MIB = 62_841
AVAIL_MIB = 58_000
NATIVE_CTX = 262_144


def _probe(
    monkeypatch,
    *,
    jetson = True,
    smi_free = "[N/A]",
    for_llama_server = True,
    cgroup = None,
):
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b = None: False))
    monkeypatch.setattr(
        gpu_query,
        "run_nvidia_smi",
        lambda argv, **_kw: subprocess.CompletedProcess(
            argv, 0, f"0, {smi_free}, {smi_free}\n", ""
        ),
    )
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_nvml", staticmethod(lambda: []))
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *_a, **_k: [])
    )
    # A generic aarch64 torch wheel cannot drive the Tegra GPU.
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(tegra, "is_tegra", lambda: jetson)
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: AVAIL_MIB)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_total_system_memory_mib", staticmethod(lambda: TOTAL_MIB)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: cgroup)
    )
    return LlamaCppBackend._get_gpu_memory("/fake/llama-server", for_llama_server = for_llama_server)


def test_jetson_na_is_priced_from_system_ram(monkeypatch):
    assert _probe(monkeypatch) == [(0, _apply_igpu_host_reserve_mib(AVAIL_MIB, True), TOTAL_MIB)]
    # One shared pool: it must not also be credited as discrete VRAM.
    assert LlamaCppBackend._integrated_cuda_unified_memory([0])


def test_cgroup_bound_jetson_publishes_no_total(monkeypatch):
    assert _probe(monkeypatch, cgroup = 30_000) == [
        (0, _apply_igpu_host_reserve_mib(AVAIL_MIB, True), 0)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jetson": False},  # a non-Jetson [N/A] row (MIG / vGPU) keeps today's answer
        {"for_llama_server": False},  # torch-backed callers cannot use this GPU
    ],
)
def test_other_hosts_and_callers_are_unchanged(monkeypatch, kwargs):
    assert _probe(monkeypatch, **kwargs) == []
    assert not LlamaCppBackend._integrated_cuda_unified_memory([0])


def test_jetson_na_caps_native_context(tmp_path, monkeypatch):
    rows = _probe(monkeypatch)
    monkeypatch.undo()
    linux = next(p for p in M.PLATFORMS if p[0] == "linux")
    backend, gguf = M.cell_backend(
        tmp_path,
        monkeypatch,
        linux,
        M.Accelerator("jetson", False, tuple(rows)),
        native_ctx = NATIVE_CTX,
    )
    backend._get_gguf_size_bytes = lambda _p: 16 << 30
    backend._estimate_kv_cache_bytes = lambda ctx, *_a, **_k: int(ctx * 0.25 * M.MIB)
    ctx = int(M._flag(M._launch(backend, gguf, n_ctx = 0)["cmd"], "-c", "--ctx-size"))
    # 16 GiB of weights plus 64 GiB of KV at native cannot fit a 64 GB pool.
    assert 0 < ctx < NATIVE_CTX
