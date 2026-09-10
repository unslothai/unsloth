# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GGUF context fit on a DGX Spark must not lose the pool to the page cache.

``cudaMemGetInfo``'s free half on an integrated SoC is the kernel's ``MemFree``, which
counts the page cache as used. Measured on a GB10: writing a 60 GiB file took it from
103.25 GiB to 41.52 GiB while ``MemAvailable`` never moved off 115.6 GiB. Downloading or
mmap'ing a GGUF is that same write, so a model the machine can hold gets its context
fitted against the bytes its own weights left in cache, and a 262k-native model comes up
at ``_FIT_MIN_CTX`` (#9889).

Hermetic: torch, nvidia-smi and host memory are stubbed, so these run anywhere.
"""

from __future__ import annotations

import sys
import types

from core.inference.llama_cpp import LlamaCppBackend

GIB = 1 << 30
MIB = 1 << 20
# What a DGX Spark actually reports.
SPARK_TOTAL_BYTES = 124609 * MIB
SPARK_TOTAL_GB = round(SPARK_TOTAL_BYTES / GIB, 2)


class _SparkProps:
    """cudaDeviceProp as torch surfaces it for a GB10."""

    name = "NVIDIA GB10"
    total_memory = SPARK_TOTAL_BYTES
    is_integrated = 1
    gcnArchName = ""


class _DiscreteProps:
    name = "NVIDIA GB200"
    total_memory = 183 * GIB
    is_integrated = 0
    gcnArchName = ""
def _spark_torch(driver_free_mib: int, total_mib: int) -> types.ModuleType:
    """torch as it answers on a GB10: mem_get_info's free half is MemFree."""
    module = types.ModuleType("torch")
    module.version = types.SimpleNamespace(hip = None)
    module.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 1,
        mem_get_info = lambda ordinal: (driver_free_mib * MIB, total_mib * MIB),
        get_device_properties = lambda ordinal: _SparkProps(),
    )
    return module


def _spark_gpu_memory(monkeypatch, driver_free_mib, available_mib, total_mib = 124609):
    from core.inference.llama_cpp import LlamaCppBackend

    import sys as _sys

    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)
    monkeypatch.setitem(_sys.modules, "torch", _spark_torch(driver_free_mib, total_mib))
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: available_mib)
    )
    # nvidia-smi answers [N/A] for both memory columns on a Spark, so every row is
    # unparseable and the probe falls through to torch. Absent is the same path.
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    return LlamaCppBackend._get_gpu_memory()


def test_gguf_fit_does_not_lose_the_pool_to_the_page_cache(monkeypatch):
    """The measured case: a 60 GiB download leaves the driver reporting 29.5 GiB free."""
    gpus = _spark_gpu_memory(monkeypatch, driver_free_mib = 29509, available_mib = 118451)

    index, free_mib, total_mib = gpus[0]
    assert index == 0
    # 118451 available, less the 1 GiB integrated host reserve. Was 28485.
    assert free_mib == 118451 - 1024
    # An integrated part keeps its total: unlike a ROCm APU's, it IS the whole pool.
    assert total_mib == 124609


def test_gguf_fit_keeps_the_host_reserve(monkeypatch):
    """A genuinely full Spark is not talked up, and still gives the OS its margin."""
    gpus = _spark_gpu_memory(monkeypatch, driver_free_mib = 4096, available_mib = 4096)

    assert gpus[0][1] == 4096 - 1024


def test_gguf_fit_never_exceeds_the_pool(monkeypatch):
    """MemAvailable can exceed a masked or smaller device total; the pool is the cap."""
    gpus = _spark_gpu_memory(
        monkeypatch, driver_free_mib = 1024, available_mib = 200000, total_mib = 124609
    )

    assert gpus[0][1] == 124609 - 1024




# ── a host that is not one of these parts must be untouched ──────────────────


def test_discrete_cuda_keeps_the_whole_free_reading(monkeypatch):
    """No host reserve, no MemAvailable credit, and the driver's own total."""
    from core.inference.llama_cpp import LlamaCppBackend

    import sys as _sys

    module = _spark_torch(29509, 81559)
    module.cuda.get_device_properties = lambda ordinal: _DiscreteProps()
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)
    monkeypatch.setitem(_sys.modules, "torch", module)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 4096)
    )
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )

    # Untouched by both the 1 GiB reserve and the low host figure beside it.
    assert LlamaCppBackend._get_gpu_memory() == [(0, 29509, 81559)]


