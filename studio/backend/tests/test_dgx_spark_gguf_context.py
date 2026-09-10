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


def _spark_gpu_memory(
    monkeypatch, driver_free_mib, available_mib, total_mib = 124609, cgroup_mib = None
):
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
    # The cgroup probe is a SECOND, independent read of the host, so leaving it live
    # would let the machine running the suite decide the answer: under a container with
    # a memory.max below the mocked pool these cases fail while claiming to be hermetic.
    # Stubbed to the value the case is about, None (unconstrained) unless it says.
    monkeypatch.setattr(
        LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: cgroup_mib)
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




def test_gguf_fit_is_bounded_by_an_enforcing_cgroup(monkeypatch):
    """A container's limit is a ceiling, not a floor.

    ``_available_system_memory_mib`` already caps host MemAvailable by the cgroup
    remainder, but reading it only as a lower bound throws that away whenever the
    driver's host-wide MemFree is larger, which is the normal case in a container.
    Host-backed GPU allocations are charged to the cgroup here, so a fit sized above it
    is killed at memory.max.
    """
    gpus = _spark_gpu_memory(
        monkeypatch, driver_free_mib = 102400, available_mib = 16384, cgroup_mib = 16384
    )

    assert gpus[0][1] == 16384 - 1024


def test_an_unconstrained_host_is_not_capped(monkeypatch):
    """No cgroup limit means no ceiling: the credited pool stands."""
    gpus = _spark_gpu_memory(
        monkeypatch, driver_free_mib = 29509, available_mib = 118451, cgroup_mib = None
    )

    assert gpus[0][1] == 118451 - 1024


def test_the_unified_preflight_reaches_an_integrated_cuda_soc(monkeypatch):
    """The oversize-load guard was AMD-only, so a Spark's pool read as dedicated VRAM.

    ``_shared_gpu_ids`` is populated for Vulkan alone, so without this the downstream
    guard credits the SoC's reported pool against the weights, finds no spill, and the
    unmapped oversize load that would have been remapped is not.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "torch", _spark_torch(29509, 124609))
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)

    assert LlamaCppBackend._integrated_cuda_unified_memory(None) is True
    assert LlamaCppBackend._integrated_cuda_unified_memory([0]) is True
    # 180 GiB of weights against 118 GiB of pool: the message the preflight now reaches.
    message = LlamaCppBackend._apu_ram_shortfall_message(
        180 * GIB, 118 * 1024, part = "SoC"
    )
    assert message is not None
    assert "unified-memory SoC" in message
    # A Spark is aarch64 Linux and a Jetson is not a PC: neither runs under WSL.
    assert ".wslconfig" not in message


def test_the_apu_message_is_unchanged(monkeypatch):
    """The AMD wording and its WSL hint are what they were."""
    from core.inference.llama_cpp import LlamaCppBackend

    message = LlamaCppBackend._apu_ram_shortfall_message(64 * GIB, 46 * 1024)

    assert "unified-memory APU" in message
    assert ".wslconfig" in message


def test_a_discrete_cuda_host_reaches_no_unified_preflight(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    import sys as _sys

    module = _spark_torch(29509, 81559)
    module.cuda.get_device_properties = lambda ordinal: _DiscreteProps()
    monkeypatch.setitem(_sys.modules, "torch", module)
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)

    assert LlamaCppBackend._integrated_cuda_unified_memory(None) is False


def test_the_preflight_probe_is_skipped_on_a_discrete_x86_host(monkeypatch):
    """The gate that keeps an ordinary NVIDIA load from paying for a CUDA context.

    ``_integrated_cuda_gpu_ids`` calls ``get_device_properties`` on every visible card,
    which initialises CUDA and pins a primary context per device in this long-lived
    process. The launch preflight runs after the VRAM budget was taken, so a probe there
    can OOM a tightly fitted child on a host whose answer is False regardless.
    """
    import platform as _platform
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    torch_module = _spark_torch(29509, 81559)
    torch_module.cuda.is_initialized = lambda: False

    def _refuse(ordinal):
        raise AssertionError("the preflight touched a device on an x86 discrete host")

    torch_module.cuda.get_device_properties = _refuse
    monkeypatch.setitem(_sys.modules, "torch", torch_module)
    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")

    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def test_the_preflight_probe_runs_on_arm_and_once_cuda_is_up(monkeypatch):
    """ARM is where an integrated part can exist, and an initialised CUDA is free."""
    import platform as _platform
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    torch_module = _spark_torch(29509, 124609)
    torch_module.cuda.is_initialized = lambda: False
    monkeypatch.setitem(_sys.modules, "torch", torch_module)

    monkeypatch.setattr(_platform, "machine", lambda: "aarch64")
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is True

    # Already paid for: the context exists, so the reading costs nothing new.
    monkeypatch.setattr(_platform, "machine", lambda: "x86_64")
    torch_module.cuda.is_initialized = lambda: True
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is True


def test_repricing_keeps_the_soc_wording(monkeypatch):
    """A text-only retry must not turn a Spark's notice into a .wslconfig hint.

    The repriced message is rebuilt from scratch after the CPU-pinned projector is
    dropped, so the hardware kind has to travel with it.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    original = LlamaCppBackend._apu_ram_shortfall_message(
        200 * GIB, 118 * 1024, part = "SoC"
    )
    backend._last_load_warning = original

    backend._reprice_after_dropping_pinned_projector(
        apu_msg = original,
        host_msg = None,
        model_size = 180 * GIB,
        pinned_bytes = 20 * GIB,
        avail_mib = 118 * 1024,
        part = "SoC",
    )

    assert backend._last_load_warning is not None
    assert "unified-memory SoC" in backend._last_load_warning
    assert ".wslconfig" not in backend._last_load_warning


def test_repricing_still_says_apu_for_an_apu():
    """The AMD path keeps the wording and the WSL hint it has always had."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    original = LlamaCppBackend._apu_ram_shortfall_message(64 * GIB, 46 * 1024)
    backend._last_load_warning = original

    backend._reprice_after_dropping_pinned_projector(
        apu_msg = original,
        host_msg = None,
        model_size = 60 * GIB,
        pinned_bytes = 4 * GIB,
        avail_mib = 46 * 1024,
    )

    assert "unified-memory APU" in backend._last_load_warning
    assert ".wslconfig" in backend._last_load_warning
