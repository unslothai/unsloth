# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spark's cudaMemGetInfo counts page cache as used, so GGUF context fits can drop to the minimum."""

from __future__ import annotations

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend

GIB = 1 << 30
MIB = 1 << 20
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


@pytest.fixture(autouse = True)
def _forget_the_last_machine(monkeypatch):
    """The integrated classification is cached for the life of the process, which is
    right for one machine and wrong for a file that describes several. Each case starts
    with an empty cache so it cannot inherit the hardware the previous one invented."""
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})


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
    monkeypatch,
    driver_free_mib,
    available_mib,
    total_mib = 124609,
    cgroup_mib = None,
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
    # Stub the cgroup probe too, or the runner's own memory.max decides the answer.
    monkeypatch.setattr(
        LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: cgroup_mib)
    )
    # A Spark's nvidia-smi answers [N/A] for memory, so the probe falls through to torch.
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
    assert free_mib == 118451 - 1024
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
    monkeypatch.setattr(LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 4096))
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )

    assert LlamaCppBackend._get_gpu_memory() == [(0, 29509, 81559)]


def test_gguf_fit_is_bounded_by_an_enforcing_cgroup(monkeypatch):
    """A container's cgroup limit is a ceiling; host MemFree can exceed it, and fits above it are killed."""
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
    """Shared-pool detection must cover integrated CUDA SoCs, or their pool reads as dedicated VRAM."""
    from core.inference.llama_cpp import LlamaCppBackend

    import sys as _sys

    monkeypatch.setitem(_sys.modules, "torch", _spark_torch(29509, 124609))
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)

    assert LlamaCppBackend._integrated_cuda_unified_memory(None) is True
    assert LlamaCppBackend._integrated_cuda_unified_memory([0]) is True
    message = LlamaCppBackend._apu_ram_shortfall_message(180 * GIB, 118 * 1024, part = "SoC")
    assert message is not None
    assert "unified-memory SoC" in message
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


def test_the_preflight_never_probes_a_device_itself(monkeypatch):
    """The preflight must never probe a device itself: get_device_properties pins a CUDA context per
    card."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def test_the_memory_probe_pays_for_the_classification_up_front(monkeypatch):
    """Classify before reading free memory: on a Spark nvidia-smi reports [N/A], so torch pays the cost."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    _spark_gpu_memory(monkeypatch, driver_free_mib = 29509, available_mib = 118451)

    assert LlamaCppBackend._integrated_cuda_probe_is_free() is True
    assert LlamaCppBackend._integrated_cuda_unified_memory([0]) is True


def test_a_different_mask_is_a_different_question(monkeypatch):
    """The cache is keyed by the visibility mask, which decides which devices it is
    about. A cached answer for one mask must not be read as an answer for another."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    _spark_gpu_memory(monkeypatch, driver_free_mib = 29509, available_mib = 118451)
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is True

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def test_a_failed_probe_is_not_remembered(monkeypatch):
    """A torch that raised says nothing about the hardware, so caching its empty answer
    would make the miss permanent for the life of the process."""
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    broken = types.ModuleType("torch")
    broken.version = types.SimpleNamespace(hip = None)

    def _raise():
        raise RuntimeError("driver not loaded")

    broken.cuda = types.SimpleNamespace(is_available = _raise)
    monkeypatch.setitem(_sys.modules, "torch", broken)

    assert LlamaCppBackend._integrated_cuda_gpu_ids() == set()
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def test_repricing_keeps_the_soc_wording(monkeypatch):
    """The repriced message is rebuilt from scratch, so the SoC hardware kind must be passed along."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    original = LlamaCppBackend._apu_ram_shortfall_message(200 * GIB, 118 * 1024, part = "SoC")
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


def test_a_device_that_did_not_answer_is_not_settled(monkeypatch):
    """A probe that raised must not be cached as settled, or a retry initialises a GPU after budgeting."""
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)

    def _properties(ordinal):
        if ordinal == 1:
            raise RuntimeError("device 1 did not answer")
        return _SparkProps()

    module = _spark_torch(29509, 124609)
    module.cuda.device_count = lambda: 2
    module.cuda.get_device_properties = _properties
    monkeypatch.setitem(_sys.modules, "torch", module)

    assert LlamaCppBackend._integrated_cuda_gpu_ids() == {0}
    assert LlamaCppBackend._integrated_cuda_probe_is_free() is False


def test_a_settled_answer_is_never_probed_again(monkeypatch):
    """Integratedness cannot change under a fixed mask, so one pass is the whole cost."""
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    calls = []
    module = _spark_torch(29509, 124609)
    original = module.cuda.get_device_properties

    def _counted(ordinal):
        calls.append(ordinal)
        return original(ordinal)

    module.cuda.get_device_properties = _counted
    monkeypatch.setitem(_sys.modules, "torch", module)

    assert LlamaCppBackend._integrated_cuda_gpu_ids() == {0}
    assert len(calls) == 1
    assert LlamaCppBackend._integrated_cuda_gpu_ids() == {0}
    assert LlamaCppBackend._integrated_cuda_unified_memory([0]) is True
    assert len(calls) == 1


def test_a_cgroup_bound_row_publishes_no_total(monkeypatch):
    """A cgroup-bound row publishes no total; a host-wide total reserves memory the container cannot use."""
    gpus = _spark_gpu_memory(
        monkeypatch, driver_free_mib = 102400, available_mib = 16384, cgroup_mib = 16384
    )

    index, free_mib, total_mib = gpus[0]
    assert (index, free_mib) == (0, 16384 - 1024)
    assert total_mib == 0


def test_an_unconstrained_row_keeps_its_total(monkeypatch):
    """The pool is only the container's where a container is what bounds it."""
    gpus = _spark_gpu_memory(monkeypatch, driver_free_mib = 29509, available_mib = 118451)

    assert gpus[0][2] == 124609


def test_the_preflight_needs_every_credited_device_to_share_the_pool(monkeypatch):
    """The guard charges the WHOLE model against system RAM, and layers placed on a
    discrete card never touch the shared pool. Answering for ANY integrated device in a
    mixed selection calls an otherwise fitting load oversized."""
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    module = _spark_torch(29509, 124609)
    module.cuda.device_count = lambda: 2
    module.cuda.get_device_properties = (
        lambda ordinal: _SparkProps() if ordinal == 0 else _DiscreteProps()
    )
    monkeypatch.setitem(_sys.modules, "torch", module)

    assert LlamaCppBackend._integrated_cuda_unified_memory([0, 1]) is True
    assert LlamaCppBackend._integrated_cuda_selection_is_all_shared([0, 1]) is False
    assert LlamaCppBackend._integrated_cuda_selection_is_all_shared([0]) is True
    assert LlamaCppBackend._integrated_cuda_selection_is_all_shared(None) is False


def test_an_all_integrated_selection_still_reaches_the_preflight(monkeypatch):
    import sys as _sys

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setitem(_sys.modules, "torch", _spark_torch(29509, 124609))

    assert LlamaCppBackend._integrated_cuda_selection_is_all_shared(None) is True
    assert LlamaCppBackend._integrated_cuda_selection_is_all_shared([0]) is True


def test_an_equal_cgroup_remainder_is_still_the_ceiling(monkeypatch):
    """_available_system_memory_mib is cgroup-capped, so it hands back exactly the
    remainder whenever MemAvailable exceeds it: equality is the ordinary result on a
    constrained Spark whose driver-free reading is no larger, not a coincidence."""
    gpus = _spark_gpu_memory(
        monkeypatch, driver_free_mib = 8192, available_mib = 16384, cgroup_mib = 16384
    )

    assert gpus[0][1] == 16384 - 1024
    assert gpus[0][2] == 0


def _two_integrated_torch(free_mib: int, total_mib: int) -> types.ModuleType:
    """Two devices that both report integrated, sharing ONE host pool."""
    module = types.ModuleType("torch")
    module.version = types.SimpleNamespace(hip = None)
    module.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: 2,
        mem_get_info = lambda ordinal: (free_mib * MIB, total_mib * MIB),
        get_device_properties = lambda ordinal: _SparkProps(),
    )
    return module


def test_two_integrated_devices_do_not_each_claim_the_whole_pool(monkeypatch):
    for mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(mask, raising = False)
    monkeypatch.setitem(sys.modules, "torch", _two_integrated_torch(1590, 124609))
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 61850)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no nvidia-smi")),
    )
    gpus = LlamaCppBackend._get_gpu_memory()

    assert len(gpus) == 2
    assert sum(free for _idx, free, _total in gpus) <= 61850
    assert sum(total for _idx, _free, total in gpus) <= 124609


def test_a_single_integrated_device_is_not_divided(monkeypatch):
    gpus = _spark_gpu_memory(monkeypatch, driver_free_mib = 1590, available_mib = 61850)
    assert gpus == [(0, 61850 - 1024, 124609)]
