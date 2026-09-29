# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import main


def _shared_setup_1(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(main, "_system_gpu_cache", None)


def test_system_gpu_info_preserves_vulkan_visibility_metrics(monkeypatch):
    import utils.hardware as hardware

    vulkan_device = {
        "index": 0,
        "index_kind": "relative",
        "visible_ordinal": 0,
        "name": "Vulkan0",
        "memory_total_gb": 8.0,
        "vram_used_gb": 0.77,
        "vram_free_gb": 7.23,
        "vram_utilization_pct": 9.6,
        "shared_memory": False,
    }
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": False,
            "backend": "cpu",
            "devices": [],
            "index_kind": "relative",
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(
        hardware,
        "get_vulkan_inference_gpu_info",
        lambda: {
            "available": True,
            "backend": "vulkan",
            "devices": [vulkan_device],
            "index_kind": "relative",
        },
    )

    _shared_setup_1(monkeypatch)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["available"] is False
    assert gpu["backend"] == "cpu"
    assert gpu["index_kind"] == "relative"
    # The training inventory must not advertise physical pins for a Vulkan
    # llama.cpp build. Its ordinals live in inference_gpu below.
    assert gpu["gguf_gpu_ids_supported"] is False
    # Torch's view stays empty; the ggml ordinals stay in inference_gpu.
    assert gpu["devices"] == []
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"] == [vulkan_device]

    fresh_device = {**vulkan_device, "vram_free_gb": 3.0}
    monkeypatch.setattr(
        hardware,
        "get_vulkan_inference_gpu_info",
        lambda: {**inference_gpu, "devices": [fresh_device]},
    )
    logger = SimpleNamespace(debug = lambda *args: None)
    assert main._get_cached_system_gpu_info(logger)[1]["devices"] == [vulkan_device]
    refreshed = main._get_cached_system_gpu_info(logger, refresh_memory = True)
    assert refreshed[1]["devices"] == [fresh_device]
    assert main._get_cached_system_gpu_info(logger) is refreshed


def test_system_gpu_info_withholds_gguf_pin_when_the_vulkan_probe_enumerates_nothing(monkeypatch):
    """A Vulkan build whose probe returns no ordinals has nothing valid to pin,
    so the picker must be told pins are unsupported rather than offered an empty
    namespace it would 400 on."""
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {"available": False, "backend": "cpu", "devices": [], "index_kind": "relative"},
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)

    _shared_setup_1(monkeypatch)

    gpu, _ = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["gguf_gpu_ids_supported"] is False


def test_system_gpu_info_keeps_forced_vulkan_separate_from_training_metrics(monkeypatch):
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [{"index": 0, "name": "CUDA0", "memory_total_gb": 24.0}],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "vram_total_gb": 24.0,
                    "vram_used_gb": 6.0,
                    "vram_utilization_pct": 25.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_vulkan_inference_gpu_info",
        lambda: {
            "available": True,
            "backend": "vulkan",
            "devices": [
                {
                    "index": 0,
                    "name": "Vulkan0",
                    "memory_total_gb": 8.0,
                    "vram_used_gb": 1.0,
                    "vram_free_gb": 7.0,
                    "vram_utilization_pct": 12.5,
                    "shared_memory": False,
                }
            ],
            "index_kind": "relative",
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware import DeviceType

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(hardware, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["backend"] == "cuda"
    assert gpu["devices"][0]["vram_used_gb"] == 6.0
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"][0]["vram_used_gb"] == 1.0
    # Probed devices exist, so the ordinals are known and picks are offered.
    assert inference_gpu["gguf_gpu_ids_supported"] is True


def test_system_gpu_info_does_not_merge_metrics_across_backend_index_spaces(monkeypatch):
    import utils.hardware as hardware

    vulkan_device = {
        "index": 0,
        "name": "Vulkan0",
        "memory_total_gb": 8.0,
        "vram_used_gb": 1.0,
        "vram_free_gb": 7.0,
        "vram_utilization_pct": 12.5,
    }
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {"available": True, "backend": "vulkan", "devices": [vulkan_device]},
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "vram_total_gb": 24.0,
                    "vram_used_gb": 20.0,
                    "vram_utilization_pct": 83.3,
                }
            ],
        },
    )

    _shared_setup_1(monkeypatch)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["devices"] == [vulkan_device]
    assert inference_gpu == gpu


def test_vulkan_inference_gpu_uses_real_device_names_and_igpu_flag(monkeypatch):
    """The picker and the GPU labels need ggml's real device description, not a
    Vulkan<i> placeholder, and an explicit iGPU flag rather than inferring one
    from a zero total. Memory still comes from _get_gpu_memory so the iGPU host
    reserve is applied; budgeting off the raw shared total would hand out the
    whole machine's RAM with no OS headroom.
    """
    from core.inference import llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware.hardware import get_vulkan_inference_gpu_info

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
    )
    monkeypatch.setattr(
        llama_cpp,
        "_apply_igpu_host_reserve_mib",
        lambda free_mib, is_igpu: 12 * 1024 if is_igpu else free_mib,
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "vulkan_device_inventory",
        staticmethod(
            lambda binary = None: [
                {
                    "index": 0,
                    "name": "AMD Radeon RX 9070 XT",
                    "free_mib": 15 * 1024,
                    "total_mib": 16 * 1024,
                    "is_igpu": False,
                },
                {
                    "index": 1,
                    "name": "AMD Radeon(TM) 8060S Graphics",
                    "free_mib": 89 * 1024,
                    "total_mib": 91 * 1024,
                    "is_igpu": True,
                },
            ]
        ),
    )

    info = get_vulkan_inference_gpu_info()
    assert info is not None and info["index_kind"] == "vulkan"
    dgpu, igpu = info["devices"]

    assert dgpu["name"] == "AMD Radeon RX 9070 XT"
    assert dgpu["index_kind"] == "vulkan"
    assert dgpu["shared_memory"] is False
    assert dgpu["memory_total_gb"] == 16.0

    assert igpu["name"] == "AMD Radeon(TM) 8060S Graphics"
    assert igpu["shared_memory"] is True
    # The capped free budget from _get_gpu_memory, NOT the 91 GiB raw total.
    assert igpu["memory_total_gb"] == 12.0


def test_vulkan_inference_gpu_uses_inventory_fallback_names(monkeypatch):
    """The inventory's fallback name must flow through unchanged."""
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware.hardware import get_vulkan_inference_gpu_info

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "vulkan_device_inventory",
        staticmethod(
            lambda binary = None: [
                {
                    "index": 0,
                    "name": "Vulkan0",
                    "free_mib": 15 * 1024,
                    "total_mib": 16 * 1024,
                    "is_igpu": False,
                }
            ]
        ),
    )

    info = get_vulkan_inference_gpu_info()
    assert info["devices"][0]["name"] == "Vulkan0"
    assert info["devices"][0]["memory_total_gb"] == 16.0


def test_system_gpu_info_keeps_a_reported_free_over_total_minus_used(monkeypatch):
    """Apple unified memory reports free directly because it is not total - used.
    Recomputing it here put the overstated figure back on the Resources tab while
    /api/system/hardware served the honest one."""
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "mlx",
            "index_kind": "relative",
            "devices": [
                {
                    "index": 0,
                    "index_kind": "relative",
                    "visible_ordinal": 0,
                    "name": "Apple Silicon (Apple M2)",
                    "memory_total_gb": 16.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "mlx",
            "index_kind": "relative",
            "parent_visible_gpu_ids": [0],
            "devices": [
                {
                    "index": 0,
                    "vram_total_gb": 16.0,
                    "vram_used_gb": 1.2,
                    "vram_free_gb": 6.0,
                    "vram_utilization_pct": 7.5,
                }
            ],
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, _inference_gpu = main._get_cached_system_gpu_info(
        SimpleNamespace(debug = lambda *args: None)
    )

    assert gpu["devices"][0]["vram_free_gb"] == 6.0


def test_system_gpu_info_still_derives_free_when_the_probe_reports_none(monkeypatch):
    """CUDA's utilization probe reports no free, so the subtraction has to stay."""
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "cuda",
            "index_kind": "relative",
            "devices": [
                {
                    "index": 0,
                    "index_kind": "relative",
                    "visible_ordinal": 0,
                    "name": "NVIDIA GeForce RTX 4090",
                    "memory_total_gb": 24.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "cuda",
            "index_kind": "relative",
            "parent_visible_gpu_ids": [0],
            "devices": [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 6.0}],
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, _inference_gpu = main._get_cached_system_gpu_info(
        SimpleNamespace(debug = lambda *args: None)
    )

    assert gpu["devices"][0]["vram_free_gb"] == 18.0


def _refuse(*args, **kwargs):
    raise AssertionError("this vendor's reader must not run")


def _mixed_host(monkeypatch, *, torch_rocm, llama_backend):
    """Stubs a host whose torch and llama.cpp are pinned by the test, never read off the runner."""
    import utils.hardware as hardware
    import utils.hardware.hardware as hw
    from core.inference.llama_cpp import LlamaCppBackend

    torch_backend = "rocm" if torch_rocm else "cuda"
    torch_card = "AMD Radeon AI PRO R9700" if torch_rocm else "NVIDIA GeForce RTX 3080"
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": torch_backend,
            "index_kind": "physical",
            "devices": [
                {"index": 0, "index_kind": "physical", "name": torch_card, "memory_total_gb": 32.0}
            ],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": True, "backend": torch_backend, "devices": []},
    )
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", torch_rocm)
    monkeypatch.setattr(hw, "_installed_llama_backend", lambda: llama_backend)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda binary = None: False)
    )
    monkeypatch.setattr(main, "_system_gpu_cache", None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)


def test_rocm_torch_with_a_cuda_llama_cpp_reports_the_nvidia_card(monkeypatch):
    """RTX 3080 beside an R9700, ROCm torch, CUDA llama.cpp: only the AMD card was listed."""
    import utils.hardware.amd as amd
    import utils.hardware.nvidia as nvidia

    _mixed_host(monkeypatch, torch_rocm = True, llama_backend = "cuda")
    monkeypatch.setattr(
        nvidia,
        "get_physical_gpu_inventory",
        lambda: {
            "available": True,
            "devices": [
                {
                    "vendor": "nvidia",
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3080",
                    "memory_total_gb": 10.0,
                }
            ],
        },
    )
    asked = []

    def _usage(parent_visible_ids, parent_cuda_visible_devices = None):
        asked.append(parent_visible_ids)
        return {"devices": [{"index": 0, "vram_used_gb": 7.88, "vram_total_gb": 10.0}]}

    monkeypatch.setattr(nvidia, "get_visible_gpu_utilization", _usage)
    monkeypatch.setattr(amd, "get_gpu_vram_report", _refuse)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["backend"] == "rocm"
    assert inference_gpu is not gpu
    assert inference_gpu["backend"] == "cuda"
    assert inference_gpu["available"] is True
    assert inference_gpu["index_kind"] == "physical"
    # nvidia-smi row numbers are not the ordinals a pin is applied in.
    assert inference_gpu["gguf_gpu_ids_supported"] is False
    assert asked == [[0]]
    (card,) = inference_gpu["devices"]
    assert card["name"] == "NVIDIA GeForce RTX 3080"
    assert card["index"] == 0
    assert card["memory_total_gb"] == 10.0
    assert card["vram_used_gb"] == 7.88
    assert card["vram_free_gb"] == 2.12
    assert card["vram_utilization_pct"] == 78.8


def test_rocm_torch_with_a_rocm_llama_cpp_keeps_the_training_inventory(monkeypatch):
    import utils.hardware.amd as amd
    import utils.hardware.nvidia as nvidia

    _mixed_host(monkeypatch, torch_rocm = True, llama_backend = "rocm")
    monkeypatch.setattr(nvidia, "get_physical_gpu_inventory", _refuse)
    monkeypatch.setattr(nvidia, "get_visible_gpu_utilization", _refuse)
    monkeypatch.setattr(amd, "get_gpu_vram_report", _refuse)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert inference_gpu is gpu
    assert inference_gpu["backend"] == "rocm"


def test_cuda_torch_with_a_rocm_llama_cpp_reports_the_amd_card(monkeypatch):
    import utils.hardware.amd as amd
    import utils.hardware.nvidia as nvidia

    _mixed_host(monkeypatch, torch_rocm = False, llama_backend = "rocm")
    monkeypatch.setattr(nvidia, "get_physical_gpu_inventory", _refuse)
    # {amd-smi id: (free MiB, total MiB)}, plus every id the call enumerated.
    monkeypatch.setattr(amd, "get_gpu_vram_report", lambda: ({0: (8192, 32768)}, [0]))

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["backend"] == "cuda"
    assert inference_gpu["backend"] == "rocm"
    (card,) = inference_gpu["devices"]
    assert card["memory_total_gb"] == 32.0
    assert card["vram_used_gb"] == 24.0
    assert card["vram_free_gb"] == 8.0


def test_a_silent_nvidia_smi_falls_back_to_the_training_inventory(monkeypatch):
    """An empty cross-vendor list would read as "this host has no GPU" to the load estimate."""
    import utils.hardware.nvidia as nvidia

    _mixed_host(monkeypatch, torch_rocm = True, llama_backend = "cuda")
    monkeypatch.setattr(
        nvidia, "get_physical_gpu_inventory", lambda: {"available": False, "devices": []}
    )
    monkeypatch.setattr(nvidia, "get_visible_gpu_utilization", _refuse)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert inference_gpu is gpu


def test_a_vulkan_llama_cpp_never_asks_the_cross_vendor_probe(monkeypatch):
    """CUDA torch with a Vulkan llama.cpp keeps the Vulkan inventory exactly as before."""
    import utils.hardware as hardware

    vulkan_info = {
        "available": True,
        "backend": "vulkan",
        "devices": [
            {"index": 0, "index_kind": "vulkan", "name": "Vulkan0", "memory_total_gb": 8.0}
        ],
        "index_kind": "vulkan",
    }
    _mixed_host(monkeypatch, torch_rocm = False, llama_backend = "vulkan")
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: vulkan_info)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", _refuse)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["backend"] == "cuda"
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"] == vulkan_info["devices"]
    assert inference_gpu["gguf_gpu_ids_supported"] is True


def _two_nvidia_cards(monkeypatch, names_and_totals):
    import utils.hardware.amd as amd
    import utils.hardware.nvidia as nvidia

    monkeypatch.setattr(
        nvidia,
        "get_physical_gpu_inventory",
        lambda: {
            "available": True,
            "devices": [
                {"vendor": "nvidia", "index": i, "name": name, "memory_total_gb": total}
                for i, (name, total) in enumerate(names_and_totals)
            ],
        },
    )
    asked = []

    def _usage(parent_visible_ids, parent_cuda_visible_devices = None):
        asked.append(parent_visible_ids)
        return {"devices": [{"index": i, "vram_used_gb": 1.0} for i in parent_visible_ids]}

    monkeypatch.setattr(nvidia, "get_visible_gpu_utilization", _usage)
    monkeypatch.setattr(amd, "get_gpu_vram_report", _refuse)
    return asked


def test_cross_vendor_nvidia_cards_follow_cuda_visible_devices(monkeypatch):
    _mixed_host(monkeypatch, torch_rocm = True, llama_backend = "cuda")
    asked = _two_nvidia_cards(monkeypatch, [("RTX 3080", 10.0), ("RTX 4090", 24.0)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")

    _, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert [d["name"] for d in inference_gpu["devices"]] == ["RTX 4090"]
    assert asked == [[1]]


def test_capacity_less_nvidia_rows_fall_back_to_the_training_inventory(monkeypatch):
    """procfs placeholder rows (nvidia-smi failed, driver loaded) carry no capacity."""
    _mixed_host(monkeypatch, torch_rocm = True, llama_backend = "cuda")
    _two_nvidia_cards(monkeypatch, [(None, None), (None, None)])

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert inference_gpu is gpu
