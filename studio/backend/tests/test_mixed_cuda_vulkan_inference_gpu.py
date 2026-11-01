# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A llama.cpp built with CUDA and Vulkan also runs on the cards only Vulkan reaches (#15444): an
AMD card beside NVIDIA ones belongs in the GGUF inventory, a Vulkan view of an NVIDIA card does
not."""

import ctypes
import subprocess
from types import SimpleNamespace

import pytest

import main

GIB_MIB = 1024


def _cuda_device(
    index,
    name,
    total_gb = 7.71,
):
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": index,
        "name": name,
        "memory_total_gb": total_gb,
        "vram_used_gb": 0.3,
        "vram_free_gb": round(total_gb - 0.3, 2),
        "vram_utilization_pct": 3.9,
    }


CUDA_INFO = {
    "available": True,
    "backend": "cuda",
    "backend_cuda_visible_devices": None,
    "parent_visible_gpu_ids": [0, 1],
    "devices": [
        _cuda_device(0, "NVIDIA GeForce RTX 3070"),
        _cuda_device(1, "NVIDIA GeForce RTX 3060"),
    ],
    "index_kind": "physical",
    "gguf_gpu_ids_supported": True,
    "vram_used_gb_aggregate": None,
}


def _vulkan_row(index, name, total_mib, free_mib, pci_id):
    return {
        "index": index,
        "free_mib": free_mib,
        "is_igpu": False,
        "total_mib": total_mib,
        "name": name,
        "type_known": True,
        "pci_id": pci_id,
    }


# The reporter's llama.cpp --list-devices: Vulkan0 and Vulkan2 are the CUDA cards again.
REPORTED_ROWS = [
    _vulkan_row(0, "NVIDIA GeForce RTX 3060", 8438, 7932, "0000:a1:00.0"),
    _vulkan_row(1, "AMD Radeon Pro V620 (RADV NAVI21)", 30704, 30687, "0000:04:00.0"),
    _vulkan_row(2, "NVIDIA GeForce RTX 3070", 8438, 7778, "0000:05:00.0"),
]
# nvidia-smi rows: 0 is the 3070 at 05:00.0, 1 the 3060 at a1:00.0.
SMI_BUS_IDS = {0: (0, "05:00.0"), 1: (0, "A1:00.0")}


@pytest.fixture
def mixed_build(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware import nvidia

    state = {
        "backends": frozenset({"base", "cpu", "cuda", "vulkan"}),
        "rows": REPORTED_ROWS,
        "bus_ids": SMI_BUS_IDS,
        "probed": 0,
    }

    def inventory(binary = None):
        state["probed"] += 1
        return [dict(row) for row in state["rows"]]

    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda binary = None: state["backends"]),
    )
    monkeypatch.setattr(LlamaCppBackend, "vulkan_device_inventory", staticmethod(inventory))
    monkeypatch.setattr(nvidia, "get_pci_bus_ids", lambda: dict(state["bus_ids"]))
    return state


def _mixed(info = CUDA_INFO):
    from utils.hardware import get_mixed_vulkan_inference_gpu_info
    return get_mixed_vulkan_inference_gpu_info(info)


def test_amd_card_beside_cuda_cards_joins_the_inference_inventory(mixed_build):
    info = _mixed()

    assert info["backend"] == "cuda+vulkan"
    names = [d["name"] for d in info["devices"]]
    assert names == [
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3060",
        "AMD Radeon Pro V620 (RADV NAVI21)",
    ]
    amd = info["devices"][2]
    assert amd["index_kind"] == "vulkan"
    assert amd["index"] == 1
    assert amd["memory_total_gb"] == round(30704 / GIB_MIB, 2)
    assert amd["vram_free_gb"] == round(30687 / GIB_MIB, 2)
    # 7.71 + 7.71 + 29.98: the budget the model registry compares a GGUF against.
    assert round(sum(d["memory_total_gb"] for d in info["devices"]), 2) == 45.4
    # The CUDA rows are untouched and the input is not mutated.
    assert info["devices"][:2] == CUDA_INFO["devices"]
    assert len(CUDA_INFO["devices"]) == 2


def test_pci_ids_match_even_when_names_differ(mixed_build):
    mixed_build["rows"] = [
        _vulkan_row(0, "NVIDIA GeForce RTX 3060 (Vulkan)", 8438, 7932, "0000:a1:00.0"),
        _vulkan_row(1, "AMD Radeon Pro V620", 30704, 30687, "0000:04:00.0"),
        _vulkan_row(2, "NVIDIA RTX 3070 (Vulkan)", 8438, 7778, "0000:05:00.0"),
    ]

    info = _mixed()

    assert [d["name"] for d in info["devices"][2:]] == ["AMD Radeon Pro V620"]


def test_cuda_hidden_nvidia_card_is_still_counted_as_llama_cpp_uses_it(mixed_build):
    """llama.cpp only skips a Vulkan device a CUDA one already holds; a card CUDA_VISIBLE_DEVICES
    hides is not held, so Vulkan runs on it."""
    info = _mixed({**CUDA_INFO, "devices": [CUDA_INFO["devices"][1]]})

    assert [d["name"] for d in info["devices"]] == [
        "NVIDIA GeForce RTX 3060",
        "AMD Radeon Pro V620 (RADV NAVI21)",
        "NVIDIA GeForce RTX 3070",
    ]


def test_probe_without_pci_ids_falls_back_to_names(mixed_build):
    mixed_build["rows"] = [{**row, "pci_id": ""} for row in REPORTED_ROWS]

    info = _mixed()

    assert [d["name"] for d in info["devices"][2:]] == ["AMD Radeon Pro V620 (RADV NAVI21)"]


def test_unreadable_nvidia_bus_ids_fall_back_to_names(mixed_build):
    mixed_build["bus_ids"] = {}

    info = _mixed()

    assert [d["name"] for d in info["devices"][2:]] == ["AMD Radeon Pro V620 (RADV NAVI21)"]


def test_vulkan_seeing_only_the_cuda_cards_changes_nothing(mixed_build):
    mixed_build["rows"] = [REPORTED_ROWS[0], REPORTED_ROWS[2]]

    assert _mixed() is None


def test_cuda_only_build_never_runs_the_vulkan_probe(mixed_build):
    mixed_build["backends"] = frozenset({"base", "cpu", "cuda"})

    assert _mixed() is None
    assert mixed_build["probed"] == 0


def _torch_backend(
    monkeypatch,
    label,
    smi_devices = (),
):
    import utils.hardware.hardware as hw
    monkeypatch.setattr(hw, "_backend_label", lambda _device: label)
    monkeypatch.setattr(hw, "_nvidia_inference_devices", lambda: [dict(d) for d in smi_devices])


def test_cuda_torch_with_no_cards_is_left_alone(monkeypatch, mixed_build):
    _torch_backend(monkeypatch, "cuda", CUDA_INFO["devices"])

    assert _mixed({**CUDA_INFO, "devices": []}) is None
    assert mixed_build["probed"] == 0


@pytest.mark.parametrize("info", [None, {"available": False, "backend": "cpu", "devices": []}])
def test_without_cuda_torch_the_nvidia_smi_rows_stand_in(monkeypatch, mixed_build, info):
    _torch_backend(monkeypatch, "cpu", CUDA_INFO["devices"])

    mixed = _mixed(info)

    assert [d["name"] for d in mixed["devices"]] == [
        "NVIDIA GeForce RTX 3070",
        "NVIDIA GeForce RTX 3060",
        "AMD Radeon Pro V620 (RADV NAVI21)",
    ]
    # nvidia-smi row numbers are not pinnable ordinals.
    assert mixed["gguf_gpu_ids_supported"] is False


def test_without_cuda_torch_or_nvidia_smi_nothing_changes(monkeypatch, mixed_build):
    _torch_backend(monkeypatch, "cpu", ())

    assert _mixed(None) is None
    assert mixed_build["probed"] == 0


def test_integrated_gpu_is_left_out_as_llama_cpp_skips_it_beside_discrete_cards(mixed_build):
    mixed_build["rows"] = [
        *REPORTED_ROWS,
        {**_vulkan_row(3, "AMD Radeon(TM) Graphics", 8142, 7734, ""), "is_igpu": True},
    ]

    info = _mixed()

    assert [d["name"] for d in info["devices"][2:]] == ["AMD Radeon Pro V620 (RADV NAVI21)"]


def test_only_an_integrated_gpu_beside_cuda_changes_nothing(mixed_build):
    mixed_build["rows"] = [
        REPORTED_ROWS[0],
        {**_vulkan_row(1, "AMD Radeon(TM) Graphics", 8142, 7734, ""), "is_igpu": True},
    ]

    assert _mixed() is None


def test_failed_vulkan_probe_keeps_the_cuda_inventory(mixed_build):
    mixed_build["rows"] = []

    assert _mixed() is None


def _system_info(monkeypatch):
    import utils.hardware as hardware
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {k: v for k, v in CUDA_INFO.items() if k != "gguf_gpu_ids_supported"},
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"backend": "cuda", "devices": []},
    )
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", lambda: None)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda *a: False))
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda *a: False))
    monkeypatch.setattr(main, "_system_gpu_cache", None)
    return main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *a: None))


def test_system_endpoint_reports_the_amd_card_for_gguf(monkeypatch, mixed_build):
    gpu, inference_gpu = _system_info(monkeypatch)

    # Training still sees the CUDA cards only; GGUF inference sees all three.
    assert [d["name"] for d in gpu["devices"]] == [d["name"] for d in CUDA_INFO["devices"]]
    assert inference_gpu["backend"] == "cuda+vulkan"
    assert len(inference_gpu["devices"]) == 3
    assert inference_gpu["devices"][2]["name"] == "AMD Radeon Pro V620 (RADV NAVI21)"
    # CUDA pins keep working through the training inventory.
    assert inference_gpu["gguf_gpu_ids_supported"] == gpu["gguf_gpu_ids_supported"]


def test_system_endpoint_is_unchanged_on_a_cuda_only_build(monkeypatch, mixed_build):
    mixed_build["backends"] = frozenset({"base", "cpu", "cuda"})

    gpu, inference_gpu = _system_info(monkeypatch)

    assert inference_gpu is gpu


def test_probe_rows_carry_the_pci_id_column(monkeypatch, tmp_path):
    from core.inference import llama_cpp

    lib = "ggml-vulkan.dll" if llama_cpp.sys.platform == "win32" else "libggml-vulkan.so"
    (tmp_path / lib).write_bytes(b"stub")
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"stub")
    stdout = "\n".join(
        [
            "0\t8847884288\t0\t8847884288\tNVIDIA GeForce RTX 3060\t1\t0000:a1:00.0",
            "1\t32177455104\t0\t32195477504\tAMD Radeon Pro V620 (RADV NAVI21)\t1\t0000:04:00.0",
            "2\t8155824128\t0\t8847884288\tOld Build Card\t1",
        ]
    )
    monkeypatch.setattr(
        llama_cpp.subprocess,
        "run",
        lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 0, stdout = stdout, stderr = ""),
    )

    rows = llama_cpp.LlamaCppBackend._run_vulkan_probe(str(binary))

    assert [r["pci_id"] for r in rows] == ["0000:a1:00.0", "0000:04:00.0", ""]
    assert rows[1]["name"] == "AMD Radeon Pro V620 (RADV NAVI21)"


def _fake_get_props(device_id):
    def get_props(_dev, props_ref):
        props_ref._obj.device_id = device_id

    return get_props


def test_probe_reads_the_pci_id_from_device_props():
    from core.inference._vulkan_probe import _pci_device_id
    raw = ctypes.create_string_buffer(b"0000:A1:00.0")
    assert _pci_device_id(_fake_get_props(ctypes.addressof(raw)), 1) == "0000:a1:00.0"


@pytest.mark.parametrize("device_id", [None, 0, 1])
def test_probe_ignores_a_props_slot_an_older_ggml_leaves_blank(device_id):
    """A ggml without device_id writes a caps bool, or nothing, where the pointer would be."""
    from core.inference._vulkan_probe import _pci_device_id
    assert _pci_device_id(_fake_get_props(device_id), 1) == ""


def test_probe_rejects_a_device_id_that_is_not_a_pci_address():
    from core.inference._vulkan_probe import _pci_device_id
    raw = ctypes.create_string_buffer(b"virtual-device-7")
    assert _pci_device_id(_fake_get_props(ctypes.addressof(raw)), 1) == ""
