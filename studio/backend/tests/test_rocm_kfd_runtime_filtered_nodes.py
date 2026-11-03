# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.

"""KFD retains nodes that HIP omits when their render device cannot open."""

import pytest

from utils.hardware import amd, hardware


def _node(root, idx, *, simd, minor):
    d = root / str(idx)
    d.mkdir(parents = True)
    lines = [f"simd_count {simd}", "vendor_id 4098" if simd else "vendor_id 0"]
    if minor is not None:
        lines.append(f"drm_render_minor {minor}")
    (d / "properties").write_text("\n".join(lines) + "\n")


@pytest.fixture
def two_boards(monkeypatch, tmp_path):
    nodes = tmp_path / "nodes"
    dri = tmp_path / "dri"
    dri.mkdir()
    _node(nodes, 0, simd = 0, minor = None)
    _node(nodes, 1, simd = 96, minor = 128)
    _node(nodes, 2, simd = 32, minor = 129)
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware, "_KFD_TOPOLOGY_NODES", str(nodes), raising = False)
    monkeypatch.setattr(hardware, "_DRI_DIR", str(dri), raising = False)
    monkeypatch.setattr(hardware, "_rocm_kfd_gpu_pci_ids", lambda: ["0000:03:00.0", "0000:04:00.0"])
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda *a, **k: ["gfx1100", "gfx1102"])
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_amdgpu_cards",
        lambda: [("0000:03:00.0", 0, "x"), ("0000:04:00.0", 1, "x")],
    )
    monkeypatch.setattr(
        hardware,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {"0000:03:00.0": (4.0, 24.0), "0000:04:00.0": (6.0, 8.0)},
    )
    return dri


def test_every_render_node_present_keeps_the_kfd_order(two_boards):
    (two_boards / "renderD128").write_text("")
    (two_boards / "renderD129").write_text("")
    devices = hardware._rocm_context_free_visible_devices()
    assert [(d["visible_ordinal"], d["vram_total_gb"]) for d in devices] == [(0, 24.0), (1, 8.0)]


def test_a_render_node_missing_from_the_container_declines(two_boards):
    (two_boards / "renderD129").write_text("")
    assert hardware._rocm_context_free_visible_devices() == []


def test_smi_fallback_does_not_restore_the_hidden_cards_memory(two_boards, monkeypatch):
    from core.inference import video

    (two_boards / "renderD129").write_text("")
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(
        hardware, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": [0], "raw": None}
    )
    monkeypatch.setattr(hardware, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {0: 0, 1: 1})
    monkeypatch.setattr(amd, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(hardware, "_torch_get_physical_gpu_count", lambda: 1)
    runtime = [{"index": 0, "visible_ordinal": 0, "used_gb": 6.0, "total_gb": 8.0}]
    monkeypatch.setattr(hardware, "_torch_get_per_device_info", lambda ids: runtime)
    monkeypatch.setattr(hardware, "_torch_get_device_inventory", lambda ids: runtime)
    queried = []

    def smi(*args, **kwargs):
        queried.append(args)
        return {
            "devices": [
                {"index": 0, "visible_ordinal": 0, "vram_used_gb": 4.0, "vram_total_gb": 24.0}
            ]
        }

    monkeypatch.setattr(hardware, "_smi_query", smi)
    monkeypatch.setattr(video, "_user_reserve_bytes", lambda *args: 0)
    assert hardware._amd_smi_ids_for_hip_ids([0]) is None
    assert video._h3_card_memory_bytes("cuda", 0) == (2 * 1024**3, 8 * 1024**3)
    assert queried == []
    monkeypatch.setattr(hardware, "_rocm_linux_sysfs_vram_gb", lambda: (10.0, 32.0))
    displayed = hardware.get_gpu_utilization()
    assert displayed["vram_total_gb"] == 8.0
    assert displayed["vram_used_gb"] == 6.0


@pytest.mark.parametrize(
    "rocm,ids,mapping",
    [(True, [0], [1]), (True, [0], [0]), (True, [], None), (True, None, None), (False, [0], None)],
)
def test_valid_smi_queries_keep_their_existing_mapping(monkeypatch, rocm, ids, mapping):
    monkeypatch.setattr(hardware, "IS_ROCM", rocm)
    monkeypatch.setattr(hardware, "_amd_smi_ids_for_hip_ids", lambda selected: mapping)
    queried = []

    def smi(name, selected, **kwargs):
        queried.append(selected)
        return {"devices": [{"index": index} for index in (selected or [])]}

    monkeypatch.setattr(hardware, "_smi_query", smi)
    result = hardware._smi_visible_utilization({"numeric_ids": ids, "raw": None})
    assert queried == [mapping if mapping is not None else ids]
    assert [row["index"] for row in result["devices"]] == (ids or [])


@pytest.mark.parametrize(
    "arch,initialized,expected",
    [("gfx1100", True, 12 * 1024**3), ("gfx1100", False, None), ("gfx1151", True, None)],
)
def test_filtered_topology_keeps_the_discrete_card_user_reserve(
    two_boards, monkeypatch, arch, initialized, expected
):
    from types import SimpleNamespace
    from core.inference import diffusion_memory, video

    (two_boards / "renderD128").write_text("")
    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(hardware, "IS_ROCM", True)
    monkeypatch.setattr(hardware, "_torch_get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda: [arch, "gfx1102"])
    properties = SimpleNamespace(total_memory = 24 * 1024**3, gcnArchName = arch)
    property_reads = []

    def no_occupancy(*args):
        pytest.fail("a reserve read must not create an occupancy context")

    def get_properties(index):
        property_reads.append(index)
        return properties

    module = SimpleNamespace(
        get_device_properties = get_properties,
        is_initialized = lambda: initialized,
        mem_get_info = no_occupancy,
    )
    monkeypatch.setattr(hardware, "_torch_get_device_module", lambda: (module, "cuda"))
    monkeypatch.setattr(diffusion_memory, "_vram_budget_fraction", lambda: 0.5)
    reserve = video._h3_card_reserve_bytes("cuda", 0)
    assert reserve == expected
    if not initialized:
        assert property_reads == []
    if expected is not None:
        runtime = SimpleNamespace(vram_reserve_probe = lambda: reserve)
        flags = video._h3_spare_user_reserve(runtime, ["--max-vram", "-1"])
        assert flags == ["--max-vram", "-13.00"]
