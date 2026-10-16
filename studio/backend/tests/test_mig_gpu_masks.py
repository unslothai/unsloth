# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""MIG hosts: a MIG mask is honoured, MIG slices are not GPUs, and a MIG card is sized by its slice.

nvidia-smi rows are whole cards and ``nvidia-smi -L`` lists each MIG instance as an indented
row under its card. CUDA sees a MIG card as one slice, so whole-card numbers over-fit it, and
a numeric pin over a MIG mask hands the child the whole host.
"""

from __future__ import annotations

import json
import os
import sys
import types

import pytest

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend
import utils.hardware.hardware as hw
from utils.hardware import nvidia

GPU0 = "GPU-5c89852c-d268-c3f3-1b07-005d5ae1dc3f"
GPU1 = "GPU-7d3a9e11-4b2c-11ee-8c90-0242ac120002"
GPU2 = "GPU-0f1e2d3c-aaaa-bbbb-cccc-123456789abc"
MIG0A = "MIG-c6d4f1ef-42e4-5de3-91c7-45d71c87eb3f"
MIG0B = "MIG-1b2c3d4e-0000-5abc-8def-0123456789ab"
MIG0C = "MIG-99aa88bb-7777-5ccc-9ddd-fedcba987654"

A100_MIB = 40960


def _smi_list(cards):
    """``nvidia-smi -L``: cards = [(uuid, [(profile, mig_uuid), ...]), ...]; old_form uses MIG-GPU-<uuid>/gi/ci."""
    lines = []
    for i, (uuid, slices) in enumerate(cards):
        lines.append(f"GPU {i}: NVIDIA A100-SXM4-40GB (UUID: {uuid})")
        for j, (profile, mig_uuid) in enumerate(slices):
            lines.append(f"  MIG {profile:<12} Device  {j}: (UUID: {mig_uuid})")
    return "\n".join(lines) + "\n"


def _nvml_row(
    index,
    uuid,
    free,
    total,
    mig = False,
):
    row = {
        "index": str(index),
        "uuid": uuid,
        "name": "NVIDIA A100-SXM4-40GB" + (" MIG" if mig else ""),
        "compute_cap": "8.0",
        "memory_total_mib": str(total),
        "memory_free_mib": str(free),
    }
    if mig:
        row["mig"] = "1"
    return row


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A fake nvidia-smi (``-L``, memory and uuid queries) plus a fake NVML probe script."""
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_CACHE", "0")
    monkeypatch.setattr(nvidia, "_uuid_mask_cache", {})
    monkeypatch.setattr(LlamaCppBackend, "_NVML_ROWS", [])
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/opt/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: [])
    )
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_torch", staticmethod(lambda *a, **k: []))
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(mod.sys, "platform", "linux")
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)

    payload_path = tmp_path / "payload.json"
    script = tmp_path / "nvidia_probe.py"
    script.write_text(
        f"import sys; sys.stdout.write(open({str(payload_path)!r}).read())\n", encoding = "utf-8"
    )
    monkeypatch.setenv("UNSLOTH_NVIDIA_PROBE", str(script))
    monkeypatch.delenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", raising = False)

    state = {"cards": [], "free": [], "nvml": None}
    real_run = mod.subprocess.run

    def run(cmd, *args, **kwargs):
        if not cmd or os.path.basename(str(cmd[0])) != "nvidia-smi":
            return real_run(cmd, *args, **kwargs)
        ok = lambda out: types.SimpleNamespace(returncode = 0, stdout = out, stderr = "")
        if "-L" in cmd:
            return ok(_smi_list(state["cards"]))
        query = next((a for a in cmd if a.startswith("--query-gpu=")), "")
        if query == "--query-gpu=index,uuid":
            return ok("".join(f"{i}, {u}\n" for i, (u, _) in enumerate(state["cards"])))
        if query.startswith("--query-gpu=index,memory.free,memory.total"):
            # Whole-card numbers, also for a card in MIG mode.
            uuids = [u for u, _ in state["cards"]] if query.endswith(",uuid") else None
            return ok(
                "".join(
                    f"{i}, {f}, {A100_MIB}" + (f", {uuids[i]}" if uuids else "") + "\n"
                    for i, f in enumerate(state["free"])
                )
            )
        return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "")

    monkeypatch.setattr(mod.subprocess, "run", run)

    def set_host(
        cards,
        free,
        nvml_rows,
        driver = "",
    ):
        state["cards"], state["free"] = cards, free
        payload = (
            {"source": "nvml", "driver_version": driver, "devices": nvml_rows}
            if nvml_rows is not None
            else None
        )
        payload_path.write_text(json.dumps(payload), encoding = "utf-8")

    return set_host


# An A100 split into two 3g.20gb slices (19968 MiB each).
TWO_SLICES = [(GPU0, [("3g.20gb", MIG0A), ("3g.20gb", MIG0B)])]
TWO_SLICES_NVML = [
    _nvml_row(0, GPU0, 38000, A100_MIB),
    _nvml_row(0, MIG0A, 15000, 19968, mig = True),
    _nvml_row(0, MIG0B, 19000, 19968, mig = True),
]
# Three whole A100s, card 0 split into three 2g.10gb slices (9856 MiB each).
MIXED = [
    (GPU0, [("2g.10gb", MIG0A), ("2g.10gb", MIG0B), ("2g.10gb", MIG0C)]),
    (GPU1, []),
    (GPU2, []),
]
MIXED_NVML = [
    _nvml_row(0, GPU0, 39000, A100_MIB),
    _nvml_row(0, MIG0A, 9000, 9856, mig = True),
    _nvml_row(0, MIG0B, 9500, 9856, mig = True),
    _nvml_row(0, MIG0C, 9800, 9856, mig = True),
    _nvml_row(1, GPU1, 30000, A100_MIB),
    _nvml_row(2, GPU2, 40000, A100_MIB),
]


class TestAMigMaskIsHonouredByLlamaCpp:
    @pytest.mark.parametrize("mask", [MIG0B, MIG0B[:12]])
    def test_a_mig_mask_is_sized_by_its_slice_and_kept_for_the_child(self, host, monkeypatch, mask):
        host(MIXED, [39000, 30000, 40000], MIXED_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        # Not three whole 40 GB cards the parent never exposed.
        assert LlamaCppBackend._get_gpu_memory() == [(0, 9500, 9856)]
        assert LlamaCppBackend._pinned_child_visibility([0]) == MIG0B

    def test_the_old_mig_gpu_form_is_never_widened_to_a_numeric_pin(self, host, monkeypatch):
        # Pre-R470 form: NVML on a newer driver names the slice MIG-<uuid>, so nothing matches.
        host(TWO_SLICES, [38000], TWO_SLICES_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"MIG-{GPU0}/1/0")
        assert LlamaCppBackend._get_gpu_memory() == []
        # Placement from another source (torch ordinal 0) must not become CUDA_VISIBLE_DEVICES=0.
        assert LlamaCppBackend._pinned_child_visibility([0]) is None

    def test_a_mig_mask_without_nvml_keeps_the_inherited_mask(self, host, monkeypatch):
        host(TWO_SLICES, [38000], None)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", MIG0A)
        assert LlamaCppBackend._get_gpu_memory() == []
        assert LlamaCppBackend._pinned_child_visibility([0]) is None

    def test_whole_card_masks_still_pin_numerically(self, host, monkeypatch):
        host(MIXED, [39000, 30000, 40000], MIXED_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
        assert LlamaCppBackend._get_gpu_memory() == [(1, 30000, A100_MIB), (2, 40000, A100_MIB)]
        assert LlamaCppBackend._pinned_child_visibility([1, 2]) == "1,2"
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU2},{GPU1}")
        assert LlamaCppBackend._pinned_child_visibility([1, 2]) == "1,2"


class TestAMigCardIsSizedByItsSlice:
    def test_a_numeric_mask_on_a_mig_card_reads_the_slice(self, host, monkeypatch):
        host(TWO_SLICES, [38000], TWO_SLICES_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        # CUDA exposes the first slice, not the 38 GB free on the card.
        assert LlamaCppBackend._get_gpu_memory() == [(0, 15000, 19968)]

    def test_an_unmasked_mixed_host_reads_slice_and_whole_cards(self, host):
        host(MIXED, [39000, 30000, 40000], MIXED_NVML)
        assert LlamaCppBackend._get_gpu_memory() == [
            (0, 9000, 9856),
            (1, 30000, A100_MIB),
            (2, 40000, A100_MIB),
        ]

    def test_auto_select_ranks_a_mig_card_by_its_slice(self, host, monkeypatch):
        host(MIXED, [39000, 30000, 40000], MIXED_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
        monkeypatch.setattr(hw, "IS_ROCM", False)
        monkeypatch.setattr(
            hw,
            "estimate_required_model_memory_gb",
            lambda *a, **k: (20.0, {"required_gb": 20.0, "model_size_source": "config"}),
        )
        # Whole-card telemetry: card 0 looks freer (39 GB) than card 1 (30 GB).
        monkeypatch.setattr(
            hw,
            "get_visible_gpu_utilization",
            lambda: {
                "devices": [
                    {"index": 0, "vram_total_gb": 40.0, "vram_used_gb": 1.0},
                    {"index": 1, "vram_total_gb": 40.0, "vram_used_gb": 10.0},
                ]
            },
        )
        selected, metadata = hw.auto_select_gpu_ids("unsloth/test")
        # The 9.6 GB slice cannot hold 20 GB; card 1 can, alone.
        assert selected == [1]
        assert metadata["selection_mode"] == "auto"

    def test_auto_select_without_a_slice_reading_does_not_trust_the_card(self, host, monkeypatch):
        host(MIXED, [39000, 30000, 40000], None)
        devices = [
            {"index": 0, "vram_total_gb": 40.0, "vram_used_gb": 1.0},
            {"index": 1, "vram_total_gb": 40.0, "vram_used_gb": 10.0},
        ]
        out = hw._with_mig_instance_memory(devices)
        assert out[0]["vram_total_gb"] is None and out[0]["vram_used_gb"] is None
        assert out[1] == devices[1]

    def test_a_host_without_mig_is_unchanged(self, host):
        host([(GPU0, []), (GPU1, [])], [39000, 30000], None)
        devices = [
            {"index": 0, "vram_total_gb": 40.0, "vram_used_gb": 1.0},
            {"index": 1, "vram_total_gb": 40.0, "vram_used_gb": 10.0},
        ]
        assert hw._with_mig_instance_memory(devices) == devices
        assert LlamaCppBackend._get_gpu_memory() == [(0, 39000, A100_MIB), (1, 30000, A100_MIB)]


MIG1A = "MIG-2e3f4a5b-1111-5bcd-9eef-112233445566"
# Two GPU instances on one card, as an R570+ host enumerates them.
TWO_GPU_INSTANCES = [dict(r) for r in TWO_SLICES_NVML]
TWO_GPU_INSTANCES[1]["gpu_instance_id"], TWO_GPU_INSTANCES[2]["gpu_instance_id"] = "1", "2"


class TestOneSlicePerMigCard:
    # R570+ CUDA enumerates every slice; Studio still sizes and pins one per card, so a
    # tensor split or a single-device load never counts memory the child cannot use.
    @pytest.mark.parametrize(
        "mask, expected, uuid",
        [
            (None, (0, 15000, 19968), MIG0A),
            ("0", (0, 15000, 19968), MIG0A),
            (f"{MIG0A},{MIG0B}", (0, 15000, 19968), MIG0A),
            (f"{MIG0B},{MIG0A}", (0, 19000, 19968), MIG0B),
        ],
    )
    def test_a_mig_card_is_one_slice(self, host, monkeypatch, mask, expected, uuid):
        host(TWO_SLICES, [38000], TWO_GPU_INSTANCES, driver = "570.86.15")
        if mask is not None:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert LlamaCppBackend._get_gpu_memory() == [expected]
        assert LlamaCppBackend._pinned_child_visibility([0]) == uuid

    @pytest.mark.parametrize("mask", [None, "0"])
    def test_an_unnamed_slice_is_the_least_free_one(self, host, monkeypatch, mask):
        # The budget must not exceed the slice the child is pinned to, whichever CUDA lists first.
        rows = [dict(r) for r in TWO_SLICES_NVML]
        rows[1]["memory_free_mib"], rows[2]["memory_free_mib"] = "19000", "4000"
        host(TWO_SLICES, [38000], rows)
        if mask is not None:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 4000, 19968)]
        assert LlamaCppBackend._pinned_child_visibility([0]) == MIG0B

    @pytest.mark.parametrize("mask", ["1,1", "1gpu,1gpu", "0,2,0"])
    def test_a_repeated_index_hides_every_gpu(self, host, monkeypatch, mask):
        # CUDA empties the device set on a repeat (torch's _parse_visible_devices agrees).
        host(MIXED, [39000, 30000, 40000], MIXED_NVML)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        assert LlamaCppBackend._nvml_rows_visible(MIXED_NVML) == []

    def test_torch_ordinals_ignore_rows_cached_by_an_earlier_probe(self, host, monkeypatch):
        # Mask order is card 1's slice then card 0's: torch ordinal 0 is MIG1A, while the cached
        # rows are keyed by physical index and would name card 0's slice.
        rows = [
            _nvml_row(0, GPU0, 38000, A100_MIB),
            _nvml_row(0, MIG0A, 15000, 19968, mig = True),
            _nvml_row(1, GPU1, 38000, A100_MIB),
            _nvml_row(1, MIG1A, 17000, 19968, mig = True),
        ]
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG1A},{MIG0A}")
        monkeypatch.setattr(LlamaCppBackend, "_NVML_ROWS", rows)
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
        fake = types.SimpleNamespace(cuda = types.SimpleNamespace(device_count = lambda: 2))
        monkeypatch.setitem(sys.modules, "torch", fake)
        assert LlamaCppBackend._pinned_child_visibility([0]) == MIG1A

    @pytest.mark.parametrize(
        "count, picked, expected", [(2, [1], MIG0B), (2, [0, 1], None), (1, [0], None)]
    )
    def test_torch_ordinals_narrow_a_multi_mig_mask(
        self, host, monkeypatch, count, picked, expected
    ):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{MIG0A},{MIG0B}")
        fake = types.SimpleNamespace(cuda = types.SimpleNamespace(device_count = lambda: count))
        monkeypatch.setitem(sys.modules, "torch", fake)
        monkeypatch.setattr(LlamaCppBackend, "_GPU_IDS_ARE_PCI_INDICES", False)
        if expected is None and len(picked) == 2:
            expected = f"{MIG0A},{MIG0B}"
        assert LlamaCppBackend._pinned_child_visibility(picked) == expected


class TestTelemetryFailuresNeverDropCards:
    @pytest.mark.parametrize("mask, expected", [("1gpu2", [1]), ("1,2,abc", [1, 2])])
    def test_a_numeric_prefix_mask_keeps_its_cards(self, host, monkeypatch, mask, expected):
        host(MIXED, [39000, 30000, 40000], None)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
        free = {1: 30000, 2: 40000}
        assert LlamaCppBackend._get_gpu_memory() == [(i, free[i], A100_MIB) for i in expected]

    def test_whole_cards_survive_a_missing_slice_reading(self, host):
        host(MIXED, [39000, 30000, 40000], None)
        assert LlamaCppBackend._get_gpu_memory() == [(1, 30000, A100_MIB), (2, 40000, A100_MIB)]

    def test_a_mig_only_host_without_any_reading_keeps_the_card_rows(self, host):
        host(TWO_SLICES, [38000], None)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 38000, A100_MIB)]


class TestMigSlicesAreNotGpus:
    @pytest.fixture
    def count_state(self, monkeypatch):
        monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
        monkeypatch.setattr(hw, "IS_ROCM", False)
        monkeypatch.setattr(hw, "_physical_gpu_count", None)
        monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)
        monkeypatch.setattr(hw, "_physical_gpu_count_checked_at", None, raising = False)

    @pytest.mark.parametrize(
        "cards, expected",
        [
            (TWO_SLICES, 1),
            (MIXED, 3),
            ([(GPU0, [("3g.20gb", f"MIG-{GPU0}/1/0"), ("3g.20gb", f"MIG-{GPU0}/2/0")])], 1),
        ],
    )
    def test_the_physical_count_skips_mig_rows(self, host, count_state, cards, expected):
        host(cards, [38000] * len(cards), None)
        assert hw.get_physical_gpu_count() == expected
        assert hw.get_parent_visible_gpu_ids() == list(range(expected))
        with pytest.raises(ValueError):
            hw.resolve_requested_gpu_ids([expected])

    def test_mig_gpu_indices(self, host):
        host(MIXED, [0, 0, 0], None)
        assert nvidia.get_mig_gpu_indices() == {0}
        host([(GPU0, []), (GPU1, [("1g.5gb", MIG0A)])], [0, 0], None)
        # Cached until the inventory generation moves (hotplug, driver reset).
        assert nvidia.get_mig_gpu_indices() == {0}
        nvidia.gpu_query.invalidate_static("test")
        assert nvidia.get_mig_gpu_indices() == {1}


def test_auto_select_and_llama_cpp_budget_the_same_slice(host):
    # One selected id is one worker device: training and llama.cpp both see one slice per card.
    host(TWO_SLICES, [38000], TWO_GPU_INSTANCES, driver = "570.86.15")
    devices = [{"index": 0, "vram_total_gb": 40.0, "vram_used_gb": 2.0}]
    out = hw._with_mig_instance_memory(devices)
    assert out[0]["vram_total_gb"] == round(19968 / 1024, 2)
    assert out[0]["vram_used_gb"] == round((19968 - 15000) / 1024, 2)
    assert LlamaCppBackend._get_gpu_memory() == [(0, 15000, 19968)]


def test_mig_overlay_skips_a_non_cuda_backend(host, monkeypatch):
    # Intel XPU indices are not NVIDIA indices: an NVIDIA MIG card at 0 must not touch XPU 0.
    host(TWO_SLICES, [38000], TWO_SLICES_NVML)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.XPU)
    devices = [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 2.0}]
    assert hw._with_mig_instance_memory(devices) == devices


T4_MIB = 15360


class TestOrdinaryHostsKeepTheirReadings:
    """Whole-GPU uuid masks and plain polls on hosts without MIG behave as before MIG support."""

    @pytest.fixture
    def calls(self, monkeypatch):
        seen = {"torch": 0, "list": 0, "uuid_query_fails": False}
        run = mod.subprocess.run

        def counting_run(cmd, *args, **kwargs):
            if cmd and os.path.basename(str(cmd[0])) == "nvidia-smi":
                if "-L" in cmd:
                    seen["list"] += 1
                if seen["uuid_query_fails"] and "--query-gpu=index,uuid" in cmd:
                    return types.SimpleNamespace(returncode = 1, stdout = "", stderr = "")
            return run(cmd, *args, **kwargs)

        monkeypatch.setattr(mod.subprocess, "run", counting_run)

        def torch_probe(*a, **k):
            seen["torch"] += 1
            return [(0, 1, 1)]

        monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_torch", staticmethod(torch_probe))
        return seen

    @pytest.mark.parametrize("token", [GPU0, GPU0[:12], GPU0.lower()])
    def test_a_uuid_mask_under_fastest_first_reads_its_card(self, host, monkeypatch, calls, token):
        host([(GPU0, [])], [14000], None)
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", token)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 14000, A100_MIB)]
        assert calls["torch"] == 0

    def test_a_uuid_mask_reads_its_card_when_the_uuid_query_fails(self, host, monkeypatch, calls):
        host([(GPU0, [])], [14000], None)
        calls["uuid_query_fails"] = True
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU0)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 14000, A100_MIB)]
        assert calls["torch"] == 0

    def test_a_multi_gpu_uuid_mask_under_fastest_first_keeps_only_its_cards(
        self, host, monkeypatch, calls
    ):
        host([(GPU0, []), (GPU1, []), (GPU2, [])], [39000, 30000, 40000], None)
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", f"{GPU2},{GPU1}")
        assert LlamaCppBackend._get_gpu_memory() == [(1, 30000, A100_MIB), (2, 40000, A100_MIB)]
        assert calls["torch"] == 0

    def test_an_unknown_uuid_still_reads_nothing_from_the_cli(self, host, monkeypatch, calls):
        host([(GPU0, []), (GPU1, [])], [39000, 30000], None)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU2)
        assert LlamaCppBackend._get_gpu_memory() == [(0, 1, 1)]

    def test_a_lone_card_without_a_uuid_column(self, host):
        host([(GPU0, [])], [14000], None)
        assert LlamaCppBackend._whole_gpu_mask_rows(GPU0, [], {0}) == {0}
        assert LlamaCppBackend._whole_gpu_mask_rows(f"{GPU0},{GPU1}", [], {0}) is None
        assert LlamaCppBackend._whole_gpu_mask_rows(MIG0A, [], {0}) is None
        nvidia.gpu_query.invalidate_static("test")
        host(TWO_SLICES, [38000], None)
        assert LlamaCppBackend._whole_gpu_mask_rows(GPU0, [], {0}) is None

    def test_memory_polls_reuse_one_mig_listing(self, host, calls):
        host([(GPU0, [])], [14000], None)
        devices = [{"index": 0, "vram_total_gb": 15.0, "vram_used_gb": 1.0}]
        for _ in range(3):
            assert LlamaCppBackend._get_gpu_memory() == [(0, 14000, A100_MIB)]
            assert hw._with_mig_instance_memory(devices) == devices
        assert calls["list"] == 1

    def test_the_physical_count_probe_publishes_the_mig_set(self, host, monkeypatch, calls):
        monkeypatch.setattr(hw, "IS_ROCM", False)
        monkeypatch.setattr(hw, "_physical_gpu_count", None)
        monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)
        monkeypatch.setattr(hw, "_physical_gpu_count_checked_at", None, raising = False)
        host([(GPU0, [])], [14000], None)
        assert hw.get_physical_gpu_count() == 1
        assert calls["list"] == 1
        for _ in range(3):
            assert LlamaCppBackend._get_gpu_memory() == [(0, 14000, A100_MIB)]
        assert calls["list"] == 1
