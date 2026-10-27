# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""R570+ MIG: ids in the ``cuda_device`` namespace are CUDA instance ordinals, so every free-memory reader
sizes an instance from its own slice row, never from the nvidia-smi row of a physical card that happens to
share the number. CPU fixtures only: nvidia-smi, the NVML probe and torch are faked."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from core.inference import diffusion_device as dd
from core.inference import diffusion_multigpu as multi
from utils.hardware import gpu_query, nvidia


def _fixture_module():
    # The MIG host fixture lives beside this file; "tests" also names the repo-root package.
    name = "_studio_mig_multi_instance_fixture"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("test_mig_multi_instance.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


fx = _fixture_module()

hw = fx.hw

OTHER = "GPU-dddddddd-0000-0000-0000-000000000000"
THIRD = "GPU-eeeeeeee-0000-0000-0000-000000000000"
SLICE_B = (18000, 19968)  # instance B's own free / total MiB


@pytest.fixture(autouse = True)
def _probe_script(monkeypatch):
    import utils.prebuilt.update_flow as update_flow
    monkeypatch.setattr(update_flow, "find_installer_script", lambda **k: Path("mock-probe.py"))


@contextlib.contextmanager
def mig_host(mig_cards = frozenset({0}), b_row = None):
    """Two inherited instances (A, B) plus whole cards 1 and 2, masked out. nvidia-smi rows 1 and 2
    report far more free memory than either slice, so a physical-row join is visible."""
    rows = copy.deepcopy(fx.ROWS) + [
        {**fx.row(OTHER, 72000, 81920), "index": "1"},
        {**fx.row(THIRD, 60000, 81920), "index": "2"},
    ]
    if b_row is not None:
        rows[2] = b_row
    real_run = gpu_query.run_nvidia_smi

    def run_smi(cmd, *args, **kwargs):
        if any("utilization.gpu" in str(arg) for arg in cmd):
            return types.SimpleNamespace(
                returncode = 0,
                stderr = "",
                stdout = (
                    "0, 10, 40, 40000, 81920, 100, 400\n"
                    "1, 10, 40, 9920, 81920, 100, 400\n"
                    "2, 10, 40, 21920, 81920, 100, 400\n"
                ),
            )
        return real_run(cmd, *args, **kwargs)

    listing = (
        fx.LISTING
        + f"GPU 1: NVIDIA A100-SXM4-80GB (UUID: {OTHER})\n"
        + f"GPU 2: NVIDIA A100-SXM4-80GB (UUID: {THIRD})\n"
    )
    real_smi_visible = hw._smi_visible_utilization
    with contextlib.ExitStack() as s:
        s.enter_context(patch.object(fx, "LISTING", listing))
        s.enter_context(fx.host(mask = f"{fx.A},{fx.B_UUID}", rows = rows))
        s.enter_context(
            patch.object(nvidia, "get_mig_mode_gpu_indices_or_none", lambda: set(mig_cards))
        )
        s.enter_context(patch.object(nvidia, "get_mig_gpu_indices_or_none", lambda: set(mig_cards)))
        s.enter_context(
            patch.object(nvidia, "get_compute_capabilities", lambda: {i: (8, 0) for i in range(3)})
        )
        s.enter_context(
            patch.object(
                nvidia,
                "get_card_identities",
                lambda: {i: ("NVIDIA A100-SXM4-80GB", "81920", "8.0", "1410") for i in range(3)},
            )
        )
        s.enter_context(patch.object(hw, "_integrated_cuda_rows", lambda ids: ({}, "index")))
        s.enter_context(patch.object(hw, "_smi_visible_utilization", real_smi_visible))
        s.enter_context(patch.object(gpu_query, "run_nvidia_smi", run_smi))
        s.enter_context(patch.object(hw, "_cuda_card_identities", {}))
        s.enter_context(patch.object(hw, "_cuda_card_identities_gen", None))
        s.enter_context(
            patch.object(dd, "_PLANNED_ORDINAL", dd.ContextVar("t", default = dd._UNPLANNED))
        )
        s.enter_context(patch.object(dd, "_PINNED_AWAY_FROM_ZERO", False))
        assert hw.get_parent_visible_gpu_ids() == [0, 1]
        assert hw._get_parent_visible_gpu_spec()["index_kind"] == "cuda_device"
        yield


def test_diffusion_smi_reader_sizes_the_instance_by_its_slice():
    with mig_host():
        assert dd.smi_free_total_mib(1) == SLICE_B
        assert dd.smi_free_total_mib(0) == (36000, 40192)
        assert dd.smi_free_total_mib(2) == (None, None)


def test_automatic_companion_budget_reads_the_slice():
    with mig_host():
        token = multi._SELECTION.set(multi.MultiGpuSelection(0, (1,), automatic = True))
        try:
            assert multi._free_total_mib(1) == SLICE_B
        finally:
            multi._SELECTION.reset(token)


def test_an_unread_slice_is_unknown_not_another_cards_memory():
    # A failed refresh keeps identities with zero memory: that is unknown, never physical card 1.
    with mig_host(b_row = fx.row(fx.B_UUID, 0, 0, 1)):
        assert dd.smi_free_total_mib(1) == (None, None)
        token = multi._SELECTION.set(multi.MultiGpuSelection(0, (1,), automatic = True))
        try:
            assert multi._free_total_mib(1) == (None, None)
        finally:
            multi._SELECTION.reset(token)


def test_auto_ordinal_never_ranks_instances_by_physical_rows():
    # The instances sit on card 2, so instance ordinals 0 and 1 collide with whole cards 0 and 1, and
    # whole card 1 has the most free memory: ranking those rows would move the load to instance 1.
    with mig_host(mig_cards = {2}):
        assert dd.resolve_auto_cuda_ordinal() == 0


def test_instance_ids_are_never_proven_identical_from_physical_rows():
    with mig_host(mig_cards = {2}):
        assert hw._cuda_cards_identical([0, 1]) is None
        assert hw._cuda_cards_identical() is False


def test_training_utilization_lists_instance_rows():
    with mig_host():
        util = hw.get_gpu_utilization()
        assert util["index_kind"] == "cuda_device"
        totals = [d["vram_total_gb"] for d in util["devices"]]
        assert totals == [round(40192 / 1024, 2), round(19968 / 1024, 2)]


def test_context_free_reader_uses_the_slice_row():
    with mig_host():
        total = SLICE_B[1] * 1024**2
        free = hw._context_free_cuda_memory_info(1, total)
        assert free == SLICE_B[0] * 1024**2
    with mig_host(b_row = fx.row(fx.B_UUID, 0, 0, 1)):
        assert hw._context_free_cuda_memory_info(1, SLICE_B[1] * 1024**2) is None


def test_a_single_whole_card_keeps_its_nvidia_smi_reading(monkeypatch):
    spec = {
        "raw": None,
        "numeric_ids": [0],
        "supports_explicit_gpu_ids": True,
        "index_kind": "physical",
    }
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: dict(spec))
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "_integrated_cuda_rows", lambda ids: ({}, "index"))
    monkeypatch.setattr(hw, "_nvidia_mig_gpu_indices_or_none", lambda: set())
    monkeypatch.setattr(
        hw,
        "_smi_visible_utilization",
        lambda s: {"devices": [{"index": 0, "vram_total_gb": 24.0, "vram_used_gb": 4.0}]},
    )
    monkeypatch.setattr(dd, "_PLANNED_ORDINAL", dd.ContextVar("t", default = dd._UNPLANNED))
    assert hw._cuda_instance_ids_active() is False
    assert dd.smi_free_total_mib(0) == (20480, 24576)
    assert dd.smi_free_total_mib(1) == (None, None)
    assert dd.resolve_auto_cuda_ordinal() is None
