# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""#8235: an AUTOMATIC image / video load ranks the visible GPUs by free VRAM.

The picker on "auto" sends no gpu_ids, and bare "cuda" is the current device, PCI
slot 0 under Studio's PCI_BUS_ID order, so an 8 GB card 0 beside an idle 24 GB
card 1 budgeted the load against 8 GB and offloaded to RAM. Explicit picks and
masks keep winning; one visible card, UUID masks and ROCm are unchanged."""

import ast
import asyncio
import types
from pathlib import Path

import pytest

from core.inference import diffusion_device as dd
from utils.hardware import hardware as hw

def _dev(index, total, used):
    return {"index": index, "vram_total_gb": total, "vram_used_gb": used}


def _host(monkeypatch, visible, devices, rocm = False, order_matches = True):
    monkeypatch.setattr(hw, "IS_ROCM", rocm)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(visible))
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: order_matches)
    monkeypatch.setattr(hw, "_get_parent_visible_gpu_spec", lambda: {"numeric_ids": list(visible)})
    smi = {"available": True, "devices": devices}
    monkeypatch.setattr(hw, "_smi_visible_utilization", lambda spec: smi)
    monkeypatch.setattr(hw, "get_visible_gpu_utilization", lambda: smi)


def test_auto_picks_the_card_with_the_most_free_vram(monkeypatch):
    _host(monkeypatch, [0, 1], [_dev(0, 8.0, 1.0), _dev(1, 24.0, 0.5)])
    assert dd.resolve_auto_cuda_ordinal() == 1


def test_auto_maps_physical_ids_through_the_mask(monkeypatch):
    # CUDA_VISIBLE_DEVICES=5,2: physical 2 is torch ordinal 1.
    _host(monkeypatch, [5, 2], [_dev(2, 80.0, 0.0), _dev(5, 80.0, 70.0)])
    assert dd.resolve_auto_cuda_ordinal() == 1


def test_auto_ties_keep_the_lowest_ordinal(monkeypatch):
    # An idle uniform host stays on card 0, MiB-level noise included.
    _host(monkeypatch, [0, 1, 2], [_dev(0, 80.0, 0.6), _dev(1, 80.0, 0.5), _dev(2, 80.0, 0.6)])
    assert dd.resolve_auto_cuda_ordinal() == 0


@pytest.mark.parametrize(
    "visible, devices, rocm",
    [
        ([0], [_dev(0, 24.0, 1.0)], False),  # one card: unchanged
        ([], [_dev(0, 24.0, 1.0), _dev(1, 24.0, 0.0)], False),  # UUID/MIG mask
        ([0, 1], [_dev(0, 8.0, 7.0), _dev(1, 96.0, 0.0)], True),  # ROCm APU pool
        ([0, 1], [], False),  # no telemetry
    ],
)
def test_auto_keeps_the_current_device(monkeypatch, visible, devices, rocm):
    _host(monkeypatch, visible, devices, rocm = rocm)
    assert dd.resolve_auto_cuda_ordinal() is None


def test_auto_never_raises(monkeypatch):
    _host(monkeypatch, [0, 1], [])

    def _boom(*_a):
        raise RuntimeError("nvidia-smi hung")

    monkeypatch.setattr(hw, "_smi_visible_utilization", _boom)
    assert dd.resolve_auto_cuda_ordinal() is None


def test_auto_keeps_the_current_device_when_cuda_order_is_not_smi_order(monkeypatch):
    # CUDA_DEVICE_ORDER=FASTEST_FIRST: nvidia-smi rows are not torch ordinals, so no ranking.
    _host(monkeypatch, [0, 1], [_dev(0, 8.0, 1.0), _dev(1, 24.0, 0.5)], order_matches = False)
    assert dd.resolve_auto_cuda_ordinal() is None


def test_auto_never_falls_back_to_torch_telemetry(monkeypatch):
    # No nvidia-smi: the hardware layer's torch fallback opens a CUDA context per card.
    _host(monkeypatch, [0, 1], [_dev(0, 8.0, 1.0), _dev(1, 24.0, 0.5)])
    monkeypatch.setattr(hw, "_smi_visible_utilization", lambda spec: None)
    assert dd.resolve_auto_cuda_ordinal() is None


def _route(monkeypatch, device = "cuda"):
    from routes import inference as routes

    monkeypatch.setattr(
        dd, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = device)
    )
    monkeypatch.setattr(dd, "resolve_auto_cuda_ordinal", lambda: 1)
    monkeypatch.setattr(dd, "resolve_selected_cuda_ordinal", lambda ids, **_k: min(ids))
    return routes._selected_gpu_ordinal


def test_route_ranks_an_automatic_load(monkeypatch):
    resolve = _route(monkeypatch)
    assert asyncio.run(resolve(None)) == 1
    assert asyncio.run(resolve([])) == 1


def test_route_explicit_pick_still_wins(monkeypatch):
    resolve = _route(monkeypatch)
    assert asyncio.run(resolve([0])) == 0


def test_route_auto_stays_off_cuda_and_while_ranking_is_barred(monkeypatch):
    resolve = _route(monkeypatch, device = "mps")
    assert asyncio.run(resolve(None)) is None
    resolve = _route(monkeypatch)
    assert asyncio.run(resolve(None, allow_ranking = False)) is None


def test_route_uses_the_ordinal_an_auto_switch_planned(monkeypatch):
    from core.inference import media_locality

    resolve = _route(monkeypatch)
    token = media_locality.pin_planned_gpu_ordinal(0)
    try:
        assert media_locality.plan_gpu_ordinal() == 0
        assert asyncio.run(resolve(None)) == 0
        assert asyncio.run(resolve([1])) == 1  # an explicit pick still wins
    finally:
        media_locality._PLANNED_ORDINAL.reset(token)
    assert asyncio.run(resolve(None)) == 1


def test_plan_sizes_the_card_the_load_will_rank(monkeypatch):
    from core.inference import media_locality

    monkeypatch.setattr(
        dd, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(device = "cuda")
    )
    _host(monkeypatch, [0, 1], [_dev(0, 8.0, 1.0), _dev(1, 24.0, 0.5)])
    assert media_locality.plan_gpu_ordinal() == 1


@pytest.mark.parametrize(
    "rel", ["core/inference/diffusion.py", "core/inference/video.py", "core/inference/sd_cpp_backend.py"]
)
def test_direct_backend_loads_rank_automatic_too(rel):
    # A direct caller (MCP, plugin) passes neither gpu_ids nor an ordinal.
    src = (Path(dd.__file__).resolve().parents[2] / rel).read_text(encoding = "utf-8")
    calls = {
        node.func.id
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "resolve_auto_cuda_ordinal" in calls
