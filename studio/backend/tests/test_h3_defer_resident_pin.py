# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3: a fully resident initial fit defers the up-front block pin (UNSLOTH_H3_DEFER_RESIDENT_PIN)."""

from __future__ import annotations

import logging
import types

import pytest

torch = pytest.importorskip("torch")

DEFER_ENV = "UNSLOTH_H3_DEFER_RESIDENT_PIN"


class _Stream:
    def synchronize(self):
        return None


class _LazyGroup:
    """A lazily streamed diffusers block group: ``cpu_param_dict`` maps each weight to its pageable host copy (the
    weight itself, as diffusers' lazy ``_to_cpu`` gives a torchao weight)."""

    def __init__(
        self,
        seed: int,
        n: int = 32,
    ):
        g = torch.Generator().manual_seed(seed)
        lin = torch.nn.Linear(n, n, bias = True)
        with torch.no_grad():
            lin.weight.copy_(torch.randn(n, n, generator = g))
            lin.bias.copy_(torch.randn(n, generator = g))
        self.modules = [lin]
        self.parameters = []
        self.buffers = []
        self.stream = _Stream()
        self.low_cpu_mem_usage = True
        self.offload_to_disk_path = None
        self.cpu_param_dict = {p: p for p in lin.parameters()}
        self.original = {p: p.detach().clone() for p in lin.parameters()}
        self.onloads = 0
        self.processed_with: list = []

    def _process_tensors_from_modules(
        self,
        pinned_memory = None,
        default_stream = None,
    ):
        self.processed_with.append((pinned_memory, default_stream))
        for p in list(self.modules[0].parameters()):
            p.data = pinned_memory[p].detach().clone()  # the "device" copy: distinct storage

    def onload_(self):
        self.onloads += 1
        self._process_tensors_from_modules(
            {p: t.detach().clone() for p, t in self.cpu_param_dict.items()}
        )

    def offload_(self):
        for p in list(self.modules[0].parameters()):
            p.data = self.cpu_param_dict[p]

    def _init_cpu_param_dict(self):
        return {p: p.detach().clone() for p in self.modules[0].parameters()}


def _residency(n_blocks: int, logger = None):
    from core.inference.video_minimax_h3_residency import H3Residency, group_payload_bytes

    r = H3Residency.__new__(H3Residency)
    r.top = None
    r.blocks = [_LazyGroup(i) for i in range(n_blocks)]
    r.device = "cuda"
    r.logger = logger
    r.top_bytes = 0
    r.block_bytes = [group_payload_bytes(g) for g in r.blocks]
    r.max_blocks = 0
    r.host_yield = None
    r.host_top_up = None
    r.released_bytes = 0
    r.topped_bytes = 0
    r.compacted_bytes = 0
    r.compact_pending = False
    return r


_CURRENT = object()


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    from contextlib import nullcontext

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: _CURRENT)
    for name in (
        DEFER_ENV,
        "UNSLOTH_H3_RESIDENT_UNPIN",
        "UNSLOTH_H3_DIT_RESIDENT",
        "UNSLOTH_H3_STREAM_BLOCK_PIN",
    ):
        monkeypatch.delenv(name, raising = False)


# -- prediction before placement -------------------------------------------------------------------------------------


def _fam():
    return types.SimpleNamespace(
        resolution_presets = ((640, 384), (1344, 768)), default_num_frames = 121
    )


def _predict(monkeypatch, budget):
    from core.inference import video

    seen = {}

    def fake_budget(residency, device, **kw):
        seen["resident"] = residency.resident_bytes()
        seen.update(kw)
        return budget

    monkeypatch.setattr(video, "_h3_residency_budget_bytes", fake_budget)
    net = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.Linear(64, 64))
    total = sum(p.numel() * p.element_size() for p in net.parameters())
    got = video._h3_resident_pin_deferral(_fam(), net, "cuda", te_streamed = True, te_scheme = "int8")
    return got, total, seen


def test_prediction_defers_when_the_initial_fit_holds_the_whole_denoiser(monkeypatch):
    got, total, seen = _predict(monkeypatch, budget = 10**9)
    assert got == total
    # the fit's own sizing: largest preset, default length, nothing resident yet, streamed conditioner
    assert (seen["width"], seen["height"], seen["frames"]) == (1344, 768, 121)
    assert seen["resident"] == 0 and seen["te_streamed"] is True and seen["te_scheme"] == "int8"


def test_prediction_keeps_the_up_front_pin_on_a_partial_fit(monkeypatch):
    # 12 / 16 GB cards: the budget holds only part of the denoiser
    _got, total, _seen = _predict(monkeypatch, budget = 0)
    assert _got is None
    got, _total, _seen = _predict(monkeypatch, budget = total - 1)
    assert got is None


@pytest.mark.parametrize(
    "env",
    [
        {DEFER_ENV: "0"},
        {"UNSLOTH_H3_DIT_RESIDENT": "0"},
        {"UNSLOTH_H3_RESIDENT_UNPIN": "0"},
    ],
)
def test_prediction_kill_switches(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    got, _total, _seen = _predict(monkeypatch, budget = 10**12)
    assert got is None


def test_prediction_needs_a_cuda_device(monkeypatch):
    from core.inference import video

    monkeypatch.setattr(video, "_h3_residency_budget_bytes", lambda *a, **k: 10**12)
    net = torch.nn.Linear(8, 8)
    assert (
        video._h3_resident_pin_deferral(_fam(), net, "cpu", te_streamed = True, te_scheme = None)
        is None
    )
    assert (
        video._h3_resident_pin_deferral(_fam(), None, "cuda", te_streamed = True, te_scheme = None)
        is None
    )


# -- streaming without the up-front pin ------------------------------------------------------------------------------


def _patch_stream(monkeypatch, plan):
    pytest.importorskip("diffusers.hooks.group_offloading")
    import diffusers.hooks as hooks

    from core.inference import diffusion_memory as dm
    from core.inference import diffusion_prequant as dp
    from core.inference import host_pinning

    monkeypatch.setattr(dp, "torchao_group_offload_supported", lambda: True)
    monkeypatch.setattr(dp, "_unhook_from_manager", lambda *a, **k: True)
    monkeypatch.setattr(dp, "_weights_pinnable", lambda m: True)
    monkeypatch.setattr(dp, "_move_groups_outside_inference_mode", lambda m: None)
    monkeypatch.setattr(dp, "_evict_rotation_hook", lambda *a: (lambda m, a: None))
    monkeypatch.setattr(dm, "install_group_offload_buffer_restore", lambda: None)
    monkeypatch.setattr(dm, "install_group_offload_hooks_eager", lambda: None)
    monkeypatch.setattr(host_pinning, "host_pinning_allowed", lambda *a, **k: True)
    plans = []

    def fake_plan(*a, **k):
        plans.append(a)
        return plan

    monkeypatch.setattr(dm, "_streamed_pin_plan", fake_plan)
    applied = {}

    def fake_apply(
        module,
        *,
        onload_device,
        offload_device,
        offload_type,
        num_blocks_per_group,
        use_stream,
        non_blocking = False,
        record_stream = False,
        low_cpu_mem_usage = False,
    ):
        applied["low_cpu_mem_usage"] = low_cpu_mem_usage

    monkeypatch.setattr(hooks, "apply_group_offloading", fake_apply)
    return dp, plans, applied


def test_deferred_stream_skips_the_up_front_arena_pin(monkeypatch):
    dp, plans, applied = _patch_stream(monkeypatch, plan = (True, True))
    net = torch.nn.Sequential(torch.nn.Linear(64, 64))
    mode = dp.stream_prequantized_module(object(), net, "cuda", defer_pin = True)
    assert mode == "stream_lazy"
    assert applied["low_cpu_mem_usage"] is True
    assert plans == []  # no pin plan, no arena
    assert getattr(net, "_unsloth_pin_arena_bytes", None) is None


def test_undeferred_stream_still_pins_up_front(monkeypatch):
    dp, plans, applied = _patch_stream(monkeypatch, plan = (True, True))
    net = torch.nn.Sequential(torch.nn.Linear(64, 64))
    mode = dp.stream_prequantized_module(object(), net, "cuda")
    assert mode == "stream" and applied["low_cpu_mem_usage"] is False and len(plans) == 1


# -- the residency fit -----------------------------------------------------------------------------------------------


def test_deferral_engages_when_the_fit_holds_every_block(caplog):
    from core.inference import video_minimax_h3_residency as res

    logger = logging.getLogger("test_h3_defer")
    r = _residency(4, logger)
    with caplog.at_level(logging.INFO, logger = "test_h3_defer"):
        assert res.defer_resident_block_pins(r, sum(r.block_bytes), logger = logger)
    assert any(
        "4 block groups will be resident, pin deferred" in rec.getMessage()
        for rec in caplog.records
    )
    for g in r.blocks:
        assert g.low_cpu_mem_usage is False
        # host copy is its own object over the same storage: the onload's swap cannot carry it to the device
        for p, host in g.cpu_param_dict.items():
            assert host is not p and host.data_ptr() == p.data_ptr()


def test_deferral_does_not_engage_on_a_partial_fit():
    from core.inference import video_minimax_h3_residency as res

    r = _residency(4)
    assert not res.defer_resident_block_pins(r, sum(r.block_bytes) - 1)
    for g in r.blocks:
        assert g.low_cpu_mem_usage is True and not getattr(g, "_unsloth_pin_deferred", False)
        assert all(host is p for p, host in g.cpu_param_dict.items())


def test_kill_switch_keeps_the_blocks_lazy(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    monkeypatch.setenv(DEFER_ENV, "0")
    r = _residency(3)
    assert not res.defer_resident_block_pins(r, 10**12)
    assert all(g.low_cpu_mem_usage is True for g in r.blocks)


def test_deferred_blocks_promote_from_the_pageable_copy_and_release_it():
    from core.inference import video_minimax_h3_residency as res

    r = _residency(4)
    assert res.defer_resident_block_pins(r, sum(r.block_bytes))
    hosts = [dict(g.cpu_param_dict) for g in r.blocks]
    top, n = r.fit(sum(r.block_bytes), initial = True)
    assert n == 4 and r.resident_blocks() == 4
    for g, host in zip(r.blocks, hosts):
        # one direct copy from the pageable host copy, not diffusers' per-tensor pinning onload
        assert g.onloads == 0 and len(g.processed_with) == 1
        assert g.processed_with[0][0] == host
        # recorded on the compute stream, as diffusers' stream onload does
        assert g.processed_with[0][1] is _CURRENT
        assert not getattr(g, "_unsloth_pin_deferred", False)
        assert g.cpu_param_dict == {} and res.released_host_bytes(g) == res.group_payload_bytes(g)
        for p in g.modules[0].parameters():
            assert torch.equal(p, g.original[p])
    assert r.released_bytes == sum(r.block_bytes)


def test_a_demoted_deferred_block_streams_from_a_rebuilt_host_copy():
    from core.inference import video_minimax_h3_residency as res

    r = _residency(3)
    assert res.defer_resident_block_pins(r, sum(r.block_bytes))
    r.fit(sum(r.block_bytes), initial = True)
    r.fit(r.block_bytes[0])  # a bigger request
    assert r.resident_blocks() == 1
    for g in r.blocks[1:]:
        assert res.released_host_bytes(g) == 0 and len(g.cpu_param_dict) == 2
        for p in g.modules[0].parameters():
            assert p.data_ptr() == g.cpu_param_dict[p].data_ptr()
            assert torch.equal(p, g.original[p])
        g.onload_()  # streams again through the normal onload
        assert g.onloads == 1


def test_a_smaller_fit_than_predicted_pins_the_leftover_blocks(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    r = _residency(4)
    assert res.defer_resident_block_pins(r, sum(r.block_bytes))
    r.fit(sum(r.block_bytes[:2]), initial = True)  # the prediction was wrong
    assert r.resident_blocks() == 2
    seen = {}

    def fake_pin(transformer, logger = None):
        seen["lazy"] = [g for g in r.blocks if g.low_cpu_mem_usage]
        return len(seen["lazy"]), len(r.blocks)

    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (None, list(r.blocks)))
    monkeypatch.setattr(res, "pin_streamed_block_groups", fake_pin)
    assert res.pin_deferred_block_groups(object()) == (2, 4)
    assert seen["lazy"] == r.blocks[2:]
    assert not any(getattr(g, "_unsloth_pin_deferred", False) for g in r.blocks)


def test_leftover_pin_is_a_no_op_when_every_block_went_resident(monkeypatch):
    from core.inference import video_minimax_h3_residency as res

    r = _residency(3)
    assert res.defer_resident_block_pins(r, sum(r.block_bytes))
    r.fit(sum(r.block_bytes), initial = True)
    called = []
    monkeypatch.setattr(res, "h3_offload_groups", lambda t: (None, list(r.blocks)))
    monkeypatch.setattr(res, "pin_streamed_block_groups", lambda *a, **k: called.append(1))
    assert res.pin_deferred_block_groups(object()) == (0, 3)
    assert called == []


def test_a_torchao_host_alias_survives_the_onload_swap():
    torchao = pytest.importorskip("torchao")
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    from core.inference.video_minimax_h3_residency import _host_alias

    lin = torch.nn.Sequential(torch.nn.Linear(64, 64))
    try:
        quantize_(lin, Int8DynamicActivationInt8WeightConfig(version = 2))
    except TypeError:
        quantize_(lin, Int8DynamicActivationInt8WeightConfig())
    w = lin[0].weight
    if not hasattr(w, "__tensor_flatten__"):
        pytest.skip("not a wrapper subclass in this torchao")
    names, _ctx = w.__tensor_flatten__()
    before = {n: getattr(w, n).data_ptr() for n in names}
    alias = _host_alias(w)
    assert alias is not w and type(alias) is type(w)
    moved = _host_alias(w)
    for n in names:
        setattr(moved, n, getattr(w, n).clone())  # the "device" copy
    torch.utils.swap_tensors(w, moved)
    assert {n: getattr(alias, n).data_ptr() for n in names} == before
    assert all(getattr(w, n).data_ptr() != before[n] for n in names)
    del torchao


def test_the_direct_promote_drives_diffusers_own_tensor_transfer():
    """Guards diffusers' private ``_process_tensors_from_modules(pinned_memory, default_stream)``: the promote reads the
    group's host-copy dict as its sources."""
    go = pytest.importorskip("diffusers.hooks.group_offloading")
    from core.inference import video_minimax_h3_residency as res

    lin = torch.nn.Linear(16, 16)
    group = go.ModuleGroup(
        modules = [lin],
        offload_device = torch.device("cpu"),
        onload_device = torch.device("cpu"),
        offload_leader = lin,
    )
    host = {p: p.detach().clone() for p in lin.parameters()}
    group.cpu_param_dict = dict(host)
    assert res._onload_pageable(group)
    for p in lin.parameters():
        assert p.data_ptr() == host[p].data_ptr()
