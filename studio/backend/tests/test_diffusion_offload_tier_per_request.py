# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whole-module vs streamed offload chosen per request."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import core.inference.diffusion as dif
import core.inference.diffusion_memory as dm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_diffusion_placement_monotonic import (  # noqa: E402, F401 - the autouse env fixture
    _card_probe,
    _clean_env,
    _klein_above_floor,
    _priced,
)

KLEIN_DIT_MIB = 7393


@pytest.fixture(autouse = True)
def _per_request_env(monkeypatch):
    monkeypatch.delenv(dm.WHOLE_MODULE_OFFLOAD_ENV, raising = False)
    monkeypatch.delenv(dm.WHOLE_MODULE_PER_REQUEST_ENV, raising = False)


def _tier_pipe(card):
    h2d, tflops = _card_probe(card)
    pipe = types.SimpleNamespace()
    setattr(
        pipe,
        dm.WHOLE_TIER_ATTR,
        {"dit_mib": KLEIN_DIT_MIB, "h2d_gib_s": h2d, "bf16_tflops": tflops},
    )
    return pipe


def _whole(pipe, steps, **kw):
    costs = dm.whole_tier_request_costs(
        pipe, steps = steps, width = kw.pop("width", 1024), height = kw.pop("height", 1024), **kw
    )
    return None if costs is None else costs[0] < costs[1]


def test_streamed_load_records_the_probe_for_requests(monkeypatch):
    """RTX PRO 6000 at 4 steps streams and keeps the probe; a card that priced whole keeps the #15129 plan."""
    flat = _klein_above_floor(monkeypatch)
    refined, plan = _priced(monkeypatch, 12544, flat, steps = 4, card = "rtx_pro_6000")
    assert plan.offload_policy == dm.OFFLOAD_GROUP
    import torch

    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = torch.bfloat16))
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: _card_probe("rtx_pro_6000"))
    dm.prefer_whole_module_offload(pipe, refined, steps = 4)
    tier = getattr(pipe, dm.WHOLE_TIER_ATTR)
    assert tier["dit_mib"] == KLEIN_DIT_MIB
    assert (tier["h2d_gib_s"], tier["bf16_tflops"]) == _card_probe("rtx_pro_6000")

    for card in ("b200", "a100"):
        pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = torch.bfloat16))
        monkeypatch.setattr(dm, "_offload_cost_measurements", lambda card = card: _card_probe(card))
        assert (
            dm.prefer_whole_module_offload(pipe, refined, steps = 4).offload_policy
            == dm.OFFLOAD_MODEL
        )
        assert not hasattr(pipe, dm.WHOLE_TIER_ATTR)

    monkeypatch.setenv(dm.WHOLE_MODULE_PER_REQUEST_ENV, "0")
    pipe = types.SimpleNamespace(transformer = types.SimpleNamespace(dtype = torch.bfloat16))
    monkeypatch.setattr(dm, "_offload_cost_measurements", lambda: _card_probe("rtx_pro_6000"))
    assert dm.prefer_whole_module_offload(pipe, refined, steps = 4).offload_policy == dm.OFFLOAD_GROUP
    assert not hasattr(pipe, dm.WHOLE_TIER_ATTR)


def test_request_step_count_picks_the_tier():
    """The measured RTX PRO 6000 calls: streamed at 2 and 4 steps, whole at 8."""
    pipe = _tier_pipe("rtx_pro_6000")
    assert [_whole(pipe, n) for n in (1, 2, 4, 8, 16)] == [False, False, False, True, True]


def test_request_size_moves_the_crossover():
    """More tokens per step (canvas, batch, CFG) let streaming win to more steps."""
    pipe = _tier_pipe("rtx_pro_6000")
    assert _whole(pipe, 8) is True
    assert _whole(pipe, 8, batch_size = 2) is False
    assert _whole(pipe, 8, cfg = True) is False
    assert _whole(pipe, 8, width = 1536, height = 1536) is False
    assert _whole(pipe, 4, width = 512, height = 512) is True


def test_no_tier_or_kill_switch_prices_nothing(monkeypatch):
    assert (
        dm.whole_tier_request_costs(types.SimpleNamespace(), steps = 8, width = 1024, height = 1024)
        is None
    )
    pipe = _tier_pipe("rtx_pro_6000")
    assert dm.whole_tier_request_costs(pipe, steps = None, width = 1024, height = 1024) is None
    monkeypatch.setenv(dm.WHOLE_MODULE_PER_REQUEST_ENV, "0")
    assert dm.whole_tier_request_costs(pipe, steps = 8, width = 1024, height = 1024) is None


class _Bg:
    def __init__(self, pending):
        self._pending = pending

    def pending(self):
        return self._pending


class _Graph:
    def __init__(
        self,
        declined,
        poisoned = False,
    ):
        self._declined = declined
        self.poisoned = poisoned
        self.fallback_handle = None
        self.fallback = None

    def _placement_declined(self):
        return self._declined


def _state(**kw):
    pipe = _tier_pipe("rtx_pro_6000")
    pipe._unsloth_cuda_graphs = kw.pop("graphs", ())
    base = dict(
        pipe = pipe,
        offload_policy = dm.OFFLOAD_GROUP,
        transformer_cache = None,
        bg_compile = None,
        cuda_graphs = (),
        device = "cuda:0",
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def _request(state, steps = 8):
    return dif._whole_tier_for_request(
        state, steps = steps, width = 1024, height = 1024, batch_size = 1, cfg = False, condition_pixels = 0
    )


LIVE = ("flux.2-klein", "dense:bfloat16|attn:_native_cudnn", "default")
LOADED = ("flux.2-klein", "dense:bfloat16|attn:native", "default")
RECORDED = ("dense:bfloat16|attn:native", "off")  # the eager renders' key


@pytest.fixture
def _learned(monkeypatch):
    seen = []

    def fake_peak(family, speed, **kw):
        seen.append((kw.get("quant"), speed, kw.get("phase")))
        return 2560 if (kw.get("quant"), speed) == RECORDED else None

    monkeypatch.setattr(dif, "_live_peak_identity", lambda state: LIVE)
    monkeypatch.setattr(dif, "measured_image_runtime_mib", fake_peak)
    monkeypatch.setattr(dif.cuda_graph, "live_graph_count", lambda index: 0)
    monkeypatch.setattr(dif, "_peak_device_index", lambda state: 0)
    return seen


def test_request_gate_prices_and_reads_the_eager_peak(_learned):
    """A deferred compile tier records no peak: the eager renders' pre-decode peak sizes the pinned call."""
    state = _state()
    state.pipe._unsloth_peak_identity = LOADED
    whole, streamed, activation = _request(state)
    assert whole < streamed and activation == 2560
    assert _learned == [
        (LIVE[1], "default", "pre_decode"),
        (LIVE[1], "off", "pre_decode"),
        (LOADED[1], "default", "pre_decode"),
        (LOADED[1], "off", "pre_decode"),
    ]
    assert _request(state, steps = 4) is None


def test_request_gate_reads_the_live_key_first(_learned, monkeypatch):
    monkeypatch.setattr(dif, "_live_peak_identity", lambda state: ("flux.2-klein", *RECORDED))
    state = _state()
    state.pipe._unsloth_peak_identity = LOADED
    assert _request(state)[2] == 2560
    assert _learned == [(*RECORDED, "pre_decode")]


@pytest.mark.parametrize(
    "kw",
    [
        dict(offload_policy = dm.OFFLOAD_MODEL),
        dict(transformer_cache = "fbcache"),
        dict(bg_compile = _Bg(True)),
        dict(graphs = (_Graph(declined = False),)),
    ],
)
def test_request_gate_keeps_the_load_plan(_learned, kw):
    assert _request(_state(**kw)) is None


def test_request_gate_after_graphs_settle(_learned, monkeypatch):
    def state(**kw):
        out = _state(**kw)
        out.pipe._unsloth_peak_identity = LOADED
        return out

    assert _request(state(bg_compile = _Bg(False), graphs = (_Graph(declined = True),))) is not None
    assert _request(state(graphs = (_Graph(declined = False, poisoned = True),))) is not None
    monkeypatch.setattr(dif.cuda_graph, "live_graph_count", lambda index: 1)
    assert _request(state(graphs = (_Graph(declined = True),))) is None


def test_request_gate_without_a_learned_peak_streams(_learned, monkeypatch):
    monkeypatch.setattr(dif, "measured_image_runtime_mib", lambda *a, **k: None)
    assert _request(_state()) is None


def test_request_gate_declines_condition_images(_learned):
    """Condition-image requests stream."""
    state = _state()
    state.pipe._unsloth_peak_identity = LOADED
    assert (
        dif._whole_tier_for_request(
            state,
            steps = 8,
            width = 1024,
            height = 1024,
            batch_size = 1,
            cfg = False,
            condition_pixels = 1 << 20,
        )
        is None
    )
    assert _learned == []


def test_request_gate_reads_the_workflow_peak(_learned):
    """An img2img request is sized from the peak recorded under its own workflow, never the text-to-image one."""
    state = _state()
    state.pipe._unsloth_peak_identity = LOADED
    out = dif._whole_tier_for_request(
        state,
        steps = 8,
        width = 1024,
        height = 1024,
        batch_size = 1,
        cfg = False,
        condition_pixels = 0,
        workflow = "img2img",
    )
    assert out is None  # only the text-to-image key holds a peak in the fixture
    assert _learned and all(q.endswith("|wf:img2img") for q, _, _ in _learned)


def test_the_request_gate_prices_the_largest_chunk():
    """The tier is priced on the forward the request runs, not the activation guard's singleton floor."""
    import inspect

    src = inspect.getsource(dif)
    call = src.split("whole_tier = _whole_tier_for_request(", 1)[1].split(")\n", 1)[0]
    assert "batch_size = max((len(chunk) for chunk in chunks), default = 1)" in call
    assert "guard_batch" not in call


def _cuda_net():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA-only: diffusers group offloading with a copy stream")
    pytest.importorskip("diffusers.hooks")
    from diffusers.hooks import apply_group_offloading

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj_in = torch.nn.Linear(64, 1024)
            self.blocks = torch.nn.ModuleList(torch.nn.Linear(1024, 1024) for _ in range(6))
            self.proj_out = torch.nn.Linear(1024, 64)

        def forward(self, x):
            x = self.proj_in(x)
            for block in self.blocks:
                x = torch.nn.functional.gelu(block(x))
            return self.proj_out(x)

    torch.manual_seed(0)
    net = Net()
    x = torch.randn(3, 64)
    ref = net.to("cuda")(x.cuda()).cpu()
    net.to("cpu")
    apply_group_offloading(
        net,
        onload_device = torch.device("cuda"),
        offload_device = torch.device("cpu"),
        offload_type = "block_level",
        num_blocks_per_group = 1,
        use_stream = True,
        record_stream = True,
        non_blocking = True,
    )
    return torch, net, x, ref


def test_pin_for_call_is_bit_identical_and_streams_again():
    """Every group resident for the call, same output, then the exact pre-call placement (same host tensors)."""
    torch, net, x, ref = _cuda_net()
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    hosts = {id(p): p.data for p in net.blocks.parameters()}
    release = dm.pin_denoiser_for_call(pipe, "cuda", activation_mib = 0)
    assert release is not None
    assert all(p.device.type == "cuda" for p in net.parameters())
    assert dm.denoisers_pinned_resident(pipe)
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)
    release()
    release()  # idempotent
    assert all(p.device.type == "cpu" for p in net.blocks.parameters())
    assert all(p.data.data_ptr() == hosts[id(p)].data_ptr() for p in net.blocks.parameters())
    assert not getattr(net, "_unsloth_resident_room", None)
    assert not any(getattr(g, "_unsloth_resident", False) for g in dm._offload_groups(net))
    assert dm.resident_group_mib(pipe) == 0
    for _ in range(2):
        assert torch.equal(net(x.cuda()).cpu(), ref)


def test_pin_for_call_needs_the_room(monkeypatch):
    torch, net, x, ref = _cuda_net()
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net})
    torch.cuda.empty_cache()  # the room counts the allocator's reusable cache
    free, total = torch.cuda.mem_get_info()
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (1 << 20, total))
    assert dm.pin_denoiser_for_call(pipe, "cuda", activation_mib = 0) is None
    assert all(p.device.type == "cpu" for p in net.blocks.parameters())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device = None: (free, total))
    assert dm.pin_denoiser_for_call(pipe, "cuda", activation_mib = 1 << 30) is None
    release = dm.pin_denoiser_for_call(pipe, "cuda", activation_mib = 64)
    assert release is not None
    release()
    assert torch.equal(net(x.cuda()).cpu(), ref)


def test_request_gate_keeps_the_eager_key_once_the_profile_rewrites_the_load_key(
    _learned, monkeypatch
):
    """The deferred profile rewrites the load's key and nothing records after capture: the eager key is remembered."""
    live = {"id": ("flux.2-klein", *RECORDED)}
    monkeypatch.setattr(dif, "_live_peak_identity", lambda state: live["id"])
    state = _state()
    state.pipe._unsloth_peak_identity = ("flux.2-klein", *RECORDED)
    assert _request(state, steps = 2) is None  # streamed, but the eager key is kept
    live["id"] = LIVE
    state.pipe._unsloth_peak_identity = LIVE
    _learned.clear()
    assert _request(state)[2] == 2560
    assert _learned[-1] == (*RECORDED, "pre_decode")
