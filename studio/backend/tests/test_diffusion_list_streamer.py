# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared pinned ModuleList streamer: CPU refusals, plus CUDA parity / prefetch / remove() checks."""

from __future__ import annotations

import pytest
import torch

import core.inference.diffusion_offload_prefetch as op


class _Stop(Exception):
    pass


class _BufferLinear(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randint(-127, 127, (width, width), dtype = torch.int8))
        self.register_buffer("weight_scale", torch.rand(width, 1) / 64)

    def forward(self, x):
        return x @ (self.weight.to(x.dtype) * self.weight_scale.to(x.dtype)).t()


class _Stack(torch.nn.Module):
    def __init__(
        self,
        n: int = 6,
        width: int = 64,
    ) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.inp = torch.nn.Linear(width, width)
        self.layers = torch.nn.ModuleList(
            torch.nn.Sequential(
                torch.nn.Linear(width, width * 2),
                torch.nn.GELU(),
                torch.nn.Linear(width * 2, width),
                _BufferLinear(width),
            )
            for _ in range(n)
        )
        self.stop_after = None

    def forward(self, x):
        x = self.inp(x)
        for i, layer in enumerate(self.layers):
            x = x + layer(x)
            if self.stop_after is not None and i == self.stop_after:
                raise _Stop()
        return x


def test_cpu_device_is_refused_and_nothing_changes():
    stack = _Stack()
    before = [p.data_ptr() for p in stack.parameters()]
    assert op.stream_module_list(stack, stack.layers, "cpu", 2) is None
    assert [p.data_ptr() for p in stack.parameters()] == before
    assert not stack._forward_pre_hooks and not stack.layers[3]._forward_pre_hooks


def test_kill_switch_refuses(monkeypatch):
    monkeypatch.setenv(op.ASYNC_PREFETCH_ENV, "0")
    assert op._list_stream_refusal("cuda").startswith(op.ASYNC_PREFETCH_ENV)


def test_h3_image_falls_back_to_the_pageable_streamer_off_cuda(monkeypatch):
    pytest.importorskip("diffusers")
    from core.inference import diffusion_minimax_h3_image as h3i

    class _Pipe:
        _unsloth_h3_streamers: list = []

    pipe = _Pipe()
    pipe._unsloth_h3_streamers = []
    stack = _Stack()
    streamer = h3i.stream_list(pipe, stack, stack.layers, "cpu", 2, "transformer")
    assert isinstance(streamer, h3i.ModuleListStreamer) and streamer.streamed == 4
    assert pipe._unsloth_h3_streamers == [streamer] and streamer._unsloth_role == "transformer"
    monkeypatch.setenv(h3i.PINNED_STREAM_ENV, "0")
    assert h3i.stream_inflight() == 1
    monkeypatch.delenv(h3i.PINNED_STREAM_ENV)
    assert h3i.stream_inflight() == 1 + op.LIST_STREAM_DEPTH


def test_a_stream_larger_than_the_pin_budget_is_refused(monkeypatch):
    import core.inference.diffusion_memory as dm

    stack = _Stack()
    went_on = []
    monkeypatch.setattr(op, "_list_stream_refusal", lambda device: None)
    # no diffusers needed: the budget check runs before ModuleGroup is looked up
    monkeypatch.setattr(op, "_go", lambda: type("_Go", (), {"ModuleGroup": object}))
    # the refusal must come from the budget, before any pinning
    monkeypatch.setattr(
        dm, "install_group_offload_buffer_restore", lambda: went_on.append(1) or 1 / 0
    )
    # compared in bytes, not rounded MiB
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 0)
    assert op.stream_module_list(stack, stack.layers, "cuda", 2) is None
    assert not went_on and not stack.layers[3]._forward_pre_hooks
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1)
    assert op.stream_module_list(stack, stack.layers, "cuda", 2) is None
    assert went_on == [1] and not stack.layers[3]._forward_pre_hooks


def test_the_plan_keeps_room_for_the_prefetched_member():
    pytest.importorskip("diffusers")
    from core.inference import diffusion_minimax_h3_image as h3i

    args = dict(vram_gb = 23.46, host_gb = 53.97, dit_gb = 21.9, te_gb = 27.2)
    one = h3i.plan_placement(**args)
    two = h3i.plan_placement(**args, inflight = 2)
    assert one.placement == two.placement == h3i.PLACE_STREAM
    assert one.dit_resident_blocks == 28  # the #17453 L4 plan, unchanged at one in flight
    assert two.dit_resident_blocks in (26, 27)
    a100 = dict(vram_gb = 41.96, host_gb = 86.55, dit_gb = 21.9, te_gb = 27.2)
    assert h3i.plan_placement(**a100).te_resident_layers == 20
    assert h3i.plan_placement(**a100, inflight = 2).te_resident_layers == 19


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")


@cuda
@pytest.mark.parametrize("resident", [0, 2, 5])
def test_streamed_stack_is_bit_identical_and_prefetches(resident):
    ref = _Stack().cuda().eval()
    stack = _Stack().eval()
    stack.inp.cuda()
    streamer = op.stream_module_list(stack, stack.layers, "cuda", resident)
    assert streamer is not None and streamer.streamed == 6 - resident
    for g in streamer.groups:
        assert all(t.is_pinned() for t in g.cpu_param_dict.values())
    x = torch.randn(8, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(3):
            assert torch.equal(stack(x), want)
        stack.stop_after = 3
        with pytest.raises(_Stop):
            stack(x)
        stack.stop_after = None
        assert torch.equal(stack(x), want)
    torch.cuda.synchronize()
    stats = streamer.stats()
    assert stats["forwards"] == 5 and stats["prefetched"] > 0
    assert streamer.prefetcher.inflight_bytes == 0 and not streamer.prefetcher.ready
    for layer in stack.layers[resident:]:
        assert all(p.device.type == "cpu" for p in layer.parameters())
        assert all(b.device.type == "cpu" for b in layer.buffers())
    assert torch.cuda.memory_allocated() <= before + (1 << 20)
    streamer.remove()
    assert not stack._forward_pre_hooks and not stack._forward_hooks
    with torch.no_grad():
        stack.cuda()
        assert torch.equal(stack(x), want)


def test_a_stream_over_the_callers_host_cap_is_refused_before_pinning(monkeypatch):
    import core.inference.diffusion_memory as dm

    stack = _Stack()
    went_on = []
    monkeypatch.setattr(op, "_list_stream_refusal", lambda device: None)
    monkeypatch.setattr(
        dm, "install_group_offload_buffer_restore", lambda: went_on.append(1) or 1 / 0
    )
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1 << 20)
    assert op.stream_module_list(stack, stack.layers, "cuda", 2, max_pin_bytes = 0) is None
    assert not went_on


def _h3():
    pytest.importorskip("diffusers")
    from core.inference import diffusion_minimax_h3_image as h3i
    return h3i


@pytest.mark.parametrize(
    "host, dit_room, te_pinned",
    [
        (
            86.55,
            True,
            True,
        ),  # A100 VM: DiT resident, the encoder's streamed layers fit the plan's room
        (
            53.97,
            True,
            False,
        ),  # L4 VM: the streamed DiT blocks pin, the 24 GB encoder stays pageable
        (
            40.0,
            False,
            False,
        ),  # 40 GB host (the smallest that loads on L4): nothing pins, everything streams pageable
    ],
)
def test_pins_fit_the_plans_host_room(host, dit_room, te_pinned):
    """Pinned copies are counted on top of what the plan holds on the host, so a host the plan fits still fits."""
    h3i = _h3()
    vram = 41.96 if host > 80 else 23.46
    plan = h3i.plan_placement(vram_gb = vram, host_gb = host, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    room = h3i.pin_room_bytes(plan)
    assert room is not None and room >= 0
    assert room == max(0, int((host - plan.host_need_gb) * 1e9))
    dit_stream = (
        (50 - int(plan.dit_resident_blocks or 50)) * 0.398e9
        if plan.placement == h3i.PLACE_STREAM
        else 0
    )
    te_stream = (51 - int(plan.te_resident_layers or 0)) * 0.478e9
    if plan.placement == h3i.PLACE_STREAM:
        assert (dit_stream <= room) == dit_room

    class _Pinned:
        pinned_bytes = int(dit_stream) if dit_room else 0

    left = h3i.pin_room_bytes(plan, [_Pinned()])
    assert (plan.te_kept_on_host and te_stream <= left) == te_pinned
    assert (
        h3i.pin_room_bytes(h3i.plan_placement(vram_gb = vram, host_gb = None, dit_gb = 21.9, te_gb = 27.2))
        is None
    )


def test_a_small_host_streams_pageable_and_still_renders(monkeypatch):
    """40 GB host, stream tier (about 37 GB is the planner floor): the DiT gets no pin room and runs pageable."""
    h3i = _h3()
    import core.inference.diffusion_offload_prefetch as opf

    seen = []
    real = opf.stream_module_list
    monkeypatch.setattr(
        opf,
        "stream_module_list",
        lambda *a, **k: seen.append(k.get("max_pin_bytes")) or real(*a, **k),
    )
    with pytest.raises(RuntimeError, match = "free system memory"):
        h3i.plan_placement(vram_gb = 23.46, host_gb = 32.0, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    plan = h3i.plan_placement(vram_gb = 23.46, host_gb = 40.0, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    assert plan.placement == h3i.PLACE_STREAM and not plan.te_kept_on_host
    stack = _Stack()

    class _T(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = stack.inp
            self.transformer_blocks = stack.layers

    class _V(torch.nn.Module):
        pass

    t = _T()
    streamer = h3i._place_streamed_dit(t, _V(), "cpu", plan)
    assert seen == [0] and isinstance(streamer, h3i.ModuleListStreamer)
    x = torch.randn(4, 64)
    with torch.no_grad():
        out = stack(x)
    streamer.remove()
    with torch.no_grad():
        assert torch.equal(out, stack(x))


@pytest.mark.parametrize("value", ["0", "off"])
def test_the_global_pin_opt_out_keeps_the_pageable_streamer(monkeypatch, value):
    import core.inference.diffusion_memory as dm

    monkeypatch.setenv(dm.GROUP_OFFLOAD_PIN_ENV, value)
    assert op._list_stream_refusal("cuda") == f"{dm.GROUP_OFFLOAD_PIN_ENV}=0"
    stack = _Stack()
    assert op.stream_module_list(stack, stack.layers, "cuda", 2) is None
    assert not stack.layers[3]._forward_pre_hooks
