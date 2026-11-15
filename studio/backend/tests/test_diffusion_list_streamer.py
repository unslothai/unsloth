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
        ),  # L4 VM, old accounting: the streamed DiT blocks pin, the 24 GB encoder stays pageable
        (
            40.0,
            False,
            False,
        ),  # 40 GB host (the smallest that loads on L4): nothing pins, everything streams pageable
    ],
)
def test_pins_fit_the_plans_host_room(host, dit_room, te_pinned):
    """Old accounting (pins on top of every copy the plan holds), kept for the plan room arithmetic."""
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


class _Plan:
    def __init__(self, pinned: int, replaced: int) -> None:
        self.pinned_bytes = pinned
        self.replaced_bytes = replaced


@pytest.mark.parametrize("host, credited", [(86.55, True), (53.97, True), (40.0, False)])
def test_pins_that_replace_counted_copies_cost_the_plan_nothing(host, credited):
    """L4: the 24 GB encoder's pins take the place of the host copy the plan counts, so it pins beside the DiT."""
    h3i = _h3()
    vram = 41.96 if host > 80 else 23.46
    plan = h3i.plan_placement(vram_gb = vram, host_gb = host, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    assert h3i.pins_replace_counted(plan) == credited
    base = h3i.pin_room_bytes(plan)
    # a lock that replaced its counted copy leaves the room as it was; one on top still counts
    assert h3i.pin_room_bytes(plan, [_Plan(9_000_000_000, 9_000_000_000)]) == base
    assert h3i.pin_room_bytes(plan, [_Plan(3_000_000_000, 0)]) == max(0, base - 3_000_000_000)
    assert h3i.pin_room_bytes(plan, [_Plan(3_000_000_000, 1_000_000_000), None]) == max(
        0, base - 2_000_000_000
    )


def _budget_only(monkeypatch, budget_mib: int) -> list:
    """Records reaching the pinning step, then raises so the call returns None."""
    import core.inference.diffusion_memory as dm

    went_on: list = []
    monkeypatch.setattr(op, "_list_stream_refusal", lambda device: None)
    monkeypatch.setattr(op, "_go", lambda: type("_Go", (), {"ModuleGroup": object}))
    monkeypatch.setattr(
        dm, "install_group_offload_buffer_restore", lambda: went_on.append(1) or 1 / 0
    )
    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: budget_mib)
    return went_on


def test_replaces_counted_skips_the_callers_cap_but_not_the_live_budget(monkeypatch):
    stack = _Stack()
    went_on = _budget_only(monkeypatch, 1 << 20)
    assert op.stream_module_list(stack, stack.layers, "cuda", 2, max_pin_bytes = 0) is None
    assert went_on == []
    assert (
        op.stream_module_list(
            stack, stack.layers, "cuda", 2, max_pin_bytes = 0, replaces_counted = True
        )
        is None
    )
    assert went_on == [1]
    went_on = _budget_only(monkeypatch, 0)
    assert (
        op.stream_module_list(
            stack, stack.layers, "cuda", 2, max_pin_bytes = 0, replaces_counted = True
        )
        is None
    )
    assert went_on == [] and not stack.layers[3]._forward_pre_hooks


@pytest.mark.parametrize("value", ["0", "off"])
def test_the_replace_kill_switch_restores_the_old_cap(monkeypatch, value):
    stack = _Stack()
    went_on = _budget_only(monkeypatch, 1 << 20)
    monkeypatch.setenv(op.PIN_REPLACES_HOST_ENV, value)
    assert not op.pin_replaces_host_enabled()
    assert (
        op.stream_module_list(
            stack, stack.layers, "cuda", 2, max_pin_bytes = 0, replaces_counted = True
        )
        is None
    )
    assert went_on == [] and not stack.layers[3]._forward_pre_hooks


class _Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        stack = _Stack()
        lm = torch.nn.Module()
        lm.inp = stack.inp
        lm.layers = stack.layers
        model = torch.nn.Module()
        model.language_model = lm
        self.model = model
        self.stack = [stack]


def _record_stream_calls(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(op, "stream_module_list", lambda *a, **k: seen.append(k) or None)
    return seen


@pytest.mark.parametrize(
    "host, per_prompt, want",
    [
        (53.97, False, True),  # L4: the encoder's pins replace its counted host copy
        (40.0, False, False),  # host barely fits: pins stay on top of the copies
        (53.97, True, None),  # per-prompt encoder: never pinned, no call
    ],
)
def test_the_encoder_routes_through_the_counted_copy_credit(monkeypatch, host, per_prompt, want):
    h3i = _h3()
    seen = _record_stream_calls(monkeypatch)
    plan = h3i.plan_placement(vram_gb = 23.46, host_gb = host, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    assert plan.placement == h3i.PLACE_STREAM

    class _Pipe:
        pass

    pipe = _Pipe()
    pipe._unsloth_h3_streamers = []
    enc = _Encoder()
    h3i._place_text_encoder(pipe, enc, plan, "cpu", per_prompt = per_prompt)
    (streamer,) = pipe._unsloth_h3_streamers
    assert isinstance(streamer, h3i.ModuleListStreamer) and streamer._unsloth_role == "text_encoder"
    if want is None:
        assert seen == []
    else:
        assert len(seen) == 1 and seen[0]["replaces_counted"] is want
        # the encoder runs once per request: its copy stream's cache goes back after each encode
        assert seen[0]["release_after_forward"] is True
    x = torch.randn(4, 64)
    stack = enc.stack[0]
    with torch.no_grad():
        got = stack(x)
    streamer.remove()
    with torch.no_grad():
        assert torch.equal(got, stack(x))


def test_the_dit_routes_through_the_counted_copy_credit(monkeypatch):
    h3i = _h3()
    seen = _record_stream_calls(monkeypatch)
    plan = h3i.plan_placement(vram_gb = 23.46, host_gb = 53.97, dit_gb = 21.9, te_gb = 27.2, inflight = 2)
    stack = _Stack()

    class _T(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = stack.inp
            self.transformer_blocks = stack.layers

    class _V(torch.nn.Module):
        pass

    streamer = h3i._place_streamed_dit(_T(), _V(), "cpu", plan, pinned_already = [_Plan(5, 5)])
    assert len(seen) == 1 and seen[0]["replaces_counted"] is True
    assert seen[0]["max_pin_bytes"] == h3i.pin_room_bytes(plan)
    # the DiT streams every step: its copy stream keeps its cache
    assert seen[0]["release_after_forward"] is False
    streamer.remove()


@cuda
def test_a_counted_stream_pins_bit_identically_and_reports_its_bytes(monkeypatch):
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1 << 20)
    ref = _Stack().cuda().eval()
    stack = _Stack().eval()
    stack.inp.cuda()
    streamer = op.stream_module_list(
        stack, stack.layers, "cuda", 1, max_pin_bytes = 0, replaces_counted = True
    )
    assert streamer is not None and streamer.replaced_bytes == streamer.pinned_bytes > 0
    for g in streamer.groups:
        assert all(t.is_pinned() for t in g.cpu_param_dict.values())
    x = torch.randn(8, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for _ in range(3):
            assert torch.equal(stack(x), want)
    assert streamer.stats()["pinned_bytes"] == streamer.pinned_bytes
    streamer.remove()


@cuda
@pytest.mark.parametrize(
    "release, credit, releases",
    [
        (False, True, False),
        (True, True, True),
        (True, False, False),
    ],  # (True, False): pinned before, keeps its cache
)
def test_an_encoder_stream_hands_its_cache_back_after_each_forward(
    monkeypatch, release, credit, releases
):
    import core.inference.diffusion_memory as dm

    monkeypatch.setattr(dm, "_pin_budget_mib", lambda: 1 << 20)
    ref = _Stack().cuda().eval()
    stack = _Stack().eval()
    stack.inp.cuda()
    kw = {"max_pin_bytes": 0, "replaces_counted": True} if credit else {}
    streamer = op.stream_module_list(
        stack, stack.layers, "cuda", 0, release_after_forward = release, **kw
    )
    assert streamer is not None
    calls = []
    real = torch.cuda.empty_cache
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1) or real())
    x = torch.randn(8, 64, device = "cuda")
    with torch.no_grad():
        want = ref(x)
        for _ in range(2):
            assert torch.equal(stack(x), want)
        stack.stop_after = 2
        with pytest.raises(_Stop):
            stack(x)
        stack.stop_after = None
        assert torch.equal(stack(x), want)
    assert len(calls) == (4 if releases else 0)
    streamer.remove()


class _CapturingPrefetcher:
    def __init__(self, capturing):
        self.capturing = capturing
        self.ended = 0

    def end(self):
        self.ended += 1
        self.capturing = False


@pytest.mark.parametrize(
    "release, pf_capturing, cuda_capturing, want",
    [
        (True, False, False, 1),
        (True, True, False, 0),  # end() clears the flag: it must be read before end()
        (True, False, True, 0),
        (False, False, False, 0),
    ],
)
def test_the_end_of_a_streamed_forward_never_empties_the_cache_inside_a_capture(
    monkeypatch, release, pf_capturing, cuda_capturing, want
):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(1))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: cuda_capturing)
    pf = _CapturingPrefetcher(pf_capturing)
    op._end_stream_forward(pf, release)
    assert pf.ended == 1 and len(calls) == want
