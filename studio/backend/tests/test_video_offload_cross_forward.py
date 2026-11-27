# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-forward carry on the video path: two-expert (Wan2.2-A14B) routing and the render wiring."""

from __future__ import annotations

import copy
import os
import sys
import types

import pytest
import torch

import core.inference.diffusion_memory as dm
import core.inference.diffusion_offload_prefetch as op

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_offload_cross_forward import (
    _Net,
    _clean_env,  # noqa: F401 - autouse fixture
    _cpu_prefetcher,
    _cuda,
    _forward,
)  # noqa: E402


def _wan_class(module = "diffusers.pipelines.wan.pipeline_wan"):
    def call(self, **kwargs):
        return None

    return type("WanPipeline", (), {"__call__": call, "__module__": module})


def _wan_pipe(
    first,
    second,
    timesteps,
    ratio = 0.875,
    cls = None,
):
    pipe = (cls or _wan_class())()
    pipe.transformer = first
    pipe.transformer_2 = second
    pipe.components = {"transformer": first, "transformer_2": second}
    pipe.config = types.SimpleNamespace(boundary_ratio = ratio)
    pipe.scheduler = types.SimpleNamespace(
        timesteps = torch.tensor(timesteps, dtype = torch.float32),
        config = types.SimpleNamespace(num_train_timesteps = 1000),
    )
    pipe._interrupt = False
    pipe._num_timesteps = len(timesteps)
    return pipe


def _holder(pf):
    return types.SimpleNamespace(**{op.PREFETCHER_ATTR: pf})


def _two_experts(depth = 2):
    a, ga, la = _cpu_prefetcher("abcdef", depth = depth)
    b, gb, lb = _cpu_prefetcher("uvwxyz", depth = depth)
    _forward(a, ga)  # records the orders
    _forward(b, gb)
    return (a, ga, la), (b, gb, lb)


def _wan_loop(
    pipe,
    experts,
    cfg = True,
    timesteps_seen = None,
):
    """diffusers' WanPipeline denoise loop: the expert picked per step by ``t >= boundary``."""
    boundary = pipe.config.boundary_ratio * 1000 if pipe.config.boundary_ratio is not None else None
    ran = []
    with op.cross_forward_render(pipe):
        seq = (
            torch.tensor(timesteps_seen, dtype = torch.float32)
            if timesteps_seen
            else pipe.scheduler.timesteps
        )
        for i, t in enumerate(seq):
            pf, groups = experts[0] if boundary is None or t >= boundary else experts[1]
            for _ in range(2 if cfg else 1):
                _forward(pf, groups)
            ran.append(pf)
            op.cross_forward_step(pipe, i, timestep = t)
    return ran


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("switch_at", [1, 2, 4, 5])
def test_two_experts_never_carry_across_the_switch(cfg, switch_at):
    (a, ga, la), (b, gb, lb) = _two_experts()
    steps = 6
    ts = [999.0 - i for i in range(switch_at)] + [500.0 - i for i in range(steps - switch_at)]
    pipe = _wan_pipe(_holder(a), _holder(b), ts)
    la.clear()
    lb.clear()
    begun_b = b.begin

    def begin_b():
        assert (
            not a.ready and not a.carried
        ), "the high-noise expert's copies sat into the low-noise forward"
        begun_b()

    b.begin = begin_b
    ran = _wan_loop(pipe, [(a, ga), (b, gb)], cfg = cfg)
    assert ran == [a] * switch_at + [b] * (steps - switch_at)
    for pf in (a, b):
        # every carried copy was read by the next forward of the same expert, except a switch right after step 0:
        # no callback runs before step 0's carry, so step 0's callback drops it (before the other expert begins)
        wasted = 2 if pf is a and switch_at == 1 else 0
        assert pf.stats.get("carried", 0) == pf.stats.get("carried_hits", 0) + wasted
        assert pf.stats.get("carried_dropped", 0) == wasted
        assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
        assert pf.stats["missed"] == 0
    # the low-noise expert carries within its own run of steps (none when it runs only the last step)
    assert (b.stats.get("carried", 0) > 0) == (steps - switch_at >= 2)
    if switch_at > 1 or cfg:
        assert a.stats.get("carried", 0) > 0


def test_unreadable_schedule_carries_nothing_more():
    (a, ga, _), (b, gb, _) = _two_experts()
    ts = [999.0, 998.0, 997.0, 500.0]
    pipe = _wan_pipe(_holder(a), _holder(b), ts)
    # the timesteps the loop runs are not the scheduler's: the routing cannot be read back
    _wan_loop(pipe, [(a, ga), (b, gb)], cfg = False, timesteps_seen = [999.0, 600.0, 599.0, 500.0])
    assert a.stats.get("carried_hits", 0) == 0 and not a.armed and not b.armed
    assert not a.ready and not b.ready and a.inflight_bytes == 0 and b.inflight_bytes == 0


def test_second_denoiser_under_another_loop_is_not_armed():
    (a, _, _), (b, _, _) = _two_experts()
    pipe = _wan_pipe(
        _holder(a), _holder(b), [999.0, 500.0], cls = _wan_class("diffusers.pipelines.lucy.x")
    )
    with op.cross_forward_render(pipe) as pfs:
        assert pfs == [] and not a.armed and not b.armed


def test_subclass_overriding_the_loop_is_not_armed():
    base = _wan_class()
    sub = type("Sub", (base,), {"__call__": lambda self, **k: None, "__module__": "studio.x"})
    (a, _, _), (b, _, _) = _two_experts()
    pipe = _wan_pipe(_holder(a), _holder(b), [999.0, 500.0], cls = sub)
    with op.cross_forward_render(pipe) as pfs:
        assert pfs == []
    sub2 = type("Sub2", (base,), {"__module__": "studio.x"})  # inherits Wan's loop unchanged
    pipe = _wan_pipe(_holder(a), _holder(b), [999.0, 500.0], cls = sub2)
    with op.cross_forward_render(pipe) as pfs:
        assert len(pfs) == 2


def test_no_boundary_runs_the_first_expert_throughout():
    (a, ga, _), (b, gb, _) = _two_experts()
    pipe = _wan_pipe(_holder(a), _holder(b), [999.0, 500.0, 100.0, 50.0], ratio = None)
    ran = _wan_loop(pipe, [(a, ga), (b, gb)], cfg = False)
    assert ran == [a] * 4
    assert (
        a.stats["carried"] == a.stats["carried_hits"] > 0 and a.stats.get("carried_dropped", 0) == 0
    )


def test_video_render_arms_only_with_the_step_callback(monkeypatch):
    import core.inference.video as video

    pf, groups, _ = _cpu_prefetcher("abcdef")
    seen = []

    class _Pipe:
        def __init__(self):
            self.components = {"transformer": _holder(pf)}
            self._interrupt = False

        def __call__(self, **kwargs):
            seen.append(pf.armed)
            return "out"

    assert video._carried_call(_Pipe(), {}, True) == "out"
    assert video._carried_call(_Pipe(), {}, False) == "out"
    monkeypatch.setenv(op.CROSS_FORWARD_ENV, "0")
    assert video._carried_call(_Pipe(), {}, True) == "out"
    assert seen == [True, False, False] and not pf.armed


def _streamed(net):
    pipe = types.SimpleNamespace(transformer = net, components = {"transformer": net}, _interrupt = False)
    assert dm._apply_group_offload(pipe, "cuda", None)
    return op.module_prefetcher(net)


@pytest.mark.parametrize("cfg", [False, True])
def test_streamed_two_experts_bit_identical(cfg):
    _cuda()
    torch.manual_seed(0)
    nets = [_Net(), _Net()]
    refs = [copy.deepcopy(n).cuda() for n in nets]
    pfs = [_streamed(n) for n in nets]
    pipe = _wan_pipe(nets[0], nets[1], [999.0, 950.0, 900.0, 600.0, 400.0, 200.0])
    pipe.scheduler.timesteps = (
        pipe.scheduler.timesteps.cuda()
    )  # the device schedule diffusers keeps
    ts = pipe.scheduler.timesteps
    xs = [torch.randn(4, 64, device = "cuda") for _ in range(2)]

    def run(models, on_step):
        x = xs[0]
        for i, t in enumerate(ts):
            model = models[0] if t >= 875.0 else models[1]  # the Wan loop's own routing
            v = model(x)
            if cfg:
                v = v + 0.5 * model(xs[1] + x)
            x = x - 0.1 * v
            on_step(i, t)
        return x

    with torch.no_grad():
        want = run(refs, lambda i, t: None)
        for n in nets:
            n(xs[0])  # records the order
        with op.cross_forward_render(pipe):
            got = run(nets, lambda i, t: op.cross_forward_step(pipe, i, timestep = t))
            torch.cuda.synchronize()
            for n, pf in zip(nets, pfs):
                assert all(b.weight.device.type == "cpu" for b in n.blocks)
                assert not pf.ready and not pf.carried and pf.inflight_bytes == 0
    assert torch.equal(got, want)
    for pf in pfs:
        assert pf.stats["carried_hits"] == pf.stats["carried"] > 0
        assert pf.stats.get("carried_dropped", 0) == 0 and pf.stats["missed"] == 0


def test_a_new_schedule_is_routed_afresh():
    (a, ga, _), (b, gb, _) = _two_experts()
    pipe = _wan_pipe(_holder(a), _holder(b), [999.0, 998.0, 997.0, 500.0, 499.0])
    assert _wan_loop(pipe, [(a, ga), (b, gb)], cfg = False) == [a] * 3 + [b] * 2
    pipe.scheduler.timesteps = torch.tensor([999.0, 500.0, 499.0, 498.0, 497.0])
    assert _wan_loop(pipe, [(a, ga), (b, gb)], cfg = False) == [a] + [b] * 4
    for pf in (a, b):
        assert pf.stats.get("carried", 0) == pf.stats.get("carried_hits", 0) + pf.stats.get(
            "carried_dropped", 0
        )
        assert not pf.ready and pf.inflight_bytes == 0
