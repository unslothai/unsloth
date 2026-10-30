# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-block U-Net compile (SDXL) and the background compile that a render pauses between blocks instead of waiting
for the whole compile. CPU only: ``Module.compile`` is lazy, and the "compiled" block calls below are stand-ins
that take time like a block graph's first compile."""

from __future__ import annotations

import logging
import threading
import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_bg_compile as bg  # noqa: E402
from core.inference import diffusion_speed as ds  # noqa: E402
from core.inference.diffusion_autoopt import targets  # noqa: E402

LOG = logging.getLogger("test_unet_block_compile")


# ---------------------------------------------------------------- switches


@pytest.mark.parametrize(
    "raw, on",
    [
        (None, True),
        ("", True),
        ("1", True),
        ("0", False),
        ("off", False),
        ("False", False),
        ("no", False),
    ],
)
def test_unet_regional_switch(monkeypatch, raw, on):
    if raw is None:
        monkeypatch.delenv(targets.UNET_REGIONAL_ENV, raising = False)
    else:
        monkeypatch.setenv(targets.UNET_REGIONAL_ENV, raw)
    assert targets.unet_regional_enabled() is on
    unet = _tiny_unet()
    assert ds.unet_compiles_regionally(unet) is on
    assert ds.unet_compiles_regionally(unet, max_autotune = True) is False
    assert ds.unet_compiles_regionally(None) is False
    # nothing to compile per block (not a torch module, or no repeated block): the whole-module compile
    assert ds.unet_compiles_regionally(object()) is False
    assert ds.unet_compiles_regionally(torch.nn.Linear(2, 2)) is False


@pytest.mark.parametrize(
    "raw, eager_s, want",
    [
        (None, None, 0.0),
        (None, 2.8, 2.8),
        (None, 99.0, 30.0),
        (None, -1.0, 0.0),
        (None, float("nan"), 0.0),
        ("5", 2.8, 5.0),
        ("0", 2.8, 0.0),
        ("junk", 2.8, 2.8),
    ],
)
def test_slice_seconds(monkeypatch, raw, eager_s, want):
    if raw is None:
        monkeypatch.delenv(bg.SLICE_ENV, raising = False)
    else:
        monkeypatch.setenv(bg.SLICE_ENV, raw)
    assert bg.slice_seconds(eager_s) == pytest.approx(want)


# ---------------------------------------------------------------- the per-block compile of a real (tiny) U-Net


def _tiny_unet():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.UNet2DConditionModel(
        sample_size = 8,
        in_channels = 4,
        out_channels = 4,
        layers_per_block = 1,
        block_out_channels = (8, 16),
        down_block_types = ("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types = ("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim = 8,
        attention_head_dim = 2,
        norm_num_groups = 4,
    ).eval()


def _compiled_blocks(unet):
    return [
        m
        for m in unet.modules()
        if m is not unet and getattr(m, "_compiled_call_impl", None) is not None
    ]


@pytest.fixture
def _restore_dynamo_limits():
    import torch._dynamo

    cfg = torch._dynamo.config
    saved = {a: getattr(cfg, a) for a in ("recompile_limit", "cache_size_limit") if hasattr(cfg, a)}
    yield
    for a, v in saved.items():
        setattr(cfg, a, v)


def test_unet_compiles_its_repeated_blocks_not_the_whole_module(
    monkeypatch, _restore_dynamo_limits
):
    monkeypatch.delenv(targets.UNET_REGIONAL_ENV, raising = False)
    unet = _tiny_unet()
    pipe = SimpleNamespace(unet = unet)
    assert ds._compile_repeated_blocks(pipe, LOG) is True
    assert getattr(unet, "_compiled_call_impl", None) is None
    names = {type(m).__name__ for m in _compiled_blocks(unet)}
    assert names == set(targets.UNET_REGIONAL_BLOCKS)
    # every compiled block sits behind the compile guard (fallback to eager, background-compile gating)
    assert all(
        getattr(m._compiled_call_impl, "_unsloth_compile_guard", None)
        for m in _compiled_blocks(unet)
    )
    assert unet._unsloth_bg_yield is True
    assert unet._unsloth_regional_compile_kwargs == {"fullgraph": True, "dynamic": False}
    # the guard is found like a DiT's, so a compile failure is reported
    assert unet in ds._guarded_dits(pipe)
    assert ds.compile_fallback_error(pipe) is None
    unet._unsloth_compile_guard.error = "InductorError: boom"
    assert ds.compile_fallback_error(pipe) == "InductorError: boom"


def test_unet_kill_switch_keeps_the_whole_module_compile(monkeypatch, _restore_dynamo_limits):
    monkeypatch.setenv(targets.UNET_REGIONAL_ENV, "0")
    unet = _tiny_unet()
    assert ds._compile_repeated_blocks(SimpleNamespace(unet = unet), LOG) is True
    assert getattr(unet, "_compiled_call_impl", None) is not None
    assert _compiled_blocks(unet) == []  # no block compiled on its own
    assert not getattr(unet, "_unsloth_bg_yield", False)


def test_max_autotune_keeps_the_whole_module_compile(monkeypatch, _restore_dynamo_limits):
    monkeypatch.delenv(targets.UNET_REGIONAL_ENV, raising = False)
    unet = _tiny_unet()
    assert ds._compile_repeated_blocks(SimpleNamespace(unet = unet), LOG, max_autotune = True) is True
    assert getattr(unet, "_compiled_call_impl", None) is not None


def test_compile_cache_key_names_the_block_graphs(monkeypatch):
    from core.inference.diffusion_autoopt import plan

    unet = _tiny_unet()
    pipe = SimpleNamespace(unet = unet, vae = None)
    kw = dict(speed_mode = "default", fullgraph = True, offload_below_hooks = False, deferred = True)
    monkeypatch.delenv(targets.UNET_REGIONAL_ENV, raising = False)
    assert plan.compile_cache_kwargs(pipe, **kw)["unet_graph"] == "blocks"
    monkeypatch.setenv(targets.UNET_REGIONAL_ENV, "0")
    assert "unet_graph" not in plan.compile_cache_kwargs(pipe, **kw)
    # a DiT pipe's key is unchanged
    monkeypatch.delenv(targets.UNET_REGIONAL_ENV, raising = False)
    assert "unet_graph" not in plan.compile_cache_kwargs(SimpleNamespace(vae = None), **kw)


# ---------------------------------------------------------------- the yielding background compile


class _Blocks(torch.nn.Module):
    """A denoiser of N guarded blocks whose first compiled call takes ``compile_s`` (a block graph compile)."""

    def __init__(
        self,
        n: int = 6,
        compile_s: float = 0.15,
        yields: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(4, 4) for _ in range(n))
        self.compiled_calls = 0
        self.compiled_blocks: set = set()
        for i, blk in enumerate(self.blocks):
            blk._compiled_call_impl = self._fake_compiled(i, blk, compile_s)
        ds.guard_compiled_blocks(self, LOG)
        if yields:
            self._unsloth_bg_yield = True

    def _fake_compiled(self, index, blk, compile_s):
        def compiled(*args, **kwargs):
            if index not in self.compiled_blocks:
                time.sleep(compile_s)
                self.compiled_blocks.add(index)
            self.compiled_calls += 1
            return blk.forward(*args, **kwargs)

        return compiled

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


def _armed(net):
    job = bg.arm(net, logger = LOG)
    assert job is not None
    with torch.no_grad(), bg.force_eager():
        net(torch.randn(2, 4))  # records the sample; the guard routes the blocks eager
    assert net.compiled_calls == 0 and len(job.samples) == 1
    return job


def _wait_for(pred, timeout = 10.0):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise AssertionError("timed out")
        time.sleep(0.01)


def test_a_render_pauses_the_compile_between_blocks_and_it_resumes(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 6, compile_s = 0.15)
    job = _armed(net)
    assert job.yields()
    assert job.kick() and job.compiling()
    _wait_for(lambda: len(net.compiled_blocks) >= 1)
    waited, paused = job.pause(budget_s = 0.0)
    assert paused
    assert waited < 0.15 + 0.5  # at most the block graph in flight
    done_at_pause = len(net.compiled_blocks)
    assert done_at_pause < 6
    time.sleep(0.5)  # parked: no progress while the render runs
    assert len(net.compiled_blocks) == done_at_pause
    assert job.compiling()
    # the render runs eager beside the parked compile
    with torch.no_grad(), bg.force_eager():
        calls = net.compiled_calls
        net(torch.randn(2, 4))
        assert net.compiled_calls == calls
    job.resume()
    _wait_for(lambda: job.state == "done")
    assert len(net.compiled_blocks) == 6
    assert job.pauses == 1 and job.paused_s >= 0.4
    assert job.describe()["pauses"] == 1


def test_a_compile_that_ends_inside_the_slice_is_not_paused(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 4, compile_s = 0.3)
    job = _armed(net)
    job.kick()
    _wait_for(lambda: job.graph_s is not None)  # one block graph timed
    assert job.graph_s == pytest.approx(0.3, abs = 0.15)
    waited, paused = job.pause(budget_s = 5.0)
    assert not paused and job.state == "done"
    assert waited < 5.0
    job.resume()  # harmless without a pause held


def test_the_pause_is_asked_one_block_graph_before_the_slice_ends(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 12, compile_s = 0.4)
    job = _armed(net)
    job.kick()
    _wait_for(lambda: job.graph_s is not None)
    waited, paused = job.pause(budget_s = 1.0)
    assert paused
    # asked at ~0.6 s, the graph in flight ends by ~1.0 s (+ its own jitter); never slice + a whole graph
    assert 0.5 <= waited < 1.0 + 0.3
    job.resume()
    _wait_for(lambda: job.state == "done")


def test_back_to_back_renders_cannot_starve_the_compile(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 5, compile_s = 0.3)
    job = _armed(net)
    job.kick()
    renders = 0
    while job.compiling() and renders < 50:
        _, paused = job.pause(budget_s = 0.0)  # every render asks at once
        if paused:
            job.resume()
        renders += 1
    _wait_for(lambda: job.state == "done")
    # each resume owes one block graph before the next park: at most one render per graph (+ the first)
    assert renders <= 5 + 1
    assert len(net.compiled_blocks) == 5


def test_nested_pause_runs_no_second_slice(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 6, compile_s = 0.2)
    job = _armed(net)
    job.kick()
    _, paused = job.pause(budget_s = 0.0)
    assert paused
    t0 = time.time()
    _, paused2 = job.pause(budget_s = 5.0)  # a hires pass inside the paused render
    assert paused2 and time.time() - t0 < 1.0
    job.resume()
    time.sleep(0.4)
    assert job.compiling()  # still held by the outer render
    job.resume()
    _wait_for(lambda: job.state == "done")


def test_close_ends_a_parked_compile(monkeypatch):
    monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    net = _Blocks(n = 6, compile_s = 0.15)
    job = _armed(net)
    job.kick()
    _, paused = job.pause(budget_s = 0.0)
    assert paused
    thread = job._thread
    job.close(timeout = 5.0)
    assert not thread.is_alive()
    assert job.state == "failed" and "closed" in (job.error or "")


@pytest.mark.parametrize("env, flag", [("0", True), (None, False)])
def test_without_yield_a_pause_waits_for_the_whole_compile(monkeypatch, env, flag):
    if env is None:
        monkeypatch.delenv(bg.YIELD_ENV, raising = False)
    else:
        monkeypatch.setenv(bg.YIELD_ENV, env)
    net = _Blocks(n = 4, compile_s = 0.1, yields = flag)
    job = _armed(net)
    assert not job.yields()
    job.kick()
    _wait_for(lambda: len(net.compiled_blocks) >= 1)
    _, paused = job.pause(budget_s = 0.0)
    # no yield point is armed: the thread never parks, the pause returns when the compile ends
    assert job.state == "done" and len(net.compiled_blocks) == 4
    job.resume()


def test_yield_point_is_a_no_op_off_the_compile_thread():
    bg.yield_point()  # no active compile on this thread


# ---------------------------------------------------------------- the render side


class _FakeBg:
    def __init__(
        self,
        yields: bool,
        compiling: bool = True,
    ) -> None:
        self._yields, self._compiling = yields, compiling
        self.calls: list = []

    def compiling(self):
        return self._compiling

    def yields(self):
        return self._yields

    def pause(
        self,
        cancel = None,
        budget_s = 0.0,
    ):
        self.calls.append(("pause", budget_s))
        return 0.1, True

    def wait(self, cancel = None):
        self.calls.append(("wait",))
        return 0.2


def _state(loras = ()):
    pipe = SimpleNamespace(_unsloth_loras = loras, _unsloth_eager_render_s = 2.5)
    return SimpleNamespace(pipe = pipe, speed_optims = ())


def test_render_pauses_a_yielding_compile_for_one_slice(monkeypatch):
    from core.inference import diffusion as d

    monkeypatch.delenv(bg.SLICE_ENV, raising = False)
    fake = _FakeBg(yields = True)
    assert d._wait_or_pause_bg_compile(fake, _state(), None, threading.Event()) is fake
    assert fake.calls == [("pause", 2.5)]


@pytest.mark.parametrize(
    "yields, loras, attached",
    [(False, None, ()), (True, [("some/lora", 1.0)], ()), (True, None, (("a", "p", 1.0),))],
)
def test_render_waits_for_the_whole_compile_otherwise(yields, loras, attached):
    from core.inference import diffusion as d

    fake = _FakeBg(yields = yields)
    assert d._wait_or_pause_bg_compile(fake, _state(attached), loras, threading.Event()) is None
    assert fake.calls == [("wait",)]


def test_no_compile_in_flight_neither_waits_nor_pauses():
    from core.inference import diffusion as d

    fake = _FakeBg(yields = True, compiling = False)
    assert d._wait_or_pause_bg_compile(fake, _state(), None, threading.Event()) is None
    assert d._wait_or_pause_bg_compile(None, _state(), None, threading.Event()) is None
    assert fake.calls == []


def test_note_eager_render_keeps_the_fastest_eager_render():
    from core.inference import diffusion as d

    state = SimpleNamespace(pipe = SimpleNamespace(), speed_optims = ())
    d._note_eager_render(state, False, 4.1)
    d._note_eager_render(state, False, 2.8)
    d._note_eager_render(state, True, 10.1)
    assert state.pipe._unsloth_eager_render_s == pytest.approx(2.8)
    compiled = SimpleNamespace(pipe = state.pipe, speed_optims = ("compiled",))
    d._note_eager_render(compiled, False, 1.0)  # a compiled render is not an eager one
    assert state.pipe._unsloth_eager_render_s == pytest.approx(2.8)
    d._note_eager_render(compiled, True, 2.0)  # eager beside a pending compile
    assert state.pipe._unsloth_eager_render_s == pytest.approx(2.0)


# ---------------------------------------------------------------- compile worker warm-up on the 2nd image


@pytest.fixture
def _fake_pool(monkeypatch):
    from torch._inductor import async_compile

    calls: list = []
    monkeypatch.setattr(
        async_compile.AsyncCompile, "warm_pool", classmethod(lambda cls: calls.append("warm"))
    )
    monkeypatch.setattr(
        async_compile.AsyncCompile, "wakeup", classmethod(lambda cls: calls.append("wakeup"))
    )
    monkeypatch.setattr(bg, "_POOL_WARMED", False)
    monkeypatch.delenv(bg.WARM_POOL_ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    return calls


def test_warm_compile_pool_starts_the_workers_once(_fake_pool):
    assert bg.warm_compile_pool(LOG) is True
    assert bg.warm_compile_pool(LOG) is False
    assert _fake_pool == ["warm", "wakeup"]


@pytest.mark.parametrize(
    "env", ["UNSLOTH_DIFFUSION_BG_COMPILE_WARM_POOL", "UNSLOTH_DIFFUSION_BG_COMPILE"]
)
def test_warm_compile_pool_switches(monkeypatch, _fake_pool, env):
    monkeypatch.setenv(env, "0")
    assert bg.warm_compile_pool(LOG) is False
    assert _fake_pool == []
