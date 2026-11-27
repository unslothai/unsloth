# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Background compile of a denoiser whose call carries a state object (Qwen-Image-2.1's prefix KV cache) or is
offloaded: the recording keeps such calls, and a planned CUDA-graph step replays the way the render captures it.
CPU only."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_bg_compile as bg  # noqa: E402
from core.inference import diffusion_cuda_graph as cg  # noqa: E402


class _LayerCache:
    def __init__(self) -> None:
        self.k = None
        self.v = None


class _KVCache:
    """Shaped like diffusers' ``QwenImage21KVCache``: a list of per-layer holders, filled by the prefill."""

    def __init__(self, layers: int) -> None:
        self.layer_caches = [_LayerCache() for _ in range(layers)]

    def get_layer(self, index: int) -> _LayerCache:
        return self.layer_caches[index]


class _CachedNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(
        self,
        hidden_states,
        *,
        kv_cache = None,
        kv_cache_mode = "extract",
        return_dict = True,
    ):
        out = self.lin(hidden_states)
        layer = kv_cache.get_layer(0)
        if kv_cache_mode == "extract":
            layer.k, layer.v = out.detach().clone(), (out * 2).detach().clone()
            return out
        return out + layer.k


def _prefill_and_decode(net):
    cache = _KVCache(1)
    x = torch.randn(2, 8)
    with torch.no_grad(), bg.force_eager():
        net(x, kv_cache = cache, kv_cache_mode = "extract", return_dict = False)
        net(x, kv_cache = cache, kv_cache_mode = "cached", return_dict = False)
        net(x, kv_cache = cache, kv_cache_mode = "cached", return_dict = False)  # same key: deduped
    return cache


def test_a_state_object_in_the_call_is_recorded_as_a_fresh_copy(monkeypatch):
    monkeypatch.delenv(bg.OBJECTS_ENV, raising = False)
    net = _CachedNet()
    job = bg.arm(net)
    cache = _prefill_and_decode(net)
    assert job.state == "recording" and len(job.samples) == 2
    spec, clones, *_ = job.samples[1]
    args, kwargs = bg._rebuild_sample(spec, clones)
    rebuilt = kwargs["kv_cache"]
    assert type(rebuilt) is _KVCache and rebuilt is not cache
    assert rebuilt.layer_caches[0] is not cache.layer_caches[0]
    assert torch.equal(rebuilt.get_layer(0).k, cache.get_layer(0).k)
    assert rebuilt.get_layer(0).k.data_ptr() != cache.get_layer(0).k.data_ptr()
    # the prefill sample holds the cache as the prefill saw it: empty, so its replay fills its own copy
    spec0, clones0, *_ = job.samples[0]
    assert bg._rebuild_sample(spec0, clones0)[1]["kv_cache"].get_layer(0).k is None
    assert job.kick()
    job._thread.join(60)
    assert job.state == "done", job.error
    job.close()


def test_the_object_kill_switch_restores_the_old_refusal(monkeypatch, caplog):
    monkeypatch.setenv(bg.OBJECTS_ENV, "0")
    net = _CachedNet()
    job = bg.arm(net, logger = logging.getLogger("bgtest"))
    with caplog.at_level(logging.INFO, logger = "bgtest"):
        _prefill_and_decode(net)
    assert job.state == "failed" and "_KVCache" in (job.error or "")
    assert not job.samples and not job.pending()
    # the silent failure hid a 47-77 s inline compile: it is logged now
    assert any("cannot be recorded" in r.getMessage() for r in caplog.records)
    job.close()


def test_an_oversized_state_object_is_not_copied(monkeypatch):
    monkeypatch.delenv(bg.OBJECTS_ENV, raising = False)
    monkeypatch.setattr(bg, "_OBJECT_MAX_BYTES", 16)
    net = _CachedNet()
    job = bg.arm(net)
    _prefill_and_decode(net)
    assert job.state == "failed" and "above" in (job.error or "")
    job.close()


def test_modules_and_callables_are_never_copied():
    for leaf in (torch.nn.Linear(2, 2), lambda: None, _KVCache):
        with pytest.raises(TypeError):
            bg._flatten_sample({"x": leaf}, [])


def test_select_module_takes_an_offloaded_denoiser_unless_switched_off(monkeypatch):
    import types

    monkeypatch.delenv("UNSLOTH_DIFFUSION_BG_COMPILE", raising = False)
    monkeypatch.setattr(cg, "_denoiser_modules", lambda pipe: [pipe.transformer])
    pipe = types.SimpleNamespace(transformer = torch.nn.Linear(2, 2))
    kwargs = dict(
        speed_optims = ("compiled",),
        default_tier = True,
        quantized = False,
        gguf = False,
        step_cache = False,
        device = "cuda",
        backend = "cuda",
        denoiser_hooked = True,
    )
    monkeypatch.delenv(bg.OFFLOADED_ENV, raising = False)
    assert bg.select_module(pipe, **kwargs) is pipe.transformer
    monkeypatch.setenv(bg.OFFLOADED_ENV, "0")
    assert bg.select_module(pipe, **kwargs) is None
    assert bg.select_module(pipe, **{**kwargs, "denoiser_hooked": False}) is pipe.transformer


class _PlainNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)

    def forward(
        self,
        x,
        *,
        return_dict = True,
    ):
        return self.lin(x)


@pytest.mark.parametrize("planned_replay", [True, False])
def test_a_background_replay_takes_the_planned_step_on_fresh_statics(monkeypatch, planned_replay):
    if not planned_replay:
        monkeypatch.setenv(cg.BG_PLANNED_REPLAY_ENV, "0")
    else:
        monkeypatch.delenv(cg.BG_PLANNED_REPLAY_ENV, raising = False)
    net = _PlainNet()
    graphed = cg.GraphedForward(net).enable()
    seen = []

    def step(*, x):
        seen.append(x)
        return (x,)

    graphed.plan = lambda module, args, kwargs: (step, {"x": args[0][:, :4]})
    try:
        x = torch.randn(2, 8)
        token = bg._NO_CAPTURE.set(True)
        try:
            with torch.no_grad():
                net(x, return_dict = False)
        finally:
            bg._NO_CAPTURE.reset(token)
        assert graphed.stats["captures"] == 0
        if planned_replay:
            # a fresh offset-0 copy, as the capture's warm-up steps on: not the caller's strided view
            assert len(seen) == 1 and seen[0].storage_offset() == 0 and seen[0].is_contiguous()
            assert torch.equal(seen[0], x[:, :4])
        else:
            assert seen == []
    finally:
        graphed.free()


def test_nested_state_objects_charge_each_tensor_once(monkeypatch):
    monkeypatch.delenv(bg.OBJECTS_ENV, raising = False)
    cache = _KVCache(2)
    for layer in cache.layer_caches:
        layer.k = torch.zeros(4, dtype = torch.float32)
        layer.v = torch.zeros(4, dtype = torch.float32)
    total = 2 * 2 * 4 * 4
    monkeypatch.setattr(bg, "_OBJECT_MAX_BYTES", total)
    out: list = []
    budget = [0, set()]
    bg._flatten_sample(((), {"cache": cache}), out, 0, budget)
    assert budget[0] == total and len(out) == 4
    monkeypatch.setattr(bg, "_OBJECT_MAX_BYTES", total - 1)
    with pytest.raises(TypeError, match = "above"):
        bg._flatten_sample(((), {"cache": cache}), [], 0, [0, set()])


def test_the_background_kick_waits_for_the_resident_restore():
    """An offloaded denoiser's groups go back on the card before its background compile can start."""
    import inspect

    from core.inference import diffusion as D

    src = inspect.getsource(D)
    assert "bg.kick()" not in src.split("kick_bg = True")[0].split("kick_bg = False")[-1]
    tail = src.split("restore_resident()", 2)[-1]
    assert tail.index("if kick_bg and bg is not None:") < tail.index("bg.kick()")
