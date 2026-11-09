# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Growable KV cache (unsloth/models/_kv_cache.py, UNSLOTH_GROWABLE_KV=1): full-attention layers allocate a
power-of-two bucket and double when a step would pass it, so a call that stops early never allocates its worst
case. Checked on the host through the stand-in CUDA graph backend of test_generic_decode_graphs: rows and tokens
equal a StaticCache of the full length, growth is eager and re-captures once per bucket, a kept cache grows for a
longer call, sliding layers stay transformers' own, left padding, and a compiled update is one graph per bucket."""

import contextlib
import functools
import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
import unsloth.models._kv_cache as KV
from unsloth.models import vision

from transformers import AutoModelForCausalLM, StaticCache
from transformers import cache_utils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_generic_decode_graphs import GEMMA3, LLAMA, _FakeGraph  # noqa: E402
from test_generic_decode_graphs_default import _DefaultHost  # noqa: E402

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"
# transformers 4.x passes the step's cache positions to get_mask_sizes, 5.x the query length.
_MASK_TAKES_POSITIONS = (
    "cache_position" in cache_utils.StaticLayer.get_mask_sizes.__code__.co_varnames[:2]
)


@pytest.fixture
def host(monkeypatch):
    backend = type("Host", (_DefaultHost,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    monkeypatch.setattr(G, "_STUDIO_DEFAULT_OK", True)
    for name in (ENV, "UNSLOTH_DECODE_GENERIC_GRAPHS", "UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP"):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", "16")
    # As a real replay, which runs no Python: a growable layer must not grow while a step replays (the stand-in
    # re-runs the step to fill the outputs). Every growth is logged with whether a capture or replay was running.
    backend.growths = []
    replaying = [False]
    real_replay = _FakeGraph.replay

    def replay(self):
        replaying[0] = True
        try:
            with getattr(KV, "frozen", contextlib.nullcontext)():
                return real_replay(self)
        finally:
            replaying[0] = False

    monkeypatch.setattr(_FakeGraph, "replay", replay)
    real_grow = KV._grow_group

    def grow(group, size):
        backend.growths.append((size, backend.recording, replaying[0]))
        return real_grow(group, size)

    monkeypatch.setattr(KV, "_grow_group", grow)
    return backend


def _load(name, **kwargs):
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(name, dtype = torch.float32, **kwargs).eval()
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


@pytest.fixture
def llama():
    return _load(LLAMA)


@pytest.fixture
def gemma3():
    # A short window, so a short generation runs past it; one full-attention layer to grow.
    return _load(GEMMA3, sliding_window = 8, layer_types = ["sliding_attention", "full_attention"])


def _ids(
    bsz = 1,
    length = 6,
    pad = 0,
):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 1000, (bsz, length), generator = g)
    mask = torch.ones_like(ids)
    if pad:
        ids[0, :pad] = 0
        mask[0, :pad] = 0
    return ids, mask


def _gen(
    model,
    new = 10,
    bsz = 1,
    pad = 0,
    length = 6,
    **kwargs,
):
    ids, mask = _ids(bsz, length = length, pad = pad)
    kwargs.setdefault("min_new_tokens", new)
    with torch.inference_mode():
        return model.generate(
            input_ids = ids,
            attention_mask = mask,
            max_new_tokens = new,
            do_sample = False,
            pad_token_id = 0,
            **kwargs,
        )


def _both(model, monkeypatch, **kwargs):
    """Tokens with a plain StaticCache, then with the growable one (same graph path)."""
    monkeypatch.delenv("UNSLOTH_GROWABLE_KV", raising = False)
    GG.release(model)
    static = _gen(model, **kwargs)
    GG.release(model)
    model.__dict__.pop(GG._ATTR, None)
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    grown = _gen(model, **kwargs)
    return static, grown


def _stats(model):
    return GG.stats(model) or {"captures": 0, "replays": 0, "graphs": 0, "refusals": {}}


def _full_layers(cache):
    return [layer for layer in cache.layers if type(layer) is KV.GrowableLayer]


def test_buckets():
    os.environ.pop("UNSLOTH_GROWABLE_KV_MIN", None)
    assert (
        KV.bucket(1, 10_000) == 256
        and KV.bucket(256, 10_000) == 256
        and KV.bucket(257, 10_000) == 512
    )
    assert KV.bucket(3000, 2100) == 2100  # clamped to the limit generate sized for


def test_layer_rows_match_a_static_cache_across_growth(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", "16")
    config = AutoModelForCausalLM.from_pretrained(LLAMA).config
    limit = 70
    static = StaticCache(config = config, max_cache_len = limit)
    grown = KV.make_growable(StaticCache(config = config, max_cache_len = limit))
    assert len(_full_layers(grown)) == config.num_hidden_layers
    g = torch.Generator().manual_seed(0)
    heads, dim = config.num_key_value_heads, config.hidden_size // config.num_attention_heads
    written, caps = 0, []
    for q in [5] + [1] * 60:
        k = torch.randn(2, heads, q, dim, generator = g)
        v = torch.randn(2, heads, q, dim, generator = g)
        kwargs = {"cache_position": torch.arange(written, written + q)}
        size = grown.layers[0].get_mask_sizes(
            kwargs["cache_position"] if _MASK_TAKES_POSITIONS else q
        )[0]
        assert size >= written + q
        for idx in range(config.num_hidden_layers):
            a = static.update(k, v, idx, kwargs)
            b = grown.update(k, v, idx, kwargs)
            assert b[0].shape[-2] == grown.layers[idx].capacity
            assert torch.equal(a[0][:, :, : written + q], b[0][:, :, : written + q])
            assert torch.equal(a[1][:, :, : written + q], b[1][:, :, : written + q])
        written += q
        caps.append(grown.layers[0].capacity)
    assert sorted(set(caps)) == [16, 32, 64, 70], caps
    assert all(layer.cumulative_length_int == written for layer in _full_layers(grown))
    # 4.57 and 5.x both report the limit as the cache's length, so generate reuses it.
    assert grown.layers[0].max_cache_len == limit and grown.layers[0].get_max_cache_shape() == limit


def test_generate_matches_static_and_recaptures_once_per_bucket(host, llama, monkeypatch):
    static, grown = _both(llama, monkeypatch, new = 40)
    assert torch.equal(static, grown)
    s = _stats(llama)
    cache = GG.kept_cache(llama)
    assert KV.is_growable(cache)
    # prompt 6 + 39 written slots: buckets 16 -> 32 -> 45 (the limit). generate grows the cache before the step's
    # mask, so the growth step itself is captured on the new bucket: one capture per bucket, no eager step.
    assert KV.capacity(cache) == 45, KV.capacity(cache)
    # Each growth moves the tensors, so the previous bucket's graph is dropped: one live graph.
    assert s["captures"] == 3 and s["graphs"] == 1 and not s["refusals"], s


def test_early_stop_allocates_only_its_bucket(host, llama, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    first = _gen(llama, new = 8)
    stop = int(first[0, 6 + 4])
    GG.release(llama)
    llama.__dict__.pop(GG._ATTR, None)
    llama.__dict__.pop("_cache", None)
    # The same greedy tokens with a stop token four steps in and a far larger budget.
    out = _gen(llama, new = 500, min_new_tokens = 0, eos_token_id = stop)
    assert out.shape[1] <= 6 + 5, out.shape
    cache = GG.kept_cache(llama)
    assert KV.is_growable(cache)
    assert KV.capacity(cache) == 16
    assert all(layer.keys.shape[-2] == 16 for layer in _full_layers(cache))
    # A StaticCache of this call holds 505 slots per layer.
    assert _full_layers(cache)[0].max_cache_len >= 505


def test_kept_cache_grows_for_a_longer_call(host, llama, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    _gen(llama, new = 10)
    kept = GG.kept_cache(llama)
    assert KV.capacity(kept) <= 16
    before = GG.PLANS["reused kept cache"]
    longer = _gen(llama, new = 30)
    if hasattr(
        llama, "_prepare_static_cache"
    ):  # 5.x plans the call's cache; 4.x reuses model._cache itself
        assert GG.PLANS["reused kept cache"] == before + 1
    assert GG.kept_cache(llama) is kept and KV.capacity(kept) == 35
    monkeypatch.delenv("UNSLOTH_GROWABLE_KV")
    GG.release(llama)
    llama.__dict__.pop(GG._ATTR, None)
    assert torch.equal(longer, _gen(llama, new = 30))


def test_sliding_layers_stay_transformers_own(host, gemma3, monkeypatch):
    static, grown = _both(gemma3, monkeypatch, new = 30)
    assert torch.equal(static, grown)
    cache = GG.kept_cache(gemma3)
    kinds = {type(layer).__name__ for layer in cache.layers}
    assert "GrowableLayer" in kinds and kinds - {"GrowableLayer"}, kinds
    assert all(
        layer.keys.shape[-2] == 8 for layer in cache.layers if type(layer) is not KV.GrowableLayer
    )


def test_left_padding(host, llama, monkeypatch):
    static, grown = _both(llama, monkeypatch, new = 30, bsz = 2, pad = 2)
    assert torch.equal(static, grown)


def test_compiled_update_is_one_graph_per_bucket(monkeypatch):
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV_MIN", "16")
    torch._logging.set_logs(graph_breaks = True, recompiles = True)
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    layer = KV.GrowableLayer(max_cache_len = 64)
    layer.group.members.append(layer)
    k = torch.randn(1, 2, 4, 8)
    layer.update(
        k, k, {"cache_position": torch.arange(4)}
    )  # eager prefill: allocates the first bucket
    counter = CompileCounter()

    @torch.compile(backend = counter, fullgraph = True)
    def step(layer, k, pos):
        keys, _ = layer.update(k, k, {"cache_position": pos})
        return keys.sum()

    written = 4
    for _ in range(6):
        k1 = torch.randn(1, 2, 1, 8)
        step(layer, k1, torch.tensor([written]))
        layer.cumulative_length_int += (
            1  # the decode graph path advances the host counter outside the graph
        )
        written += 1
    assert counter.frame_count == 1, counter.frame_count
    layer.get_mask_sizes(
        torch.arange(written, written + 10) if _MASK_TAKES_POSITIONS else 10
    )  # eager growth
    assert layer.capacity == 32
    step(layer, torch.randn(1, 2, 1, 8), torch.tensor([written]))
    assert counter.frame_count == 2, counter.frame_count
    torch._dynamo.reset()


def test_a_short_call_after_a_long_one_replays_the_kept_bucket(host, llama, monkeypatch):
    # As test_generic_decode_graphs_default's static version, but the long call captures once per bucket.
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    _gen(llama, new = 40)
    kept = GG.kept_cache(llama)
    assert KV.is_growable(kept) and _stats(llama)["captures"] == 3
    monkeypatch.setattr(
        llama, "_previous_max_cache_length", GG._kept_len(llama.__dict__[GG._ATTR]), raising = False
    )
    s2 = _stats(llama)
    outs = [_gen(llama, new = 10) for _ in range(2)]
    s3 = _stats(llama)
    assert torch.equal(outs[0], outs[1])
    assert GG.kept_cache(llama) is kept and KV.capacity(kept) == 45
    assert s3["captures"] == s2["captures"], (s2, s3)
    assert s3["replays"] - s2["replays"] == 2 * 9, (s2, s3)


def _no_growth_in_graphs(host):
    assert host.growths and not [g for g in host.growths if g[1] or g[2]], host.growths


def test_no_growth_inside_a_capture(host, llama, monkeypatch):
    # A 15-token prompt fills its 16-slot bucket on the first captured step: the capture runs after the warm-up
    # advanced the host counter, so a growth decision there would see one token too many and grow inside the
    # capture (on CUDA: the copy into the new buffers recorded, never run; the device counter one step ahead).
    static, grown = _both(llama, monkeypatch, new = 30, length = 15)
    assert torch.equal(static, grown)
    _no_growth_in_graphs(host)
    # That first step runs eagerly instead (a graph captured there could never replay); the next one captures.
    assert _stats(llama)["refusals"] == {"kv cache fills its bucket": 1}, _stats(llama)
    assert KV.capacity(GG.kept_cache(llama)) == 44


def test_kept_cache_grows_from_a_clamped_bucket(host, llama, monkeypatch):
    # The first call's bucket is clamped to its limit (15); the longer call grows it to 16, a step that fills it.
    monkeypatch.setenv("UNSLOTH_GROWABLE_KV", "1")
    _gen(llama, new = 10)
    kept = GG.kept_cache(llama)
    longer = _gen(llama, new = 30)
    assert GG.kept_cache(llama) is kept and KV.capacity(kept) == 35
    _no_growth_in_graphs(host)
    monkeypatch.delenv("UNSLOTH_GROWABLE_KV")
    GG.release(llama)
    llama.__dict__.pop(GG._ATTR, None)
    llama.__dict__.pop("_cache", None)
    assert torch.equal(longer, _gen(llama, new = 30))


def test_stale_host_counter_and_a_wrapped_prepare(host, llama, monkeypatch):
    # A decode update that runs no Python counter code (torch.compile'd decode, a replay path that does not advance
    # it) leaves the host counter behind, and another wrapper may own prepare_inputs_for_generation already: growth
    # must still follow generate's own lengths, never the stale counter.
    real_update = KV.GrowableLayer.update

    def compiled_like(self, key_states, value_states, *args, **kwargs):
        if key_states.shape[-2] != 1:
            return real_update(self, key_states, value_states, *args, **kwargs)
        return KV._StaticLayer.update(self, key_states, value_states, *args, **kwargs)

    monkeypatch.setattr(KV.GrowableLayer, "update", compiled_like)
    inner = llama.prepare_inputs_for_generation
    wrapper = functools.wraps(inner)(lambda *a, **k: inner(*a, **k))
    llama.prepare_inputs_for_generation = wrapper
    static, grown = _both(llama, monkeypatch, new = 40)
    assert torch.equal(static, grown)
    cache = GG.kept_cache(llama)
    assert KV.capacity(cache) == 45, KV.capacity(cache)
    # The other wrapper is back in place after each call.
    assert llama.__dict__.get("prepare_inputs_for_generation") is wrapper
