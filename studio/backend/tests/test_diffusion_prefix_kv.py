# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The prefix KV cache a compiled denoiser keeps across steps holds only the prefix bytes."""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from core.inference.diffusion_prefix_kv import (  # noqa: E402
    compact_prefix_kv_cache,
    install_prefix_kv_compaction,
    release_prefix_kv,
    track_prefix_kv,
)


class _Layer:
    def __init__(self):
        self.k = None
        self.v = None

    def store(self, k, v):
        self.k = k
        self.v = v


class _Cache:
    def __init__(self, n):
        self.layer_caches = [_Layer() for _ in range(n)]


def _prefill_like_inductor(
    cache,
    seq = 64,
    prefix = 3,
):
    for layer in cache.layer_caches:
        full_k = torch.randn(1, seq, 4, 8)
        full_v = torch.randn(1, seq, 4, 8)
        layer.store(full_k[:, :prefix], full_v[:, :prefix])


def test_views_into_a_larger_buffer_are_copied_out_with_the_same_values():
    cache = _Cache(3)
    _prefill_like_inductor(cache)
    before = [(layer.k.clone(), layer.v.clone()) for layer in cache.layer_caches]

    released = compact_prefix_kv_cache(cache)

    assert released == 3 * 2 * (64 - 3) * 4 * 8 * 4
    for layer, (k, v) in zip(cache.layer_caches, before):
        assert layer.k.untyped_storage().nbytes() == layer.k.numel() * layer.k.element_size()
        assert torch.equal(layer.k, k) and torch.equal(layer.v, v)


def test_a_cache_that_is_already_compact_is_left_alone():
    cache = _Cache(2)
    for layer in cache.layer_caches:
        layer.store(torch.randn(1, 3, 4, 8), torch.randn(1, 3, 4, 8))
    ids = [(id(layer.k), id(layer.v)) for layer in cache.layer_caches]

    assert compact_prefix_kv_cache(cache) == 0
    assert ids == [(id(layer.k), id(layer.v)) for layer in cache.layer_caches]


def test_other_cache_layouts_are_walked_too():
    class _RefLayer:
        def __init__(
            self,
            k = None,
            v = None,
        ):
            self.k_ref = k
            self.v_ref = v

    class _Flux2Like:
        def __init__(self):
            full = torch.randn(1, 32, 2, 4)
            self.double_block_caches = [_RefLayer(full[:, :2], full[:, 2:4]), _RefLayer()]
            self.single_block_caches = [_RefLayer(full[:, :2].clone(), full[:, :2].clone())]
            self.num_ref_tokens = 2

    cache = _Flux2Like()
    assert compact_prefix_kv_cache(cache) > 0
    first = cache.double_block_caches[0]
    assert first.k_ref.untyped_storage().nbytes() == first.k_ref.numel() * 4
    assert cache.double_block_caches[1].k_ref is None


class _PrefixKVDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        hidden_states,
        kv_cache = None,
        kv_cache_mode = None,
    ):
        self.calls.append(kv_cache_mode)
        if kv_cache_mode == "extract":
            _prefill_like_inductor(kv_cache)
        return hidden_states


def test_the_hook_compacts_after_the_prefill_call_only():
    module = _PrefixKVDenoiser()
    assert install_prefix_kv_compaction(module) is True
    cache = _Cache(2)
    x = torch.zeros(1)

    module(x, kv_cache = cache, kv_cache_mode = "extract")
    layer = cache.layer_caches[0]
    assert layer.k.untyped_storage().nbytes() == layer.k.numel() * layer.k.element_size()

    planted = torch.randn(1, 64, 4, 8)[:, :3]
    layer.k = planted
    module(x, kv_cache = cache, kv_cache_mode = "cached")
    assert layer.k is planted


class _ReturnsItsCache(torch.nn.Module):
    def __init__(self, as_tuple):
        super().__init__()
        self.as_tuple = as_tuple

    def forward(
        self,
        hidden_states,
        kv_cache = None,
        kv_cache_mode = None,
    ):
        if kv_cache_mode != "extract":
            return (hidden_states,)
        cache = _Cache(2)
        _prefill_like_inductor(cache)
        if self.as_tuple:
            return (hidden_states, cache)
        return types.SimpleNamespace(sample = hidden_states, kv_cache = cache)


@pytest.mark.parametrize("as_tuple", [True, False])
def test_a_cache_returned_by_the_prefill_call_is_compacted(as_tuple):
    module = _ReturnsItsCache(as_tuple)
    install_prefix_kv_compaction(module)

    out = module(torch.zeros(1), kv_cache_mode = "extract")

    cache = out[1] if as_tuple else out.kv_cache
    for layer in cache.layer_caches:
        assert layer.k.untyped_storage().nbytes() == layer.k.numel() * layer.k.element_size()
        assert layer.v.untyped_storage().nbytes() == layer.v.numel() * layer.v.element_size()


def test_install_is_idempotent_and_skips_denoisers_without_a_prefix_cache():
    module = _PrefixKVDenoiser()
    assert install_prefix_kv_compaction(module) is True
    assert install_prefix_kv_compaction(module) is False
    assert len(module._forward_hooks) == 1

    class _Plain(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            timestep = None,
        ):
            return hidden_states

    assert install_prefix_kv_compaction(_Plain()) is False


def test_the_real_qwen_image_21_cache_is_compacted():
    tq = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    cache = tq.QwenImage21KVCache(2)
    full = torch.randn(1, 64, 4, 8)
    for index in range(2):
        cache.get_layer(index).store(full[:, :3], full[:, 3:6])
    assert compact_prefix_kv_cache(cache) > 0
    k, v = cache.get_layer(0).get()
    assert k.untyped_storage().nbytes() == k.numel() * k.element_size()
    assert torch.equal(k, full[:, :3]) and torch.equal(v, full[:, 3:6])


def test_inductor_really_stores_the_clone_as_a_view_of_the_full_buffer():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile unavailable")

    class _Holder:
        k = None

    def prefill(x, w, holder):
        k = (x @ w).view(1, x.shape[1], 4, 8)
        holder.k = k[:, :3].clone()
        return (k * 2).sum()

    holder = _Holder()
    x = torch.randn(1, 64, 32)
    w = torch.randn(32, 32)
    try:
        torch.compile(prefill)(x, w, holder)
    except Exception as exc:  # noqa: BLE001 - no C++ toolchain for the CPU backend here
        pytest.skip(f"inductor unavailable: {exc}")
    held = holder.k.untyped_storage().nbytes()
    used = holder.k.numel() * holder.k.element_size()
    if held == used:
        pytest.skip("this inductor already copies the slice out")
    cache = _Cache(1)
    cache.layer_caches[0].store(holder.k, holder.k)
    compact_prefix_kv_cache(cache)
    assert cache.layer_caches[0].k.untyped_storage().nbytes() == used


def test_a_regional_compile_installs_the_hook(monkeypatch):
    from core.inference import diffusion_speed as ds_mod

    class _CompiledDenoiser(_PrefixKVDenoiser):
        def compile_repeated_blocks(self, **kwargs):
            self.compile_kwargs = kwargs

    pipe = type("P", (), {})()
    pipe.transformer = _CompiledDenoiser()
    monkeypatch.setattr(ds_mod, "_inductor_config", lambda: None)
    monkeypatch.setattr(ds_mod, "guard_compiled_blocks", lambda transformer, logger = None: 0)

    assert ds_mod._compile_repeated_blocks(pipe, None) is True
    cache = _Cache(1)
    pipe.transformer(torch.zeros(1), kv_cache = cache, kv_cache_mode = "extract")
    k = cache.layer_caches[0].k
    assert k.untyped_storage().nbytes() == k.numel() * k.element_size()


def test_tracked_caches_are_emptied_on_release_and_only_once():
    module = _PrefixKVDenoiser()
    assert track_prefix_kv(module) is True
    assert track_prefix_kv(module) is False
    cond, neg = _Cache(2), _Cache(2)
    x = torch.zeros(1)
    for cache in (cond, neg):
        module(x, kv_cache = cache, kv_cache_mode = "extract")
    module(x, kv_cache = cond, kv_cache_mode = "cached")

    released = release_prefix_kv(module)

    assert released == 2 * 2 * 2 * 64 * 4 * 8 * 4
    for cache in (cond, neg):
        assert all(layer.k is None and layer.v is None for layer in cache.layer_caches)
    assert release_prefix_kv(module) == 0


def test_release_frees_the_cache_storage_and_does_not_keep_dead_caches_alive():
    import gc
    import weakref

    module = _PrefixKVDenoiser()
    track_prefix_kv(module)
    cache = _Cache(1)
    module(torch.zeros(1), kv_cache = cache, kv_cache_mode = "extract")
    k = weakref.ref(cache.layer_caches[0].k)
    release_prefix_kv(module)
    gc.collect()
    assert k() is None

    other = _Cache(1)
    module(torch.zeros(1), kv_cache = other, kv_cache_mode = "extract")
    ref = weakref.ref(other)
    del other
    gc.collect()
    assert ref() is None
    assert release_prefix_kv(module) == 0


def test_tracking_skips_denoisers_without_a_prefix_cache():
    class _Plain(torch.nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    assert track_prefix_kv(_Plain()) is False
    assert track_prefix_kv(None) is False
    assert release_prefix_kv(None) == 0


def test_the_real_qwen_image_21_cache_is_released():
    tq = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

    class _Q21Like(torch.nn.Module):
        def forward(
            self,
            hidden_states,
            kv_cache = None,
            kv_cache_mode = None,
        ):
            if kv_cache_mode == "extract":
                for index in range(2):
                    kv_cache.get_layer(index).store(
                        torch.randn(1, 3, 4, 8), torch.randn(1, 3, 4, 8)
                    )
            return hidden_states

    module = _Q21Like()
    track_prefix_kv(module)
    cache = tq.QwenImage21KVCache(2)
    module(torch.zeros(1), kv_cache = cache, kv_cache_mode = "extract")
    assert release_prefix_kv(module) == 2 * 2 * 3 * 4 * 8 * 4
    assert all(layer.k is None and layer.v is None for layer in cache.layer_caches)


def test_the_render_helper_also_drops_a_cfg_parallel_replica_copy(monkeypatch):
    from core.inference import diffusion as diffusion_mod
    from core.inference import diffusion_multigpu as mg

    pipe = type("P", (), {})()
    pipe.transformer = _PrefixKVDenoiser()
    track_prefix_kv(pipe.transformer)
    cache = _Cache(1)
    pipe.transformer(torch.zeros(1), kv_cache = cache, kv_cache_mode = "extract")
    router = types.SimpleNamespace(
        kv_mirrors = {id(cache): (cache, [], None)}, kv_declined = {1: (cache, [])}
    )
    monkeypatch.setattr(mg, "_CFG_ROUTERS", {pipe: router})

    diffusion_mod._release_prefix_kv(pipe)

    assert cache.layer_caches[0].k is None
    assert router.kv_mirrors == {} and router.kv_declined == {}
