# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests for the MiniMax-H3 condition latent cache on a tiny random H3 VAE."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
encoders = pytest.importorskip("diffusers.modular_pipelines.minimax_h3.encoders")
vae_mod = pytest.importorskip("diffusers.models.autoencoders.autoencoder_kl_minimax_h3")

from core.inference import video_minimax_h3_ref_cache as ref_cache  # noqa: E402
from core.inference import video_minimax_h3_ref_encode as ref_encode  # noqa: E402

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _vae(seed = 0):
    torch.manual_seed(seed)
    return vae_mod.AutoencoderKLMiniMaxH3(
        latent_channels = 4,
        block_out_channels = [8, 8, 16, 16, 16, 16],
        layers_per_block = 1,
        norm_num_groups = 4,
        decoder_num_layers = 1,
        decoder_num_attention_heads = 2,
        decoder_attention_head_dim = 8,
        latents_mean = (0.1, -0.2, 0.3, 0.0),
        latents_std = (1.5, 0.5, 2.0, 1.0),
    ).eval()


def _pixels(
    frames,
    height = 32,
    width = 48,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (1, 3, frames, height, width), generator = g, dtype = torch.uint8)


def _stock_encode():
    current = encoders.encode_vae_condition
    while getattr(current, "_unsloth_chunked_ref_encode", False):
        current = current.__wrapped__
    return current


class _Counted:
    """Counts encoder forwards through a forward hook (kept out of the key: hooks are not attributes)."""

    def __init__(self, vae):
        self.n = 0
        vae.encoder.register_forward_hook(self._hook)

    def _hook(self, *_):
        self.n += 1


@pytest.fixture
def encode(monkeypatch):
    for name in (
        ref_encode.H3_CHUNKED_REF_ENCODE_ENV,
        ref_encode.H3_REFERENCE_LATENT_ENV,
        ref_cache.H3_REF_LATENT_CACHE_ENV,
        ref_cache.H3_REF_LATENT_CACHE_ENTRIES_ENV,
        ref_cache.H3_REF_LATENT_CACHE_MB_ENV,
    ):
        monkeypatch.delenv(name, raising = False)
    original = _stock_encode()
    if ref_encode._sha(original) not in ref_encode._KNOWN_SOURCES["encode_vae_condition"]:
        pytest.skip("diffusers H3 encoder source this shim does not mirror")
    monkeypatch.setattr(encoders, "encode_vae_condition", original)
    ref_cache.clear("test")
    assert ref_encode.install_h3_chunked_reference_encode()
    installed = encoders.encode_vae_condition
    yield lambda vae, pixels, seed = 42: installed(vae, pixels, _MEAN, _STD, seed)
    ref_cache.clear("test")


@pytest.mark.parametrize("frames", [1, 22, 39])
def test_same_inputs_hit_and_are_bit_identical(encode, monkeypatch, frames):
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        first = encode(vae, _pixels(frames))
        n_first = counted.n
        second = encode(
            vae, _pixels(frames)
        )  # a fresh tensor with the same bytes, as a new request makes
    assert n_first > 0 and counted.n == n_first  # no encoder forward on the hit
    assert torch.equal(first, second) and second.dtype == torch.float32
    assert ref_cache.stats()["hits"] >= 1
    # the hit is a copy: the caller mutating it never reaches the cache
    second.add_(1.0)
    with torch.no_grad():
        third = encode(vae, _pixels(frames))
    assert torch.equal(first, third)
    # and equal to the encode with the cache off, bit for bit
    monkeypatch.setenv(ref_cache.H3_REF_LATENT_CACHE_ENV, "0")
    with torch.no_grad():
        uncached = encode(vae, _pixels(frames))
    assert torch.equal(first, uncached)


def test_sampled_posterior_hit_is_exact(encode, monkeypatch):
    monkeypatch.setenv(ref_encode.H3_REFERENCE_LATENT_ENV, "sample")
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        first = encode(vae, _pixels(22))
        n_first = counted.n
        second = encode(vae, _pixels(22))
        other_seed = encode(vae, _pixels(22), seed = 7)
    assert counted.n > n_first  # a different encode seed missed
    assert torch.equal(first, second)
    assert not torch.equal(first, other_seed)


def _misses(encode, vae, pixels, change):
    counted = _Counted(vae)
    with torch.no_grad():
        encode(vae, pixels)
        before = counted.n
        vae2, pixels2 = change(vae, pixels)
        if vae2 is not vae:
            counted = _Counted(vae2)
            before = 0
        encode(vae2, pixels2)
    return counted.n > before


def _one_pixel(vae, pixels):
    p = pixels.clone()
    p[0, 0, -1, -1, -1] ^= 1
    return vae, p


@pytest.mark.parametrize(
    "name,change",
    [
        ("file bytes", _one_pixel),
        ("size", lambda vae, p: (vae, _pixels(p.shape[2], height = 48, width = 48))),
        ("frames", lambda vae, p: (vae, _pixels(39))),
        ("vae tiling", lambda vae, p: (vae.disable_tiling() or vae, p)),
        ("vae tile size", lambda vae, p: (vae.enable_tiling(tile_sample_min_height = 512) or vae, p)),
        ("another vae load", lambda vae, p: (_vae(), p)),
    ],
)
def test_key_changes_miss(encode, name, change):
    assert _misses(encode, _vae(), _pixels(22), change), name


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_vae_dtype_changes_the_key(dtype):
    # a CPU encode in half precision is not runnable here, so the key itself is checked
    vae, pixels = _vae(), _pixels(22)
    before = ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ("mean", True, 0))
    assert before == ref_cache.condition_key(
        vae, pixels.clone(), _MEAN, _STD, 42, ("mean", True, 0)
    )
    vae.to(dtype)
    assert before != ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ("mean", True, 0))


def test_patched_encoder_forward_changes_the_key():
    import types

    vae, pixels = _vae(), _pixels(1)
    before = ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ())
    stock = vae.encoder.forward
    vae.encoder.forward = types.MethodType(lambda self, x: stock(x), vae.encoder)
    assert before != ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ())


def test_latent_mode_change_misses(encode, monkeypatch):
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        mean = encode(vae, _pixels(22))
        n = counted.n
        monkeypatch.setenv(ref_encode.H3_REFERENCE_LATENT_ENV, "sample")
        sampled = encode(vae, _pixels(22))
    assert counted.n > n and not torch.equal(mean, sampled)


def test_unload_evicts(encode, monkeypatch):
    from core.inference import video

    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        encode(vae, _pixels(22))
    assert ref_cache.stats()["entries"] == 1
    monkeypatch.setattr(video, "release_load_budget_fraction", lambda *_: None)

    class _Backend:
        _state = None
        _float_seeds = None

    video.VideoBackend._teardown_state_locked(_Backend())
    assert ref_cache.stats()["entries"] == 0
    n = counted.n
    with torch.no_grad():
        encode(vae, _pixels(22))
    assert counted.n > n


def test_kill_switch_disables(encode, monkeypatch):
    monkeypatch.setenv(ref_cache.H3_REF_LATENT_CACHE_ENV, "0")
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        encode(vae, _pixels(22))
        n = counted.n
        encode(vae, _pixels(22))
    assert counted.n == 2 * n
    assert ref_cache.stats()["entries"] == 0


def test_lru_bound(encode, monkeypatch):
    monkeypatch.setenv(ref_cache.H3_REF_LATENT_CACHE_ENTRIES_ENV, "2")
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        for seed in (1, 2, 3):
            encode(vae, _pixels(1, seed = seed))
        assert ref_cache.stats()["entries"] == 2
        n = counted.n
        encode(vae, _pixels(1, seed = 3))  # most recent: kept
        assert counted.n == n
        encode(vae, _pixels(1, seed = 1))  # oldest: evicted
        assert counted.n > n
    monkeypatch.setenv(ref_cache.H3_REF_LATENT_CACHE_MB_ENV, "0")
    ref_cache.clear("test")
    with torch.no_grad():
        encode(vae, _pixels(1, seed = 4))
    assert ref_cache.stats()["entries"] == 0  # over the byte bound: not kept


def test_large_pixels_tree_digest_is_exact():
    a = torch.zeros(ref_cache._PARALLEL_HASH_BYTES + 7, dtype = torch.uint8)
    b = a.clone()
    b[-1] = 1
    assert ref_cache._pixel_digest(a) == ref_cache._pixel_digest(a.clone())
    assert ref_cache._pixel_digest(a) != ref_cache._pixel_digest(b)


def test_decode_bookkeeping_between_renders_still_hits(encode):
    import types

    # Studio's decode side writes counters / last-tile / canvas state on the shared VAE every render
    vae = _vae()
    counted = _Counted(vae)
    with torch.no_grad():
        first = encode(vae, _pixels(22))
        n = counted.n
        vae._unsloth_vae_full_map_calls = vae.__dict__.get("_unsloth_vae_full_map_calls", 0) + 1
        vae._unsloth_last_decode_tile = (256, 256, 65536)
        vae._unsloth_canvas_graphs = False
        vae._unsloth_vae_oom_active = False
        stock_decode = type(vae).decode
        # Studio re-binds a decode wrapper on the VAE every render
        vae.decode = types.MethodType(lambda self, *a, **k: stock_decode(self, *a, **k), vae)
        second = encode(vae, _pixels(22))
    assert counted.n == n and torch.equal(first, second)


def test_encoder_fallback_flag_misses_with_reason(encode):
    lines = []

    class _Log:
        def info(self, msg, *args):
            lines.append(msg % args)

    vae, pixels = _vae(), _pixels(1)
    recipe = ("mean", False, 0)
    with torch.no_grad():
        ref_cache.cached_encode(
            lambda: torch.zeros(1), vae, pixels, _MEAN, _STD, 42, recipe, logger = _Log()
        )
        vae.encoder._unsloth_fast_failed = (
            True  # the fused encoder fell back to stock: other numerics
        )
        ref_cache.cached_encode(
            lambda: torch.ones(1), vae, pixels, _MEAN, _STD, 42, recipe, logger = _Log()
        )
    assert lines[0].startswith("video.h3_ref_latent_cache: miss")
    assert "miss" in lines[1] and "encoder_attrs['_unsloth_fast_failed']" in lines[1], lines


@pytest.mark.parametrize("name", ["encode", "_encode", "_encode_clip"])
def test_instance_encode_wrapper_changes_the_key(name):
    import types

    vae, pixels = _vae(), _pixels(22)
    before = ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ())
    stock = getattr(type(vae), name)
    setattr(vae, name, types.MethodType(lambda self, *a, **k: stock(self, *a, **k), vae))
    assert before != ref_cache.condition_key(vae, pixels, _MEAN, _STD, 42, ())
