# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Engagement counters: the untiled VAE wrapper's per-clip decode routes and FBCache's computed / skipped steps."""

import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers.hooks.first_block_cache")

from core.inference import diffusion_cache as C
from core.inference import video_vae_untiled as U


class _VAE:
    def __init__(self):
        self.decoder = torch.nn.Conv3d(1, 1, 1).to(torch.bfloat16)
        self.use_tiling = True

    def decode(
        self,
        z,
        return_dict = False,
    ):
        return ("tiled" if self.use_tiling else "untiled",)


Z = torch.zeros(1, 48, 31, 44, 80)


def test_untiled_stats_per_clip(monkeypatch):
    pipe = types.SimpleNamespace(vae = _VAE())
    assert U.untiled_decode_stats(pipe) is None
    free = [100 * 2**30]
    monkeypatch.setattr(U, "_free_bytes", lambda device: free[0])
    assert U.install_untiled_decode(pipe, "wan2.2-ti2v-5b")
    before = U.untiled_decode_stats(pipe)
    assert pipe.vae.decode(Z) == ("untiled",)
    free[0] = 0
    monkeypatch.setattr(U, "wide_tile_plan", lambda *a, **k: None)
    assert pipe.vae.decode(Z) == ("tiled",)
    delta = U.decode_stats_delta(before, U.untiled_decode_stats(pipe))
    assert delta == {"untiled": 1, "tiled": 1, "wide": 0, "oom_fallback": 0}
    U.untiled_decode_stats(pipe)["untiled"] = 99
    assert U.untiled_decode_stats(pipe)["untiled"] == 1


def _tiny_flux():
    from diffusers import FluxTransformer2DModel
    torch.manual_seed(0)
    return FluxTransformer2DModel(
        patch_size = 1,
        in_channels = 4,
        num_layers = 2,
        num_single_layers = 2,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = 16,
        pooled_projection_dim = 8,
        axes_dims_rope = (2, 2, 4),
    ).eval()


def _flux_inputs(t):
    torch.manual_seed(1)
    return dict(
        hidden_states = torch.randn(1, 16, 4),
        encoder_hidden_states = torch.randn(1, 4, 16),
        pooled_projections = torch.randn(1, 8),
        timestep = torch.tensor([t]),
        img_ids = torch.zeros(16, 3),
        txt_ids = torch.zeros(4, 3),
        return_dict = False,
    )


def test_fbcache_counts_computed_and_skipped():
    from diffusers import FirstBlockCacheConfig

    transformer = _tiny_flux()
    pipe = types.SimpleNamespace(transformer = transformer)
    assert C.fbcache_counts(pipe) is None
    assert C.install_fbcache_counter()
    assert C.install_fbcache_counter()  # idempotent: no double count below
    transformer.enable_cache(FirstBlockCacheConfig(threshold = 0.5))
    with torch.no_grad(), transformer.cache_context("cond"):
        for _ in range(4):
            transformer(**_flux_inputs(1.0))
    assert C.fbcache_counts(pipe, reset = True) == {"computed": 1, "skipped": 3}
    assert C.fbcache_counts(pipe) == {"computed": 0, "skipped": 0}
    transformer.disable_cache()


def test_fbcache_counter_keeps_length_guard_marker():
    from diffusers.hooks.first_block_cache import FBCHeadBlockHook

    assert C.install_fbcache_counter()
    assert C.install_fbcache_length_guard()
    fn = FBCHeadBlockHook._should_compute_remaining_blocks
    assert fn._unsloth_fbc_counter and fn._unsloth_length_guard
    assert C.install_fbcache_counter() and FBCHeadBlockHook._should_compute_remaining_blocks is fn


def test_counter_does_not_coerce_a_non_bool_decision(monkeypatch):
    from diffusers.hooks.first_block_cache import FBCHeadBlockHook

    class _NoBool:
        def __bool__(self):
            raise AssertionError("coerced the decision (a host sync on a device tensor)")

    decision = _NoBool()
    monkeypatch.setattr(
        FBCHeadBlockHook, "_should_compute_remaining_blocks", lambda self, residual: decision
    )
    assert C.install_fbcache_counter()
    hook = types.SimpleNamespace()
    assert FBCHeadBlockHook._should_compute_remaining_blocks(hook, None) is decision
    assert hook.__dict__[C._FBC_COUNTS_ATTR] == {"computed": 0, "skipped": 0}


def test_video_render_engagement_recorded(monkeypatch):
    from core.inference import video

    class _Pipe:
        def __init__(self):
            self.vae = _VAE()

    vae_pipe = _Pipe()
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    assert U.install_untiled_decode(vae_pipe, "wan2.2-ti2v-5b")
    before = U.untiled_decode_stats(vae_pipe)
    keys = set(vae_pipe.__dict__)
    vae_pipe.vae.decode(Z)
    lines = []
    log = types.SimpleNamespace(
        info = lambda *a: lines.append(a[0] % a[1:]), debug = lambda *a: lines.append(a)
    )
    video._record_render_engagement(vae_pipe, vae_pipe, before, log)
    rec = video._render_engagement(vae_pipe)
    assert rec == {
        "vae_decode": {"untiled": 1, "tiled": 0, "wide": 0, "oom_fallback": 0},
        "fbcache": None,
    }
    assert lines and "untiled': 1" in lines[0]
    assert set(vae_pipe.__dict__) == keys


class _FluxPipe:
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, steps):
        for _ in range(steps):
            with self.transformer.cache_context("cond"):
                self.transformer(**_flux_inputs(1.0))


def test_apply_step_cache_counts_each_decision_on_the_head_hook():
    transformer = _tiny_flux()
    pipe = _FluxPipe(transformer)
    assert C.apply_step_cache(pipe, mode = C.TC_FBCACHE, threshold = 0.5) == C.TC_FBCACHE
    with torch.no_grad():
        pipe(5)
    from diffusers.hooks.first_block_cache import _FBC_LEADER_BLOCK_HOOK

    heads = [
        m._diffusers_hook.hooks[_FBC_LEADER_BLOCK_HOOK]
        for m in transformer.modules()
        if _FBC_LEADER_BLOCK_HOOK in getattr(getattr(m, "_diffusers_hook", None), "hooks", {})
    ]
    assert len(heads) == 1
    assert heads[0].__dict__["_unsloth_fbc_counts"] == {"computed": 1, "skipped": 4}
    transformer.disable_cache()


def test_install_untiled_decode_exposes_counters_on_the_vae(monkeypatch):
    pipe = types.SimpleNamespace(vae = _VAE())
    monkeypatch.setattr(U, "_free_bytes", lambda device: 100 * 2**30)
    assert U.install_untiled_decode(pipe, "wan2.2-ti2v-5b")
    pipe.vae.decode(Z)
    assert pipe.vae._unsloth_untiled_stats["untiled"] == 1


@pytest.mark.parametrize("order", ["counter_first", "guard_first"])
def test_guard_recomputes_are_counted_in_either_install_order(monkeypatch, order):
    # The guard's own shape-change True must be counted, whichever was installed first.
    from diffusers.hooks.first_block_cache import FBCHeadBlockHook

    monkeypatch.setattr(FBCHeadBlockHook, "_should_compute_remaining_blocks", lambda self, r: False)
    steps = [C.install_fbcache_counter, C.install_fbcache_length_guard]
    for install in steps if order == "counter_first" else steps[::-1]:
        assert install()
    fn = FBCHeadBlockHook._should_compute_remaining_blocks
    assert fn._unsloth_fbc_counter and fn._unsloth_length_guard
    state = types.SimpleNamespace(head_block_residual = torch.zeros(1, 4), tail_block_residuals = None)
    hook = types.SimpleNamespace(state_manager = types.SimpleNamespace(get_state = lambda: state))
    assert fn(hook, torch.zeros(1, 8)) is True
    assert fn(hook, torch.zeros(1, 4)) is False
    assert hook.__dict__[C._FBC_COUNTS_ATTR] == {"computed": 1, "skipped": 1}
