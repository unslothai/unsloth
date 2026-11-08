# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest
import torch

from core.inference import diffusion_cfg_batch as cb
from core.inference import diffusion_cfg_pair as cp

diffusers = pytest.importorskip("diffusers")
QwenImageTransformer2DModel = diffusers.QwenImageTransformer2DModel
from diffusers.hooks import apply_group_offloading  # noqa: E402


def _tiny(zero_cond_t = True, layers = 3):
    torch.manual_seed(0)
    m = QwenImageTransformer2DModel(
        patch_size = 2,
        in_channels = 16,
        out_channels = 4,
        num_layers = layers,
        attention_head_dim = 8,
        num_attention_heads = 2,
        joint_attention_dim = 12,
        axes_dims_rope = (2, 2, 4),
        zero_cond_t = zero_cond_t,
    ).eval()
    return m


def _offload(m):
    apply_group_offloading(
        m, onload_device = torch.device("cpu"), offload_type = "block_level", num_blocks_per_group = 1
    )
    return m


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def _edit_loop(
    model,
    steps = 4,
    *,
    neg_len = 3,
    scale = 4.0,
    swap_neg_at = None,
):
    """The diffusers Qwen-Image-Edit-Plus loop: one ``latent_model_input`` per step (target + source image tokens),
    cond then uncond call with their own prompt lengths and masks, normalised true CFG."""
    g = torch.Generator().manual_seed(1)
    latents = torch.randn(1, 16, 16, generator = g)
    image_latents = torch.randn(1, 16, 16, generator = g)
    pos = torch.randn(1, 7, 12, generator = g)
    neg = torch.randn(1, neg_len, 12, generator = g)
    pos_mask = torch.ones(1, 7, dtype = torch.long)
    neg_mask = torch.ones(1, neg_len, dtype = torch.long)
    img_shapes = [[(1, 4, 4), (1, 4, 4)]]
    outs = []
    with torch.no_grad():
        for i in range(steps):
            if i == swap_neg_at:
                neg = torch.randn(1, neg_len, 12, generator = g)
            latent_model_input = torch.cat([latents, image_latents], dim = 1)
            t = torch.full((1,), 1000.0 - 200 * i)
            common = dict(
                hidden_states = latent_model_input,
                timestep = t / 1000,
                guidance = None,
                img_shapes = img_shapes,
                attention_kwargs = None,
                return_dict = False,
            )
            cond = model(encoder_hidden_states_mask = pos_mask, encoder_hidden_states = pos, **common)[
                0
            ][:, : latents.size(1)]
            uncond = model(
                encoder_hidden_states_mask = neg_mask, encoder_hidden_states = neg, **common
            )[0][:, : latents.size(1)]
            comb = uncond + scale * (cond - uncond)
            pred = comb * (cond.norm(dim = -1, keepdim = True) / comb.norm(dim = -1, keepdim = True))
            latents = latents - 0.1 * pred
            outs.append(pred)
    return latents, outs


def _block_calls(model):
    counts = [0] * len(model.transformer_blocks)
    for i, b in enumerate(model.transformer_blocks):
        b.register_forward_pre_hook(lambda *_a, _i = i: counts.__setitem__(_i, counts[_i] + 1))
    return counts


@pytest.mark.parametrize("zero_cond_t", [True, False])
def test_paired_steps_are_bit_identical_and_upload_each_block_once(zero_cond_t):
    ref, ref_outs = _edit_loop(_offload(_tiny(zero_cond_t)))
    model = _offload(_tiny(zero_cond_t))
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    assert router.paired is not None
    counts = _block_calls(model)
    got, outs = _edit_loop(model, steps = 4)
    assert torch.equal(got, ref)
    assert all(torch.equal(a, b) for a, b in zip(outs, ref_outs))
    # step 1 plain (2 calls per block), steps 2-4 paired (1 call per block)
    assert counts == [2 + 3] * len(counts)
    assert router.stats["paired_steps"] == 3 and router.stats["hits"] == 3
    assert router.describe()["mode"] == "pair"
    cb.end_cfg_batch_render(pipe)
    got2, _ = _edit_loop(model, steps = 4)
    assert torch.equal(got2, ref)
    assert router.stats["paired_steps"] == 6


def test_a_changed_second_branch_reruns_plain():
    ref, ref_outs = _edit_loop(_offload(_tiny()), steps = 5, swap_neg_at = 3)
    model = _offload(_tiny())
    pipe = _Pipe(model)
    cb.install_cfg_batch(pipe, offload_policy = "group")
    router = cb.router_for(pipe)
    got, outs = _edit_loop(model, steps = 5, swap_neg_at = 3)
    assert torch.equal(got, ref)
    assert all(torch.equal(a, b) for a, b in zip(outs, ref_outs))
    assert (
        router.stats["misses"] == 1
        and router.stats["paired_steps"] == 4
        and router.stats["hits"] == 3
    )


def test_kill_switch_keeps_the_batch_router(monkeypatch):
    monkeypatch.setenv(cp.CFG_PAIR_ENV, "0")
    model = _offload(_tiny())
    pipe = _Pipe(model)
    assert cb.cfg_pair_decline(pipe, offload_policy = "group") == f"{cp.CFG_PAIR_ENV}=0"
    cb.install_cfg_batch(pipe, offload_policy = "group")
    router = cb.router_for(pipe)
    assert router is None or router.paired is None
    if router is not None:
        cb.uninstall_cfg_batch(pipe)


def test_declines_a_resident_or_unhooked_denoiser():
    assert (
        cb.cfg_pair_decline(_Pipe(_tiny()), offload_policy = "none")
        == "the denoiser stays on the card"
    )
    model = _offload(_tiny())
    assert cb.cfg_pair_decline(_Pipe(model), offload_policy = "group") is None
    pipe = _Pipe(model)
    pipe._unsloth_cuda_graphs = (object(),)
    assert cb.cfg_pair_decline(pipe, offload_policy = "group") == "the denoiser replays CUDA graphs"
    assert cb.cfg_pair_decline(pipe, offload_policy = "group", graphs = False) is None


def test_a_lora_scale_or_controlnet_call_stays_plain():
    model = _offload(_tiny())
    pf = cp.PairedForward(model)
    with pytest.raises(cp.NotPairable):
        pf.run(model, model.forward, None, ((), {"attention_kwargs": {"scale": 0.5}}), ((), {}))
    with pytest.raises(cp.NotPairable):
        pf.run(model, model.forward, None, ((), {"controlnet_block_samples": [1]}), ((), {}))


# ---------------------------------------------------------------------------------------------- Qwen-Image-2.1


def _q21_mod():
    return pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")


def _q21(layers = 3):
    qmod = _q21_mod()
    torch.manual_seed(0)
    m = qmod.QwenImage21Transformer2DModel(
        num_layers = layers,
        attention_head_dim = 16,
        num_attention_heads = 2,
        context_in_dim = 24,
        in_channels = 8,
        out_channels = 8,
        axes_dims_rope = (4, 6, 6),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.2)
    return m


def _q21_loop(
    model,
    steps = 5,
    *,
    cond = True,
    cache = True,
    swap_neg_at = None,
    scale = 4.0,
):
    """The diffusers Qwen-Image-2.1 loop: per-branch prompts, masks and prefix KV caches; step 1 prefills."""
    qmod = _q21_mod()
    g = torch.Generator().manual_seed(3)
    th, tw, ch, cw = 4, 6, 4, 4
    target = th * tw

    def prompt(text):
        vlm = torch.zeros(text, dtype = torch.bool)
        if cond:
            vlm = torch.cat(
                [vlm[: text // 2], torch.ones(ch * cw // 4, dtype = torch.bool), vlm[text // 2 :]]
            )
        mask = torch.cat([vlm, torch.ones(target // 4, dtype = torch.bool)])[None]
        return torch.randn(1, vlm.numel(), 24, generator = g), mask

    pos, pos_mask = prompt(9)
    neg, neg_mask = prompt(5)
    shapes = [[(1, ch, cw), (1, th, tw)] if cond else [(1, th, tw)]]
    image = torch.randn(1, ch * cw, 8, generator = g) if cond else None
    latents = torch.randn(1, target, 8, generator = g)
    n = len(model.transformer_blocks)
    caches = (qmod.QwenImage21KVCache(n), qmod.QwenImage21KVCache(n)) if cache else (None, None)
    outs = []
    with torch.inference_mode():
        for i in range(steps):
            if i == swap_neg_at:
                neg = torch.randn_like(neg)
            mode = None if not cache else ("extract" if i == 0 else "cached")
            hs = latents if image is None else torch.cat([image, latents], 1)
            common = dict(
                hidden_states = hs,
                timestep = torch.full((1,), 1.0 - i / steps),
                img_shapes = shapes,
                kv_cache_mode = mode,
                return_dict = False,
            )
            c = model(encoder_hidden_states = pos, img_mask = pos_mask, kv_cache = caches[0], **common)[0]
            u = model(encoder_hidden_states = neg, img_mask = neg_mask, kv_cache = caches[1], **common)[0]
            c, u = c[:, -target:], u[:, -target:]
            pred = u + scale * (c - u)
            latents = latents - 0.1 * pred
            outs.append((c.clone(), u.clone()))
    return latents, outs


@pytest.fixture
def _q21_fast(request, monkeypatch):
    from core.inference import diffusion_qwenimage21 as q21

    q21.uninstall()
    if request.param:
        assert q21.install()
    yield request.param
    q21.uninstall()


@pytest.mark.parametrize("_q21_fast", [False, True], indirect = True, ids = ["stock", "fast_step"])
@pytest.mark.parametrize("cond", [True, False])
def test_q21_decode_steps_pair_bit_identical_one_upload_per_block(_q21_fast, cond):
    ref, ref_outs = _q21_loop(_offload(_q21()), cond = cond)
    model = _offload(_q21())
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    router = cb.router_for(pipe)
    assert router.paired is not None and router.paired.branch_cache is not None
    counts = _block_calls(model)
    shared = []
    model.modulation[1].register_forward_pre_hook(lambda *_a: shared.append(1))
    got, outs = _q21_loop(model, cond = cond)
    assert torch.equal(got, ref)
    for (c, u), (rc, ru) in zip(outs, ref_outs):
        assert torch.equal(c, rc) and torch.equal(u, ru)
    # step 1 prefills both caches plain (2 calls per block); steps 2-5 pair (1 call per block)
    assert counts == [2 + 4] * len(counts)
    assert len(shared) == 2 + 4
    assert router.stats["paired_steps"] == 4 and router.stats["hits"] == 4
    assert router.stats["misses"] == 0
    cb.end_cfg_batch_render(pipe)
    got2, _ = _q21_loop(model, cond = cond)
    assert torch.equal(got2, ref)
    assert router.stats["paired_steps"] == 8


def test_q21_a_changed_negative_prompt_reruns_plain():
    ref, _ = _q21_loop(_offload(_q21()), steps = 6, swap_neg_at = 3)
    model = _offload(_q21())
    pipe = _Pipe(model)
    cb.install_cfg_batch(pipe, offload_policy = "group")
    router = cb.router_for(pipe)
    got, _ = _q21_loop(model, steps = 6, swap_neg_at = 3)
    assert torch.equal(got, ref)
    assert router.stats["misses"] == 1 and router.stats["hits"] == 4


def test_q21_without_a_prefix_cache_stays_plain_and_exact():
    ref, _ = _q21_loop(_offload(_q21()), cache = False)
    model = _offload(_q21())
    pipe = _Pipe(model)
    cb.install_cfg_batch(pipe, offload_policy = "group")
    router = cb.router_for(pipe)
    counts = _block_calls(model)
    got, _ = _q21_loop(model, cache = False)
    assert torch.equal(got, ref)
    assert router.stats["paired_steps"] == 0 and counts == [2 * 5] * len(counts)


def test_q21_kill_switch(monkeypatch):
    monkeypatch.setenv(cp.CFG_PAIR_Q21_ENV, "0")
    model = _offload(_q21())
    assert cb.cfg_pair_decline(_Pipe(model), offload_policy = "group") == (
        "QwenImage21Transformer2DModel has no paired block loop"
    )
    assert cb.cfg_pair_decline(_Pipe(_offload(_tiny())), offload_policy = "group") is None
