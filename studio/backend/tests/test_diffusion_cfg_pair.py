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
