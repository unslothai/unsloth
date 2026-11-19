# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HunyuanVideo-1.5 padded-text trim under the CFG batch router: the doubled call bypasses the trim pre-hook, so a
branch the router pads must keep its mask. CPU only."""

from __future__ import annotations

import types

import torch

from core.inference import diffusion_attention as da
from core.inference import diffusion_cfg_batch as cb


class _Attn(torch.nn.Module):
    pass


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()


class HunyuanVideo15Transformer3DModel(torch.nn.Module):
    """The flag picks the attention branch exactly as the null-mask processor does: set, the mask is ignored."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.linspace(0.5, 1.5, 4))
        self.transformer_blocks = torch.nn.ModuleList([_Block(), _Block()])
        self.calls = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        encoder_hidden_states_2 = None,
        encoder_attention_mask_2 = None,
        image_embeds = None,
        return_dict = True,
    ):
        self.calls.append(int(hidden_states.shape[0]))
        if getattr(self.transformer_blocks[0].attn, da._NULL_ATTN_FLAG, False):
            text = encoder_hidden_states.mean(1)
        else:
            m = encoder_attention_mask.float().unsqueeze(-1)
            text = (encoder_hidden_states * m).sum(1) / m.sum(1)
        return (
            hidden_states * self.w.view(1, -1, 1, 1)
            + text.mean(-1).view(-1, 1, 1, 1)
            + timestep.view(-1, 1, 1, 1),
        )


HunyuanVideo15Transformer3DModel.__module__ = "diffusers.models.fake"


def _trimmed(model):
    pre = model.register_forward_pre_hook(da._hunyuan_trim_pre_hook, with_kwargs = True)
    post = model.register_forward_hook(da._hunyuan_trim_post_hook, always_call = True)
    model._unsloth_trim_hook = (pre, post)
    return model


def _offloaded(model):
    sub = torch.nn.Linear(1, 1)
    model.add_module("offloaded_group", sub)
    sub._diffusers_hook = types.SimpleNamespace(hooks = {"group_offloading": object()})
    return model


def _loop(model, steps = 6):
    g = torch.Generator().manual_seed(0)
    lat = torch.randn(1, 4, 2, 2, generator = g)
    pos, neg = torch.randn(1, 8, 6, generator = g), torch.randn(1, 8, 6, generator = g)
    # The text encoder pads both prompts to one length; the trim cuts each to its own (6 vs 2 valid tokens).
    pm = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])
    nm = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]])
    byt5 = torch.zeros(1, 4, 6)
    bm = torch.zeros(1, 4, dtype = torch.long)
    for i in range(steps):
        t = torch.tensor([1.0 - 0.1 * i])
        kw = dict(encoder_hidden_states_2 = byt5, encoder_attention_mask_2 = bm)
        c = model(
            hidden_states = lat,
            timestep = t,
            encoder_hidden_states = pos,
            encoder_attention_mask = pm,
            **kw,
        )[0]
        u = model(
            hidden_states = lat,
            timestep = t,
            encoder_hidden_states = neg,
            encoder_attention_mask = nm,
            **kw,
        )[0]
        lat = lat - 0.1 * (u + 5.0 * (c - u))
    return lat


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def test_a_padded_branch_keeps_its_mask_in_the_doubled_call(monkeypatch):
    monkeypatch.delenv(cb.CFG_BATCH_ENV, raising = False)
    ref = _loop(_trimmed(HunyuanVideo15Transformer3DModel()))
    model = _offloaded(_trimmed(HunyuanVideo15Transformer3DModel()))
    pipe = _Pipe(model)
    assert cb.install_cfg_batch(pipe, offload_policy = "group") is None
    got = _loop(model)
    assert 2 in model.calls, cb.router_for(pipe).describe()
    torch.testing.assert_close(got, ref)
    assert not da._hunyuan_null_mask_state(model), "the flag never latches past a call"
