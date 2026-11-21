# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2.x all-ones text masks: dropped before the DiT forward (same output, no bias on text cross-attention)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import video_ltx2 as L  # noqa: E402

_TINY_DIT = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 8,
    "attention_out_bias": True,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "audio_cross_attn_mod": True,
    "audio_ff_bias": True,
    "audio_gated_attn": True,
    "audio_hop_length": 160,
    "audio_in_channels": 4,
    "audio_num_attention_heads": 2,
    "audio_out_channels": 4,
    "audio_patch_size": 1,
    "audio_patch_size_t": 1,
    "audio_pos_embed_max_pos": 20,
    "audio_sampling_rate": 16000,
    "audio_scale_factor": 4,
    "base_height": 2048,
    "base_width": 2048,
    "caption_channels": 16,
    "causal_offset": 1,
    "cross_attention_dim": 16,
    "cross_attn_mod": True,
    "cross_attn_timestep_scale_multiplier": 1000,
    "ff_bias": False,
    "gated_attn": True,
    "in_channels": 8,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 8,
    "patch_size": 1,
    "patch_size_t": 1,
    "perturbed_attn": True,
    "pos_embed_max_pos": 20,
    "qk_norm": "rms_norm_across_heads",
    "rope_double_precision": False,
    "rope_theta": 10000.0,
    "rope_type": "split",
    "timestep_scale_multiplier": 1000,
    "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True,
    "use_prompt_embeddings": False,
    "vae_scale_factors": [8, 32, 32],
}


def _dit():
    torch.manual_seed(0)
    return diffusers.LTX2VideoTransformer3DModel.from_config(_TINY_DIT).eval()


def _inputs(mask):
    g = torch.Generator().manual_seed(1)
    b, f, h, w, a, t = mask.shape[0], 2, 2, 2, 3, mask.shape[1]
    return dict(
        hidden_states = torch.randn(b, f * h * w, 8, generator = g),
        audio_hidden_states = torch.randn(b, a, 4, generator = g),
        encoder_hidden_states = torch.randn(b, t, 16, generator = g),
        audio_encoder_hidden_states = torch.randn(b, t, 16, generator = g),
        timestep = torch.full((b,), 500.0),
        sigma = torch.full((b,), 0.5),
        encoder_attention_mask = mask,
        audio_encoder_attention_mask = mask,
        num_frames = f,
        height = h,
        width = w,
        audio_num_frames = a,
        return_dict = False,
    )


class _SdpaSpy:
    """Records the attn_mask of every SDPA call diffusers' native dispatch makes."""

    def __init__(self, monkeypatch):
        self.masks = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(*args, **kwargs):
            self.masks.append(kwargs.get("attn_mask"))
            return real(*args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)


def _run(dit, mask):
    with torch.no_grad():
        return dit(**_inputs(mask))


def test_all_ones_mask_is_dropped_and_the_output_is_unchanged(monkeypatch):
    monkeypatch.delenv(L.LTX2_FULL_MASK_ENV, raising = False)
    dit = _dit()
    mask = torch.ones(1, 5, dtype = torch.int64)
    spy = _SdpaSpy(monkeypatch)
    ref = _run(dit, mask)
    stock_masked = sum(m is not None for m in spy.masks)
    assert stock_masked > 0, "the stock forward passes the text bias to SDPA"
    pipe = types.SimpleNamespace(transformer = dit)
    assert L.install_ltx2_full_mask_drop(pipe) == 1
    spy.masks.clear()
    out = _run(dit, mask)
    assert spy.masks and all(m is None for m in spy.masks)
    assert getattr(dit, L._LTX2_MASK_DROPS) == 2
    for got, want in zip(out, ref):
        torch.testing.assert_close(got, want, rtol = 1e-5, atol = 1e-6)


def test_padded_mask_is_kept(monkeypatch):
    monkeypatch.delenv(L.LTX2_FULL_MASK_ENV, raising = False)
    dit = _dit()
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    ref = _run(dit, mask)
    L.install_ltx2_full_mask_drop(types.SimpleNamespace(transformer = dit))
    spy = _SdpaSpy(monkeypatch)
    out = _run(dit, mask)
    assert sum(m is not None for m in spy.masks) > 0
    assert getattr(dit, L._LTX2_MASK_DROPS, 0) == 0
    for got, want in zip(out, ref):
        assert torch.equal(got, want)


def test_bool_and_float_masks_and_batches(monkeypatch):
    dit = _dit()
    L.install_ltx2_full_mask_drop(types.SimpleNamespace(transformer = dit))
    hook = L._ltx2_full_mask_pre_hook
    for mask, dropped in (
        (torch.ones(2, 4, dtype = torch.bool), True),
        (torch.ones(2, 4), True),
        (torch.tensor([[1.0, 1.0], [1.0, 0.0]]), False),
        (torch.tensor([[1.0, 0.5]]), False),
        (torch.zeros(1, 1, 4), False),  # already an additive bias
    ):
        _, kw = hook(dit, (), {"encoder_attention_mask": mask})
        assert (kw["encoder_attention_mask"] is None) is dropped, mask


def test_verdict_is_memoised_per_tensor_and_version(monkeypatch):
    dit = _dit()
    mask = torch.ones(1, 4)
    calls = []
    real_all = torch.all
    monkeypatch.setattr(torch, "all", lambda *a, **k: calls.append(1) or real_all(*a, **k))
    for _ in range(3):
        assert L._ltx2_mask_all_ones(dit, mask)
    assert len(calls) == 1
    mask[0, 1] = 0  # in-place edit bumps the version: decided again
    assert not L._ltx2_mask_all_ones(dit, mask)
    assert len(calls) == 2


def test_kill_switch_and_other_dits(monkeypatch):
    dit = _dit()
    monkeypatch.setenv(L.LTX2_FULL_MASK_ENV, "0")
    assert L.install_ltx2_full_mask_drop(types.SimpleNamespace(transformer = dit)) == 0
    assert getattr(dit, L._LTX2_MASK_HOOK, None) is None
    monkeypatch.delenv(L.LTX2_FULL_MASK_ENV)
    other = torch.nn.Linear(2, 2)
    assert L.install_ltx2_full_mask_drop(types.SimpleNamespace(transformer = other)) == 0
    assert not other._forward_pre_hooks


def test_install_is_idempotent():
    dit = _dit()
    pipe = types.SimpleNamespace(transformer = dit)
    assert L.install_ltx2_full_mask_drop(pipe) == 1
    assert L.install_ltx2_full_mask_drop(pipe) == 1
    assert len(dit._forward_pre_hooks) == 1


def test_a_failing_check_leaves_the_call_unchanged(monkeypatch):
    dit = _dit()
    mask = torch.ones(1, 4)

    def boom(*_a, **_k):
        raise RuntimeError("capture")

    monkeypatch.setattr(L, "_ltx2_mask_all_ones", boom)
    assert L._ltx2_full_mask_pre_hook(dit, (), {"encoder_attention_mask": mask}) is None
