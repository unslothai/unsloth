# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""LTX-2 / LTX-2.5 video-only LoRA training drops the audio stream: CPU only, bitwise against the full model."""

from __future__ import annotations

import inspect

import pytest
import torch

from core.training import diffusion_dit_trainer as dit

diffusers = pytest.importorskip("diffusers")
LTX2 = getattr(diffusers, "LTX2VideoTransformer3DModel", None)

# The LTX-2.5 transformer config (unsloth/LTX-2.5-FP8 transformer/config.json) shrunk to a few MB.
_CONF = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 16,
    "attention_out_bias": True,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "audio_cross_attn_mod": True,
    "audio_ff_bias": True,
    "audio_gated_attn": True,
    "audio_hop_length": 160,
    "audio_in_channels": 8,
    "audio_num_attention_heads": 2,
    "audio_out_channels": 8,
    "audio_sampling_rate": 16000,
    "audio_scale_factor": 4,
    "caption_channels": 16,
    "cross_attention_dim": 32,
    "cross_attn_mod": True,
    "ff_bias": False,
    "gated_attn": True,
    "in_channels": 16,
    "norm_elementwise_affine": False,
    "num_attention_heads": 2,
    "num_layers": 2,
    "out_channels": 16,
    "patch_size": 1,
    "patch_size_t": 1,
    "perturbed_attn": True,
    "qk_norm": "rms_norm_across_heads",
    "rope_type": "split",
    "use_keyframes_abs_pos_embedding": True,
    "use_prompt_adaln_single": True,
    "use_prompt_embeddings": False,
    "vae_scale_factors": [8, 32, 32],
}

pytestmark = pytest.mark.skipif(
    LTX2 is None, reason = "diffusers without LTX2VideoTransformer3DModel"
)


def _model(drop: bool):
    params = inspect.signature(LTX2.__init__).parameters
    if not all(k in params for k in ("audio_ff_bias", "use_keyframes_abs_pos_embedding")):
        pytest.skip("diffusers predates the LTX-2.5 transformer config")
    torch.manual_seed(0)
    m = LTX2(**_CONF)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.parameters():
            # nonzero everywhere: a zero-initialised gate or table would hide a leak of the audio stream
            p.copy_(torch.randn(p.shape, generator = g) * 0.05)
    dropped = dit._ltx2_drop_audio_stream(m) if drop else 0
    m.requires_grad_(False)
    from peft import LoraConfig

    torch.manual_seed(2)
    m.add_adapter(
        LoraConfig(
            r = 4, lora_alpha = 4, init_lora_weights = "gaussian", target_modules = list(dit._LTX2_TARGETS)
        )
    )
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator = g) * 0.05)
                p.requires_grad_(True)
    return m, dropped


def _step(m, gradient_checkpointing: bool):
    if gradient_checkpointing:
        m.enable_gradient_checkpointing()
    m.train()
    torch.manual_seed(10)
    noisy = torch.randn(1, 16, 2, 2, 3)
    emb = (torch.randn(1, 6, 32), torch.randn(1, 6, 16), torch.tensor([[0, 1, 1, 1, 1, 1]]))
    pred = dit._ltx2_forward(
        m,
        noisy,
        torch.full((1,), 700.0),
        torch.full((1, 1, 1, 1, 1), 0.7),
        emb,
        None,
        "cpu",
        torch.float32,
    )
    torch.nn.functional.mse_loss(pred, torch.ones_like(pred)).backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    return pred.detach(), grads, torch.randn(3)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_dropping_the_audio_stream_is_bitwise_on_video_pred_and_lora_grads(gradient_checkpointing):
    full, _ = _model(drop = False)
    lean, dropped = _model(drop = True)
    assert dropped > 0
    assert (
        sum(p.numel() for p in full.parameters()) - sum(p.numel() for p in lean.parameters())
        == dropped
    )
    pa, ga, ra = _step(full, gradient_checkpointing)
    pb, gb, rb = _step(lean, gradient_checkpointing)
    assert torch.equal(pa, pb)
    assert ga.keys() == gb.keys() and len(ga) == 8 * _CONF["num_layers"] * 2
    assert all(ga[k].abs().sum() > 0 for k in ga)
    assert all(torch.equal(ga[k], gb[k]) for k in ga)
    # the placeholder audio noise is still drawn, so the RNG stream after a step is unchanged
    assert torch.equal(ra, rb)


def test_every_block_loses_its_audio_and_cross_modal_weights():
    lean, _ = _model(drop = True)
    names = dit._LTX2_AUDIO_BLOCK_MODULES + dit._LTX2_CROSS_MODAL_BLOCK_MODULES
    for block in lean.transformer_blocks:
        for name in names:
            assert not list(getattr(block, name).parameters())
        # the video stream is untouched
        assert list(block.attn1.parameters()) and list(block.ff.parameters())
    assert not [
        n
        for n, _ in lean.named_parameters()
        if ".audio_attn" in n or "_to_audio_attn" in n or "to_video_attn" in n
    ]


def test_a_run_without_isolated_modalities_raises_instead_of_dropping_the_cross_attention():
    lean, _ = _model(drop = True)
    with torch.no_grad(), pytest.raises(RuntimeError, match = "isolate_modalities"):
        lean(
            hidden_states = torch.randn(1, 12, 16),
            audio_hidden_states = torch.randn(1, 3, 8),
            encoder_hidden_states = torch.randn(1, 6, 32),
            audio_encoder_hidden_states = torch.randn(1, 6, 16),
            timestep = torch.full((1,), 700.0),
            sigma = torch.full((1,), 700.0),
            num_frames = 2,
            height = 2,
            width = 3,
            fps = 24.0,
            audio_num_frames = 3,
            isolate_modalities = False,
            return_dict = False,
        )


def test_the_env_keeps_the_audio_stream(monkeypatch):
    monkeypatch.setenv(dit.LTX2_TRAIN_AUDIO_STREAM_ENV, "1")
    m, dropped = _model(drop = True)
    assert dropped == 0
    assert list(m.transformer_blocks[0].audio_attn1.parameters())


@pytest.mark.parametrize("loader", ["_ltx2_load_transformer", "_ltx25_load_transformer"])
def test_both_ltx_loaders_drop_before_anything_else_sees_the_model(loader, monkeypatch):
    seen = {}

    class Fake:
        transformer_blocks = []

    monkeypatch.setattr(dit, "_load_dit_transformer", lambda *a, **k: Fake())
    monkeypatch.setattr(dit, "_ltx25_dit_source", lambda cfg: ("repo", "transformer"))
    monkeypatch.setattr(dit, "_with_base", lambda cfg, repo: cfg)
    monkeypatch.setattr(dit, "_ltx2_drop_audio_stream", lambda t: seen.setdefault("t", t))
    out = getattr(dit, loader)(object(), "cpu", torch.bfloat16, "int8")
    assert seen["t"] is out
