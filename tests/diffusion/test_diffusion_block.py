# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block-diffusion profiles (LLaDA2.x, SDAR): mask layout, batch building, noise, loss vs a naive reference."""

import re
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 30
EOS_ID = 31
VOCAB = 32


class _Decoder(nn.Module):
    def __init__(self, hidden = 16):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, hidden)
        self.pos = nn.Embedding(256, hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        use_cache = False,
        return_dict = True,
    ):
        x = self.embed(input_ids) + self.pos(position_ids)
        q, k, v = self.qkv(x)[:, None].chunk(3, dim = -1)
        h = F.scaled_dot_product_attention(q, k, v, attn_mask = attention_mask)[:, 0]
        return (x + self.out(h),)


class _LM(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.model = _Decoder()
        self.lm_head = nn.Linear(16, VOCAB)
        self.config = SimpleNamespace(
            model_type = "llada2_moe", eos_token_id = EOS_ID, mask_token_id = MASK_ID
        )

    def get_decoder(self):
        return self.model

    def get_output_embeddings(self):
        return self.lm_head


def _profiles():
    from unsloth.models.diffusion_block import LLADA2_PROFILE, SDAR_PROFILE
    return LLADA2_PROFILE, SDAR_PROFILE


def _batch(prompt_lengths, real_lengths, length):
    torch.manual_seed(1)
    batch = len(real_lengths)
    ids = torch.randint(0, 29, (batch, length))
    attention = torch.zeros(batch, length, dtype = torch.long)
    labels = torch.full((batch, length), -100)
    for i, (p, r) in enumerate(zip(prompt_lengths, real_lengths)):
        attention[i, :r] = 1
        labels[i, p:r] = ids[i, p:r]
        ids[i, r:] = EOS_ID
    return {"input_ids": ids, "attention_mask": attention, "labels": labels}


def test_block_mask_small_layout():
    from unsloth.models.diffusion_block import block_diffusion_attention_mask

    # L = 4, block 2: rows/cols 0-3 noisy, 4-7 clean.
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype = torch.bool,
    )
    assert torch.equal(block_diffusion_attention_mask(4, 2), expected)


def test_profiles_resolve_and_route():
    from unsloth.models.diffusion import is_diffusion_model_type
    from unsloth.models.diffusion_profiles import resolve_diffusion_profile

    llada2, sdar = _profiles()
    assert resolve_diffusion_profile(SimpleNamespace(model_type = "llada2_moe")) is llada2
    assert (
        resolve_diffusion_profile(
            SimpleNamespace(model_type = "x", architectures = ["SDARForCausalLM"])
        )
        is sdar
    )
    assert is_diffusion_model_type("llada2_moe") and is_diffusion_model_type("sdar")
    assert not is_diffusion_model_type("qwen3")


def test_llada2_lora_targets_skip_routed_experts_and_router():
    llada2, _ = _profiles()
    pattern = re.compile(llada2.lora_target_modules)
    hit = [
        "model.layers.0.attention.query_key_value",
        "model.layers.0.attention.dense",
        "model.layers.0.mlp.gate_proj",
        "model.layers.3.mlp.shared_experts.down_proj",
    ]
    miss = [
        "model.layers.3.mlp.experts.7.up_proj",
        "model.layers.3.mlp.gate",
        "lm_head",
        "model.word_embeddings",
    ]
    assert all(pattern.fullmatch(name) for name in hit)
    assert not any(pattern.fullmatch(name) for name in miss)


def test_llada2_eos_fill_is_block_aligned_and_supervised():
    llada2, _ = _profiles()
    model, args = _LM(), SimpleNamespace()
    inputs = _batch([3, 5, 2], [40, 20, 9], 40)
    inputs["labels"][2] = -100  # a row with nothing to learn gets no EOS target
    clean, maskable, valid = llada2.build_batch(model, inputs, args)
    assert clean.shape[1] == 64  # next multiple of 32 past 40
    assert torch.equal(
        clean[:, :40][inputs["attention_mask"].bool()],
        inputs["input_ids"][inputs["attention_mask"].bool()],
    )
    assert (clean[0, 40:] == EOS_ID).all() and maskable[0, 40:].all()
    assert (clean[1, 20:] == EOS_ID).all() and maskable[1, 20:].all() and not maskable[1, :5].any()
    assert not maskable[2].any()


def test_llada2_row_clipped_at_max_length_gets_no_eos_tail():
    llada2, _ = _profiles()
    inputs = _batch([3, 5], [40, 20], 40)
    clean, maskable, valid = llada2.build_batch(_LM(), inputs, SimpleNamespace(max_length = 40))
    assert not maskable[0, 40:].any() and valid[0, 40:].all() and (clean[0, 40:] == EOS_ID).all()
    assert (clean[1, 20:] == EOS_ID).all() and maskable[1, 20:].all()


def test_left_padding_rejected():
    llada2, _ = _profiles()
    inputs = _batch([2], [10], 12)
    inputs["attention_mask"] = inputs["attention_mask"].flip(1)
    with pytest.raises(ValueError, match = "right padding"):
        llada2.build_batch(_LM(), inputs, SimpleNamespace())


def _naive_loss(profile, model, clean, noisy, masked, maskable, p, valid, block):
    from unsloth.models.diffusion_block import block_diffusion_attention_mask

    batch, length = clean.shape
    allowed = (
        block_diffusion_attention_mask(length, block)[None, None].expand(batch, 1, -1, -1).clone()
    )
    if profile.defaults["key_padding"]:
        keys = torch.cat([valid, valid], 1)[:, None, None, :]
        allowed = (allowed & keys) | torch.eye(2 * length, dtype = torch.bool)
    positions = torch.arange(length)
    hidden = model.model(
        torch.cat([noisy, clean], 1),
        allowed,
        torch.cat([positions, positions])[None].expand(batch, -1),
    )[0]
    logits = model.lm_head(hidden)[:, :length]
    nll = F.cross_entropy(logits.transpose(1, 2), clean, reduction = "none")
    if profile.defaults["diffusion_time_weighting"] == "inverse_t":
        nll = nll / p[:, None]
    nll = nll * masked
    denominator = maskable.sum() if profile.defaults["normalize"] == "supervised" else masked.sum()
    return nll.sum() / denominator


@pytest.mark.parametrize("which", ["llada2", "sdar"])
def test_loss_matches_naive_full_forward(which):
    llada2, sdar = _profiles()
    profile = llada2 if which == "llada2" else sdar
    model = _LM()
    args = SimpleNamespace(diffusion_block_size = 4)
    clean, maskable, valid = profile.build_batch(model, _batch([3, 6], [22, 13], 22), args)
    torch.manual_seed(2)
    noisy, masked, p = profile.sample_noise(clean, maskable, MASK_ID, args)
    ours = profile.loss_from_noise(model, clean, noisy, masked, maskable, p, valid, args)
    ref = _naive_loss(profile, model, clean, noisy, masked, maskable, p, valid, 4)
    torch.testing.assert_close(ours, ref, rtol = 1e-5, atol = 1e-6)


def test_noisy_block_never_sees_its_own_or_later_clean_tokens():
    llada2, _ = _profiles()
    model, args = _LM(), SimpleNamespace(diffusion_block_size = 4)
    clean = torch.randint(0, 29, (1, 12))
    noisy = clean.clone()
    noisy[:, 4:] = MASK_ID
    valid = torch.ones_like(clean, dtype = torch.bool)
    base, _ = llada2.noisy_hidden_and_head(model, noisy, clean, valid, 4)
    changed = clean.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 29
    moved, _ = llada2.noisy_hidden_and_head(model, noisy, changed, valid, 4)
    torch.testing.assert_close(base[:, :8], moved[:, :8])  # block 1 sees clean block 0 only
    assert not torch.allclose(base[:, 8:], moved[:, 8:])  # block 2 sees the changed clean block 1


def test_sdar_padding_does_not_leak_into_real_tokens():
    _, sdar = _profiles()
    model, args = _LM(), SimpleNamespace(diffusion_block_size = 4)
    short = _batch([2], [9], 9)
    padded = _batch([2], [9], 16)
    padded["input_ids"][:, :9] = short["input_ids"]
    padded["labels"][:, :9] = short["labels"]
    outs = []
    for inputs in (short, padded):
        clean, maskable, valid = sdar.build_batch(model, inputs, args)
        noisy = torch.where(maskable, MASK_ID, clean)
        hidden, _ = sdar.noisy_hidden_and_head(model, noisy, clean, valid, 4)
        outs.append(hidden[:, :9])
    torch.testing.assert_close(outs[0], outs[1], rtol = 1e-5, atol = 1e-6)


def test_noise_ranges_and_prompt_untouched():
    llada2, sdar = _profiles()
    clean = torch.randint(0, 29, (2048, 24))
    maskable = torch.ones_like(clean, dtype = torch.bool)
    maskable[:, :5] = False
    for profile, (low, high) in ((llada2, (0.3, 0.8)), (sdar, (1e-3, 1.0))):
        torch.manual_seed(0)
        noisy, masked, p = profile.sample_noise(clean, maskable, MASK_ID, SimpleNamespace())
        assert p.min() >= low and p.max() <= high
        assert not masked[:, :5].any() and torch.equal(noisy == MASK_ID, masked)
        ratio = masked[:, 5:].float().mean().item()
        assert abs(ratio - (low + high) / 2) < 0.02
    _, masked, _ = sdar.sample_noise(clean[:, :6], maskable[:, :6], MASK_ID, SimpleNamespace())
    assert masked.any(dim = 1).all()  # SDAR masks at least one supervised token per row
    _, _, p = llada2.sample_noise(clean, maskable, MASK_ID, SimpleNamespace(diffusion_eps = 0.1))
    assert p.min() >= 0.1 and p.max() <= 0.9


def test_sdar_import_shim_is_scoped():
    from unsloth.models.diffusion_block import _sdar_remote_code_imports
    import importlib.util

    if importlib.util.find_spec("flash_attn") is not None:
        pytest.skip("flash-attn installed: the shim is a no-op")
    with _sdar_remote_code_imports():
        from flash_attn.ops.triton.layer_norm import rms_norm_fn
        x = torch.randn(3, 8)
        torch.testing.assert_close(
            rms_norm_fn(x, torch.ones(8), eps = 1e-6),
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6),
        )
    assert "flash_attn" not in sys.modules


def test_restore_rotary_buffers_refills_zeroed_inv_freq():
    from unsloth.models.diffusion_profiles import restore_rotary_buffers

    def init_fn(config, device = None):
        dim = config.head_dim
        return 1.0 / (
            config.rope_theta ** (torch.arange(0, dim, 2, dtype = torch.float, device = device) / dim)
        ), 1.0

    class Rotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(head_dim = 8, rope_theta = 10000.0)
            self.rope_init_fn = init_fn
            self.register_buffer("inv_freq", torch.zeros(4), persistent = False)
            self.original_inv_freq = self.inv_freq

    holder = nn.Module()
    holder.rotary_emb = Rotary()
    assert restore_rotary_buffers(holder) == 1
    torch.testing.assert_close(holder.rotary_emb.inv_freq, init_fn(holder.rotary_emb.config)[0])
    assert restore_rotary_buffers(holder) == 0  # idempotent


def test_deepspeed_engine_keeps_its_own_forward():
    from unsloth.models.diffusion_block import _is_distributed_wrapper
    engine = type("DeepSpeedEngine", (nn.Module,), {})()
    assert _is_distributed_wrapper(engine) and not _is_distributed_wrapper(_LM())


def test_unknown_time_weighting_is_rejected():
    llada2, _ = _profiles()
    inputs = _batch([3], [20], 20)
    with pytest.raises(ValueError, match = "diffusion_time_weighting"):
        llada2.compute_loss(_LM(), inputs, SimpleNamespace(diffusion_time_weighting = "inverse-t"))
