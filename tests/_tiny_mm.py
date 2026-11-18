# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tiny random vision / audio language checkpoints built from code (no download), fp32 CPU."""

import torch

ARCHS = ("qwen3_5", "gemma4", "gemma4_e2b", "gemma4_moe", "gemma4_unified", "muse_glimmer")
VOCAB = 128
MEDIA_TOKENS = dict(image_token_id = 120, video_token_id = 121)


def _gemma4_config(layers, e2b, moe, kv_heads):
    from transformers import Gemma4Config

    text = dict(
        vocab_size = VOCAB,
        vocab_size_per_layer_input = VOCAB,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = kv_heads,
        head_dim = 8,
        global_head_dim = 16,
        layer_types = (["sliding_attention", "full_attention"] * layers)[:layers],
        sliding_window = 4,
        hidden_size_per_layer_input = 8 if e2b else 0,
        num_kv_shared_layers = 2 if e2b else 0,
        attention_k_eq_v = False,
        pad_token_id = 0,
        tie_word_embeddings = True,
    )
    if moe:
        text.update(enable_moe_block = True, num_experts = 4, top_k_experts = 2, moe_intermediate_size = 16)
    vision = dict(
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
        patch_size = 4,
    )
    audio = dict(hidden_size = 16, num_hidden_layers = 1, num_attention_heads = 2) if e2b else None
    return Gemma4Config(
        text_config = text,
        vision_config = vision,
        audio_config = audio,
        audio_token_id = 122,
        boi_token_id = 123,
        eoi_token_id = 124,
        **MEDIA_TOKENS,
    )


def _gemma4_unified_config(layers):
    from transformers.models.gemma4_unified import Gemma4UnifiedConfig

    text = dict(
        vocab_size = VOCAB,
        vocab_size_per_layer_input = VOCAB,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        layer_types = (["sliding_attention", "full_attention"] * layers)[:layers],
        sliding_window = 4,
        attention_k_eq_v = True,
        hidden_size_per_layer_input = 0,
        num_kv_shared_layers = 0,
        pad_token_id = 0,
        tie_word_embeddings = True,
    )
    vision = dict(hidden_size = 16, patch_size = 4)
    return Gemma4UnifiedConfig(text_config = text, vision_config = vision, **MEDIA_TOKENS)


def _qwen3_5_config(layers):
    from transformers import Qwen3_5Config

    text = dict(
        vocab_size = VOCAB,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
        linear_conv_kernel_dim = 4,
        layer_types = (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
        tie_word_embeddings = True,
        pad_token_id = 0,
    )
    vision = dict(
        depth = 1,
        hidden_size = 16,
        intermediate_size = 32,
        num_heads = 2,
        out_hidden_size = 32,
        patch_size = 4,
        temporal_patch_size = 2,
        spatial_merge_size = 2,
    )
    return Qwen3_5Config(
        text_config = text,
        vision_config = vision,
        vision_start_token_id = 122,
        vision_end_token_id = 123,
        **MEDIA_TOKENS,
    )


def _muse_glimmer_config(layers):
    from transformers import MuseGlimmerConfig

    text = dict(
        vocab_size = VOCAB,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        pad_token_id = 0,
        bos_token_id = 1,
        eos_token_id = 2,
    )
    vision = dict(hidden_size = 16, intermediate_size = 32, num_hidden_layers = 1, num_attention_heads = 2)
    return MuseGlimmerConfig(text_config = text, vision_config = vision, **MEDIA_TOKENS)


def config_for(arch, layers = 4):
    if arch.startswith("gemma4") and arch != "gemma4_unified":
        layers = 6 if arch == "gemma4_e2b" else layers
        return _gemma4_config(layers, arch == "gemma4_e2b", arch == "gemma4_moe", 1)
    return {
        "qwen3_5": _qwen3_5_config,
        "gemma4_unified": _gemma4_unified_config,
        "muse_glimmer": _muse_glimmer_config,
    }[arch](layers)


def build(
    arch,
    layers = 4,
    seed = 0,
    lora = True,
    target_modules = "all-linear",
    causal_lm = False,
):
    """Linear weights at std 1 / sqrt(fan in): at 0.02 Gemma 4's RMSNorms blow fp32 rounding up to O(1) grads."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    config = config_for(arch, layers)
    if causal_lm:
        config = config.get_text_config()
    config._attn_implementation = "eager"
    config.get_text_config()._attn_implementation = "eager"
    torch.manual_seed(seed)
    cls = AutoModelForCausalLM if causal_lm else AutoModelForImageTextToText
    model = cls.from_config(config, dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 2 and "embed" not in name:
                p.normal_(0, p.shape[1] ** -0.5)
            elif "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    if not lora:
        return model.train()
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed + 1)
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = target_modules, init_lora_weights = False, lora_dropout = 0.0
    )
    return get_peft_model(model, config).train()


def batch(
    bsz = 4,
    seq = 8,
    seed = 1,
):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 110, (bsz, seq), generator = g)  # no media / special tokens
    labels = ids.clone()
    labels[0, :3] = -100  # uneven token counts per micro-batch: num_items normalisation matters
    labels[-1, -2:] = -100
    return ids, labels
