# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason = "splitting the model needs two CUDA devices",
)


def test_split_gemma4_trains_with_the_vision_attention_mask(tmp_path):
    from unsloth import FastModel
    import transformers

    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        pytest.skip(reason = "this transformers has no Gemma 4")
    config = transformers.Gemma4Config(
        text_config = dict(
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 32,
            global_head_dim = 32,
            hidden_size_per_layer_input = 0,
            layer_types = ["sliding_attention", "full_attention"],
            sliding_window = 16,
            use_bidirectional_attention = "vision",
        ),
        vision_config = dict(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
            head_dim = 16,
        ),
        audio_config = None,
    )
    transformers.Gemma4ForConditionalGeneration(config).save_pretrained(tmp_path)
    transformers.AutoProcessor.from_pretrained(
        "unsloth/gemma-4-31B-it-unsloth-bnb-4bit"
    ).save_pretrained(tmp_path)

    # The head-aware planner's shape: embedding with lm_head on the last card, root hook on the first.
    device_map = {
        "model.vision_tower": 0,
        "model.embed_vision": 0,
        "model.language_model.layers.0": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.layers.1": 1,
        "model.language_model.norm": 1,
        "model.language_model.embed_tokens": 1,
        "lm_head": 1,
    }
    model, _ = FastModel.from_pretrained(
        str(tmp_path),
        load_in_4bit = False,
        device_map = device_map,
        max_seq_length = 64,
        attn_implementation = "sdpa",
    )
    model.train()
    input_ids = torch.randint(10, 1000, (1, 24), device = "cuda:0")
    loss = model(
        input_ids = input_ids,
        labels = input_ids,
        mm_token_type_ids = torch.zeros_like(input_ids),
    ).loss
    assert torch.isfinite(loss)
