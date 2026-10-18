# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ling 3.x checkpoints with `q_lora_rank` load their low-rank query on transformers' bailing_hybrid (#13948)."""

import json
import os
import shutil

import pytest
import torch

import unsloth  # noqa: F401
from unsloth.import_fixes import fix_transformers_bailing_hybrid_low_rank_query
from unsloth.save import _base_config_dir

modeling = pytest.importorskip("transformers.models.bailing_hybrid.modeling_bailing_hybrid")
from transformers import BailingHybridConfig, BailingHybridForCausalLM


def _config(q_lora_rank):
    return BailingHybridConfig(
        vocab_size = 64,
        pad_token_id = 0,
        eos_token_id = 1,
        hidden_size = 32,
        intermediate_size = 64,
        moe_intermediate_size = 16,
        num_hidden_layers = 4,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = 4,
        num_experts_per_tok = 2,
        n_group = 1,
        topk_group = 1,
        first_k_dense_replace = 1,
        layer_group_size = 4,
        kv_lora_rank = 16,
        qk_nope_head_dim = 8,
        qk_rope_head_dim = 4,
        v_head_dim = 8,
        q_lora_rank = q_lora_rank,
    )


def _mla(model):
    return model.model.layers[3].self_attn


def test_low_rank_query_replaces_q_proj():
    fix_transformers_bailing_hybrid_low_rank_query()
    torch.manual_seed(0)
    model = BailingHybridForCausalLM(_config(12)).float().eval()
    keys = set(model.state_dict())
    assert {
        "model.layers.3.self_attn.q_a_proj.weight",
        "model.layers.3.self_attn.q_a_layernorm.weight",
        "model.layers.3.self_attn.q_b_proj.weight",
    } <= keys
    assert "model.layers.3.self_attn.q_proj.weight" not in keys
    attention = _mla(model)
    assert attention.q_b_proj.weight.shape == (2 * 12, 12)
    x = torch.randn(1, 5, 32)
    expected = attention.q_b_proj(attention.q_a_layernorm(attention.q_a_proj(x)))
    torch.testing.assert_close(attention.q_proj(x), expected)


def test_checkpoint_low_rank_query_loads_without_missing_keys(tmp_path):
    fix_transformers_bailing_hybrid_low_rank_query()
    torch.manual_seed(0)
    model = BailingHybridForCausalLM(_config(12)).float()
    model.save_pretrained(tmp_path)
    reloaded, info = BailingHybridForCausalLM.from_pretrained(tmp_path, output_loading_info = True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    torch.testing.assert_close(
        _mla(reloaded).q_a_proj.weight.float(), _mla(model).q_a_proj.weight.float()
    )


def test_full_rank_query_is_untouched():
    fix_transformers_bailing_hybrid_low_rank_query()
    model = BailingHybridForCausalLM(_config(None))
    attention = _mla(model)
    assert isinstance(attention.q_proj, torch.nn.Linear)
    assert not hasattr(attention, "q_a_proj")


def test_patch_is_idempotent():
    fix_transformers_bailing_hybrid_low_rank_query()
    init = modeling.BailingHybridAttention.__init__
    fix_transformers_bailing_hybrid_low_rank_query()
    assert modeling.BailingHybridAttention.__init__ is init


def test_base_config_dir_holds_only_the_config(tmp_path, monkeypatch):
    import huggingface_hub

    cached = tmp_path / "config.json"
    cached.write_text('{"kda_lower_bound": -5}')
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(cached))
    directory = _base_config_dir("org/model")
    try:
        assert os.listdir(directory) == ["config.json"]
        with open(os.path.join(directory, "config.json")) as f:
            assert json.load(f) == {"kda_lower_bound": -5}
    finally:
        shutil.rmtree(directory)


def test_base_config_dir_unavailable(monkeypatch):
    import huggingface_hub

    def missing(*args, **kwargs):
        raise huggingface_hub.errors.LocalEntryNotFoundError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert _base_config_dir("org/model") is None
