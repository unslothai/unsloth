# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Latent-query MQA sinks (DeepSeek-V4.1) are sharded by tp_native, so the registry must not refuse tp. CPU, meta."""

import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import parallel_capabilities as caps
from unsloth.parallel_env import ParallelLayout

dsv41 = pytest.importorskip("unsloth.models.deepseek_v41.modeling")
from unsloth.models.deepseek_v41.configuration import DeepseekV41Config  # noqa: E402
from unsloth.models import tp_native  # noqa: E402


@pytest.fixture(autouse = True)
def clean_env(monkeypatch):
    for key in ("UNSLOTH_PARALLEL_STRICT", "UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "WORLD_SIZE"):
        monkeypatch.delenv(key, raising = False)


def _model():
    text = dict(
        vocab_size = 512,
        hidden_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        head_dim = 32,
        qk_rope_head_dim = 8,
        q_lora_rank = 32,
        o_groups = 4,
        o_lora_rank = 16,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        compress_ratios = [0, 2, 0],
        index_n_heads = 2,
        index_head_dim = 16,
        index_topk = 3,
        engram_layer_ids = [1],
        engram_vocab_size = 101,
        engram_max_ngram_size = 3,
        engram_n_heads = 2,
        engram_head_dim = 32,
        engram_compressed_vocab_size = 97,
        num_nextn_predict_layers = 1,
        max_position_embeddings = 1024,
    )
    primes, offsets = dsv41.engram_layout(
        tuple(text["engram_layer_ids"]),
        text["engram_max_ngram_size"],
        text["engram_n_heads"],
        text["engram_vocab_size"],
    )
    text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes)]
    with torch.device("meta"):
        model = dsv41.DeepseekV41ForCausalLM(DeepseekV41Config(text_config = text))
    return model.requires_grad_(False)


def test_latent_query_sinks_do_not_trip_the_direct_parameter_rule():
    model = _model()
    assert any(n.endswith("self_attn.sinks") for n, _ in model.named_parameters())
    tp_native.derive_tp_plan(model, 2)
    assert not caps.query_from(None, None, model).direct_params
    caps.check_layout(
        ParallelLayout(fsdp2 = False, tp = 2, dp = 1, world = 2),
        None,
        model,
        types.SimpleNamespace(packing = False),
    )


def test_a_non_latent_attention_owning_sinks_is_still_refused():
    model = _model()
    for module in model.modules():
        if "kv_proj" in module._modules:
            module._modules["kv_b_proj"] = torch.nn.Linear(
                2, 2, device = "meta"
            )  # MLA, not latent MQA
    assert caps.query_from(None, None, model).direct_params
