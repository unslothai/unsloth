# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen3.5 / Qwen3.6 (qwen3_5, qwen3_5_moe) greedy generate() with one-token steps on Unsloth's flex decode equals
SDPA decode: gated full-attention layers (output gate, partial rotary, q / k norm) between Gated DeltaNet layers,
GQA, the MoE block, batch 1 and a left-padded ragged batch. Same float64 stand-in as test_flex_decode_gemma4 (zoo's
flex decode replaced by eager flex_attention over the same mask semantics), so any difference is a routing or mask
error, not rounding."""

import pytest
import torch

transformers = pytest.importorskip("transformers")
pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")

import unsloth.models._flex_decode as FD
from unsloth.models.vision import _NonFlexGenerate

from test_flex_decode_gemma4 import _generate, _inputs, flex  # noqa: F401  (fixture)


def _qwen(moe):
    from transformers import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeTextConfig,
        Qwen3_5TextConfig,
    )

    torch.manual_seed(0)
    common = dict(
        vocab_size = 128,
        hidden_size = 64,
        num_hidden_layers = 4,
        layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads = 8,
        num_key_value_heads = 2,
        head_dim = 32,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        max_position_embeddings = 1024,
        pad_token_id = 0,
        eos_token_id = 1,
        bos_token_id = 2,
        attn_implementation = "sdpa",
    )
    if moe:
        config = Qwen3_5MoeTextConfig(
            moe_intermediate_size = 32,
            shared_expert_intermediate_size = 32,
            num_experts = 8,
            num_experts_per_tok = 2,
            **common,
        )
        model = Qwen3_5MoeForCausalLM(config)
    else:
        config = Qwen3_5TextConfig(intermediate_size = 64, **common)
        model = Qwen3_5ForCausalLM(config)
    model = model.eval().double()
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(4.0)
    return model


@pytest.mark.parametrize("zoo_mask", [False, True], ids = ["stand_in_mask", "zoo_mask"])
@pytest.mark.parametrize("moe", [False, True], ids = ["qwen3_5", "qwen3_5_moe"])
@pytest.mark.parametrize("batch", [1, 4])
def test_qwen35_flex_decode_matches_sdpa(flex, monkeypatch, moe, batch, zoo_mask):
    # Regression: a mask keeping a reference to the live counter saw one slot too many.
    if zoo_mask:
        zoo = pytest.importorskip("unsloth_zoo.flex_attention.decode")
        monkeypatch.setattr(FD, "decode_block_mask", zoo.decode_block_mask)
    model = _qwen(moe)
    assert FD._model_type(model.config) in FD.MODEL_TYPES
    ids, mask = _inputs(48, batch)
    with torch.no_grad():
        ref_seq, ref_logits = _generate(model, ids, mask, 24)
        FD.reset_stats(model)
        with _NonFlexGenerate(model):
            seq, logits = _generate(model, ids, mask, 24)
    assert model.config._attn_implementation == "sdpa"
    stats = FD.stats(model)
    assert len(stats) == 1 and all(
        s["flex"] == 23 and not s["fallback"] for s in stats.values()
    ), stats
    assert torch.equal(seq, ref_seq)
    assert (logits - ref_logits).abs().max().item() < 1e-9
