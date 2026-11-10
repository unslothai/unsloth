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
if "linear_attention" not in getattr(
    pytest.importorskip("transformers.masking_utils"),
    "LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING",
    {"linear_attention": 1},
):
    # Early 5.x: generate() builds no mask for linear-attention layers, so a Qwen3.5 static-cache call cannot run.
    pytest.skip(
        "transformers without linear_attention masks for generate()", allow_module_level = True
    )

import unsloth.models._flex_decode as FD
from unsloth.models.vision import _NonFlexGenerate

from tests.test_flex_decode_gemma4 import _generate, _inputs, flex  # noqa: F401  (fixture)


def _torch_linear_attention(monkeypatch):
    # The CPU float64 stand-in must not reach FLA / causal-conv1d Triton kernels where those packages are installed
    # (GPU hosts): transformers picks them by import, not by device, and they reject CPU tensors.
    import importlib
    for name in ("qwen3_5", "qwen3_5_moe"):
        module = importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        for fn in (
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
            "causal_conv1d_fn",
            "causal_conv1d_update",
        ):
            original = getattr(getattr(module, fn, None), "__wrapped__", None)
            if original is not None:
                monkeypatch.setattr(module, fn, original)


def _native_torch_moe(monkeypatch):
    # The float64 stand-in needs the per-expert loop: a CUDA host selects grouped_mm (half / fp32 only) process-wide.
    import sys
    monkeypatch.setenv("UNSLOTH_MOE_BACKEND", "native_torch")
    for module in list(sys.modules.values()):
        if "_SELECTED_MOE_BACKEND" in getattr(module, "__dict__", {}):
            monkeypatch.setattr(module, "_SELECTED_MOE_BACKEND", None)


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
    _torch_linear_attention(monkeypatch)
    _native_torch_moe(monkeypatch)
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
