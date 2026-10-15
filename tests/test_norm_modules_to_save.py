# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Norm layers in modules_to_save on the FastLanguageModel path (#3235).

The fast RMSNorm kernel computes dW for a trainable weight (#13594), so a norm named in
modules_to_save trains there like lm_head / embed_tokens; anything else still raises.
"""

import pytest
from real_accelerator import (
    has_real_cuda,
)  # tests/_shared, on sys.path via tests/conftest.py

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
    )
    return transformers.LlamaForCausalLM(config)


@pytest.mark.parametrize(
    "name, allowed",
    [
        ("input_layernorm", True),
        ("post_attention_layernorm", True),
        ("model.norm", True),
        # PEFT matches on a plain suffix, so "norm" also wraps every per-layer norm: all are norms.
        ("norm", True),
        ("layernorm", True),
        ("mlp", False),
        ("q_proj", False),
        ("self_attn", False),
        ("not_a_module", False),
        ("", False),
        (None, False),
    ],
)
def test_norm_modules_to_save_matches_peft_suffix_rule(name, allowed):
    from unsloth.models.llama import _norm_modules_to_save
    assert _norm_modules_to_save(_tiny_llama(), name) is allowed


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required")
def test_fast_path_trains_norm_in_modules_to_save(tmp_path):
    from unsloth import FastLanguageModel

    # head_dim 64: the fast attention kernels reject the hub tiny Llama's head_dim of 2.
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2"
    )
    config = transformers.LlamaConfig(
        vocab_size = len(tokenizer),
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)
    tokenizer.save_pretrained(tmp_path)
    model, _ = FastLanguageModel.from_pretrained(
        str(tmp_path),
        max_seq_length = 64,
        load_in_4bit = False,
        dtype = torch.bfloat16,
    )
    with pytest.raises(TypeError, match = "is not allowed"):
        FastLanguageModel.get_peft_model(
            model, r = 4, target_modules = ["q_proj"], modules_to_save = ["mlp"]
        )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 4,
        target_modules = ["q_proj", "v_proj"],
        modules_to_save = ["input_layernorm"],
        use_gradient_checkpointing = "unsloth",
    )
    copies = {n: p for n, p in model.named_parameters() if "input_layernorm.modules_to_save" in n}
    assert copies and all(p.requires_grad for p in copies.values())
    before = {n: p.detach().clone() for n, p in copies.items()}
    ids = torch.randint(10, 100, (2, 16), device = "cuda")
    model.train()
    model(input_ids = ids, labels = ids).loss.backward()
    for n, p in copies.items():
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    torch.optim.SGD(copies.values(), lr = 1.0).step()
    assert all(not torch.equal(before[n], p.detach()) for n, p in copies.items())
