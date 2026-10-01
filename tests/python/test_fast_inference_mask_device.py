# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The fast decode loops call self_attn directly, so the attention mask must follow each layer's device.

With layers split across GPUs (device_map="auto" on 2 x T4), batched generation died in SDPA with
"attn_bias is on cuda:0, different from other tensors on cuda:1".
"""

from __future__ import annotations

import importlib

import pytest
from real_accelerator import has_real_cuda
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not has_real_cuda(), reason = "fast inference needs a CUDA device"),
]

PROMPTS = [
    "Hello",
    "The capital of France is a city called",
    "One two three four five six seven eight",
]


def _load(model_id, device_map):
    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        model_id, max_seq_length = 128, load_in_4bit = False, dtype = torch.float16, device_map = device_map
    )
    FastLanguageModel.for_inference(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _generate(model, tok):
    batch = tok(PROMPTS, return_tensors = "pt", padding = True).to("cuda:0")
    with torch.no_grad():
        out = model.generate(
            **batch, max_new_tokens = 8, do_sample = False, pad_token_id = tok.pad_token_id
        )
    return out[:, batch["input_ids"].shape[1] :].tolist()


@pytest.mark.parametrize(
    "model_id",
    [
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5-Coder",
    ],
)
def test_decode_loop_moves_the_mask_to_every_layer(monkeypatch, model_id):
    mod = importlib.import_module("unsloth.models.llama")
    real = mod.move_to_device
    mask_moves = []

    def spy(device, *tensors):
        mask_moves.extend(t for t in tensors if t.dtype == torch.bool and t.dim() == 4)
        return real(device, *tensors)

    monkeypatch.setattr(mod, "move_to_device", spy)
    model, tok = _load(model_id, {"": 0})
    _generate(model, tok)
    # 7 decode steps after prefill, each moving the mask once per layer.
    assert len(mask_moves) >= 7 * model.config.num_hidden_layers


@pytest.mark.skipif(
    not has_real_cuda() or torch.cuda.device_count() < 2,
    reason = "needs two GPUs to split the layers",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "trl-internal-testing/tiny-Qwen3ForCausalLM",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5-Coder",
    ],
)
def test_split_layers_generate_like_one_gpu(model_id):
    model, tok = _load(model_id, {"": 0})
    single = _generate(model, tok)
    del model
    n = (
        importlib.import_module("transformers")
        .AutoConfig.from_pretrained(model_id)
        .num_hidden_layers
    )
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    device_map.update({f"model.layers.{i}": 0 if i < max(1, n // 2) else 1 for i in range(n)})
    model, tok = _load(model_id, device_map)
    assert {str(next(layer.parameters()).device) for layer in model.model.layers} == {
        "cuda:0",
        "cuda:1",
    }
    assert _generate(model, tok) == single
