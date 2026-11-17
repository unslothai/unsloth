# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Training forwards keep their padding mask when a pad precedes a real token (left or interior
padding), with or without labels (ms-swift rerankers, #3705; DataCollatorForSeq2Seq left padding)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

LLAMA_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"


def _lift(name):
    tree = ast.parse(LLAMA_PY.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(LLAMA_PY), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.bool, torch.float32])
def test_has_padding_before_tokens(dtype):
    has_padding_before_tokens = _lift("_has_padding_before_tokens")

    def check(rows):
        return has_padding_before_tokens(torch.tensor(rows).to(dtype))

    assert check([[1, 1, 1], [1, 1, 1]]) is False
    assert check([[1, 1, 0], [1, 0, 0]]) is False
    assert check([[0, 0, 0], [1, 1, 1]]) is False
    assert check([[0, 1, 1], [1, 1, 1]]) is True
    assert check([[1, 1, 1], [0, 0, 1]]) is True
    assert check([[1, 0, 1]]) is True
    assert check([[1, 1, 1, 1], [1, 0, 0, 1]]) is True
    assert check([[1]]) is False
    assert has_padding_before_tokens(torch.ones(2, 1, 3, 3)) is False


TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.mark.skipif(not has_real_cuda(), reason = "Unsloth fast forward needs a GPU")
def test_left_padded_label_less_training_forward_matches_unpadded():
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        TINY,
        max_seq_length = 64,
        dtype = torch.bfloat16,
        load_in_4bit = False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = ["q_proj", "v_proj"],
        use_gradient_checkpointing = "unsloth",
    )
    model.train()
    torch.manual_seed(0)
    vocab = model.config.vocab_size
    long_row = torch.randint(3, vocab, (1, 12), device = "cuda")
    short_row = torch.randint(3, vocab, (1, 7), device = "cuda")
    pad = 5
    input_ids = torch.cat(
        [long_row, torch.cat([torch.full((1, pad), 0, device = "cuda"), short_row], 1)]
    )
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :pad] = 0

    padded = model(input_ids = input_ids, attention_mask = attention_mask).logits
    alone = model(input_ids = short_row, attention_mask = torch.ones_like(short_row)).logits
    # Measured on a B200: 3.7e-4 with the mask kept, 5.5e-2 when the pads are attended (logits ~4e-2).
    torch.testing.assert_close(padded[1, pad:].float(), alone[0].float(), atol = 5e-3, rtol = 0)

    model.zero_grad()
    score = padded[1, -1, :2]
    (score[0] - score[1]).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _lora_model(dtype):
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        TINY,
        max_seq_length = 64,
        dtype = dtype,
        load_in_4bit = False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        lora_dropout = 0.0,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = False,
        random_state = 0,
    )
    generator = torch.Generator().manual_seed(1)
    for name, param in sorted(model.named_parameters()):
        if "lora_" in name:
            with torch.no_grad():
                param.copy_((torch.randn(param.shape, generator = generator) * 0.05).to(param.dtype))
    model.train()
    return model


@pytest.mark.skipif(not has_real_cuda(), reason = "Unsloth fast forward needs a GPU")
@pytest.mark.parametrize("padding", ["left", "interior"])
# 13: the compiled masked SDPA backward faulted on torch < 2.15 without the aligned call.
@pytest.mark.parametrize("length", [13, 16])
def test_padded_labelled_lora_training_matches_unpadded_rows(padding, length):
    model = _lora_model(torch.float32)
    params = [p for p in model.parameters() if p.requires_grad]
    torch.manual_seed(0)
    vocab = model.config.vocab_size
    short = length - 7
    input_ids = torch.randint(3, vocab, (2, length), device = "cuda")
    attention_mask = torch.ones_like(input_ids)
    if padding == "left":
        attention_mask[1, : length - short] = 0
    else:
        attention_mask[1, 4 : 4 + length - short] = 0
    input_ids = input_ids.masked_fill(attention_mask == 0, 0)
    # Interior gaps shift arange positions, so the padded arm gets positions that skip the pads.
    position_ids = None
    if padding == "interior":
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min = 0)
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    labels[:, 1:].masked_fill_(attention_mask[:, :-1] == 0, -100)

    def loss_and_grads(**kwargs):
        model.zero_grad(set_to_none = True)
        loss = model(**kwargs).loss
        loss.backward()
        return loss.detach(), torch.cat([p.grad.flatten() for p in params])

    # Only the padded row is trained in the batch, so its loss and grads must match it run alone.
    row_labels = labels.clone()
    row_labels[0] = -100
    padded_loss, padded_grads = loss_and_grads(
        input_ids = input_ids,
        attention_mask = attention_mask,
        labels = row_labels,
        position_ids = position_ids,
    )
    keep = attention_mask[1].bool()
    alone_ids = input_ids[1, keep][None]
    alone_loss, alone_grads = loss_and_grads(
        input_ids = alone_ids,
        attention_mask = torch.ones_like(alone_ids),
        labels = labels[1, keep][None],
    )
    torch.testing.assert_close(padded_loss, alone_loss, atol = 1e-5, rtol = 1e-5)
    relative = (padded_grads - alone_grads).norm() / alone_grads.norm()
    assert relative < 1e-3, float(relative)
