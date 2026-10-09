# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Left-padded, label-less training forwards keep their padding mask (#3705).

LlamaModel_fast_forward drops the 2D mask in training, which is exact only for right padding.
ms-swift's generative reranker left-pads, pops its per-sequence labels and reads the last token,
so every real token attended to zeroed pad embeddings and gradients went NaN.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

LLAMA_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "llama.py"


def _lift(name):
    tree = ast.parse(LLAMA_PY.read_text(encoding = "utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(LLAMA_PY), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("dtype", [torch.long, torch.int32, torch.bool, torch.float32])
def test_has_pad_before_token(dtype):
    has_pad_before_token = _lift("_has_pad_before_token")

    def check(rows):
        return has_pad_before_token(torch.tensor(rows).to(dtype))

    assert check([[1, 1, 1], [1, 1, 1]]) is False
    assert check([[1, 1, 0], [1, 0, 0]]) is False
    assert check([[0, 1, 1], [1, 1, 1]]) is True
    assert check([[1, 1, 1], [0, 0, 1]]) is True
    assert check([[1, 0, 1]]) is True
    assert check([[1]]) is False
    assert has_pad_before_token(torch.ones(2, 1, 3, 3)) is False


TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "Unsloth fast forward needs a GPU")
def test_left_padded_label_less_training_forward_matches_unpadded():
    os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")
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
