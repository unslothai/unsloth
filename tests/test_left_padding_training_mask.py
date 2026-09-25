# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import (
    has_real_accelerator,
)  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.gpu


# Online DPO feeds left-padded prompts in training mode; dropping the mask let real tokens attend pads.
@pytest.mark.skipif(not has_real_accelerator(), reason = "runs Unsloth's training forward")
@pytest.mark.parametrize(
    "model_name",
    [
        "trl-internal-testing/tiny-LlamaForCausalLM-3.2",
        "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    ],
)
def test_left_padded_training_forward_matches_unpadded(model_name):
    import torch
    from unsloth import FastLanguageModel

    model, _tokenizer = FastLanguageModel.from_pretrained(
        model_name, max_seq_length = 64, dtype = torch.float32, load_in_4bit = False
    )
    model = FastLanguageModel.get_peft_model(
        model, r = 8, lora_alpha = 8, lora_dropout = 0, target_modules = ["q_proj", "v_proj"]
    )
    FastLanguageModel.for_training(model)
    device = model.get_input_embeddings().weight.device
    generator = torch.Generator().manual_seed(0)
    rows = [torch.randint(10, 2000, (n,), generator = generator) for n in (12, 7, 4)]
    width = max(len(r) for r in rows)

    for side in ("left", "right"):
        ids = torch.zeros((len(rows), width), dtype = torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            span = slice(width - len(row), width) if side == "left" else slice(0, len(row))
            ids[i, span] = row
            mask[i, span] = 1
        # No labels: the DPO / Online DPO scoring path.
        logits = model(input_ids = ids.to(device), attention_mask = mask.to(device)).logits
        for i, row in enumerate(rows):
            alone = model(input_ids = row[None].to(device)).logits[0]
            got = logits[i][mask[i].bool().to(device)]
            torch.testing.assert_close(got, alone, atol = 1e-4, rtol = 1e-3, msg = side)
