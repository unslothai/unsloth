# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression test for unslothai/unsloth#9326.

When `embed_tokens` and/or `lm_head` are listed in `target_modules`, Unsloth's
fast regex path would silently drop them because they do not sit under an
attention/MLP ancestor.  The fix auto-moves them to `modules_to_save` and
warns once, which also makes `embedding_learning_rate` work.
"""

import os
import pytest


MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "embed_tokens",
    "lm_head",
]


def _run_check(new_model_path: bool):
    """Core assertions shared by both FastLanguageModel paths."""
    import torch
    from unsloth import FastLanguageModel

    if new_model_path:
        os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
    else:
        os.environ.pop("UNSLOTH_USE_NEW_MODEL", None)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            target_modules = list(TARGET_MODULES),
        )

        config = model.peft_config["default"]
        saved_modules = config.modules_to_save or []
        assert "embed_tokens" in saved_modules, saved_modules
        assert "lm_head" in saved_modules, saved_modules

        # target_modules should no longer carry the embedding modules
        assert "embed_tokens" not in config.target_modules, config.target_modules
        assert "lm_head" not in config.target_modules, config.target_modules

        # Attention/MLP LoRA adapters should still be present
        state = dict(model.named_parameters())
        assert any("layers.0.self_attn.q_proj.lora_A" in k for k in state)

        # embed_tokens/lm_head must be trainable via ModulesToSave, not LoRA
        assert any(
            "embed_tokens.modules_to_save.default.weight" in k and v.requires_grad
            for k, v in state.items()
        )
        assert any(
            "lm_head.modules_to_save.default.weight" in k and v.requires_grad
            for k, v in state.items()
        )
        assert not any("embed_tokens.lora_A" in k for k in state)
        assert not any("lm_head.lora_A" in k for k in state)
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.parametrize("new_model_path", [False, True])
def test_embed_lm_head_moved_to_modules_to_save(new_model_path: bool):
    """Both the legacy and new-model paths preserve embed/lm_head trainability."""
    _run_check(new_model_path)
