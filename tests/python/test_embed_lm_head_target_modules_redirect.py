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
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "embedding target redirect integration test needs a CUDA GPU",
)


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


def _run_redirect_check(new_model_path: bool, target_modules):
    """Core assertions shared by both FastLanguageModel paths."""
    from unsloth import FastLanguageModel

    previous_new_model = os.environ.pop("UNSLOTH_USE_NEW_MODEL", None)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        if new_model_path:
            os.environ["UNSLOTH_USE_NEW_MODEL"] = "1"
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            target_modules = list(target_modules),
        )

        config = model.peft_config["default"]
        saved_modules = config.modules_to_save or []
        state = dict(model.named_parameters())
        if target_modules == ["lm_head"]:
            assert not saved_modules, saved_modules
            assert "lm_head" in config.target_modules, config.target_modules
            assert any("lm_head.lora_A" in k and v.requires_grad for k, v in state.items())
            return

        assert "embed_tokens" in saved_modules, saved_modules
        assert "lm_head" in saved_modules, saved_modules

        # target_modules should no longer carry the embedding modules
        assert "embed_tokens" not in config.target_modules, config.target_modules
        assert "lm_head" not in config.target_modules, config.target_modules

        # Attention/MLP LoRA adapters should still be present
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
        if previous_new_model is None:
            os.environ.pop("UNSLOTH_USE_NEW_MODEL", None)
        else:
            os.environ["UNSLOTH_USE_NEW_MODEL"] = previous_new_model


@pytest.mark.slow
@pytest.mark.parametrize(
    ("new_model_path", "target_modules"),
    [
        pytest.param(False, TARGET_MODULES, id = "legacy-redirect"),
        pytest.param(True, TARGET_MODULES, id = "new-model-redirect"),
        pytest.param(True, ["lm_head"], id = "new-model-lm-head-only"),
    ],
)
def test_embed_lm_head_target_redirect(new_model_path: bool, target_modules):
    """Both model paths redirect embeddings without dropping the only LoRA target."""
    _run_redirect_check(new_model_path, target_modules)
