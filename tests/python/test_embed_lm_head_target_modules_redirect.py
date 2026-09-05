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
        tied = bool(getattr(model.config, "tie_word_embeddings", False))

        assert any("embed_tokens" in m for m in saved_modules), saved_modules

        assert "embed_tokens" not in config.target_modules, config.target_modules
        assert "lm_head" not in config.target_modules, config.target_modules

        assert any("layers.0.self_attn.q_proj.lora_A" in k for k in state)

        # embed_tokens/lm_head must be trainable via ModulesToSave, not LoRA
        assert any(
            "embed_tokens.modules_to_save.default.weight" in k and v.requires_grad
            for k, v in state.items()
        )
        if tied:
            # Tying keeps ONE trainable matrix instead of a second, divergent copy.
            assert "lm_head" not in saved_modules, saved_modules
            assert model.get_output_embeddings().weight.requires_grad
        else:
            assert "lm_head" in saved_modules, saved_modules
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
    ],
)
def test_embed_lm_head_target_redirect(new_model_path: bool, target_modules):
    """Both model paths train embed_tokens/lm_head via modules_to_save, not LoRA."""
    _run_redirect_check(new_model_path, target_modules)


@pytest.mark.slow
def test_a_repeat_call_with_the_same_targets_still_passes_through():
    """A tied lm_head lands in modules_to_tie, which the equality check has to count."""
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        kwargs = dict(r = 8, lora_alpha = 16, target_modules = list(TARGET_MODULES))
        model = FastLanguageModel.get_peft_model(model, **kwargs)
        assert getattr(
            model.peft_config["default"], "modules_to_tie", None
        ), "tied model did not redirect lm_head; this guard would check nothing"
        model = FastLanguageModel.get_peft_model(model, **kwargs)
        # Same configuration, written the other way round: the embeddings named directly in modules_to_save rather than
        # reached through the redirect.
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            target_modules = [m for m in TARGET_MODULES if m not in ("embed_tokens", "lm_head")],
            modules_to_save = ["embed_tokens", "lm_head"],
        )
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
@pytest.mark.parametrize(
    "first",
    [
        TARGET_MODULES,
        [m for m in TARGET_MODULES if m != "embed_tokens"],
        [m for m in TARGET_MODULES if m != "lm_head"],
    ],
)
def test_dropping_the_embeddings_from_a_repeat_call_is_not_silently_ignored(first):
    """The embeddings live in modules_to_save/modules_to_tie, so a narrowed request must
    still be seen as different rather than returning the existing adapter."""
    from unsloth import FastLanguageModel

    core = [m for m in TARGET_MODULES if m not in ("embed_tokens", "lm_head")]
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            target_modules = list(first),
        )
        with pytest.raises(TypeError, match = "parameters are different"):
            FastLanguageModel.get_peft_model(
                model,
                r = 8,
                lora_alpha = 16,
                target_modules = core,
            )
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
def test_qualified_targets_are_not_collapsed_to_their_leaf():
    """Only PEFT's model.embed_tokens alias is folded away. layers.0.q_proj is a real
    request and is not the same module as layers.1.q_proj."""
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 8,
            lora_alpha = 16,
            target_modules = ["layers.0.self_attn.q_proj"],
        )
        for different in (["layers.1.self_attn.q_proj"], ["q_proj"]):
            with pytest.raises(TypeError, match = "parameters are different"):
                FastLanguageModel.get_peft_model(
                    model,
                    r = 8,
                    lora_alpha = 16,
                    target_modules = different,
                )
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
def test_flipping_ensure_weight_tying_is_seen_as_a_different_request():
    """Tying is not in check_parameters and leaves both name lists identical, so it has
    to be compared on its own or the caller's request is silently ignored."""
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        kwargs = dict(r = 8, lora_alpha = 16, target_modules = list(TARGET_MODULES))
        model = FastLanguageModel.get_peft_model(model, **kwargs)
        assert getattr(
            model.peft_config["default"], "modules_to_tie", None
        ), "model is not tied here; this guard would check nothing"
        with pytest.raises(TypeError, match = "parameters are different"):
            FastLanguageModel.get_peft_model(model, ensure_weight_tying = False, **kwargs)
        FastLanguageModel.get_peft_model(model, ensure_weight_tying = True, **kwargs)
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
def test_a_classification_repeat_call_is_not_tripped_by_peft_added_modules():
    """PeftModelForSequenceClassification.__init__ extends modules_to_save with
    classifier/score unconditionally, so the stored config names modules the caller never
    asked for. An unchanged request must still pass through."""
    from unsloth import FastLanguageModel

    core = [m for m in TARGET_MODULES if m not in ("embed_tokens", "lm_head")]
    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
        num_labels = 2,
    )
    try:
        model = FastLanguageModel.get_peft_model(
            model, r = 8, lora_alpha = 16, target_modules = list(core)
        )
        saved = model.peft_config["default"].modules_to_save or []
        assert (
            "score" in saved or "classifier" in saved
        ), f"PEFT did not add its classifier modules ({saved}); this guard checks nothing"
        FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = list(core))
        with pytest.raises(TypeError, match = "parameters are different"):
            FastLanguageModel.get_peft_model(model, r = 8, lora_alpha = 16, target_modules = ["q_proj"])
    finally:
        del model
        torch.cuda.empty_cache()


@pytest.mark.slow
def test_embedding_only_target_list_raises_instead_of_training_nothing():
    """Redirecting every target would leave an adapter with no trainable LoRA."""
    from unsloth import FastLanguageModel

    model, _ = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        load_in_4bit = True,
        max_seq_length = 512,
    )
    try:
        with pytest.raises(RuntimeError, match = "target_modules` is now empty"):
            FastLanguageModel.get_peft_model(
                model,
                r = 8,
                lora_alpha = 16,
                target_modules = ["embed_tokens", "lm_head"],
            )
    finally:
        del model
        torch.cuda.empty_cache()
