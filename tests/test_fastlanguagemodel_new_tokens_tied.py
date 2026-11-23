# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""add_new_tokens on a tied model: FastLanguageModel trains one embedding, as FastModel does."""

import pytest

import unsloth  # noqa: F401
import unsloth.models.llama as llama_module
from peft import get_peft_model
from unsloth.models.llama import FastLlamaModel


def _tiny_llama(tie):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        tie_word_embeddings = tie,
    )
    model = LlamaForCausalLM(config)
    model.max_seq_length = 64
    model._saved_temp_tokenizer = None
    # Set by add_new_tokens.
    model._need_to_train_embeddings = True
    return model


class _Captured(Exception):
    pass


def _lora_config_for(model, monkeypatch, **kwargs):
    def capture(model, lora_config, *args, **kwargs):
        raise _Captured(lora_config)

    monkeypatch.setattr(llama_module, "_get_peft_model", capture)
    with pytest.raises(_Captured) as info:
        FastLlamaModel.get_peft_model(
            model,
            r = 4,
            target_modules = ["q_proj", "v_proj"],
            use_gradient_checkpointing = False,
            **kwargs,
        )
    return info.value.args[0]


def _leaves(names):
    return {name.rsplit(".", 1)[-1] for name in names or ()}


# Tuples: the add_new_tokens branch appends to a caller's list in place.
@pytest.mark.parametrize(
    "modules_to_save, ensure_weight_tying",
    [(None, None), (("embed_tokens",), None), (("lm_head",), None), (None, True)],
)
def test_new_tokens_train_one_tied_embedding(monkeypatch, modules_to_save, ensure_weight_tying):
    # Two trainable copies drift apart, and a merged export keeps only embed_tokens.
    config = _lora_config_for(
        _tiny_llama(tie = True),
        monkeypatch,
        modules_to_save = modules_to_save,
        ensure_weight_tying = ensure_weight_tying,
    )
    assert config.ensure_weight_tying is True
    assert _leaves(config.modules_to_save) == {"embed_tokens"}
    model = get_peft_model(_tiny_llama(tie = True), config)
    trained_input = model.get_input_embeddings().modules_to_save.default.weight
    trained_output = model.get_output_embeddings().modules_to_save.default.weight
    assert trained_output is trained_input


@pytest.mark.parametrize("tie, ensure_weight_tying", [(False, None), (True, False)])
def test_new_tokens_keep_two_copies_when_untied_or_opted_out(monkeypatch, tie, ensure_weight_tying):
    config = _lora_config_for(
        _tiny_llama(tie = tie),
        monkeypatch,
        ensure_weight_tying = ensure_weight_tying,
    )
    assert config.ensure_weight_tying is False
    assert _leaves(config.modules_to_save) == {"embed_tokens", "lm_head"}
