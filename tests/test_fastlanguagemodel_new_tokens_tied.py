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
    kwargs.setdefault("use_gradient_checkpointing", False)
    kwargs.setdefault("target_modules", ["q_proj", "v_proj"])

    def capture(model, lora_config, *args, **kwargs):
        raise _Captured(lora_config)

    monkeypatch.setattr(llama_module, "_get_peft_model", capture)
    with pytest.raises(_Captured) as info:
        FastLlamaModel.get_peft_model(
            model,
            r = 4,
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


@pytest.mark.parametrize("tie, offloads_head", [(True, False), (False, True)])
def test_new_tokens_offload_head_only_when_trained_apart(monkeypatch, tie, offloads_head):
    # A tied head is rebuilt from the embedding's copy, so writing it to disk is wasted work.
    offloaded = []
    # Short sequences fall back to plain checkpointing, which never offloads.
    monkeypatch.setattr(llama_module, "apply_unsloth_gradient_checkpointing", lambda gc, *a: gc)
    monkeypatch.setattr(
        llama_module, "offload_input_embeddings", lambda *a, **k: offloaded.append("input")
    )
    monkeypatch.setattr(
        llama_module, "offload_output_embeddings", lambda *a, **k: offloaded.append("output")
    )
    _lora_config_for(_tiny_llama(tie = tie), monkeypatch, use_gradient_checkpointing = "unsloth")
    assert ("output" in offloaded) is offloads_head
    assert "input" in offloaded


def _base_lora_config_for(model, monkeypatch, **kwargs):
    import unsloth.models.vision as vision_module
    from unsloth.models.vision import FastBaseModel

    def capture(model, lora_config, *args, **kwargs):
        raise _Captured(lora_config)

    kwargs.setdefault("target_modules", ["q_proj", "v_proj"])
    monkeypatch.setattr(vision_module, "_get_peft_model", capture)
    with pytest.raises(_Captured) as info:
        FastBaseModel.get_peft_model(
            model,
            r = 4,
            use_gradient_checkpointing = False,
            **kwargs,
        )
    return info.value.args[0]


NEW_TOKEN_IDS = [60, 61, 62, 63]


def _rows_changed_by_one_step(model):
    import torch

    def rows():
        with torch.no_grad():
            return (
                model.get_input_embeddings()(torch.arange(64)).clone(),
                model.get_output_embeddings()(torch.eye(16)).T.clone(),
            )

    before = rows()
    ids = torch.tensor([[1] + NEW_TOKEN_IDS + [2]])
    model(input_ids = ids, labels = ids).loss.backward()
    torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr = 1.0).step()
    return [
        (after - old).abs().sum(-1).nonzero().flatten().tolist()
        for old, after in zip(before, rows())
    ]


@pytest.mark.parametrize("entry", [_lora_config_for, _base_lora_config_for])
@pytest.mark.parametrize(
    "tie, trainable_token_indices, input_rows, output_rows",
    [
        (False, NEW_TOKEN_IDS, NEW_TOKEN_IDS, NEW_TOKEN_IDS),
        (False, {"embed_tokens": NEW_TOKEN_IDS}, NEW_TOKEN_IDS, []),
        (False, {"model.embed_tokens": NEW_TOKEN_IDS}, NEW_TOKEN_IDS, []),
        (False, {"lm_head": NEW_TOKEN_IDS}, [], NEW_TOKEN_IDS),
        (
            False,
            {"embed_tokens": NEW_TOKEN_IDS, "lm_head": NEW_TOKEN_IDS},
            NEW_TOKEN_IDS,
            NEW_TOKEN_IDS,
        ),
        (True, NEW_TOKEN_IDS, NEW_TOKEN_IDS, NEW_TOKEN_IDS),
        (True, {"embed_tokens": NEW_TOKEN_IDS}, NEW_TOKEN_IDS, NEW_TOKEN_IDS),
    ],
)
def test_new_tokens_train_only_trainable_token_indices(
    monkeypatch, entry, tie, trainable_token_indices, input_rows, output_rows
):
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    config = entry(
        _tiny_llama(tie = tie),
        monkeypatch,
        trainable_token_indices = trainable_token_indices,
    )
    assert not config.modules_to_save
    if isinstance(trainable_token_indices, dict):
        assert config.trainable_token_indices == trainable_token_indices
    model = get_peft_model(_tiny_llama(tie = tie), config)
    assert not any(type(m).__name__ == "ModulesToSaveWrapper" for m in model.modules())
    assert _rows_changed_by_one_step(model) == [input_rows, output_rows]


@pytest.mark.parametrize("entry", [_lora_config_for, _base_lora_config_for])
def test_new_tokens_list_indices_keep_fully_trained_head(monkeypatch, entry):
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    config = entry(
        _tiny_llama(tie = False),
        monkeypatch,
        modules_to_save = ["lm_head"],
        trainable_token_indices = NEW_TOKEN_IDS,
    )
    assert _leaves(config.modules_to_save) == {"lm_head"}
    assert config.trainable_token_indices == NEW_TOKEN_IDS
    model = get_peft_model(_tiny_llama(tie = False), config)
    input_rows, output_rows = _rows_changed_by_one_step(model)
    assert input_rows == NEW_TOKEN_IDS
    assert len(output_rows) == 64


@pytest.mark.parametrize("entry", [_lora_config_for, _base_lora_config_for])
def test_new_tokens_list_indices_skip_lora_head(monkeypatch, entry):
    import torch

    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "0")
    monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "0")
    monkeypatch.setattr(torch.cpu, "empty_cache", lambda: None, raising = False)
    config = entry(
        _tiny_llama(tie = False),
        monkeypatch,
        target_modules = r".*(q_proj|lm_head)",
        trainable_token_indices = NEW_TOKEN_IDS,
    )
    assert config.trainable_token_indices == NEW_TOKEN_IDS
    get_peft_model(_tiny_llama(tie = False), config)
