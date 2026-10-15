# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Which FSDP2 saves gather (unsloth/save.py `_fsdp2_collective_save`). No GPU work."""

import types

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")

from unsloth import save as unsloth_save  # noqa: E402
from unsloth.models import fsdp2_save  # noqa: E402


def _merged(
    self,
    save_directory,
    tokenizer = None,
    save_method = "merged_16bit",
    push_to_hub = False,
    token = None,
    is_main_process = True,
    state_dict = None,
):
    return ("wrote", state_dict)


class _Model:
    def original_model_save_pretrained(
        self,
        save_directory,
        is_main_process = True,
        state_dict = None,
        **kwargs,
    ):
        return ("wrote", state_dict)


def _forwarding(self, *args, **kwargs):
    return self.original_model_save_pretrained(*args, **kwargs)


@pytest.fixture
def no_gather(monkeypatch):
    monkeypatch.setattr(fsdp2_save, "fsdp2_units", lambda model: [object()])
    monkeypatch.setattr(fsdp2_save, "fsdp2_gather_active", lambda: False)

    def _refuse(*args, **kwargs):
        raise AssertionError("a caller's own state_dict must not start a collective gather")

    monkeypatch.setattr(fsdp2_save, "fsdp2_gathered", _refuse)


def test_a_positional_state_dict_is_passed_through(no_gather):
    sd = {"w": torch.zeros(1)}
    merged = unsloth_save._fsdp2_collective_save(lambda *a: False)(_merged)
    # (save_directory, tokenizer, save_method, push_to_hub, token, is_main_process, state_dict)
    assert merged(_Model(), "out", None, "merged_16bit", False, None, True, sd) == ("wrote", sd)
    assert merged(_Model(), "out", state_dict = sd) == ("wrote", sd)
    forwarding = unsloth_save._fsdp2_collective_save(lambda *a: False)(_forwarding)
    # PreTrainedModel.save_pretrained(save_directory, is_main_process, state_dict, ...)
    assert forwarding(_Model(), "out", True, sd) == ("wrote", sd)


def test_peft_saves_gather_the_frozen_tensors_peft_writes():
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        vocab_size = 64,
    )
    model = get_peft_model(LlamaForCausalLM(cfg), LoraConfig(r = 2, target_modules = ["q_proj"]))
    model.add_adapter("other", LoraConfig(r = 2, target_modules = ["q_proj"]))
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {})
    names = {n for n, p in model.named_parameters() if not p.requires_grad and include(n, p)}
    assert any(".other." in n for n in names), names
    assert any("embed_tokens" in n for n in names) and any("lm_head" in n for n in names), names
    assert not any("q_proj.base_layer" in n for n in names), names
    include = unsloth_save._fsdp2_peft_saved_frozen(model, {"save_embedding_layers": False})
    names = {n for n, p in model.named_parameters() if include(n, p)}
    assert not any("embed_tokens" in n for n in names), names
    model.original_model_save_pretrained = model.save_pretrained
    arguments = unsloth_save._fsdp2_save_arguments(
        _forwarding, model, ("out", True, None, False), {}
    )
    assert arguments["save_embedding_layers"] is False
