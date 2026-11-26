# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Trainers decide forced float32 from the model they train, never from UNSLOTH_FORCE_FLOAT32.

Every Unsloth load stamps `_unsloth_forced_float32`, but the flag stays at "1" after an fp16 Gemma 3 load, so a
model loaded later without Unsloth used to train with the forced float32 switches: no fp16 / bf16, and a forced
GradScaler. The stamp is looked up through the wrappers a trainer sees (PeftModel, SentenceTransformer, DDP)."""

import types

import pytest
import torch
from torch import nn

from unsloth.models._utils import _unsloth_forced_float32_grad_scaler


class _Transformer(nn.Module):
    """sentence_transformers.models.Transformer keeps the HF model at `auto_model`."""

    def __init__(self, inner):
        super().__init__()
        self.auto_model = inner


def _lora_like(stamp):
    model = nn.Module()
    model.base = nn.Linear(4, 4).to(torch.float16).requires_grad_(False)
    model.lora = nn.Linear(4, 4)
    if stamp is not None:
        model._unsloth_forced_float32 = stamp
    return model


@pytest.mark.parametrize("stamp, expected", [(None, False), (False, False), (True, True)])
def test_the_stamp_is_found_through_trainer_wrappers(stamp, expected, monkeypatch):
    from unsloth.models._utils import _model_forced_float32

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    inner = _lora_like(stamp)
    peft_like = nn.Module()
    peft_like.base_model = nn.Module()
    peft_like.base_model.model = inner
    sentence = nn.Sequential(_Transformer(peft_like), nn.Identity())
    ddp_like = nn.Module()
    ddp_like.module = sentence
    for wrapped in (inner, peft_like, sentence, ddp_like):
        assert _model_forced_float32(wrapped) is expected
    assert _model_forced_float32(None) is False


def test_a_real_peft_model_reads_the_base_stamp(monkeypatch):
    peft = pytest.importorskip("peft")
    from unsloth.models._utils import _model_forced_float32
    from transformers import LlamaConfig, LlamaForCausalLM

    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    config = LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    lora = peft.LoraConfig(r = 2, target_modules = ["q_proj"])
    plain = peft.get_peft_model(LlamaForCausalLM(config), lora)
    assert _model_forced_float32(plain) is False
    stamped = LlamaForCausalLM(config)
    stamped._unsloth_forced_float32 = True
    assert _model_forced_float32(peft.get_peft_model(stamped, lora)) is True


def _accelerator():
    from accelerate.utils import DistributedType
    return types.SimpleNamespace(
        scaler = None,
        mixed_precision = "no",
        distributed_type = DistributedType.NO,
        device = torch.device("cuda"),
        _optimizers = [],
    )


def test_an_unstamped_model_gets_no_forced_grad_scaler(monkeypatch):
    """fp16 frozen weights + fp32 trainable ones under UNSLOTH_FORCE_FLOAT32 = 1, but Unsloth never loaded it."""
    monkeypatch.setenv("UNSLOTH_FORCE_FLOAT32", "1")
    model = _lora_like(None)
    accelerator = _accelerator()
    trainer = types.SimpleNamespace(
        accelerator = accelerator, model = model, args = types.SimpleNamespace()
    )
    assert _unsloth_forced_float32_grad_scaler(trainer, model) is None
    assert accelerator.scaler is None
