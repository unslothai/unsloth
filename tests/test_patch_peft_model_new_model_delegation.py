# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

import unsloth  # noqa: F401
from unsloth.models.llama import FastLlamaModel
from unsloth.models.vision import FastBaseModel


def _spy_post_patch_model(monkeypatch):
    calls = []
    monkeypatch.setattr(
        FastBaseModel,
        "post_patch_model",
        staticmethod(lambda model, **kwargs: calls.append((model, kwargs)) or model),
    )
    return calls


def test_patch_peft_model_delegates_when_new_model_flag_set(monkeypatch):
    # Exercise the environment fallback on a model without loader metadata.
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "1")
    calls = _spy_post_patch_model(monkeypatch)
    model = SimpleNamespace()
    assert FastLlamaModel.patch_peft_model(model) is model
    assert calls == [(model, {"use_gradient_checkpointing": "unsloth"})]


def test_patch_peft_model_delegates_seq2seq_configs(monkeypatch):
    from transformers import T5Config

    calls = _spy_post_patch_model(monkeypatch)
    model = SimpleNamespace(config = T5Config())
    assert FastLlamaModel.patch_peft_model(model) is model
    assert calls == [(model, {"use_gradient_checkpointing": "unsloth"})]
