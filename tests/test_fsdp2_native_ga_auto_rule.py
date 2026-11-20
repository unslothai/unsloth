# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FSDP2_GA_SYNC=auto: fit -> defer, sharded full fine-tune -> each, sharded LoRA -> skip."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402


def _model(trainable_fraction, fits):
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 64), nn.Linear(64, 4))
    if trainable_fraction < 0.5:  # LoRA-like: only the small head trains
        for p in model[:3].parameters():
            p.requires_grad_(False)
    model._unsloth_fsdp2_layout = {"fits": fits}
    return model


@pytest.fixture
def native(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FSDP2_GA_SYNC", raising = False)
    return _load_native()


def test_auto_sharded_full_finetune_reduces_every_micro_batch(native):
    assert native._ga_mode(_model(1.0, fits = False)) == "each"


def test_auto_sharded_lora_keeps_skip(native):
    assert native._ga_mode(_model(0.0, fits = False)) == "skip"


def test_auto_fit_layout_defers(native):
    assert native._ga_mode(_model(1.0, fits = True)) == "defer"
    assert native._ga_mode(_model(0.0, fits = True)) == "defer"


def test_auto_without_model_or_layout(native):
    assert native._ga_mode(None) == "skip"
    for fraction in (0.0, 1.0):
        model = _model(fraction, fits = False)
        del model._unsloth_fsdp2_layout
        assert native._ga_mode(model) == "skip"


@pytest.mark.parametrize("mode", ["skip", "defer", "each"])
def test_explicit_mode_wins(native, monkeypatch, mode):
    monkeypatch.setenv("UNSLOTH_FSDP2_GA_SYNC", mode)
    assert native._ga_mode(_model(1.0, fits = False)) == mode
    assert native._ga_mode(_model(0.0, fits = True)) == mode


def test_frozen_quantized_storage_counts_as_frozen(native):
    model = _model(0.0, fits = False)
    model.extra = nn.Parameter(torch.zeros(4096, dtype = torch.uint8), requires_grad = False)
    assert native._ga_mode(model) == "skip"
