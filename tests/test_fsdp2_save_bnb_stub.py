# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Without bitsandbytes, unsloth_zoo's stub makes `from bitsandbytes.nn import Linear4bit` an instance, so the
FSDP2 save helpers' isinstance checks raised TypeError and broke every full DCP state dict."""

import sys
import types

import torch

from unsloth.models import fsdp2_save


def test_save_helpers_ignore_a_stubbed_linear4bit(monkeypatch):
    stub = types.ModuleType("bitsandbytes.nn")
    stub.Linear4bit = object()  # what the zoo stub returns: not a class
    monkeypatch.setitem(sys.modules, "bitsandbytes.nn", stub)
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    model._unsloth_fsdp2_native = True
    assert fsdp2_save._linear4bit_class() is None
    assert fsdp2_save._has_sharded_4bit((model,), {}) is False
    assert fsdp2_save._has_native_replicated_4bit((model,), {}) is False
    assert fsdp2_save._linear4bit_entries(model) == []
