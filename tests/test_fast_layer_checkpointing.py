# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""pre_patch's fast decoder-layer forwards keep transformers' layer gradient checkpointing.

transformers <= 5.22 checkpoints in GradientCheckpointingLayer.__call__; >= 5.23 in a wrapper that
__init_subclass__ puts on the class forward and that only checkpoints while it is type(self).forward.
Replacing the class forward with a fast forward (and the gate around it) dropped checkpointing.
"""

import inspect
import sys
import types

import pytest
import torch


def _family(monkeypatch, modeling_layers):
    modeling = types.ModuleType("transformers.models.fakegc.modeling_fakegc")
    patcher_module = types.ModuleType("unsloth.models.fakegc")

    class FakeGCDecoderLayer(modeling_layers.GradientCheckpointingLayer):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)

        def forward(
            self,
            hidden_states,
            attention_mask = None,
            use_cache = False,
            past_key_values = None,
        ):
            return self.proj(hidden_states) + 1

    FakeGCDecoderLayer.__module__ = modeling.__name__
    modeling.FakeGCDecoderLayer = FakeGCDecoderLayer
    patcher = type("FastFakeGCModel", (), {"__module__": patcher_module.__name__})
    patcher_module.FakeGCDecoderLayer, patcher_module.FastFakeGCModel = FakeGCDecoderLayer, patcher
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    monkeypatch.setitem(sys.modules, patcher_module.__name__, patcher_module)
    return FakeGCDecoderLayer, patcher


def _layer(cls, calls, marked):
    layer = cls().train()
    layer.gradient_checkpointing = True

    def checkpoint(fn, *args):
        calls.append((len(args), dict(fn.keywords)))
        return fn(*args)

    layer._gradient_checkpointing_func = checkpoint
    if marked:
        layer._unsloth_fast_path = True
    return layer


def test_fast_decoder_layer_forward_keeps_layer_checkpointing(monkeypatch):
    modeling_layers = pytest.importorskip("transformers.modeling_layers")
    from unsloth.models import llama as unsloth_llama

    cls, patcher = _family(monkeypatch, modeling_layers)
    stock = cls.forward
    seen = []

    def fast_forward(
        self,
        hidden_states,
        attention_mask = None,
        use_cache = False,
        past_key_values = None,
    ):
        seen.append((use_cache, past_key_values))
        return self.proj(hidden_states) * 2

    for _ in range(2):  # a second load re-patches; it must not wrap twice
        snapshot = unsloth_llama._snapshot_transformers_modules(patcher)
        cls.forward = fast_forward
        unsloth_llama._record_pre_patch_changes(snapshot)
        unsloth_llama._gate_pre_patch_to_unsloth_models(snapshot)

    x = torch.randn(1, 3, 4, requires_grad = True)
    for marked in (True, False):
        calls = []
        layer = _layer(cls, calls, marked)
        out = layer(x, attention_mask = None, use_cache = True, past_key_values = object())
        expected = layer.proj(x) * 2 if marked else layer.proj(x) + 1
        torch.testing.assert_close(out, expected)
        # One checkpoint per layer call, with caching turned off as transformers does.
        assert [n for n, _ in calls] == [1], (marked, calls)
        assert calls[0][1]["use_cache"] is False and calls[0][1]["past_key_values"] is None
        layer.eval()(x)
        assert len(calls) == 1
    assert seen[0] == (False, None)

    marked = _layer(cls, [], True)
    assert unsloth_llama.running_forward(marked) is fast_forward
    assert unsloth_llama.running_forward(_layer(cls, [], False)) is None
    assert list(inspect.signature(cls.forward).parameters) == [
        "self",
        "hidden_states",
        "attention_mask",
        "use_cache",
        "past_key_values",
    ]

    unsloth_llama.restore_transformers_family(["fakegc"])
    assert vars(cls)["forward"] is stock
    calls = []
    restored = _layer(cls, calls, True)
    torch.testing.assert_close(restored(x), restored.proj(x) + 1)
    assert [n for n, _ in calls] == [1]
