# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Sentence-transformer unpadding keeps the padded forward when a cache is passed under either name."""

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "unsloth", "models", "_sentence_transformer_unpadding.py"
)


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location("_u_st_unpadding", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("name", ["past_key_values", "cache"])
def test_cache_kwarg_keeps_the_padded_forward(monkeypatch, name):
    M = _load()

    def packed_route():
        raise AssertionError(f"{name}= reached the packed route")

    monkeypatch.setattr(M, "_unpadding_backend", packed_route)
    calls = []

    def original_forward(
        hidden_states,
        attention_mask = None,
        **kwargs,
    ):
        calls.append(kwargs)
        return "padded"

    encoder = SimpleNamespace(
        _unsloth_original_forward = original_forward,
        training = True,
        _unsloth_min_tokens = 0,
    )
    # A CPU tensor would bail on is_cuda first.
    hidden = SimpleNamespace(is_cuda = True, shape = torch.Size([2, 4, 8]), dtype = torch.bfloat16)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    cache = object()
    out = M._encoder_forward(encoder, hidden, attention_mask = mask, **{M._MASK: mask, name: cache})
    assert out == "padded"
    assert calls == [{name: cache}]
