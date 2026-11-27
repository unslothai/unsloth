# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Attention gets hidden_states * attention_in_multiplier, Mamba the plain norm (transformers FalconH1DecoderLayer)."""

from __future__ import annotations

import ast
import types
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "falcon_h1.py"
FUNCTION = "FalconH1DecoderLayer_fast_forward"
HELPERS = (FUNCTION, "_attention_input")


def _rms_norm(norm, x, *args, **kwargs):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim = True) + norm.variance_epsilon)
    return (x32 * norm.weight).to(x.dtype)


def _load_layer_forward():
    text = SOURCE_PATH.read_text(encoding = "utf-8")
    tree = ast.parse(text, filename = str(SOURCE_PATH))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert len(nodes) == len(HELPERS)
    namespace = {
        "torch": torch,
        "Optional": Optional,
        "Tuple": Tuple,
        "fast_rms_layernorm": _rms_norm,
        "fast_rms_layernorm_inference": _rms_norm,
        "fast_swiglu_inference": lambda mlp, x: mlp(x),
    }
    module = ast.Module(body = nodes, type_ignores = [])
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return namespace[FUNCTION]


@pytest.mark.parametrize("multiplier", [0.25, 1.0])
@pytest.mark.parametrize("generation", [False, True], ids = ["training", "generation_flag"])
def test_attention_input_is_scaled_by_attention_in_multiplier(generation, multiplier):
    torch.manual_seed(0)
    hidden = 8
    seen = {}

    def mamba(hidden_states, **kwargs):
        seen["mamba"] = hidden_states.clone()
        return hidden_states * 0.5

    def self_attn(hidden_states, **kwargs):
        seen["attn"] = hidden_states.clone()
        return hidden_states + 1.0, None, None

    layer = types.SimpleNamespace(
        input_layernorm = types.SimpleNamespace(
            weight = torch.rand(hidden) + 0.5, variance_epsilon = 1e-5
        ),
        pre_ff_layernorm = types.SimpleNamespace(
            weight = torch.rand(hidden) + 0.5, variance_epsilon = 1e-5
        ),
        mamba = mamba,
        self_attn = self_attn,
        feed_forward = lambda x: x * 2.0,
        attention_in_multiplier = multiplier,
        attn_out_multiplier = 0.9,
        ssm_out_multiplier = 0.4,
    )
    if generation:
        layer._flag_for_generation = True

    x = torch.randn(1, 3, hidden)
    forward = _load_layer_forward()
    out = forward(layer, x.clone(), use_cache = generation)[0]

    normed = _rms_norm(layer.input_layernorm, x)
    torch.testing.assert_close(seen["mamba"], normed)
    torch.testing.assert_close(seen["attn"], normed * multiplier)

    mixed = (normed * 0.5) * 0.4 + (normed * multiplier + 1.0) * 0.9
    stream = x + mixed
    expected = stream + _rms_norm(layer.pre_ff_layernorm, stream) * 2.0
    torch.testing.assert_close(out, expected)
