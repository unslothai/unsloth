# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A llama-type LongRoPE checkpoint (unsloth/Phi-3.5-mini-instruct, which
microsoft/Phi-3.5-mini-instruct maps to) must rotate like transformers' Phi3RotaryEmbedding.
LlamaModel builds one model-level rotary, so FastLlamaModel.from_pretrained has to install
LongRopeRotaryEmbedding there, and LlamaModel_fast_forward has to pick its factors from the real
length, not max_seq_length.

llama.py needs an accelerator to import, so the code is `ast`-extracted and run on CPU.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import math
import pathlib
import types

import pytest
import torch

LLAMA = pathlib.Path(__file__).resolve().parent.parent / "unsloth" / "models" / "llama.py"

DIM = 8
MAX_POSITION = 1 << 17
ORIGINAL_MAX_POSITION = 4096
MAX_SEQ_LENGTH = 8192
CONFIG = dict(
    hidden_size = DIM,
    num_attention_heads = 1,
    num_key_value_heads = 1,
    max_position_embeddings = MAX_POSITION,
    # Top level, as in Phi-3.5's config.json.
    original_max_position_embeddings = ORIGINAL_MAX_POSITION,
    rope_scaling = {
        "type": "longrope",
        "short_factor": [1.0, 1.25, 1.5, 2.0],
        "long_factor": [2.0, 4.0, 8.0, 16.0],
    },
)


class _CPUTorch:
    """`torch`, with a bare-int device pinned to CPU so the test runs anywhere."""

    def __getattr__(self, name):
        return getattr(torch, name)

    def device(
        self,
        spec,
        index = None,
    ):
        if isinstance(spec, int) and not isinstance(spec, bool):
            return torch.device("cpu", spec)
        return torch.device(spec) if index is None else torch.device(spec, index)

    def empty(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        return torch.empty(*args, **kwargs)


def _find(nodes, kind, name):
    node = next((n for n in nodes if isinstance(n, kind) and n.name == name), None)
    if node is None:
        pytest.fail(f"{name} is gone from llama.py; repoint this test rather than deleting it.")
    return node


def _llama():
    from transformers.utils import logging

    tree = ast.parse(LLAMA.read_text(encoding = "utf-8"))
    names = [
        "_get_rope_theta",
        "_rope_scaling_as_dict",
        "_compute_config_rope_inv_freq",
        "_round_rope_size",
        "_PerDeviceRopeCache",
        "LlamaRotaryEmbedding",
        "LongRopeRotaryEmbedding",
    ]
    ns = {
        "torch": _CPUTorch(),
        "math": math,
        "contextlib": contextlib,
        "logger": logging.get_logger(__name__),
        "DEVICE_COUNT": 1,
        "DEVICE_TYPE_TORCH": "cpu",
        "get_current_device": lambda: 0,
        "is_bfloat16_supported": lambda: False,
    }
    body = [_find(tree.body, (ast.FunctionDef, ast.ClassDef), name) for name in names]
    exec(compile(ast.Module(body = body, type_ignores = []), str(LLAMA), "exec"), ns)
    return tree, ns


def _block(fn, test):
    return compile(
        ast.Module(
            body = [
                next(n for n in fn.body if isinstance(n, ast.If) and test in ast.unparse(n.test))
            ],
            type_ignores = [],
        ),
        str(LLAMA),
        "exec",
    )


def _unsloth_cos_sin(
    positions,
    packed,
    cp = 1,
    config = CONFIG,
):
    from transformers import LlamaConfig

    tree, ns = _llama()
    config = LlamaConfig(num_hidden_layers = 1, **copy.deepcopy(config))
    # pre_patch puts Unsloth's LlamaRotaryEmbedding into transformers; LlamaModel builds this one.
    model = types.SimpleNamespace(
        config = config,
        model = types.SimpleNamespace(
            rotary_emb = ns["LlamaRotaryEmbedding"](config = config),
            layers = [types.SimpleNamespace(self_attn = types.SimpleNamespace())],
        ),
    )
    # FastLlamaModel.from_pretrained: the rotary handed to every attention layer.
    fast = _find(tree.body, ast.ClassDef, "FastLlamaModel")
    load = _block(_find(fast.body, ast.FunctionDef, "from_pretrained"), "model.model")
    exec(load, dict(ns, model = model, IS_ATTENTION_REFACTOR = True))
    rope = model.model.rotary_emb
    assert model.model.layers[0].self_attn.rotary_emb is rope

    # LlamaModel_fast_forward: the cos/sin every layer reuses. Under Ulysses CP this rank holds
    # the first 1 / cp of the tokens; the layers check rows against the global length.
    seq_len = positions.shape[0]
    local = positions[: seq_len // cp]
    manager = types.SimpleNamespace(size = cp, full_position_ids = positions[None] if packed else None)
    forward = _block(
        _find(ast.walk(tree), ast.FunctionDef, "LlamaModel_fast_forward"), "IS_ATTENTION_REFACTOR"
    )
    scope = dict(
        ns,
        torch = torch,
        self = types.SimpleNamespace(
            rotary_emb = rope,
            max_seq_length = MAX_SEQ_LENGTH,
            config = types.SimpleNamespace(max_position_embeddings = MAX_POSITION),
        ),
        hidden_states = types.SimpleNamespace(device = torch.device("cpu", 0), dtype = torch.float16),
        seq_length_with_past = local.shape[0],
        position_ids = local[None] if packed else None,
        get_cp_manager = lambda: manager if cp > 1 else None,
        IS_ATTENTION_REFACTOR = True,
        IS_GRANITE = False,
    )
    exec(forward, scope)
    cos, sin = scope["position_embeddings"]
    # The layers reuse these only when they hold a row per token; else each rebuilds its own.
    assert cos.shape[0] >= seq_len
    return cos[local].float(), sin[local].float()


@pytest.mark.parametrize(
    "positions, packed",
    [
        (torch.arange(4098), False),  # past 4096: transformers takes the long factors
        (torch.arange(2048), False),  # short, though max_seq_length is past 4096
        (torch.cat([torch.arange(3000), torch.arange(3000)]), True),  # packed rows all short
    ],
    ids = ["long", "short", "packed"],
)
@pytest.mark.parametrize("cp", [1, 2], ids = ["nocp", "ulysses2"])
def test_longrope_checkpoint_matches_transformers_phi3(positions, packed, cp):
    from transformers import Phi3Config
    from transformers.models.phi3.modeling_phi3 import Phi3RotaryEmbedding

    cos, sin = _unsloth_cos_sin(positions, packed, cp)
    ref_cos, ref_sin = Phi3RotaryEmbedding(Phi3Config(**copy.deepcopy(CONFIG)))(
        torch.zeros(1, positions.shape[0], DIM), positions[None]
    )
    n = cos.shape[0]
    # The caches are fp16 here (no bf16 on CPU), the reference fp32.
    torch.testing.assert_close(cos, ref_cos[0, :n], atol = 2e-3, rtol = 0)
    torch.testing.assert_close(sin, ref_sin[0, :n], atol = 2e-3, rtol = 0)


def test_longrope_honours_explicit_head_dim():
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

    # head_dim 8, not hidden_size // num_attention_heads = 16: four factors per table.
    config = dict(CONFIG, hidden_size = 32, num_attention_heads = 2, head_dim = DIM)
    positions = torch.arange(4097)
    cos, sin = _unsloth_cos_sin(positions, False, config = config)
    ref_cos, ref_sin = LlamaRotaryEmbedding(LlamaConfig(**copy.deepcopy(config)))(
        torch.zeros(1, positions.shape[0], DIM), positions[None]
    )
    torch.testing.assert_close(cos, ref_cos[0], atol = 2e-3, rtol = 0)
    torch.testing.assert_close(sin, ref_sin[0], atol = 2e-3, rtol = 0)
