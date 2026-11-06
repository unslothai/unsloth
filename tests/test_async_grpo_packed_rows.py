# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""AsyncGRPO packed rows on compiled / vision backbones: enabled only where a packed sequence cannot see the
previous one (CPU, tiny random transformers models)."""

from __future__ import annotations

import ast
import functools
import importlib
import os
import types
from importlib.metadata import version as importlib_version

import pytest
import torch
from packaging.version import Version

transformers = pytest.importorskip("transformers")

RL_SOURCE_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "rl.py")
NAMES = (
    "_packed_seq_lengths_from_position_ids",
    "_text_prepended_position_ids",
    "_packing_honored",
    "_unsloth_async_grpo_lm_head",
)


def _load():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {
        "torch": torch,
        "functools": functools,
        "importlib": importlib,
        "Version": Version,
        "importlib_version": importlib_version,
    }
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in NAMES:
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    ns["_is_unsloth_fast_backbone"] = lambda backbone: False
    return ns


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        attn_implementation = "sdpa",
    )
    return LlamaForCausalLM(config).eval()


def _tiny_qwen2_5_vl():
    try:
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
    except ImportError:
        pytest.skip("transformers without Qwen2.5-VL")
    torch.manual_seed(0)
    config = Qwen2_5_VLConfig(
        text_config = dict(
            vocab_size = 160,
            hidden_size = 64,
            intermediate_size = 128,
            num_hidden_layers = 2,
            num_attention_heads = 4,
            num_key_value_heads = 2,
            rope_scaling = {"type": "mrope", "mrope_section": [2, 3, 3]},
        ),
        vision_config = dict(
            depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64
        ),
        attn_implementation = "sdpa",
    )
    return Qwen2_5_VLForConditionalGeneration(config).eval()


class _Leaky(torch.nn.Module):
    # Every token sees every earlier token of the row, whatever the position_ids say.
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 16)
        self.config = types.SimpleNamespace(vocab_size = 128, get_text_config = lambda: self.config)

    def forward(
        self,
        input_ids,
        position_ids = None,
        use_cache = False,
    ):
        h = self.embed(input_ids).cumsum(1)
        return types.SimpleNamespace(last_hidden_state = h)


class _LeakyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Leaky()
        self.config = self.backbone.config

    @property
    def base_model(self):
        return self.backbone

    def forward(self, *a, **k):
        return k


def _fake_fused_head(model, **kwargs):
    original = model.forward

    def forward(
        *a,
        fused_lm_head = False,
        **k,
    ):
        return {"fused": fused_lm_head, **k} if fused_lm_head else original(*a, **k)

    model.forward = forward


def test_compiled_backbone_that_honors_packing_is_enabled_unchanged():
    ns = _load()
    model = _tiny_llama()
    assert ns["_packing_honored"](model.base_model)
    ns["_unsloth_async_grpo_lm_head"](_fake_fused_head, "fused_lm_head")(model)
    pos = torch.tensor([[0, 1, 2, 0, 1]])
    out = model.forward(position_ids = pos, fused_lm_head = True)
    assert out["position_ids"] is pos and "packed_seq_lengths" not in out


def test_mrope_vlm_gets_text_prepended_position_ids():
    ns = _load()
    model = _tiny_qwen2_5_vl()
    # Plain 2D ids reach no packed mask (sequences attend to each other); the text-prepended form does.
    assert not ns["_packing_honored"](model.base_model)
    assert ns["_packing_honored"](model.base_model, ns["_text_prepended_position_ids"])
    ns["_unsloth_async_grpo_lm_head"](_fake_fused_head, "fused_lm_head")(model)
    pos = torch.tensor([[0, 1, 2, 0, 1]])
    out = model.forward(position_ids = pos, fused_lm_head = True)
    assert out["position_ids"].shape == (4, 1, 5)
    assert torch.equal(out["position_ids"][0], pos)


def test_backbone_that_leaks_across_sequences_is_refused():
    ns = _load()
    with pytest.raises(ValueError, match = "attend to each other"):
        ns["_unsloth_async_grpo_lm_head"](_fake_fused_head, "fused_lm_head")(_LeakyLM())


def test_self_test_restores_training_mode():
    ns = _load()
    model = _tiny_llama().train()
    model.model.layers[0].eval()
    ns["_packing_honored"](model.base_model)
    assert model.training and model.model.training and not model.model.layers[0].training
