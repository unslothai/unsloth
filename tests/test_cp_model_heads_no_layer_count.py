# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A config without num_hidden_layers (remote-code ChatGLM keeps num_layers) still gets its head split: on
transformers >= 5.15 every config carries a per-layer view whose len() reads num_hidden_layers."""

import types

from transformers import PretrainedConfig

import unsloth  # noqa: F401
from unsloth import context_parallel as cp


class _NumLayersConfig(PretrainedConfig):
    model_type = "unsloth_test_num_layers"

    def __init__(
        self,
        num_layers = 4,
        num_attention_heads = 8,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        super().__init__(**kwargs)


def test_config_without_num_hidden_layers_keeps_its_head_split():
    model = types.SimpleNamespace(config = _NumLayersConfig())
    assert cp._model_heads(model) == [(8, 8)]
    assert cp.resolve_cp_backend("auto", 2, model) == "ulysses"
    assert cp.resolve_cp_backend("auto", 3, model) == "ring"
