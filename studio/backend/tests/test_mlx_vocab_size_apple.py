# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_mlx_vocab_size on real mlx-lm modules (random init, no download), on Apple Silicon only."""

from __future__ import annotations

import platform
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason = "MLX runs on Apple Silicon only",
)

_VOCAB = 320


def _tiny_llama():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    args = llama.ModelArgs(
        model_type = "llama",
        hidden_size = 64,
        num_hidden_layers = 1,
        intermediate_size = 128,
        num_attention_heads = 4,
        rms_norm_eps = 1e-5,
        vocab_size = _VOCAB,
        num_key_value_heads = 4,
    )
    return llama.Model(args)


def test_dense_quantized_and_lora_embeddings_report_the_vocab_size():
    import mlx.nn as nn
    from core.training.worker import _mlx_vocab_size

    model = _tiny_llama()
    assert _mlx_vocab_size(model) == _VOCAB

    # The packed weight has fewer columns, never fewer rows than the vocabulary.
    nn.quantize(model, group_size = 64, bits = 4)
    assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    assert _mlx_vocab_size(model) == _VOCAB

    from mlx_lm.tuner.lora import LoRAEmbedding

    model = _tiny_llama()
    model.model.embed_tokens = LoRAEmbedding.from_base(model.model.embed_tokens, r = 4)
    assert _mlx_vocab_size(model) == _VOCAB
