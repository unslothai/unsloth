# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What TurboQuant does to a real cache, on the installed mlx-vlm rather than the fake-mlx shim."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mx = pytest.importorskip("mlx.core", reason = "needs a real MLX runtime")
pytest.importorskip("mlx_vlm.turboquant", reason = "needs an mlx-vlm with TurboQuant")

import mlx.nn as nn  # noqa: E402

from core.inference.mlx_inference import (  # noqa: E402
    MLX_KV_GROUP_SIZE,
    _copy_value,
    _turboquant_status,
)


def _llama(layer_types = None, sliding = None):
    from mlx_vlm.models.llama.config import ModelConfig
    from mlx_vlm.models.llama.language import Model

    model = Model(
        ModelConfig(
            model_type = "llama",
            hidden_size = 64,
            num_hidden_layers = 4,
            intermediate_size = 128,
            num_attention_heads = 4,
            num_key_value_heads = 4,
            rms_norm_eps = 1e-5,
            vocab_size = 128,
            layer_types = layer_types,
            sliding_window = sliding,
        )
    )
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("bits", [2, 3, 3.5, 4])
def test_every_offered_width_converts_and_decodes(bits):
    status = _turboquant_status(_llama(), bits)
    assert (status["eligibility"], status["kv_bits"], status["note"]) == ("full", bits, "")


def test_a_model_holding_layers_native_reports_partial_and_says_so():
    sliding = _llama(layer_types = ["sliding_attention"] + ["full_attention"] * 3, sliding = 64)
    status = _turboquant_status(sliding, 4)
    assert (status["eligibility"], status["kv_bits"]) == ("partial", 4) and status["note"]


def test_a_model_that_cannot_decode_is_refused_rather_than_raising():
    class FailsOnDecode(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0

        @property
        def layers(self):
            return self.inner.layers

        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("no decode here")
            return self.inner(*args, **kwargs)

    model = FailsOnDecode(_llama())
    status = _turboquant_status(model, 4)
    assert model.calls == 2
    assert status["eligibility"] == "refused" and status["kv_bits"] is None
    assert "RuntimeError" in status["reason"]


def test_a_copied_turboquant_cache_can_still_be_extended():
    from mlx_vlm.generate.common import maybe_quantize_kv_cache
    from mlx_vlm.models.cache import make_prompt_cache
    from mlx_vlm.turboquant import TurboQuantKVCache

    model = _llama()
    entries = make_prompt_cache(model)
    maybe_quantize_kv_cache(entries, 0, MLX_KV_GROUP_SIZE, 4, kv_quant_scheme = "turboquant")
    result = model(mx.array([[1, 2, 3, 4]]), cache = entries)
    mx.eval(getattr(result, "logits", result), [entry.state for entry in entries])

    at = next(index for index, e in enumerate(entries) if isinstance(e, TurboQuantKVCache))
    original = entries[at].state
    copied = _copy_value(entries, mx)

    assert [type(half) for half in copied[at].state] == [type(half) for half in original]

    served = entries[at].offset
    follow_on = model(mx.array([[5]]), cache = entries)
    mx.eval(getattr(follow_on, "logits", follow_on), [entry.state for entry in entries])
    assert copied[at].offset == served < entries[at].offset

    grown = model(mx.array([[6]]), cache = copied)
    mx.eval(getattr(grown, "logits", grown), [entry.state for entry in copied])
    assert copied[at].offset == served + 1
