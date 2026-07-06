"""llama3 RoPE scaling must survive the transformers v5 rope repair.

transformers v5 blanks non-persistent buffers on load, so
loader._fix_rope_inv_freq rebuilds inv_freq. It must rebuild the config
scaled (llama3) inv_freq, not vanilla: recomputing unscaled divided inv_freq
by 1 instead of the config factor (8 / 32) and inflated long-context loss.
Pure CPU, no network, no GPU."""

from __future__ import annotations

import inspect

import pytest
import torch
from transformers import LlamaConfig

import unsloth.models.loader as loader
from unsloth.models.llama import LlamaRotaryEmbedding, _get_rope_theta


def _config(factor):
    rope_scaling = None if factor is None else {
        "rope_type": "llama3",
        "factor": float(factor),
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
    }
    return LlamaConfig(
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        num_hidden_layers=2,
        max_position_embeddings=131072,
        rope_theta=500000.0,
        rope_scaling=rope_scaling,
    )


def _recompute_ratio(config):
    # Bare instance so we exercise _unsloth_recompute_inv_freq without the CUDA cos/sin build.
    module = object.__new__(LlamaRotaryEmbedding)
    module.attention_scaling = 1.0
    module.base = _get_rope_theta(config, 10000.0)
    module.dim = config.head_dim
    module._unsloth_rope_config = config
    inv_freq = module._unsloth_recompute_inv_freq()
    vanilla = 1.0 / (module.base ** (torch.arange(0, module.dim, 2, dtype=torch.int64).float() / module.dim))
    return float(vanilla[-1]) / float(inv_freq.reshape(-1)[-1])


@pytest.mark.parametrize("factor", [8.0, 32.0, None])
def test_recompute_keeps_config_scaling(factor):
    # Lowest inv_freq is divided by the config factor; unscaled (the bug) gives 1.0.
    expected = 1.0 if factor is None else factor
    assert _recompute_ratio(_config(factor)) == pytest.approx(expected, abs=1e-3)


def test_v5_repair_routes_through_recompute():
    # Guard the loader wiring so a future edit cannot rebuild unscaled inv_freq again.
    assert "_unsloth_recompute_inv_freq" in inspect.getsource(loader._fix_rope_inv_freq)
