"""Regression guard for RoPE scaling being silently dropped (issue #2405).

Llama-3.1 (and any model with `config.rope_scaling`) ships an `inv_freq`
rescaling, e.g. `{"rope_type": "llama3", "factor": 8.0, ...}`. Unsloth replaces
transformers' rotary classes with its own (`unsloth/models/llama.py`), but the
*config* constructor path of the base `LlamaRotaryEmbedding` only reads
`rope_theta`, `head_dim` and `max_position_embeddings`. It never inspects
`config.rope_scaling`, so the scaled `LlamaExtendedRotaryEmbedding` is only used
when the (legacy) `LlamaAttention.__init__` rewrite in `patch_llama_rope_scaling`
fires. On modern transformers (rotary moved to `LlamaModel`) that rewrite no
longer matches, the unscaled base class wins, and long-context inference
(> ~32k tokens) collapses into repeated-pattern gibberish (issue #2405).

Two layers, both CPU-only and deterministic:

  1. AST structural tripwire (no unsloth import): parse llama.py and assert the
     config path of `LlamaRotaryEmbedding.__init__` references `rope_scaling`.
  2. Behavioral checks: instantiate the real `LlamaRotaryEmbedding(config=...)`
     and assert its `inv_freq` matches transformers'
     `ROPE_INIT_FUNCTIONS[rope_type]` (scaled for llama3, vanilla for None).

The behavioral layer FAILS on current unfixed main (inv_freq is unscaled for a
llama3 config). Pattern mirrors tests/utils/test_prepare_inputs_leftpad.py.
"""

import ast
import math
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = REPO_ROOT / "unsloth" / "models" / "llama.py"

CLASS_NAME = "LlamaRotaryEmbedding"

# A Llama-3.1-style rope_scaling block (matches the hardcoded grid-search
# constants in LlamaExtendedRotaryEmbedding._apply_inv_freq_scaling).
LLAMA3_ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
ROPE_THETA = 500000.0
HEAD_DIM = 128
MAX_POS = 131072


# --------------------------------------------------------------------------
# Layer 1: AST structural tripwire (stdlib only, no unsloth import)
# --------------------------------------------------------------------------


def _load_class_init():
    tree = ast.parse(LLAMA_PY.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == CLASS_NAME:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "__init__":
                    return sub
    raise AssertionError(
        f"{CLASS_NAME}.__init__ not found in {LLAMA_PY}; if it was renamed or "
        "moved, update this guard so RoPE scaling stays protected (issue #2405)"
    )


def _config_branch(init_fn):
    """The `if config is not None:` block at the top of __init__."""
    for node in init_fn.body:
        if isinstance(node, ast.If):
            test = node.test
            # match `config is not None`
            is_config_test = (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "config"
            )
            if is_config_test:
                return node
    return None


def test_config_path_inspects_rope_scaling():
    init_fn = _load_class_init()
    branch = _config_branch(init_fn)
    assert branch is not None, (
        f"{CLASS_NAME}.__init__ no longer has an `if config is not None:` "
        "branch; the config constructor path must read config.rope_scaling so "
        "scaled models (llama3/linear/longrope) are not silently unscaled "
        "(issue #2405)"
    )

    names = set()
    for stmt in branch.body:
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Attribute):
                names.add(sub.attr)
            elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                names.add(sub.value)
    assert "rope_scaling" in names, (
        f"{CLASS_NAME}.__init__ config path does not reference `rope_scaling`. "
        "When a rotary class is built straight from a config (the path modern "
        "transformers takes, since rotary moved to LlamaModel), the llama3 / "
        "linear / longrope scaling must still be applied; otherwise long inputs "
        "produce repeated-pattern gibberish (issue #2405)."
    )


# --------------------------------------------------------------------------
# Layer 2: behavioral guard (instantiates the real class, lazy unsloth import)
# --------------------------------------------------------------------------


def _make_config(rope_scaling):
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=HEAD_DIM,
        rope_theta=ROPE_THETA,
        max_position_embeddings=MAX_POS,
        rope_scaling=rope_scaling,
    )


def _unsloth_rotary(config):
    from unsloth.models import llama as llama_mod

    return llama_mod.LlamaRotaryEmbedding(config=config)


def _reference_inv_freq(config, rope_type):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    inv_freq, _attention_factor = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    return inv_freq.float().cpu()


def _vanilla_inv_freq():
    return 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.int64).float() / HEAD_DIM)
    )


def test_llama3_scaling_applied_to_inv_freq():
    config = _make_config(LLAMA3_ROPE_SCALING)
    rot = _unsloth_rotary(config)

    got = rot.inv_freq.float().cpu()
    expected = _reference_inv_freq(config, "llama3")
    vanilla = _vanilla_inv_freq()

    # Sanity: the scaled reference really does differ from vanilla, else the
    # test would be vacuous.
    assert not torch.allclose(expected, vanilla, rtol=1e-4), (
        "test setup error: llama3-scaled inv_freq should differ from vanilla"
    )
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-6), (
        "LlamaRotaryEmbedding built from a llama3 config produced inv_freq that "
        "does not match transformers' llama3 RoPE scaling. The config path is "
        "ignoring config.rope_scaling, so long-context inference degrades into "
        "repeated-pattern gibberish (issue #2405).\n"
        f"got[:6]={got[:6].tolist()}\nexpected[:6]={expected[:6].tolist()}"
    )


def test_unscaled_config_uses_vanilla_inv_freq():
    config = _make_config(None)
    rot = _unsloth_rotary(config)

    got = rot.inv_freq.float().cpu()
    vanilla = _vanilla_inv_freq()
    assert torch.allclose(got, vanilla, rtol=1e-4, atol=1e-6), (
        "LlamaRotaryEmbedding with no rope_scaling must use the vanilla "
        f"inv_freq; got[:6]={got[:6].tolist()} vanilla[:6]={vanilla[:6].tolist()}"
    )


def _cos_at_position(rot, position):
    """cos row for a single position, computed from the class's own inv_freq.

    Uses the same construction as _set_cos_sin_cache so we read what the model
    actually applies, but stays on CPU and independent of device caches.
    """
    inv_freq = rot.inv_freq.float().cpu()
    t = torch.tensor([position], dtype=torch.float32)
    t = rot._apply_time_scaling(t.clone()) if hasattr(rot, "_apply_time_scaling") else t
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().squeeze(0)


def test_cos_cache_differs_between_scaled_and_unscaled_at_long_position():
    scaled = _unsloth_rotary(_make_config(LLAMA3_ROPE_SCALING))
    unscaled = _unsloth_rotary(_make_config(None))

    pos = 10000
    cos_scaled = _cos_at_position(scaled, pos)
    cos_unscaled = _cos_at_position(unscaled, pos)
    assert not torch.allclose(cos_scaled, cos_unscaled, rtol=1e-4, atol=1e-5), (
        f"cos values at position {pos} are identical for a llama3-scaled and an "
        "unscaled rotary embedding, which means scaling was dropped (issue "
        "#2405). With correct llama3 scaling the low-frequency bands shrink by "
        "up to 8x and must change the angles at long positions."
    )


def test_extended_cache_keeps_scaling_after_growth():
    scaled = _unsloth_rotary(_make_config(LLAMA3_ROPE_SCALING))
    # Grow the rope cache past its initial size (mirrors long-context decode).
    dummy = torch.zeros(1, dtype=torch.float32)
    scaled.extend_rope_embedding(dummy, seq_len=40960)

    config = _make_config(LLAMA3_ROPE_SCALING)
    expected = _reference_inv_freq(config, "llama3")
    got = scaled.inv_freq.float().cpu()
    assert torch.allclose(got, expected, rtol=1e-4, atol=1e-6), (
        "growing the RoPE cache (extend_rope_embedding) must preserve llama3 "
        "scaling of inv_freq; long-context decode loses scaling otherwise "
        "(issue #2405)."
    )
