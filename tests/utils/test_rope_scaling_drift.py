"""Guard against config.rope_scaling being silently dropped (issue #2405):
the replacement rotary classes ignored it on the config path, so Llama-3.1
ran with unscaled RoPE and produced gibberish past ~32K tokens.

Three layers: (1) AST tripwire; (2) CPU checks of the pure helper
_compute_config_rope_inv_freq vs ROPE_INIT_FUNCTIONS; (3) CUDA checks on the
real class (skipped without a real device). Layers 2-3 fail on the unfixed code.
"""

import ast
import math
from pathlib import Path

import pytest
import torch


def _has_real_cuda():
    try:
        torch.zeros(1).to("cuda")
        return True
    except Exception:
        return False


HAS_REAL_CUDA = _has_real_cuda()
requires_cuda = pytest.mark.skipif(
    not HAS_REAL_CUDA,
    reason = "LlamaRotaryEmbedding builds per-device CUDA caches in __init__",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LLAMA_PY = REPO_ROOT / "unsloth" / "models" / "llama.py"

CLASS_NAME = "LlamaRotaryEmbedding"

# Llama-3.1-style rope_scaling.
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


# --- Layer 1: AST structural tripwire (stdlib only, no unsloth import) ---


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

    called = {
        sub.func.id
        for stmt in branch.body
        for sub in ast.walk(stmt)
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
    }
    assert "_compute_config_rope_inv_freq" in called, (
        f"{CLASS_NAME}.__init__ config path no longer calls "
        "_compute_config_rope_inv_freq; the CPU behavioral tests below cover "
        "that helper directly, so the constructor must stay wired to it or "
        "scaled configs silently lose RoPE scaling again (issue #2405)."
    )


# --- Layer 2: CPU behavioral guard (pure helper, no instantiation) ---


def _make_config(rope_scaling):
    from transformers import LlamaConfig
    return LlamaConfig(
        hidden_size = 256,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        head_dim = HEAD_DIM,
        rope_theta = ROPE_THETA,
        max_position_embeddings = MAX_POS,
        rope_scaling = rope_scaling,
    )


def _unsloth_rotary(config):
    from unsloth.models import llama as llama_mod
    return llama_mod.LlamaRotaryEmbedding(config = config)


def _reference_inv_freq(config, rope_type):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    inv_freq, _attention_factor = ROPE_INIT_FUNCTIONS[rope_type](config, "cpu")
    return inv_freq.float().cpu()


def _vanilla_inv_freq():
    return 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype = torch.int64).float() / HEAD_DIM)
    )


def _compute_helper(config, rope_scaling):
    from unsloth.models.llama import _compute_config_rope_inv_freq
    return _compute_config_rope_inv_freq(config, rope_scaling)


def test_llama3_scaling_applied_to_inv_freq():
    config = _make_config(LLAMA3_ROPE_SCALING)
    got, attention_scaling = _compute_helper(config, config.rope_scaling)
    expected = _reference_inv_freq(config, "llama3")
    vanilla = _vanilla_inv_freq()

    # Guard against a vacuous test: scaled inv_freq must differ from vanilla.
    assert not torch.allclose(
        expected, vanilla, rtol = 1e-4
    ), "test setup error: llama3-scaled inv_freq should differ from vanilla"
    assert got is not None, (
        "_compute_config_rope_inv_freq returned None for a llama3 config; the "
        "config path is dropping config.rope_scaling, so long-context inference "
        "degrades into repeated-pattern gibberish (issue #2405)."
    )
    got = got.float().cpu()
    assert torch.allclose(got, expected, rtol = 1e-4, atol = 1e-6), (
        "inv_freq for a llama3 config does not match transformers' llama3 RoPE "
        "scaling (issue #2405).\n"
        f"got[:6]={got[:6].tolist()}\nexpected[:6]={expected[:6].tolist()}"
    )


def test_default_rope_type_matches_vanilla_inv_freq():
    config = _make_config(None)
    got, attention_scaling = _compute_helper(config, {"rope_type": "default"})
    assert got is not None
    vanilla = _vanilla_inv_freq()
    assert torch.allclose(got.float().cpu(), vanilla, rtol = 1e-4, atol = 1e-6), (
        "default rope_type must equal the vanilla inv_freq; "
        f"got[:6]={got[:6].tolist()} vanilla[:6]={vanilla[:6].tolist()}"
    )


def _cos_at_position(rot, position):
    """cos row at one position, built like _set_cos_sin_cache but CPU-only."""
    inv_freq = rot.inv_freq.float().cpu()
    t = torch.tensor([position], dtype = torch.float32)
    t = rot._apply_time_scaling(t.clone()) if hasattr(rot, "_apply_time_scaling") else t
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim = -1)
    return emb.cos().squeeze(0)


# --- Layer 3: CUDA behavioral guard (real instantiation needs a device) ---


@requires_cuda
def test_constructor_applies_llama3_scaling():
    config = _make_config(LLAMA3_ROPE_SCALING)
    rot = _unsloth_rotary(config)
    got = rot.inv_freq.float().cpu()
    expected = _reference_inv_freq(config, "llama3")
    assert torch.allclose(
        got, expected, rtol = 1e-4, atol = 1e-6
    ), "LlamaRotaryEmbedding built from a llama3 config produced unscaled inv_freq (issue #2405)."


@requires_cuda
def test_constructor_unscaled_config_uses_vanilla_inv_freq():
    rot = _unsloth_rotary(_make_config(None))
    got = rot.inv_freq.float().cpu()
    vanilla = _vanilla_inv_freq()
    assert torch.allclose(
        got, vanilla, rtol = 1e-4, atol = 1e-6
    ), "LlamaRotaryEmbedding with no rope_scaling must use the vanilla inv_freq"


@requires_cuda
def test_cos_cache_differs_between_scaled_and_unscaled_at_long_position():
    scaled = _unsloth_rotary(_make_config(LLAMA3_ROPE_SCALING))
    unscaled = _unsloth_rotary(_make_config(None))

    pos = 10000
    cos_scaled = _cos_at_position(scaled, pos)
    cos_unscaled = _cos_at_position(unscaled, pos)
    assert not torch.allclose(cos_scaled, cos_unscaled, rtol = 1e-4, atol = 1e-5), (
        f"cos values at position {pos} are identical for a llama3-scaled and an "
        "unscaled rotary embedding, which means scaling was dropped (issue "
        "#2405). With correct llama3 scaling the low-frequency bands shrink by "
        "up to 8x and must change the angles at long positions."
    )


@requires_cuda
def test_extended_cache_keeps_scaling_after_growth():
    scaled = _unsloth_rotary(_make_config(LLAMA3_ROPE_SCALING))
    # Grow past the initial cache size (mirrors long-context decode).
    dummy = torch.zeros(1, dtype = torch.float32)
    scaled.extend_rope_embedding(dummy, seq_len = 40960)

    config = _make_config(LLAMA3_ROPE_SCALING)
    expected = _reference_inv_freq(config, "llama3")
    got = scaled.inv_freq.float().cpu()
    assert torch.allclose(got, expected, rtol = 1e-4, atol = 1e-6), (
        "growing the RoPE cache (extend_rope_embedding) must preserve llama3 "
        "scaling of inv_freq; long-context decode loses scaling otherwise "
        "(issue #2405)."
    )


def test_object_style_rope_scaling_does_not_crash():
    # Object-style rope_scaling must be normalized, not .get()'d directly.
    from dataclasses import dataclass

    from unsloth.models.llama import _compute_config_rope_inv_freq

    @dataclass
    class FakeRopeScalingConfig:
        rope_type: str = "llama3"
        factor: float = 8.0
        low_freq_factor: float = 1.0
        high_freq_factor: float = 4.0
        original_max_position_embeddings: int = 8192

    config = _make_config(LLAMA3_ROPE_SCALING)
    inv_freq, attention_scaling = _compute_config_rope_inv_freq(config, FakeRopeScalingConfig())
    assert inv_freq is not None, (
        "object-style (non-dict) config.rope_scaling must be normalized, not "
        "dropped; otherwise scaled models silently lose RoPE scaling again "
        "(issue #2405)."
    )
    expected = _reference_inv_freq(config, "llama3")
    assert torch.allclose(inv_freq.float().cpu(), expected, rtol = 1e-4, atol = 1e-6)


def test_object_style_rope_scaling_on_config_delegates_correctly():
    # 'linear' has no inline fallback; only the normalized-config retry passes this.
    from dataclasses import dataclass

    from unsloth.models.llama import _compute_config_rope_inv_freq

    @dataclass
    class FakeLinearRopeScalingConfig:
        rope_type: str = "linear"
        factor: float = 4.0

    dict_config = _make_config({"rope_type": "linear", "factor": 4.0})
    expected = _reference_inv_freq(dict_config, "linear")

    object_config = _make_config({"rope_type": "linear", "factor": 4.0})
    try:
        object_config.rope_scaling = FakeLinearRopeScalingConfig()
    except Exception:
        pytest.skip(
            "transformers strict-validates rope_scaling to dict/RopeParameters/None, "
            "so object-style config.rope_scaling (and the delegation retry it "
            "exercises) is unreachable on this version."
        )
    inv_freq, attention_scaling = _compute_config_rope_inv_freq(
        object_config, object_config.rope_scaling
    )
    assert inv_freq is not None, (
        "linear rope_scaling exposed as a config object was silently dropped; "
        "delegation must retry with a config copy carrying the normalized dict "
        "(issue #2405)."
    )
    assert torch.allclose(inv_freq.float().cpu(), expected, rtol = 1e-4, atol = 1e-6)
