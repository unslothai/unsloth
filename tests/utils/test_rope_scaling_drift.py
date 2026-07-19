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
LOADER_PY = REPO_ROOT / "unsloth" / "models" / "loader.py"

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


def _iter_names_and_calls(node):
    """(attribute/string names, bare-name calls, method-call attrs) under node."""
    names, calls, call_attrs = set(), set(), set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            names.add(sub.value)
        elif isinstance(sub, ast.Call):
            if isinstance(sub.func, ast.Name):
                calls.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                call_attrs.add(sub.func.attr)
    return names, calls, call_attrs


def _find_method(source_path, class_name, method_name):
    for node in ast.walk(ast.parse(source_path.read_text())):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
                    return sub
    return None


def _find_function(source_path, function_name):
    for node in ast.walk(ast.parse(source_path.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    return None


def test_config_path_inspects_rope_scaling():
    init_fn = _load_class_init()
    # inv_freq is derived through the shared _unsloth_recompute_inv_freq helper
    # (or still inlined in the config branch on older layouts); whichever scope
    # holds the scaling must read config.rope_scaling and call
    # _compute_config_rope_inv_freq, else scaled models run unscaled (#2405).
    _, _, init_call_attrs = _iter_names_and_calls(init_fn)
    scope = _find_method(LLAMA_PY, CLASS_NAME, "_unsloth_recompute_inv_freq")
    if scope is not None:
        assert "_unsloth_recompute_inv_freq" in init_call_attrs, (
            f"{CLASS_NAME}.__init__ no longer derives inv_freq via "
            "_unsloth_recompute_inv_freq; keep the constructor wired to the "
            "shared scaling helper or scaled configs silently lose RoPE scaling "
            "(issue #2405)."
        )
    else:
        scope = _config_branch(init_fn)
        assert scope is not None, (
            f"{CLASS_NAME}.__init__ has neither a _unsloth_recompute_inv_freq "
            "helper nor an `if config is not None:` branch; the config path must "
            "apply llama3/linear/longrope scaling (issue #2405)."
        )

    names, called, _ = _iter_names_and_calls(scope)
    assert "rope_scaling" in names, (
        f"{CLASS_NAME} inv_freq computation does not reference `rope_scaling`; "
        "scaled models (llama3/linear/longrope) would run unscaled and produce "
        "repeated-pattern gibberish past the original context (issue #2405)."
    )
    assert "_compute_config_rope_inv_freq" in called, (
        f"{CLASS_NAME} inv_freq computation no longer calls "
        "_compute_config_rope_inv_freq; keep it wired or scaled configs silently "
        "lose RoPE scaling again (issue #2405)."
    )


def test_v5_repair_reuses_recompute():
    # transformers v5 blanks non-persistent buffers on load, so
    # loader._fix_rope_inv_freq rebuilds inv_freq; it must reuse the scaled
    # recompute, since an unscaled rebuild re-drops llama3 scaling (#2405).
    fix_fn = _find_function(LOADER_PY, "_fix_rope_inv_freq")
    assert fix_fn is not None, (
        "loader._fix_rope_inv_freq not found; if it was renamed, update this "
        "guard so the v5 rope repair keeps applying config scaling (issue #2405)."
    )
    _, _, call_attrs = _iter_names_and_calls(fix_fn)
    assert "_unsloth_recompute_inv_freq" in call_attrs, (
        "loader._fix_rope_inv_freq no longer rebuilds inv_freq via "
        "_unsloth_recompute_inv_freq; transformers v5 blanks the buffer on load "
        "and an unscaled rebuild re-drops llama3 scaling (issue #2405)."
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


def test_recompute_helper_scales_on_cpu():
    # Exercise the exact method loader._fix_rope_inv_freq calls, without CUDA.
    from unsloth.models.llama import LlamaRotaryEmbedding, _get_rope_theta

    def recompute(config):
        rot = object.__new__(LlamaRotaryEmbedding)
        rot.attention_scaling = 1.0
        rot.base = _get_rope_theta(config, 10000.0)
        rot.dim = config.head_dim
        rot._unsloth_rope_config = config
        return rot._unsloth_recompute_inv_freq().float().cpu()

    config = _make_config(LLAMA3_ROPE_SCALING)
    assert torch.allclose(
        recompute(config), _reference_inv_freq(config, "llama3"), rtol = 1e-4, atol = 1e-6
    ), "_unsloth_recompute_inv_freq dropped llama3 scaling (issue #2405)."
    assert torch.allclose(
        recompute(_make_config(None)), _vanilla_inv_freq(), rtol = 1e-4, atol = 1e-6
    ), "_unsloth_recompute_inv_freq must return vanilla inv_freq when unscaled."


def test_extended_rope_scaling_keeps_llama3_and_carries_theta():
    # Long-context extension keeps native llama3, but falls back to linear for every other
    # type (the patched attention constructor only rebuilds linear/llama3/longrope), and the
    # linear dict carries rope_theta so transformers v5 does not fall back to base 10000.
    from types import SimpleNamespace

    from unsloth.models.llama import _extended_rope_scaling

    # llama3 model: keep native scaling, do not synthesize linear.
    scaling, native = _extended_rope_scaling(_make_config(LLAMA3_ROPE_SCALING), 2.0)
    assert (
        scaling is None and native == "llama3"
    ), "must keep native llama3 scaling instead of overwriting it with linear."

    # yarn is not rebuildable by the patcher -> keep the safe linear fallback, not native.
    yarn = SimpleNamespace(rope_scaling = {"rope_type": "yarn", "factor": 2.0}, rope_theta = 500000.0)
    scaling, _ = _extended_rope_scaling(yarn, 2.0)
    assert scaling == {
        "type": "linear",
        "factor": 2.0,
        "rope_theta": 500000.0,
    }, f"yarn must fall back to linear (patcher cannot rebuild it), got {scaling}."

    # plain RoPE with theta only under v5 rope_parameters: linear must carry rope_theta.
    v5 = SimpleNamespace(rope_parameters = {"rope_type": "default", "rope_theta": 1000000.0})
    scaling, _ = _extended_rope_scaling(v5, 2.0)
    assert scaling == {
        "type": "linear",
        "factor": 2.0,
        "rope_theta": 1000000.0,
    }, f"linear override dropped rope_theta on v5 (got {scaling}); base would fall back to 10000."


def test_extended_rotary_reads_config_factor():
    # LlamaExtendedRotaryEmbedding must honor the config factor, not hardcode 8
    # (Llama-3.2 uses 32); otherwise the subclass path re-drops scaling (#2405).
    from types import SimpleNamespace

    from unsloth.models.llama import LlamaExtendedRotaryEmbedding

    rot = object.__new__(LlamaExtendedRotaryEmbedding)
    rot.base = ROPE_THETA
    rot.dim = HEAD_DIM
    rot._unsloth_rope_config = SimpleNamespace(
        rope_scaling = {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )
    vanilla = _vanilla_inv_freq()
    scaled = rot._apply_inv_freq_scaling(vanilla).reshape(-1)
    ratio = float(vanilla[-1]) / float(scaled[-1])
    assert abs(ratio - 32.0) < 1e-3, (
        f"LlamaExtendedRotaryEmbedding ignored config factor 32 (ratio {ratio}); the "
        "low-frequency band must be divided by the config factor (issue #2405)."
    )


def test_extended_rotary_reads_rope_parameters_v5():
    # transformers v5 stores scaling under rope_parameters (rope_scaling is a
    # back-compat shim that may be removed); the factor must still be read.
    from types import SimpleNamespace

    from unsloth.models.llama import LlamaExtendedRotaryEmbedding

    rot = object.__new__(LlamaExtendedRotaryEmbedding)
    rot.base = ROPE_THETA
    rot.dim = HEAD_DIM
    rot._unsloth_rope_config = SimpleNamespace(
        rope_scaling = None,
        rope_parameters = {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    )
    vanilla = _vanilla_inv_freq()
    scaled = rot._apply_inv_freq_scaling(vanilla).reshape(-1)
    ratio = float(vanilla[-1]) / float(scaled[-1])
    assert abs(ratio - 32.0) < 1e-3, (
        f"Extended rotary ignored rope_parameters factor 32 (ratio {ratio}); v5 "
        "keeps the factor under rope_parameters, not rope_scaling."
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


def _blank_nonpersistent_buffers(module):
    """Mimic transformers v5 meta-load: overwrite non-persistent buffers with garbage."""
    for name, buf in list(module.named_buffers()):
        leaf = module
        *parents, attr = name.split(".")
        for part in parents:
            leaf = getattr(leaf, part)
        if attr in getattr(leaf, "_non_persistent_buffers_set", set()):
            setattr(leaf, attr, torch.rand_like(buf))


def _build_llama3_rotary():
    from unsloth.models import llama as llama_mod
    config = _make_config(LLAMA3_ROPE_SCALING)
    return llama_mod.LlamaRotaryEmbedding(config = config), config


def _build_longrope_rotary():
    from types import SimpleNamespace

    from unsloth.models import llama as llama_mod

    short_factor, long_factor = [1.05] * 48, [1.3] * 48
    rot = llama_mod.LongRopeRotaryEmbedding(
        dim = 96,
        max_position_embeddings = 131072,
        original_max_position_embeddings = 4096,
        base = ROPE_THETA,
        short_factor = short_factor,
        long_factor = long_factor,
    )
    config = SimpleNamespace(
        rope_scaling = {
            "rope_type": "longrope",
            "short_factor": short_factor,
            "long_factor": long_factor,
            "original_max_position_embeddings": 4096,
        }
    )
    return rot, config


@requires_cuda
@pytest.mark.parametrize(
    "build", [_build_llama3_rotary, _build_longrope_rotary], ids = ["llama3", "longrope"]
)
def test_v5_blank_repair_roundtrip(build):
    # Build scaled -> blank non-persistent buffers (what transformers v5 does on
    # load) -> run the repair -> every buffer must return to its scaled value.
    # Family-agnostic: encodes no scaling math, so it guards any rotary that
    # keeps scaling in a buffer (issue #2405 / PR #6907).
    from unsloth.models import loader

    # The repair only runs on transformers v5 (it is what blanks the buffers);
    # on v4 _fix_rope_inv_freq is a no-op, so the round-trip cannot restore.
    if not loader._NEEDS_ROPE_FIX:
        pytest.skip("transformers < 5 does not blank rope buffers; repair is a no-op")

    rot, config = build()
    snapshot = {name: buf.detach().clone() for name, buf in rot.named_buffers()}
    assert snapshot, "rotary registers no buffers; nothing to guard"

    _blank_nonpersistent_buffers(rot)
    assert any(
        not torch.equal(rot.get_buffer(name), snapshot[name]) for name in snapshot
    ), "blanking changed no buffer; the round-trip would be vacuous"

    wrapper = torch.nn.Module()
    wrapper.add_module("rotary_emb", rot)
    wrapper.config = config
    loader._fix_rope_inv_freq(wrapper)

    for name in snapshot:
        assert torch.allclose(
            rot.get_buffer(name).cpu(), snapshot[name].cpu(), rtol = 1e-4, atol = 1e-6
        ), (
            f"{name} was not restored to its scaled value by loader._fix_rope_inv_freq "
            "after the transformers v5 buffer blank (issue #2405 / PR #6907)."
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
