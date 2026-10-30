# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""YaRN passed to from_pretrained must reach the model, as `rope_scaling` or as transformers v5 /
TRL's `rope_parameters` spelling (SFTConfig(model_init_kwargs = {"rope_parameters": ...})).

Before: FastLanguageModel ignored `rope_parameters`, FastModel passed it to the model constructor
(TypeError), and FastModel dropped `rope_scaling` entirely, so the model trained on plain RoPE."""

import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

transformers = pytest.importorskip("transformers")
pytest.importorskip("torch")

_TESTS = str(Path(__file__).resolve().parent)

# Subprocess: the loaders patch transformers process-wide, and the CPU redirects must not leak.
_CHECK = r"""
import math, os, sys
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, sys.argv[1])
import _zoo_aggressive_cuda_spoof as _spoof
_spoof.apply()
import torch
import unsloth
# A bare index is a CUDA ordinal here (get_current_device() returns one). torch.device(0) resolves it
# through the accelerator, which raises on a CPU-only torch, so it is not asked.
_is_cuda = lambda d: (isinstance(d, int) and not isinstance(d, bool)) or (
    isinstance(d, (str, torch.device)) and torch.device(d).type == "cuda"
)
for _name in ("empty", "zeros", "ones", "full", "tensor", "arange", "randn", "rand", "randint",
              "empty_like", "zeros_like", "ones_like"):
    _orig = getattr(torch, _name)
    def _redir(*a, _orig = _orig, **k):
        if _is_cuda(k.get("device")):
            k["device"] = "cpu"
        return _orig(*a, **k)
    setattr(torch, _name, _redir)
_orig_to = torch.Tensor.to
def _to(self, *a, **k):
    a = tuple("cpu" if _is_cuda(x) else x for x in a)
    if _is_cuda(k.get("device")):
        k["device"] = "cpu"
    return _orig_to(self, *a, **k)
torch.Tensor.to = _to
torch.Tensor.cuda = lambda self, *a, **k: self
if hasattr(torch, "accelerator"):
    torch.accelerator.is_available = lambda *a, **k: False
from unsloth import FastLanguageModel, FastModel

loader, kwarg = sys.argv[2], sys.argv[3]
if loader == "unit":
    from unsloth.models.loader import _fold_rope_parameters
    from unsloth.models.vision import _apply_user_rope_scaling

    yarn = {"rope_type": "yarn", "factor": 4.0}
    kwargs = {"rope_parameters": dict(yarn), "other": 1}
    assert _fold_rope_parameters(None, kwargs) == yarn
    assert kwargs == {"other": 1}
    assert _fold_rope_parameters(dict(yarn), {"rope_parameters": dict(yarn)}) == yarn
    assert _fold_rope_parameters(yarn, {}) is yarn
    try:
        _fold_rope_parameters({"rope_type": "linear", "factor": 2.0}, {"rope_parameters": yarn})
        raise AssertionError("differing rope_scaling / rope_parameters accepted")
    except ValueError:
        pass

    class Cfg:
        rope_parameters = {
            "full_attention": {"rope_type": "default", "rope_theta": 1e6},
            "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
        }

    # A flat override applies to full attention only, as transformers' Gemma 3 config does.
    cfg = Cfg()
    assert _apply_user_rope_scaling(cfg, {"rope_type": "yarn", "factor": 4.0}) is None
    assert cfg.rope_parameters == {
        "full_attention": {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
    }, cfg.rope_parameters

    # A flat config keeps its layout keys (Qwen2-VL mrope_section, StableLM partial_rotary_factor).
    class Flat:
        max_position_embeddings = 4096
        rope_parameters = {
            "rope_type": "default", "rope_theta": 1e4, "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
        }

    flat = Flat()
    applied = _apply_user_rope_scaling(flat, {"rope_type": "yarn", "factor": 4.0})
    expected = {
        "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096,
        "rope_theta": 1e4, "partial_rotary_factor": 0.25, "mrope_section": [2, 1, 1],
    }
    assert applied == expected and flat.rope_scaling == expected, (applied, flat.rope_scaling)
    try:
        _apply_user_rope_scaling(Flat(), {"full_attention": {"rope_type": "yarn", "factor": 4.0}})
        raise AssertionError("per-layer-type override accepted for a flat config")
    except ValueError:
        pass
    new = {
        "full_attention": {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
    }
    _apply_user_rope_scaling(cfg, new)
    assert cfg.rope_parameters == new
    # Entries without rope_theta keep the layer type's base (Gemma 3's rotary init indexes it).
    cfg = Cfg()
    _apply_user_rope_scaling(
        cfg,
        {
            "full_attention": {"rope_type": "yarn", "factor": 4.0},
            "sliding_attention": {"rope_type": "default"},
        },
    )
    assert cfg.rope_parameters == {
        "full_attention": {"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1e4},
    }, cfg.rope_parameters

    from unsloth.models.vision import _vllm_rope_scaling

    assert _vllm_rope_scaling(yarn, True) == yarn
    for args in ((new, True), (yarn, False), (yarn, True, True)):
        try:
            _vllm_rope_scaling(*args)
            raise AssertionError(f"vLLM RoPE override accepted: {args}")
        except NotImplementedError:
            pass
    # fast_inference: load_vllm reads config.json itself, so the override must be forwarded.
    import ast, inspect, textwrap
    import unsloth.models.vision as vision

    source = inspect.getsource(vision.FastBaseModel.from_pretrained)
    forwarded = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript)
            and getattr(t.value, "id", None) == "load_vllm_kwargs"
            and getattr(t.slice, "value", None) == "rope_scaling"
            for t in node.targets
        )
    ]
    assert forwarded, "load_vllm_kwargs never receives rope_scaling"
    print("OK", unsloth.__file__)
    sys.exit(0)
if loader == "gemma3_legacy_type":
    # The legacy {"type": ...} spelling on a per-layer-type config (Gemma 3 on transformers v5):
    # Gemma 3's rotary init reads rope_parameters[layer_type]["rope_type"] directly.
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig
    from unsloth.models.vision import _apply_user_rope_scaling

    def tiny():
        return Gemma3TextConfig(
            vocab_size = 128, hidden_size = 32, intermediate_size = 64, num_hidden_layers = 2,
            layer_types = ["sliding_attention", "full_attention"],
            num_attention_heads = 2, num_key_value_heads = 1, head_dim = 16,
            max_position_embeddings = 256, sliding_window = 16,
        )

    for override in (
        {"type": "yarn", "factor": 4.0},
        {"full_attention": {"type": "yarn", "factor": 4.0}, "sliding_attention": {"type": "default"}},
    ):
        cfg = tiny()
        assert _apply_user_rope_scaling(cfg, dict(override)) is None
        model = Gemma3ForCausalLM(cfg)  # KeyError('rope_type') when the legacy key is kept alone
        full = cfg.rope_parameters["full_attention"]
        assert full["rope_type"] == "yarn" and full["factor"] == 4.0, cfg.rope_parameters
        assert full["rope_theta"] == 1e6, cfg.rope_parameters
        assert cfg.rope_parameters["sliding_attention"]["rope_type"] == "default", cfg.rope_parameters
        rotary = [m for m in model.modules() if "Rotary" in type(m).__name__]
        assert rotary and rotary[0].rope_type == {
            "sliding_attention": "default", "full_attention": "yarn",
        }, rotary[0].rope_type
    print("OK", unsloth.__file__)
    sys.exit(0)
yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}
if int(__import__("transformers").__version__.split(".")[0]) >= 5:
    yarn["rope_theta"] = 1000000
cls = FastLanguageModel if loader == "lang" else FastModel
model, _ = cls.from_pretrained(
    # Past the checkpoint's 40960 window, as in the long-context recipe: there the automatic linear
    # scaling used to replace a `rope_parameters` YaRN request.
    "trl-internal-testing/tiny-Qwen3ForCausalLM", max_seq_length = 163840, load_in_4bit = False,
    dtype = torch.float32, device_map = "cpu", **{kwarg: dict(yarn)},
)
cfg = model.config
rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
assert rope.get("rope_type") == "yarn", rope
assert rope.get("factor") == 4.0 and rope.get("original_max_position_embeddings") == 40960, rope
rotary = [m for m in model.modules() if "Rotary" in type(m).__name__]
assert rotary, "no rotary module"
expected = 0.1 * math.log(4.0) + 1.0
for r in rotary:
    assert abs(float(r.attention_scaling) - expected) < 1e-9, (type(r), r.attention_scaling)
    if hasattr(r, "rope_type"):
        assert r.rope_type == "yarn", r.rope_type
print("OK", unsloth.__file__)
"""

_V5 = Version(transformers.__version__) >= Version("5.0.0")


@pytest.mark.parametrize(
    "loader,kwarg",
    [
        ("lang", "rope_scaling"),
        pytest.param(
            "lang",
            "rope_parameters",
            marks = pytest.mark.skipif(not _V5, reason = "rope_parameters is the v5 spelling"),
        ),
        ("model", "rope_scaling"),
        pytest.param(
            "model",
            "rope_parameters",
            marks = pytest.mark.skipif(not _V5, reason = "rope_parameters is the v5 spelling"),
        ),
    ],
)
def test_yarn_reaches_the_model(loader, kwarg):
    result = subprocess.run(
        [sys.executable, "-c", _CHECK, _TESTS, loader, kwarg],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )


@pytest.mark.skipif(not _V5, reason = "Gemma 3 keys rope_parameters by layer type on v5")
def test_gemma3_legacy_type_key_builds():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK, _TESTS, "gemma3_legacy_type", "-"],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )


def test_rope_helpers():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK, _TESTS, "unit", "-"],
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )
