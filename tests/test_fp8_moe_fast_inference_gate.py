# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""fast_inference on FP8 sparse MoE: block FP8 is shared with vLLM by unsloth_zoo, every other FP8
MoE format is refused before vLLM allocates the engine."""

import ast
import re
import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"
TREE = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))


def _functions(*names):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    namespace = {"re": re}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(VISION_PATH), "exec"), namespace)
    return namespace


def _refusal():
    return _functions("_is_sparse_moe_config", "_fp8_moe_fast_inference_refusal")[
        "_fp8_moe_fast_inference_refusal"
    ]


BLOCK = dict(quant_method = "fp8", weight_block_size = [128, 128], activation_scheme = "dynamic")


def _moe(**quant):
    return SimpleNamespace(
        num_experts = 128, moe_intermediate_size = 768, quantization_config = {**BLOCK, **quant}
    )


@pytest.fixture
def zoo_scheme(monkeypatch):
    """A unsloth_zoo that shares block FP8 experts (exposes fp8_moe_checkpoint_scheme)."""
    seen = []

    def fp8_moe_checkpoint_scheme(config):
        seen.append(config)
        quant = config.quantization_config
        if quant.get("weight_block_size") != [128, 128]:
            return False, "weight_block_size is not [128, 128]"
        return True, None

    empty_model = types.ModuleType("unsloth_zoo.empty_model")
    empty_model.fp8_moe_checkpoint_scheme = fp8_moe_checkpoint_scheme
    zoo = types.ModuleType("unsloth_zoo")
    zoo.empty_model = empty_model
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.empty_model", empty_model)
    return seen


@pytest.fixture
def old_zoo(monkeypatch):
    """A unsloth_zoo without fp8_moe_checkpoint_scheme: it shares no FP8 experts."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "unsloth_zoo.empty_model":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_block_fp8_moe_passes_with_a_sharing_zoo(zoo_scheme):
    assert _refusal()(_moe()) is None
    assert len(zoo_scheme) == 1


def test_other_fp8_moe_schemes_are_refused_with_the_zoo_reason(zoo_scheme):
    assert "weight_block_size" in _refusal()(_moe(weight_block_size = None))


def test_old_zoo_refuses_every_fp8_moe(old_zoo):
    refusal = _refusal()
    assert "upgrade unsloth_zoo" in refusal(_moe())
    # Dense FP8, unquantized MoE and bnb MoE are not this gate's business.
    assert refusal(SimpleNamespace(num_experts = None, quantization_config = BLOCK)) is None
    assert refusal(SimpleNamespace(num_experts = 8)) is None
    assert refusal(SimpleNamespace(num_experts = 8, quantization_config = {"quant_method": "bitsandbytes"})) is None
    assert refusal(SimpleNamespace(num_experts = 8, quantization_config = {"quant_method": "mxfp4"})) is None
    # FP8 attention with unquantized experts: vLLM keeps them 16-bit and they are shared as on main.
    assert refusal(_moe(weight_block_size = None, modules_to_not_convert = ["model.layers.0.mlp.experts"])) is None
    assert "upgrade" in refusal(_moe(modules_to_not_convert = ["model.layers.0.mlp.shared_experts"]))


def test_the_refusal_runs_before_load_vllm():
    source = VISION_PATH.read_text(encoding = "utf-8")
    gate = source.index("fp8_moe_refusal = _fp8_moe_fast_inference_refusal(model_config)")
    assert gate < source.index("llm = load_vllm(**load_vllm_kwargs)")
    assert "raise NotImplementedError" in source[gate : gate + 800]
