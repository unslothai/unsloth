# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Qwen3.8-Flash-Next (qwen4_exp) under fast_inference: the vision-model gate lets it through only with an
unsloth_zoo that can share its hyper-connection / PLE / QSA weights, bitsandbytes stays refused (sparse MoE), and its
norms keep aliasing vLLM's bf16 weights instead of being copied to float32."""

import ast
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

VISION_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "vision.py"
SOURCE = VISION_PATH.read_text(encoding = "utf-8")
TREE = ast.parse(SOURCE)


def _module_constant(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found")


def _function(name):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace = {}
            exec(
                compile(ast.Module(body = [node], type_ignores = []), str(VISION_PATH), "exec"),
                namespace,
            )
            return namespace[name]
    raise AssertionError(f"{name} not found")


QWEN4_EXP = SimpleNamespace(model_type = "qwen4_exp", text_config = SimpleNamespace(num_experts = 512))


def test_qwen4_exp_passes_the_vision_allowlist():
    supported = _module_constant("VLLM_SUPPORTED_VLM")
    assert "qwen4_exp" in supported
    assert len(supported) == len(set(supported))


def test_qwen4_exp_is_a_sparse_moe_so_bitsandbytes_stays_refused():
    assert _function("_is_sparse_moe_config")(QWEN4_EXP) is True


def test_zoo_probe_needs_the_qwen4_exp_module(monkeypatch):
    probe = _function("_zoo_supports_qwen4_exp_fast_inference")
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_qwen4_exp", None)
    assert probe() is False
    fake = types.ModuleType("unsloth_zoo.vllm_qwen4_exp")
    fake.is_qwen4_exp_config = lambda config: True
    monkeypatch.setitem(sys.modules, "unsloth_zoo.vllm_qwen4_exp", fake)
    assert probe() is True


def _gate_if(marker):
    for node in ast.walk(TREE):
        if isinstance(node, ast.If) and marker in ast.get_source_segment(SOURCE, node.test):
            return node
    raise AssertionError(f"no if-statement testing {marker}")


def test_old_zoo_refuses_qwen4_exp_before_loading():
    gate = _gate_if("_zoo_supports_qwen4_exp_fast_inference")
    test = compile(ast.Expression(gate.test), str(VISION_PATH), "eval")
    base = dict(fast_inference = True, model_types = ["qwen4_exp"])
    assert eval(test, {**base, "_zoo_supports_qwen4_exp_fast_inference": lambda: False}) is True
    assert eval(test, {**base, "_zoo_supports_qwen4_exp_fast_inference": lambda: True}) is False
    assert (
        eval(
            test,
            {
                "fast_inference": False,
                "model_types": ["qwen4_exp"],
                "_zoo_supports_qwen4_exp_fast_inference": lambda: False,
            },
        )
        is False
    )
    assert any(isinstance(n, ast.Raise) for n in ast.walk(gate))


@pytest.mark.parametrize(
    "fast_inference,model_types,upcast",
    [
        (True, ["qwen4_exp"], False),
        (False, ["qwen4_exp"], True),
        (True, ["qwen3_5_moe"], True),
    ],
)
def test_high_precision_norm_upcast_skips_shared_qwen4_exp_norms(
    monkeypatch, fast_inference, model_types, upcast
):
    gate = _gate_if("UNSLOTH_HIGH_PRECISION_LAYERNORM")
    test = compile(ast.Expression(gate.test), str(VISION_PATH), "eval")
    monkeypatch.setenv("UNSLOTH_HIGH_PRECISION_LAYERNORM", "1")
    assert (
        eval(
            test,
            {
                "os": os,
                "fast_inference": fast_inference,
                "model_types": model_types,
                "model": object(),
            },
        )
        is upcast
    )
