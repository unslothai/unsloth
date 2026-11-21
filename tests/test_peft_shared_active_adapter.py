# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL DPO's ref / default adapter round trip must not invalidate compiled LoRA graphs (Gemma 4 E2B DPO)."""

import ast
import functools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model  # noqa: E402
import peft.tuners.tuners_utils as tu  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_patch():
    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    ns = {"functools": functools, "sys": sys}
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "patch_peft_shared_active_adapter":
            exec(ast.get_source_segment(src, node), ns)
    return ns["patch_peft_shared_active_adapter"]


def _pristine_set_adapter():
    # conftest imports unsloth, which may already have wrapped it.
    fn = tu.BaseTunerLayer.set_adapter
    while getattr(fn, "_unsloth_shared_active_adapter", False):
        fn = fn.__wrapped__
    return fn


@pytest.fixture
def unpatched(monkeypatch):
    monkeypatch.setattr(tu.BaseTunerLayer, "set_adapter", _pristine_set_adapter())
    yield


@pytest.fixture
def patched(unpatched):
    _load_patch()()
    yield


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(16, 16, bias = False)
        self.up_proj = torch.nn.Linear(16, 16, bias = False)

    def forward(self, x):
        return self.up_proj(torch.relu(self.q_proj(x)))


def _lora_model():
    torch.manual_seed(0)
    model = get_peft_model(_Block(), LoraConfig(r = 4, target_modules = ["q_proj", "up_proj"]))
    model.add_adapter("ref", LoraConfig(r = 4, target_modules = ["q_proj", "up_proj"]))
    return model


def _lora_layers(model):
    return [m for m in model.modules() if isinstance(m, tu.BaseTunerLayer)]


def _round_trip(model):
    # trl.trainer.utils.use_adapter(model, "ref").
    model.set_adapter("ref")
    model.set_adapter("default")


def _recompiles_across_round_trip(model):
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()
    block = model.base_model.model
    compiled = torch.compile(block.forward, backend = counter)
    x = torch.randn(2, 16)
    compiled(x)
    first = counter.frame_count
    _round_trip(model)
    compiled(x)
    return counter.frame_count - first


def test_round_trip_keeps_one_shared_active_adapter(patched):
    model = _lora_model()
    _round_trip(model)
    layers = _lora_layers(model)
    assert len({id(layer._active_adapter) for layer in layers}) == 1
    for layer in layers:
        assert layer.active_adapters == ["default"]
        assert layer.active_adapter == "default"


def test_round_trip_keeps_peft_trainability(patched):
    model = _lora_model()
    _round_trip(model)
    for layer in _lora_layers(model):
        assert all(p.requires_grad for p in layer.lora_A["default"].parameters())
        assert not any(p.requires_grad for p in layer.lora_A["ref"].parameters())


def test_multi_adapter_selection_is_left_alone(patched):
    model = _lora_model()
    layer = _lora_layers(model)[0]
    layer.set_adapter(["default", "ref"])
    assert layer.active_adapters == ["default", "ref"]


def test_patch_is_idempotent(patched):
    first = tu.BaseTunerLayer.set_adapter
    _load_patch()()
    assert tu.BaseTunerLayer.set_adapter is first


def test_compiled_lora_forward_does_not_recompile_after_round_trip(patched):
    assert _recompiles_across_round_trip(_lora_model()) == 0


def test_unpatched_peft_recompiles_after_round_trip(unpatched):
    if _recompiles_across_round_trip(_lora_model()) == 0:
        pytest.skip("this peft / torch no longer recompiles on the adapter round trip")
