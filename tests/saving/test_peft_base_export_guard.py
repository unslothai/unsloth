# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest


_SAVE_PY = Path(__file__).resolve().parents[2] / "unsloth" / "save.py"
_SOURCE = _SAVE_PY.read_text(encoding = "utf-8")
_TREE = ast.parse(_SOURCE)
_EXPORT_FUNCTIONS = {
    "unsloth_save_pretrained_gguf",
    "unsloth_push_to_hub_gguf",
    "unsloth_generic_save_pretrained_merged",
    "unsloth_generic_push_to_hub_merged",
    "unsloth_save_pretrained_torchao",
}


def _function(name: str) -> ast.FunctionDef:
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in unsloth/save.py")


def _function_source(name: str) -> str:
    node = _function(name)
    segment = ast.get_source_segment(_SOURCE, node)
    indent = len(segment) - len(segment.lstrip())
    return "\n".join(line[indent:] for line in segment.split("\n"))


class _FakeBaseTunerLayer:
    pass


class _FakePeftModel:
    pass


class _Model:
    def __init__(self, *modules):
        self._modules = modules

    def modules(self):
        return iter(self._modules)


@pytest.fixture
def peft_guard_env(monkeypatch):
    tuners_utils = types.ModuleType("peft.tuners.tuners_utils")
    tuners_utils.BaseTunerLayer = _FakeBaseTunerLayer
    saved = {name: value for name, value in sys.modules.items() if name.startswith("peft")}
    for name, module in (
        ("peft", types.ModuleType("peft")),
        ("peft.tuners", types.ModuleType("peft.tuners")),
        ("peft.tuners.tuners_utils", tuners_utils),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    yield
    for name in [name for name in sys.modules if name.startswith("peft")]:
        if name in saved:
            sys.modules[name] = saved[name]
        else:
            del sys.modules[name]


def _load_guard(monkeypatch):
    namespace = {
        "PeftModel": _FakePeftModel,
    }
    exec(
        compile(
            _function_source("_assert_export_target_is_not_base_with_lora_layers"),
            str(_SAVE_PY),
            "exec",
        ),
        namespace,
    )
    return namespace["_assert_export_target_is_not_base_with_lora_layers"]


def test_a_base_model_with_lora_layers_raises_before_export(peft_guard_env):
    guard = _load_guard(None)
    with pytest.raises(RuntimeError, match = "FastModel.from_pretrained"):
        guard(_Model(_FakeBaseTunerLayer()))


def test_a_peft_wrapper_is_allowed(peft_guard_env):
    guard = _load_guard(None)
    model = _FakePeftModel()
    model.modules = lambda: iter([_FakeBaseTunerLayer()])
    guard(model)


def test_a_full_fine_tune_is_allowed(peft_guard_env):
    guard = _load_guard(None)
    guard(_Model(object()))


@pytest.mark.parametrize("name", sorted(_EXPORT_FUNCTIONS))
def test_export_entrypoints_install_the_guard(name):
    source = ast.get_source_segment(_SOURCE, _function(name))
    assert "_assert_export_target_is_not_base_with_lora_layers(self)" in source
