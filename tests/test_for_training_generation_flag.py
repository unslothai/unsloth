# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import builtins
import functools
import inspect
import os
import types
from pathlib import Path

import pytest

# Both for_training sites must survive a delegating PEFT wrapper.
SITES = [("llama.py", "FastLlamaModel"), ("vision.py", "FastBaseModel")]


class _Namespace(dict):
    """Globals for a method lifted out of its module; unused helpers resolve to None."""

    def __missing__(self, name):
        return getattr(builtins, name, None)


def _helpers_used_by(method, tree):
    """Real module-level helpers the lifted method calls; missing ones resolve to None and raise."""
    defined = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and not node.decorator_list
    }
    wanted, pending = {}, [method]
    while pending:
        for node in ast.walk(pending.pop()):
            if isinstance(node, ast.Name) and node.id in defined and node.id not in wanted:
                wanted[node.id] = defined[node.id]
                pending.append(defined[node.id])
    return list(wanted.values())


def _torch_or_stub():
    """The fakes below are not nn.Modules, so without torch the lifted code takes its non-Module path."""
    try:
        import torch
    except ImportError:
        return types.SimpleNamespace(nn = types.SimpleNamespace(Module = type("_NoTorch", (), {})))
    return torch


def _for_training(module, class_name):
    path = Path(__file__).parents[1] / "unsloth" / "models" / module
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    model_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "for_training"
    )
    method.decorator_list = []
    utils = ast.parse((path.parent / "_utils.py").read_text(encoding = "utf-8"))
    shared = [
        node
        for node in utils.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in (
            "resolve_training_gradient_checkpointing",
            "_gradient_checkpointing_layer_class",
            "_forward_reads_checkpoint_function",
            "_calls_checkpoint_function",
            "_is_unarmed",
            "arm_gradient_checkpointing",
            "set_module_gradient_checkpointing",
            "switch_training_mode",
            "_mode_switch_class_info",
            "_has_gradient_checkpointing",
        )
    ]
    for node in shared:
        node.decorator_list = []
    compiled = ast.Module(body = shared + _helpers_used_by(method, tree) + [method], type_ignores = [])
    namespace = _Namespace(os = os, inspect = inspect, torch = _torch_or_stub(), functools = functools)
    exec(compile(ast.fix_missing_locations(compiled), str(path), "exec"), namespace)
    return namespace["for_training"]


class _Model:
    training = False
    gradient_checkpointing = False

    def __init__(self):
        self._flag_for_generation = True

    def parameters(self):
        return ()

    def modules(self):
        return ()

    def train(self):
        self.training = True


class _PeftProxy:
    training = False

    def __init__(self, model):
        self.model = model

    def __getattr__(self, name):
        return getattr(self.__dict__["model"], name)

    def parameters(self):
        return ()

    def modules(self):
        return ()

    def train(self):
        self.training = True


@pytest.mark.parametrize("module, class_name", SITES)
def test_for_training_deletes_a_generation_flag_delegated_by_a_peft_wrapper(module, class_name):
    model = _Model()
    proxy = _PeftProxy(model)
    assert hasattr(proxy, "_flag_for_generation")
    assert "_flag_for_generation" not in vars(proxy)

    _for_training(module, class_name)(proxy)

    assert not hasattr(model, "_flag_for_generation")
    assert not hasattr(proxy, "_flag_for_generation")


@pytest.mark.parametrize("module, class_name", SITES)
def test_for_training_does_not_swallow_unrelated_errors(module, class_name):
    class _Exploding(_Model):
        def __init__(self):
            pass

        @property
        def _flag_for_generation(self):
            return True

        @_flag_for_generation.deleter
        def _flag_for_generation(self):
            raise RuntimeError("must propagate")

    with pytest.raises(RuntimeError):
        _for_training(module, class_name)(_Exploding())


@pytest.mark.parametrize("module, class_name", SITES)
def test_a_custom_train_on_the_wrapper_chain_has_the_final_say(module, class_name):
    torch = pytest.importorskip("torch")

    class _StaysInEval(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)

        def train(self, mode = True):
            super().train(mode)
            self.training = False
            return self

    model = _StaysInEval()
    _for_training(module, class_name)(model, use_gradient_checkpointing = False)
    assert model.training is False
    assert model.model.training is True


@pytest.mark.parametrize("module, class_name", SITES)
def test_layers_flagged_by_arming_get_the_requested_mode(module, class_name):
    torch = pytest.importorskip("torch")

    class _Layer(torch.nn.Module):
        def forward(self, x):
            return self._gradient_checkpointing_func(x)

    class _Unarmed(_Layer):
        gradient_checkpointing = False

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Unarmed()
            self.late = _Layer()

        def gradient_checkpointing_enable(self):
            # A remote enabler that also flags a layer which had no attribute before.
            for layer in (self.model, self.late):
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = lambda f, *a: f(*a)

    model = _Model()
    _for_training(module, class_name)(model, use_gradient_checkpointing = "unsloth")
    assert model.late.gradient_checkpointing == "unsloth"
