# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import builtins
import os
from pathlib import Path

import pytest

# Both for_training sites must survive a delegating PEFT wrapper. See issue #2490.
SITES = [("llama.py", "FastLlamaModel"), ("vision.py", "FastBaseModel")]


class _Namespace(dict):
    """Globals for a method lifted out of its module; unused helpers resolve to None."""

    def __missing__(self, name):
        return getattr(builtins, name, None)


def _helpers_used_by(method, tree):
    """Module-level functions the lifted method calls, so they run for real.

    Without this they fall to `_Namespace.__missing__` and become None, and the
    first call raises `TypeError: 'NoneType' object is not callable`: the test
    then fails for a reason that has nothing to do with what it asserts.
    """
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
    compiled = ast.Module(body = _helpers_used_by(method, tree) + [method], type_ignores = [])
    namespace = _Namespace(os = os)
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
