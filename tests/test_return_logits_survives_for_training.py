# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A user's UNSLOTH_RETURN_LOGITS=1 must survive trainer.train() (#3071).

train() and the trainer's __init__ call model.for_training(), which forced the flag to "0", so a
compute_loss or compute_metrics reading logits got EmptyLogits even with the flag set. for_training()
now only undoes the "1" that for_inference() forced, and the trainer __init__'s trailing
for_inference() leaves the flag as it found it. The methods are lifted out with ``ast`` so this
runs without ``import unsloth``.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import os
import sys
import textwrap
import types
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"
FLAG = "UNSLOTH_RETURN_LOGITS"
MARK = "_UNSLOTH_RETURN_LOGITS_BEFORE_INFERENCE"


class _Namespace(dict):
    def __missing__(self, name):
        return getattr(builtins, name, None)


def _fast_base_model():
    tree = ast.parse((MODELS / "vision.py").read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastBaseModel")
    wanted = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_embeddings_or_none"
    ]
    methods = {}
    for name in ("for_inference", "for_training"):
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        fn.decorator_list = []
        wanted.append(fn)
    utils = ast.parse((MODELS / "_utils.py").read_text(encoding = "utf-8"))
    for node in utils.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "resolve_training_gradient_checkpointing",
            "_gradient_checkpointing_layer_class",
            "_forward_reads_checkpoint_function",
            "_calls_checkpoint_function",
            "_is_unarmed",
            "arm_gradient_checkpointing",
            "set_module_gradient_checkpointing",
        ):
            node.decorator_list = []
            wanted.insert(0, node)
    namespace = _Namespace(os = os, inspect = inspect)
    module = ast.fix_missing_locations(ast.Module(body = wanted, type_ignores = []))
    exec(compile(module, str(MODELS / "vision.py"), "exec"), namespace)
    for name in ("for_inference", "for_training"):
        methods[name] = namespace[name]
    return types.SimpleNamespace(**methods)


class _Model:
    training = True
    gradient_checkpointing = False

    def parameters(self):
        return ()

    def modules(self):
        return ()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


@pytest.fixture
def fbm(monkeypatch):
    zoo = types.ModuleType("unsloth_zoo")
    utils = types.ModuleType("unsloth_zoo.training_utils")
    utils.restore_use_cache = lambda model: None
    utils.disable_use_cache = lambda model: None
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.training_utils", utils)
    monkeypatch.delenv(FLAG, raising = False)
    monkeypatch.delenv(MARK, raising = False)
    return _fast_base_model()


def test_for_training_keeps_a_user_set_flag(fbm, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    fbm.for_training(_Model())
    assert os.environ[FLAG] == "1"
    # The trainer's own cycle: for_training in __init__, for_inference, for_training in train().
    model = _Model()
    fbm.for_training(model)
    fbm.for_inference(model)
    fbm.for_training(model)
    assert os.environ[FLAG] == "1"
    assert MARK not in os.environ


@pytest.mark.parametrize("before", [None, "0"])
def test_for_training_undoes_the_flag_for_inference_forced(fbm, monkeypatch, before):
    # Default training must go back to the fused CE path after generation / GRPO rollouts.
    if before is not None:
        monkeypatch.setenv(FLAG, before)
    model = _Model()
    fbm.for_inference(model)
    assert os.environ[FLAG] == "1"
    fbm.for_inference(model)  # repeated calls (every GRPO step) keep the first saved value
    fbm.for_training(model)
    assert os.environ[FLAG] == "0"
    assert MARK not in os.environ
    fbm.for_training(model)
    assert os.environ[FLAG] == "0"


def test_for_training_still_defaults_an_unset_flag_to_zero(fbm):
    fbm.for_training(_Model())
    assert os.environ[FLAG] == "0"


def _trailing_for_inference_block():
    tree = ast.parse((MODELS / "rl.py").read_text(encoding = "utf-8"))
    template = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "RLTrainer_replacement" for t in n.targets)
    )
    lines = template.splitlines()
    start = next(i for i, line in enumerate(lines) if 'hasattr(model, "for_inference")' in line)
    block = [lines[start]]
    indent = len(lines[start]) - len(lines[start].lstrip())
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    # The template is filled with str.format: a literal brace in the block breaks every trainer.
    return textwrap.dedent("\n".join(block)).format()


def test_trainer_init_leaves_the_flag_for_train(fbm, monkeypatch):
    # Building the trainer must not arm a restore of "0" over a flag the user sets before train().
    model = _Model()
    model.for_inference = lambda: fbm.for_inference(model)
    monkeypatch.setenv(FLAG, "0")
    exec(_trailing_for_inference_block(), {"os": os, "model": model})
    assert model.training is False
    assert os.environ[FLAG] == "0"
    assert MARK not in os.environ
    os.environ[FLAG] = "1"
    fbm.for_training(model)
    assert os.environ[FLAG] == "1"
