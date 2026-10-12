# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A user's UNSLOTH_RETURN_LOGITS=1 must survive trainer.train() (#3071): for_training() may only undo
for_inference()'s forced "1". Methods are lifted out with ``ast`` so this runs without ``import unsloth``.
"""

from __future__ import annotations

import ast
import builtins
import functools
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


def _torch_or_stub():
    """The fakes here are not nn.Modules, so without torch the lifted code takes its non-Module path."""
    try:
        import torch
    except ImportError:
        return types.SimpleNamespace(nn = types.SimpleNamespace(Module = type("_NoTorch", (), {})))
    return torch


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
            "switch_training_mode",
            "_mode_switch_class_info",
            "_has_gradient_checkpointing",
        ):
            node.decorator_list = []
            wanted.insert(0, node)
    namespace = _Namespace(os = os, inspect = inspect, torch = _torch_or_stub(), functools = functools)
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


def _fast_llama_for_training():
    tree = ast.parse((MODELS / "llama.py").read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "for_training")
    fn.decorator_list = []
    namespace = _Namespace(
        os = os,
        resolve_training_gradient_checkpointing = lambda model, flag: False,
        arm_gradient_checkpointing = lambda model: None,
        set_module_gradient_checkpointing = lambda module, flag: None,
    )
    module = ast.fix_missing_locations(ast.Module(body = [fn], type_ignores = []))
    exec(compile(module, str(MODELS / "llama.py"), "exec"), namespace)
    return namespace["for_training"]


@pytest.mark.parametrize("before", [None, "0", "1"])
def test_fast_language_model_for_training_undoes_generate(fbm, monkeypatch, before):
    # FastModel-backed models (Qwen3.5, Gemma 3/4) run FastBaseModel.for_inference() in generate().
    if before is not None:
        monkeypatch.setenv(FLAG, before)
    model = _Model()
    fbm.for_inference(model)
    assert os.environ[FLAG] == "1"
    _fast_llama_for_training()(model)
    assert os.environ[FLAG] == (before or "0")
    assert MARK not in os.environ


def test_fast_language_model_for_training_leaves_a_user_flag(fbm, monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    _fast_llama_for_training()(_Model())
    assert os.environ[FLAG] == "1"


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


def _prepare_for_training_mode():
    tree = ast.parse((MODELS / "rl.py").read_text(encoding = "utf-8"))
    template = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "RLTrainer_replacement" for t in n.targets)
    )
    lines = template.splitlines()
    start = lines.index("def prepare_for_training_mode(f):")
    end = lines.index("    return wrapper", start)
    namespace = _Namespace(os = os, functools = __import__("functools"))
    exec("\n".join(lines[start : end + 1]).format(), namespace)
    return namespace["prepare_for_training_mode"]


def test_flag_set_between_two_train_calls_is_kept(fbm):
    model = _Model()
    model.for_inference = lambda: fbm.for_inference(model)
    model.for_training = lambda use_gradient_checkpointing = None: fbm.for_training(model)
    seen = []

    class _Trainer:
        args = types.SimpleNamespace(gradient_checkpointing = False)

        def train(self):
            seen.append(os.environ.get(FLAG))

    trainer = _Trainer()
    trainer.model = model
    train = _prepare_for_training_mode()(_Trainer.train)
    model.eval()
    train(trainer)
    assert model.training is False
    os.environ[FLAG] = "1"
    train(trainer)
    assert seen == ["0", "1"]


def _logits_check():
    tree = ast.parse((MODELS / "rl.py").read_text(encoding = "utf-8"))
    return next(
        n.value.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "logits_check" for t in n.targets)
    )


@pytest.mark.parametrize("user_flag", [None, "1"])
def test_compute_metrics_auto_flag_does_not_outlive_the_trainer_init(fbm, monkeypatch, user_flag):
    # compute_metrics forces "1" while building the trainer; training (and the next trainer's
    # padding-free gate) must see the user's value, not a leaked "1".
    if user_flag is not None:
        monkeypatch.setenv(FLAG, user_flag)
    exec(_logits_check(), {"os": os, "compute_metrics": object()})
    assert os.environ[FLAG] == "1"
    fbm.for_training(_Model())
    assert os.environ[FLAG] == (user_flag or "0")
