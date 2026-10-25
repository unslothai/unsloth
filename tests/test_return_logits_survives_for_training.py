# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A user's UNSLOTH_RETURN_LOGITS=1 must survive trainer.train() (#3071): mode switches never write the flag
beyond defaulting it to "0". Methods are lifted out with ``ast`` so this runs without ``import unsloth``.
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import os
import sys
import types
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"
FLAG = "UNSLOTH_RETURN_LOGITS"


class _Namespace(dict):
    def __missing__(self, name):
        return getattr(builtins, name, None)


class _StrictNamespace(dict):
    # for_training has no optional globals: a helper it gained and the stubs below lack must
    # fail as a NameError naming it, not as "'NoneType' object is not callable" (#13374).
    def __missing__(self, name):
        try:
            return getattr(builtins, name)
        except AttributeError:
            raise NameError(f"for_training uses {name!r}: stub it below") from None


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
    # Follow the lifted code into every other _utils.py helper it names, so a guard either method
    # gains (for_inference's tensor parallel refusal, #15137) runs here instead of resolving to None.
    helpers = {n.name: n for n in utils.body if isinstance(n, ast.FunctionDef)}
    lifted = {n.name for n in wanted}
    todo = list(wanted)
    while todo:
        for ref in ast.walk(todo.pop()):
            if isinstance(ref, ast.Name) and ref.id in helpers and ref.id not in lifted:
                node = helpers[ref.id]
                node.decorator_list = []
                lifted.add(node.name)
                wanted.insert(0, node)
                todo.append(node)
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
    return _fast_base_model()


def _cycle(fbm, model):
    # The trainer's own cycle: for_training in __init__, for_inference, for_training in train().
    fbm.for_training(model)
    fbm.for_inference(model)
    fbm.for_training(model)


@pytest.mark.parametrize("user_flag", ["0", "1"])
def test_mode_switches_keep_a_user_set_flag(fbm, monkeypatch, user_flag):
    monkeypatch.setenv(FLAG, user_flag)
    model = _Model()
    _cycle(fbm, model)
    fbm.for_inference(model)
    assert os.environ[FLAG] == user_flag


def test_flag_set_after_for_inference_is_kept(fbm):
    # model.generate() / for_inference() first, then the flag, then train(): the residual #3071 path.
    model = _Model()
    fbm.for_inference(model)
    assert FLAG not in os.environ
    os.environ[FLAG] = "1"
    fbm.for_training(model)
    assert os.environ[FLAG] == "1"


def test_lifted_for_inference_runs_its_real_helpers(fbm):
    # The lifted for_inference calls the real _utils.py guards, not a None the namespace made up.
    model = _Model()
    model._unsloth_tp = True
    with pytest.raises(RuntimeError, match = "tensor parallel"):
        fbm.for_inference(model)


def _fast_llama_for_training():
    tree = ast.parse((MODELS / "llama.py").read_text(encoding = "utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "for_training")
    fn.decorator_list = []
    namespace = _StrictNamespace(
        os = os,
        # for_training's helpers since #13374: one walk returns (checkpointing modules, apply).
        switch_training_mode = lambda model, training, defer = False: ([], lambda: None),
        resolve_training_gradient_checkpointing = lambda model, flag, checkpointing = None: False,
        arm_gradient_checkpointing = lambda model, checkpointing = None: False,
        set_module_gradient_checkpointing = lambda module, flag: None,
    )
    module = ast.fix_missing_locations(ast.Module(body = [fn], type_ignores = []))
    exec(compile(module, str(MODELS / "llama.py"), "exec"), namespace)
    return namespace["for_training"]


@pytest.mark.parametrize("before", [None, "0", "1"])
def test_fast_language_model_after_fastmodel_generate(fbm, monkeypatch, before):
    # FastModel-backed models (Qwen3.5, Gemma 3/4) run FastBaseModel.for_inference() in generate(),
    # then FastLanguageModel.for_training(): the flag is whatever the user left, "0" when unset.
    if before is not None:
        monkeypatch.setenv(FLAG, before)
    model = _Model()
    fbm.for_inference(model)
    _fast_llama_for_training()(model)
    assert os.environ.get(FLAG, "0") == (before or "0")
    os.environ[FLAG] = "1"
    _fast_llama_for_training()(model)
    assert os.environ[FLAG] == "1"


def test_unset_flag_defaults_to_zero_for_training(fbm):
    # Default training keeps the fused CE path after generation / GRPO rollouts.
    model = _Model()
    fbm.for_inference(model)
    fbm.for_training(model)
    assert os.environ[FLAG] == "0"
    fbm.for_inference(model)
    assert os.environ[FLAG] == "0"


def _template():
    tree = ast.parse((MODELS / "rl.py").read_text(encoding = "utf-8"))
    return next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "RLTrainer_replacement" for t in n.targets)
    )


def _prepare_for_training_mode():
    lines = _template().splitlines()
    start = lines.index("def prepare_for_training_mode(f):")
    end = lines.index("    return wrapper", start)
    namespace = _Namespace(os = os, functools = functools)
    # The template is filled with str.format: a literal brace in the block breaks every trainer.
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


def test_generated_trainer_init_never_writes_the_flag():
    # compute_metrics used to force "1" here, leaking into training and the next trainer; the eval
    # prediction_step forces it per call instead.
    tree = ast.parse((MODELS / "rl.py").read_text(encoding = "utf-8"))
    patcher = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_patch_trl_rl_trainers_impl"
    )
    assert FLAG not in ast.unparse(patcher)
    assert FLAG not in _template()
