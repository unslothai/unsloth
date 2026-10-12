# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""switch_training_mode replaces model.train() / eval() plus three tree scans in for_inference /
for_training (every GRPO step), so its result must be exactly model.train(mode)'s. Lifted out of
_utils.py with ``ast`` so this runs without ``import unsloth``.
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

UTILS = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
HELPERS = ("_mode_switch_class_info", "_has_gradient_checkpointing", "switch_training_mode")


class _Namespace(dict):
    def __missing__(self, name):
        return getattr(builtins, name)


@pytest.fixture(scope = "module")
def switch():
    tree = ast.parse(UTILS.read_text(encoding = "utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {n.name for n in nodes} == set(HELPERS)
    namespace = _Namespace(torch = torch, functools = functools, inspect = inspect)
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(UTILS), "exec"), namespace)
    return namespace["switch_training_mode"]


class _Layer(nn.Module):
    gradient_checkpointing = False  # class attribute, as GradientCheckpointingLayer

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.drop = nn.Dropout(0.1)


class _Backbone(nn.Module):
    def __init__(self, shared):
        super().__init__()
        self.gradient_checkpointing = False  # instance attribute, as Qwen3Model
        self.layers = nn.ModuleList([_Layer(), _Layer()])
        self.shared = shared
        self.missing = None  # an empty submodule slot


class _Delegating(nn.Module):
    """A PEFT-style wrapper: attribute reads fall through to the wrapped model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _build():
    shared = nn.Linear(4, 4)
    backbone = _Backbone(shared)
    backbone.head = shared  # the same module reachable twice
    return _Delegating(backbone)


def _flags(model):
    return [(name, m.training) for name, m in model.named_modules(remove_duplicate = False)]


@pytest.mark.parametrize("start", [True, False])
@pytest.mark.parametrize("training", [True, False])
def test_matches_model_train(switch, start, training):
    ref, ours = _build(), _build()
    ref.train(start)
    ours.train(start)
    ref.train(training)
    switch(ours, training)
    assert _flags(ours) == _flags(ref)


def test_returns_exactly_the_modules_hasattr_finds(switch):
    model = _build()
    expected = [m for m in model.modules() if hasattr(m, "gradient_checkpointing")]
    got = switch(model, True)
    assert sorted(map(id, got)) == sorted(map(id, expected))
    assert model in got  # the delegating wrapper answers hasattr through the wrapped model


def test_drops_fast_lora_only_when_entering_training(switch):
    model = _build()
    weight = model.model.layers[0].proj.weight
    weight._fast_lora = weight.detach().clone()
    switch(model, False)
    assert hasattr(weight, "_fast_lora")
    switch(model, True)
    assert not hasattr(weight, "_fast_lora")


def test_a_class_with_its_own_train_gets_the_real_call(switch):
    calls = []

    class _Custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(2, 2)

        def train(self, mode = True):
            calls.append(mode)
            return super().train(mode)

    model = nn.Sequential(_Custom(), nn.Dropout())
    switch(model, False)
    assert calls == [False]
    assert not any(m.training for m in model.modules())


def test_a_class_with_its_own_setattr_still_sees_the_write(switch):
    seen = []

    class _Watching(nn.Linear):
        def __setattr__(self, name, value):
            if name == "training":
                seen.append(value)
            super().__setattr__(name, value)

    model = nn.Sequential(_Watching(2, 2))
    seen.clear()
    switch(model, False)
    assert seen == [False]
    assert model[0].training is False


@pytest.mark.parametrize("use_kernels", [False, True])
def test_pretrained_model_recasts_kernels_only_when_they_are_on(switch, monkeypatch, use_kernels):
    transformers = pytest.importorskip("transformers")
    if transformers.PreTrainedModel.train is nn.Module.train:
        pytest.skip("this transformers has no PreTrainedModel.train override")

    class _Config(transformers.PretrainedConfig):
        model_type = "unsloth_switch_test"

    class _Model(transformers.PreTrainedModel):
        config_class = _Config

        def __init__(self, config):
            super().__init__(config)
            self.lin = nn.Linear(2, 2)

    model = _Model(_Config())
    model.__dict__["_use_kernels"] = use_kernels
    recasts = []
    monkeypatch.setattr(
        _Model, "set_use_kernels", lambda self, *a, **k: recasts.append(a), raising = False
    )
    model.train(True)
    switch(model, False)
    assert not model.training and not model.lin.training
    assert recasts == ([(True,)] if use_kernels else [])


def _reference_and_ours(build, switch, training):
    ref, ours = build(), build()
    ref.train() if training else ref.eval()
    switch(ours, training)
    return ref, ours


def _keep_eval(module, mode = True):
    # What _unsloth_freeze_norm_running_stats installs on frozen running-stat norms.
    return torch.nn.Module.train(module, False)


@pytest.mark.parametrize("training", [True, False])
def test_a_per_instance_train_partial_is_honoured(switch, training):
    def build():
        model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
        model[1].train = functools.partial(_keep_eval, model[1])
        return model

    ref, ours = _reference_and_ours(build, switch, training)
    assert _flags(ours) == _flags(ref)
    assert ours[1].training is False


def test_a_root_eval_override_runs_on_inference(switch):
    calls = []

    class _Root(nn.Sequential):
        def eval(self):
            calls.append("eval")
            return super().eval()

    model = _Root(nn.Linear(2, 2))
    switch(model, False)
    assert calls == ["eval"] and not model[0].training


def test_a_property_backed_training_flag_goes_through_its_setter(switch):
    class _Prop(nn.Linear):
        @property
        def training(self):
            return self.__dict__.get("_mode", True)

        @training.setter
        def training(self, value):
            self.__dict__["_mode"] = value

    model = nn.Sequential(_Prop(2, 2))
    switch(model, False)
    assert model[0].training is False


def test_a_custom_child_traversal_is_followed(switch):
    class _Hiding(nn.Module):
        def __init__(self):
            super().__init__()
            self.visible = nn.Dropout()
            self.hidden = nn.Dropout()

        def named_children(self):
            yield "visible", self.visible

    for training in (True, False):

        def build():
            model = nn.Sequential(_Hiding())
            for module in model.modules():
                module.training = not training
            return model

        ref, ours = _reference_and_ours(build, switch, training)
        assert _flags(ours) == _flags(ref)
        assert ours[0].hidden.training is (not training)


def test_pretrained_train_override_only_recasts_kernels():
    """Drift guard: the flat path treats PreTrainedModel.train as nn.Module.train unless use_kernels is
    set. If transformers adds anything else there, this fails and the flat path must learn it."""
    transformers = pytest.importorskip("transformers")
    train = transformers.PreTrainedModel.train
    if train is nn.Module.train:
        pytest.skip("this transformers has no PreTrainedModel.train override")
    tree = ast.parse(inspect.cleandoc("\n" + inspect.getsource(train)))
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert attrs <= {"training", "train", "use_kernels", "set_use_kernels"}, attrs
    assert transformers.PreTrainedModel.eval is nn.Module.eval or "train" in inspect.getsource(
        transformers.PreTrainedModel.eval
    )
