# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""The off-mode offload_layers opt-in hint (UNSLOTH_OFFLOAD_LAYER_MAJOR unset, gradient_accumulation_steps > 1) walks
model.modules() once per train() call for a model without a block swap, not once per optimizer step, and still prints
exactly once when a swap exists, including one built after training started (CPU, no transformers)."""

import ast
import importlib.util
import os
import sys
import types

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HINT = "It is opt-in: set UNSLOTH_OFFLOAD_LAYER_MAJOR=1"


@pytest.fixture
def mod(monkeypatch):
    engine = types.ModuleType("unsloth_zoo.block_swap_layer_major")
    for name in (
        "boundary_bytes",
        "layer_major_mode",
        "run_layer_major",
        "stall_fraction",
        "swap_totals",
    ):
        setattr(engine, name, lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.block_swap_layer_major", engine)
    spec = importlib.util.spec_from_file_location(
        "layer_major_hint_under_test",
        os.path.join(_ROOT, "unsloth", "models", "_layer_major_ga.py"),
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    monkeypatch.delenv("UNSLOTH_OFFLOAD_LAYER_MAJOR", raising = False)
    assert m._mode() == "off"
    return m


def _trainer_class(m, on_step = None):
    class T:
        def __init__(self, model):
            self.model = model

        def get_batch_samples(
            self,
            epoch_iterator,
            num_batches,
            device = None,
        ):
            return [object() for _ in range(num_batches)], 6

        def training_step(
            self,
            model,
            inputs,
            num_items_in_batch = None,
        ):
            return None

        def train(
            self,
            steps = 8,
            ga = 3,
        ):
            for step in range(steps):
                if on_step is not None:
                    on_step(self, step)
                self.get_batch_samples(None, ga)

    m.patch_layer_major_gradient_accumulation(T)
    assert getattr(T.training_step, "_unsloth_layer_major", False)
    return T


def _model():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
    return model


def _count_walks(m, monkeypatch):
    calls = [0]
    real = m._block_swap_layers

    def counted(model):
        calls[0] += 1
        return real(model)

    monkeypatch.setattr(m, "_block_swap_layers", counted)
    return calls


def test_no_swap_walks_once_per_train_call(mod, monkeypatch, capsys):
    walks = _count_walks(mod, monkeypatch)
    trainer = _trainer_class(mod)(_model())
    trainer.train(steps = 10)
    assert walks[0] == 1
    trainer.train(steps = 10)
    assert walks[0] == 2
    # gradient_accumulation_steps = 1 never asks.
    trainer.train(steps = 5, ga = 1)
    assert walks[0] == 2
    assert _HINT not in capsys.readouterr().out


def test_hint_prints_once_with_a_swap(mod, monkeypatch, capsys):
    walks = _count_walks(mod, monkeypatch)
    model = _model()
    model.layers._unsloth_block_swap = object()
    trainer = _trainer_class(mod)(model)
    trainer.train(steps = 6)
    assert capsys.readouterr().out.count(_HINT) == 1
    assert walks[0] == 1
    # Once per train() call, as before.
    trainer.train(steps = 6)
    assert capsys.readouterr().out.count(_HINT) == 1


def test_hint_prints_for_a_swap_built_mid_training(mod, monkeypatch, capsys):
    walks = _count_walks(mod, monkeypatch)

    def on_step(trainer, step):
        if step == 3:
            # What unsloth.models._utils._new_block_swap does before every attach / auto re-plan.
            mod.note_block_swap_installed()
            trainer.model.layers._unsloth_block_swap = object()

    trainer = _trainer_class(mod, on_step)(_model())
    trainer.train(steps = 8)
    assert capsys.readouterr().out.count(_HINT) == 1
    assert walks[0] == 2


def test_a_different_model_is_walked(mod, monkeypatch, capsys):
    def on_step(t, step):
        if step == 2:
            swapped = _model()
            swapped.layers._unsloth_block_swap = object()
            t.model = swapped

    trainer = _trainer_class(mod, on_step)(_model())
    trainer.train(steps = 4)
    assert capsys.readouterr().out.count(_HINT) == 1


def test_new_block_swap_bumps_the_install_counter(mod, monkeypatch):
    path = os.path.join(_ROOT, "unsloth", "models", "_utils.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_new_block_swap"
    )
    pkg = types.ModuleType("_hint_fake_pkg")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "_hint_fake_pkg", pkg)
    monkeypatch.setitem(sys.modules, "_hint_fake_pkg._layer_major_ga", mod)

    class BlockSwap:
        stats = None

        def __init__(self, layers, n, *args, **kwargs):
            self.layers = layers

    ns = {
        "__name__": "_hint_fake_pkg._utils",
        "__package__": "_hint_fake_pkg",
        "BlockSwap": BlockSwap,
    }
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    before = mod._BLOCK_SWAP_INSTALLS[0]
    assert isinstance(ns["_new_block_swap"]([], 1, 2, placement = "tail"), BlockSwap)
    assert mod._BLOCK_SWAP_INSTALLS[0] == before + 1


def test_every_swap_attach_builds_through_new_block_swap():
    """Each function under unsloth/ that sets `x._unsloth_block_swap = <not None>` builds the swap through
    _new_block_swap, so the hint's remembered miss is invalidated by every install."""
    offenders = []
    for root, _, files in os.walk(os.path.join(_ROOT, "unsloth")):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            tree = ast.parse(open(path, encoding = "utf-8").read())
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                sets = [
                    n
                    for n in ast.walk(fn)
                    if isinstance(n, ast.Assign)
                    and not (isinstance(n.value, ast.Constant) and n.value.value is None)
                    and any(
                        isinstance(t, ast.Attribute) and t.attr == "_unsloth_block_swap"
                        for t in n.targets
                    )
                ]
                if not sets:
                    continue
                builds = any(
                    isinstance(n, ast.Call)
                    and getattr(n.func, "id", getattr(n.func, "attr", None))
                    in ("_new_block_swap", "_attach_block_swap")
                    for n in ast.walk(fn)
                )
                if not builds:
                    offenders.append(f"{os.path.relpath(path, _ROOT)}:{fn.name}")
    assert offenders == []
