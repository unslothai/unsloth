# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""gc.freeze() scope around Trainer.train(): CPU only, the module is loaded by path (no unsloth import)."""

import ast
import gc
import importlib.util
import os
import weakref

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCOPE_PATH = os.path.join(_ROOT, "unsloth", "models", "_gc_train_scope.py")
_UTILS_PATH = os.path.join(_ROOT, "unsloth", "models", "_utils.py")


def _load_scope():
    spec = importlib.util.spec_from_file_location("unsloth_gc_train_scope_under_test", _SCOPE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scope(monkeypatch):
    monkeypatch.delenv("UNSLOTH_GC_FREEZE", raising = False)
    if gc.get_freeze_count() != 0:
        pytest.skip("something in this process already froze the heap")
    module = _load_scope()
    yield module
    gc.unfreeze()
    assert module._depth == 0


class _Cycle:
    def __init__(self):
        self.me = self


def _make_trainer(scope, body = None):
    class FakeTrainer:
        def train(self, *args, **kwargs):
            self.seen = gc.get_freeze_count()
            if body is not None:
                return body(self, *args, **kwargs)
            return "done"

    assert scope.patch_trainer_train_gc_scope(FakeTrainer) is True
    return FakeTrainer


def test_frozen_during_train_unfrozen_after(scope):
    Trainer = _make_trainer(scope)
    t = Trainer()
    assert t.train() == "done"
    assert t.seen > 0
    assert gc.get_freeze_count() == 0


@pytest.mark.parametrize("exc_type", [RuntimeError, KeyboardInterrupt])
def test_unfrozen_after_exception(scope, exc_type):
    def body(self):
        assert gc.get_freeze_count() > 0
        raise exc_type("boom")

    Trainer = _make_trainer(scope, body)
    with pytest.raises(exc_type):
        Trainer().train()
    assert gc.get_freeze_count() == 0
    assert scope._depth == 0


def test_nested_train_freezes_once(scope, monkeypatch):
    calls = {"freeze": 0, "unfreeze": 0}
    real_freeze, real_unfreeze = gc.freeze, gc.unfreeze

    def freeze():
        calls["freeze"] += 1
        real_freeze()

    def unfreeze():
        calls["unfreeze"] += 1
        real_unfreeze()

    monkeypatch.setattr(scope.gc, "freeze", freeze)
    monkeypatch.setattr(scope.gc, "unfreeze", unfreeze)

    inner_after = {}

    def body(self, depth = 0):
        if depth < 2:
            # Same shape as the RL resume guard / per-instance wrappers: call train() again.
            type(self).train(self, depth + 1)
            inner_after[depth] = gc.get_freeze_count()
        return depth

    Trainer = _make_trainer(scope, body)
    Trainer().train()
    assert calls == {"freeze": 1, "unfreeze": 1}
    # The inner returns must not have unfrozen the outer scope.
    assert inner_after[0] > 0 and inner_after[1] > 0
    assert gc.get_freeze_count() == 0


def test_wrapper_captured_after_install_freezes_once(scope, monkeypatch):
    calls = {"freeze": 0}
    real_freeze = gc.freeze

    def freeze():
        calls["freeze"] += 1
        real_freeze()

    monkeypatch.setattr(scope.gc, "freeze", freeze)
    Base = _make_trainer(scope)

    class Sub(Base):
        pass

    original = Sub.train  # what rl.py's resume guard captures

    def guarded(self, *args, **kwargs):
        with scope.unsloth_gc_train_scope():
            return original(self, *args, **kwargs)

    Sub.train = guarded
    t = Sub()
    t.train()
    assert t.seen > 0
    assert calls["freeze"] == 1
    assert gc.get_freeze_count() == 0


def test_external_freeze_is_left_alone(scope):
    gc.collect()
    gc.freeze()
    before = gc.get_freeze_count()
    assert before > 0
    Trainer = _make_trainer(scope)
    t = Trainer()
    t.train()
    # Not ours: still frozen after train(), and train() did not freeze anything on top.
    assert t.seen == before
    assert gc.get_freeze_count() == before


def test_gc_enabled_state_preserved(scope):
    was = gc.isenabled()
    try:
        gc.disable()
        Trainer = _make_trainer(scope)
        t = Trainer()
        t.train()
        assert t.seen > 0
        assert not gc.isenabled()
    finally:
        if was:
            gc.enable()
    Trainer().train()
    assert gc.isenabled() == was


@pytest.mark.parametrize("value", ["0", "false", "OFF"])
def test_opt_out(scope, monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_GC_FREEZE", value)
    Trainer = _make_trainer(scope)
    t = Trainer()
    t.train()
    assert t.seen == 0


def test_install_is_idempotent(scope):
    Trainer = _make_trainer(scope)
    wrapped = Trainer.train
    assert scope.patch_trainer_train_gc_scope(Trainer) is False
    assert Trainer.train is wrapped

    class Sub(Trainer):
        pass

    # A subclass without its own train() is covered through the base; nothing to wrap.
    assert scope.patch_trainer_train_gc_scope(Sub) is False
    assert "train" not in Sub.__dict__


def test_cycle_dropped_after_train_is_collected(scope):
    obj = _Cycle()
    ref = weakref.ref(obj)

    def body(self):
        # While frozen, a dropped pre-existing cycle is not collectable: this is why we must unfreeze.
        holder = self.holder
        self.holder = None
        del holder
        gc.collect()
        self.alive_while_frozen = ref() is not None

    Trainer = _make_trainer(scope, body)
    t = Trainer()
    t.holder = obj
    del obj
    t.train()
    assert t.alive_while_frozen
    # Notebook pattern after training: `del model; gc.collect()` must free it.
    gc.collect()
    assert ref() is None


def test_utils_installs_on_transformers_trainer_at_import():
    with open(_UTILS_PATH, encoding = "utf-8") as f:
        tree = ast.parse(f.read())
    # A module-level call, so it runs before llama.py's PatchFastRL captures `train` for the
    # RL resume guard (a later install would be bypassed by that captured reference).
    found = False
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "_patch_trainer_train_gc_scope":
                found = ast.unparse(node.value.args[0]) == "transformers.trainer.Trainer"
    assert found
