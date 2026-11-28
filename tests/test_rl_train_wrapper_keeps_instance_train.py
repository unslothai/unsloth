# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The generated trainer's post-init wraps train() in prepare_for_training_mode. An instance-level train set
before it (the context parallel wrapper bound with MethodType, or a plain closure an __init__ hook such as
optimizers/auto_fit.py put over that) must stay in the chain: falling back to the class method skipped the
context parallel train scope, so every cp rank read its own rows. Lifted out of the template with ``ast``, so
this runs without ``import unsloth``."""

import ast
import functools
import types
from pathlib import Path

RL = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _helpers():
    tree = ast.parse(RL.read_text(encoding = "utf-8"))
    template = next(
        n.value.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "RLTrainer_replacement" for t in n.targets)
    )
    lines = template.splitlines()
    start = lines.index("def _unsloth_train_to_wrap(self):")
    end = lines.index("    return instance_train", start)
    namespace = {"functools": functools}
    # The template is filled with str.format: a literal brace in the block breaks every trainer.
    exec("\n".join(lines[start : end + 1]).format(), namespace)
    return namespace["_unsloth_train_to_wrap"]


class _Trainer:
    def train(self, *args, **kwargs):
        return ("class", args, kwargs)


def _post_init(trainer):
    # The generated post-init line, with a pass-through prepare_for_training_mode.
    trainer.train = types.MethodType(_helpers()(trainer), trainer)
    return trainer


def test_no_instance_train_wraps_the_class_method():
    trainer = _post_init(_Trainer())
    assert trainer.train(1, k = 2) == ("class", (1,), {"k": 2})


def test_bound_instance_train_is_kept():
    calls = []

    def cp_train(self, *args, **kwargs):
        calls.append(self)
        return _Trainer.train(self, *args, **kwargs)

    trainer = _Trainer()
    trainer.train = types.MethodType(cp_train, trainer)
    _post_init(trainer)
    assert trainer.train(3) == ("class", (3,), {})
    assert calls == [trainer]


def test_closure_over_the_bound_train_is_kept():
    calls = []

    def cp_train(self, *args, **kwargs):
        calls.append("cp")
        return _Trainer.train(self, *args, **kwargs)

    trainer = _Trainer()
    trainer.train = types.MethodType(cp_train, trainer)
    inner = trainer.train

    def recorded(*args, **kwargs):  # an __init__ hook's closure: no __func__, trainer already bound
        calls.append("hook")
        return inner(*args, **kwargs)

    trainer.train = recorded
    _post_init(trainer)
    assert trainer.train(resume_from_checkpoint = None) == (
        "class",
        (),
        {"resume_from_checkpoint": None},
    )
    assert calls == ["hook", "cp"]
