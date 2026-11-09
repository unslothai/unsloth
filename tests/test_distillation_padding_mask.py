# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Distillation's direct backbone forward must keep the left-padding mask on the training student."""

from __future__ import annotations

import ast
import functools
import inspect
from pathlib import Path

import pytest

RL_PY = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _wrap():
    tree = ast.parse(RL_PY.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_wrap_distillation_padding_mask"
    )
    namespace = {"functools": functools}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(RL_PY), "exec"), namespace)
    return namespace["_wrap_distillation_padding_mask"]


class _Module:
    def __init__(
        self,
        *children,
        base = False,
    ):
        self.children = children
        if base:
            self.embed_tokens = object()

    def modules(self):
        yield self
        for child in self.children:
            yield from child.modules()


def test_flag_set_for_the_call_then_restored():
    fresh, preset = _Module(base = True), _Module(base = True)
    preset._unsloth_keep_padding_mask = False
    wrapper = _Module(_Module(fresh), preset)
    seen = {}

    class _Trainer:
        def _get_last_hidden_state(
            self, unwrapped_model, input_ids, attention_mask, logits_to_keep
        ):
            seen["fresh"] = fresh._unsloth_keep_padding_mask
            seen["preset"] = preset._unsloth_keep_padding_mask
            seen["wrapper"] = hasattr(wrapper, "_unsloth_keep_padding_mask")
            return "hidden"

    _wrap()(_Trainer)
    assert _Trainer()._get_last_hidden_state(wrapper, 1, 2, 3) == "hidden"
    assert seen == {"fresh": True, "preset": True, "wrapper": False}
    assert not hasattr(fresh, "_unsloth_keep_padding_mask")
    assert preset._unsloth_keep_padding_mask is False


def test_flag_restored_on_error_and_wrapped_once():
    base = _Module(base = True)

    class _Trainer:
        def _get_last_hidden_state(self, unwrapped_model, *args):
            assert base._unsloth_keep_padding_mask is True
            raise RuntimeError("oom")

    wrap = _wrap()
    wrap(_Trainer)
    first = _Trainer._get_last_hidden_state
    wrap(_Trainer)
    assert _Trainer._get_last_hidden_state is first
    with pytest.raises(RuntimeError):
        _Trainer()._get_last_hidden_state(_Module(base))
    assert not hasattr(base, "_unsloth_keep_padding_mask")


def test_trainer_without_the_helper_is_left_alone():
    class _Trainer:
        pass

    _wrap()(_Trainer)
    assert not hasattr(_Trainer, "_get_last_hidden_state")


def test_trl_helper_takes_the_model_first():
    distillation = pytest.importorskip("trl.trainer.distillation_trainer")
    helper = getattr(distillation.DistillationTrainer, "_get_last_hidden_state", None)
    if helper is None:
        pytest.skip("TRL DistillationTrainer without _get_last_hidden_state")
    params = list(inspect.signature(helper).parameters)
    assert params[:2] == ["self", "unwrapped_model"]
