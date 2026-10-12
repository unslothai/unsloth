# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The GRPO loss must not divide an eval pass by the training accumulation window.

Every GRPO loss type except dapo/cispo/vespo ends with

    loss = loss / current_gradient_accumulation_steps

and TRL sets that divisor to `self.current_gradient_accumulation_steps` in train mode and to
1.0 in eval, because an eval pass accumulates nothing. `Trainer` assigns the attribute inside
the training loop and never clears it, so during an in-training evaluation it still holds the
training window's size: reading it unconditionally makes the reported `eval_loss` smaller than
the train loss by exactly `gradient_accumulation_steps`, and moves it when that setting changes.

The helper is lifted from `unsloth/models/rl_replacements.py` with `ast` so the test tracks the
shipped source without importing unsloth, the same trick as `tests/_grpo_dispatch_source.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
HELPER_NAME = "_unsloth_grpo_accumulation_steps"


def _load_helper():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    found = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == HELPER_NAME
    ]
    assert len(found) == 1, f"expected one module-level def {HELPER_NAME}, found {len(found)}"
    namespace: dict = {}
    exec(compile(ast.Module(body = found, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    return namespace[HELPER_NAME]


class _Model:
    def __init__(self, training):
        self.training = training


class _Trainer:
    def __init__(
        self,
        training = None,
        steps = None,
    ):
        if training is not None:
            self.model = _Model(training)
        if steps is not None:
            self.current_gradient_accumulation_steps = steps


@pytest.mark.parametrize(
    ("training", "steps", "expected"),
    [
        # Training: the accumulation window is the divisor, as before.
        (True, 4, 4),
        (True, 1, 1),
        # Evaluating mid-run: the stale training window must not reach the loss.
        (False, 4, 1),
        (False, 16, 1),
        # Standalone evaluate(): the attribute never existed (#2464).
        (False, None, 1),
        (True, None, 1),
        # No model to ask: keep reading the attribute rather than guessing.
        (None, 4, 4),
    ],
)
def test_grpo_accumulation_divisor_is_one_outside_training(training, steps, expected):
    assert _load_helper()(_Trainer(training, steps)) == expected


def test_compute_loss_uses_the_helper():
    """The generated trainer's `compute_loss` must go through the helper, not the raw attribute."""
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"), filename = str(SOURCE_PATH))
    outer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "grpo_trainer_compute_loss"
    )
    compute_loss = next(
        node
        for node in ast.walk(outer)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_loss"
    )
    calls = [
        node.func.id
        for node in ast.walk(compute_loss)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert HELPER_NAME in calls
    reads = [
        node
        for node in ast.walk(compute_loss)
        if isinstance(node, ast.Attribute) and node.attr == "current_gradient_accumulation_steps"
    ]
    assert reads == [], "compute_loss reads the attribute directly, bypassing the eval guard"
