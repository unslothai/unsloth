# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Eval must not divide by current_gradient_accumulation_steps, which Trainer leaves stale in eval."""

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
        (True, 4, 4),
        (True, 1, 1),
        # Evaluating mid-run: the stale training window must not reach the loss.
        (False, 4, 1),
        (False, 16, 1),
        (False, None, 1),
        (True, None, 1),
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
