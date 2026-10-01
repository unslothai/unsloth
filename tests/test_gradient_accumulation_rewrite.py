# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import ast
import re
from pathlib import Path

import pytest

# Pre huggingface/transformers#35808 shape: backward() on the undivided loss.
OLD_TRAINING_STEP = """\
def _unsloth_training_step(self, model, inputs, num_items_in_batch = None):
    loss = self.compute_loss(model, inputs, num_items_in_batch = num_items_in_batch)
    if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        self.accelerator.backward(loss, **kwargs)
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
    return loss.detach()
"""


@pytest.fixture
def rewrite():
    # Read via ast: importing _utils.py needs an accelerator.
    source = Path(__file__).resolve().parents[1] / "unsloth/models/_utils.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "sub"
        and len(node.args) >= 2
        and isinstance(node.args[0], ast.Constant)
        and "accelerator" in str(node.args[0].value)
    ]
    assert len(calls) == 1, "expected one accelerator.backward rewrite in _utils.py"
    pattern = ast.literal_eval(calls[0].args[0])
    replacement = ast.literal_eval(calls[0].args[1])
    return lambda text: re.sub(pattern, replacement, text)


def test_rewrite_moves_the_backward_below_the_division(rewrite):
    rewritten = rewrite(OLD_TRAINING_STEP)

    assert rewritten != OLD_TRAINING_STEP, "the pattern no longer matches the shape it targets"
    assert rewritten.index("loss = loss / self.args.gradient_accumulation_steps") < rewritten.index(
        "self.accelerator.backward(loss, **kwargs)"
    ), "backward must run on the divided loss"
    assert "return loss.detach() / self.args.gradient_accumulation_steps" not in rewritten


def test_rewrite_keeps_the_source_parseable(rewrite):
    rewritten = rewrite(OLD_TRAINING_STEP)

    assert not any(character in rewritten for character in "\x01\x02\x03")
    ast.parse(rewritten)

    body = rewritten.splitlines()
    indent = {line: len(line) - len(line.lstrip()) for line in body if line.strip()}
    division = next(line for line in body if "loss = loss /" in line)
    backward = next(
        line for line in body if "accelerator.backward" in line and "scaled" not in line
    )
    assert indent[division] == indent[backward] + 4
