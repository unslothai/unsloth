# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""TRL >= 1.15 SFTTrainer.prediction_step must still reach Unsloth's Trainer.prediction_step; ast-loaded, CPU-only."""

from __future__ import annotations

import ast
import contextlib
import inspect
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"

# TRL 1.15.0 to 1.18.2, verbatim: a full override with no super() call.
TRL_115_PREDICTION_STEP = """\
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs, return_outputs=False)
                logits, labels = None, None
            else:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                logits, labels = outputs.logits, inputs.get("labels")
        return loss, logits, labels
"""

# TRL 1.14.0, verbatim: already calls super().
TRL_114_PREDICTION_STEP = """\
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Preserve the eval loop intent so compute_loss can decide whether logits are needed.
        inputs["_prediction_loss_only"] = prediction_loss_only
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
"""


def _patched(function):
    # Every RL_FUNCTIONS["sft_trainer"] edit, in registration order, as patch_functions applies them.
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    names = [
        node.value.args[0].id
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "RL_FUNCTIONS['sft_trainer'].append"
    ]
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    for name in names:
        function = namespace[name]("prediction_step", function)
    return function


class _PatchedTrainer:
    # Stands in for Trainer.prediction_step, which PatchRL replaces with unsloth_prediction_step.
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        return ("unsloth_prediction_step", prediction_loss_only, ignore_keys)

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return contextlib.nullcontext()

    def compute_loss(self, *args, **kwargs):
        raise AssertionError(
            "TRL's own prediction_step ran, so compute_metrics / predict get EmptyLogits"
        )


@pytest.mark.parametrize("prediction_loss_only", [True, False])
def test_trl_115_prediction_step_reaches_the_unsloth_step(prediction_loss_only):
    source = _patched(TRL_115_PREDICTION_STEP)
    # Compiled into a class body, as in the generated trainer, so the zero-argument super() resolves.
    namespace = {"torch": torch, "_PatchedTrainer": _PatchedTrainer}
    exec(
        "class Generated(_PatchedTrainer):\n" + textwrap.indent(textwrap.dedent(source), "    "),
        namespace,
    )
    out = namespace["Generated"]().prediction_step(
        None, {"labels": torch.zeros(1)}, prediction_loss_only, ignore_keys = ["past_key_values"]
    )
    assert out == ("unsloth_prediction_step", prediction_loss_only, ["past_key_values"])


def test_trl_114_prediction_step_is_left_alone():
    assert _patched(TRL_114_PREDICTION_STEP) == TRL_114_PREDICTION_STEP
