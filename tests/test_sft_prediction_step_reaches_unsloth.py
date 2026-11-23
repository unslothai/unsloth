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


def _skip_check():
    rl_path = SOURCE_PATH.with_name("rl.py")
    tree = ast.parse(rl_path.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_sft_prediction_step_skips_unsloth"
    )
    namespace = {"inspect": inspect}
    exec(compile(ast.Module(body = [node], type_ignores = []), str(rl_path), "exec"), namespace)
    return namespace["_sft_prediction_step_skips_unsloth"]


def test_skip_check_names_a_base_class_that_shadows_the_unsloth_step():
    transformers = pytest.importorskip("transformers")
    skips = _skip_check()

    class ReachesTrainer(transformers.Trainer):
        def prediction_step(self, *args, **kwargs):
            return super().prediction_step(*args, **kwargs)

    class Shadows(transformers.Trainer):
        def prediction_step(self, *args, **kwargs):
            return None, None, None

    class Generated(Shadows):
        def compute_loss(self, *args, **kwargs):
            return None

    assert skips(ReachesTrainer) is None
    assert skips(type("Plain", (transformers.Trainer,), {})) is None
    assert skips(Generated).endswith("Shadows")
    assert skips(type("Wrapper", (ReachesTrainer, Shadows), {})).endswith("Shadows")


class _CollectingTrainer(_PatchedTrainer):
    # Like unsloth_prediction_step: every output field except loss becomes the logits payload.
    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
    ):
        loss, outputs = self.compute_loss(model, inputs, return_outputs = True)
        return loss, tuple(v for k, v in outputs.items() if k != "loss"), inputs["labels"]

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
    ):
        outputs = {
            "loss": torch.tensor(1.0),
            "aux_loss": torch.tensor(0.5),
            "logits": torch.ones(1, 2, 3, requires_grad = True),
            "router_logits": (torch.zeros(2, 4),),
        }
        return (outputs["loss"], outputs) if return_outputs else outputs["loss"]


def test_trl_115_prediction_step_keeps_the_logits_only_payload():
    source = _patched(TRL_115_PREDICTION_STEP)
    namespace = {"torch": torch, "_CollectingTrainer": _CollectingTrainer}
    exec(
        "class Generated(_CollectingTrainer):\n" + textwrap.indent(textwrap.dedent(source), "    "),
        namespace,
    )
    trainer = namespace["Generated"]()
    loss, logits, labels = trainer.prediction_step(None, {"labels": torch.zeros(1)}, False)
    assert isinstance(logits, torch.Tensor) and logits.shape == (1, 2, 3)
    assert not logits.requires_grad
    assert "compute_loss" not in vars(trainer)
