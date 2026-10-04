# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The eval prediction_step forces UNSLOTH_RETURN_LOGITS=1 and must put the caller's value back.

UNSLOTH_RETURN_LOGITS=1 also blocks packing and padding-free when the next trainer is built, so
an evaluate() that raised part way used to leave logits forced on for the rest of the process.
The step is lifted out of PatchRL with ``ast`` so this runs without ``import unsloth``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _prediction_step():
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    patch_rl = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "PatchRL")
    step = next(
        n
        for n in ast.walk(patch_rl)
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_prediction_step"
    )
    namespace = {"os": os, "torch": torch, "nested_detach": lambda x: x}
    exec(compile(ast.Module(body = [step], type_ignores = []), str(SOURCE), "exec"), namespace)
    return namespace["unsloth_prediction_step"]


class _Trainer:
    label_names = ["labels"]
    can_return_loss = False
    args = SimpleNamespace(device = "cpu", past_index = -1)

    def __init__(self, compute_loss):
        self.model = SimpleNamespace()
        self.compute_loss = compute_loss
        self.seen = []

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        import contextlib
        return contextlib.nullcontext()

    def _get_num_items_in_batch(self, batches, device):
        return None


@pytest.mark.parametrize("before", [None, "0", "1"])
def test_a_failed_eval_step_restores_the_callers_setting(monkeypatch, before):
    if before is None:
        monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", before)
    seen = []

    def compute_loss(model, inputs, **kwargs):
        seen.append(os.environ.get("UNSLOTH_RETURN_LOGITS"))
        raise RuntimeError("CUDA out of memory")

    trainer = _Trainer(compute_loss)
    with pytest.raises(RuntimeError, match = "out of memory"):
        _prediction_step()(trainer, trainer.model, {"labels": torch.zeros(1)}, True, None)

    assert seen == ["1"], "the step no longer forces logits on while it runs"
    # An unset variable comes back as "0", the value the step treats as the default.
    assert os.environ.get("UNSLOTH_RETURN_LOGITS") == (before or "0")


def test_a_successful_eval_step_still_restores_it(monkeypatch):
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "0")

    def compute_loss(model, inputs, **kwargs):
        return torch.tensor(1.0), (None, torch.zeros(1, 2))

    trainer = _Trainer(compute_loss)
    loss, logits, labels = _prediction_step()(
        trainer, trainer.model, {"labels": torch.zeros(1)}, True, None
    )
    assert float(loss) == 1.0 and logits is None and labels is None
    assert os.environ["UNSLOTH_RETURN_LOGITS"] == "0"
