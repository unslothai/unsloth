# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused-CE label smoothing proxy fidelity: tuple-output fallback and non-stock smoothers keep the stock result."""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")
zoo_ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
if not hasattr(zoo_ce, "unsloth_label_smoothing_override"):
    pytest.skip("unsloth_zoo without unsloth_label_smoothing_override", allow_module_level = True)

from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_unsloth_fused_label_smoothing_fallback",
    ROOT / "unsloth" / "models" / "fused_label_smoothing.py",
)
fls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fls)

VOCAB, HIDDEN = 31, 8


class _Config:
    model_type = "llama"
    output_router_logits = False


class LlamaForCausalLM(torch.nn.Module):
    """return_dict=False style: (loss, logits) with labels, (logits,) without. `fused` routes labels to the fused CE."""

    def __init__(self, fused):
        super().__init__()
        torch.manual_seed(0)
        self.config = _Config()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype = torch.float64)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False, dtype = torch.float64)
        self.fused = fused

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels = None,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        if labels is not None and self.fused:
            loss = zoo_ce.unsloth_fused_ce_loss(
                None, hidden, self.lm_head.weight, None, labels, n_items = None, torch_compile = False
            )
            return (loss, torch.empty(0))
        logits = self.lm_head(hidden)
        if labels is None:
            return (logits,)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1)
        )
        return (loss, logits)


class _FusedTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        parent = super().compute_loss
        return fls.fused_label_smoothing_compute_loss(
            self,
            model,
            inputs,
            return_outputs,
            num_items_in_batch,
            lambda: parent(
                model, inputs, return_outputs = return_outputs, num_items_in_batch = num_items_in_batch
            ),
        )


class DoubledLabelSmoother(LabelSmoother):
    def __call__(self, *args, **kwargs):
        return 2 * super().__call__(*args, **kwargs)


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (2, 7), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _run(
    cls,
    fused,
    tmp_path,
    smoother = None,
):
    model = LlamaForCausalLM(fused)
    args = TrainingArguments(
        output_dir = str(tmp_path), label_smoothing_factor = 0.1, report_to = [], use_cpu = True
    )
    trainer = cls(model = model, args = args)
    if smoother is not None:
        trainer.label_smoother = smoother
    model.train()
    loss = trainer.compute_loss(model, _batch())
    loss.backward()
    return loss.detach().double(), [p.grad.double() for p in model.parameters()]


def test_tuple_output_fallback_uses_logits_not_loss(tmp_path):
    ref_loss, ref_grads = _run(Trainer, False, tmp_path)
    before = dict(fls.STATS)
    loss, grads = _run(_FusedTrainer, False, tmp_path)
    assert fls.STATS["fallback"] == before["fallback"] + 1 and fls.STATS["fused"] == before["fused"]
    torch.testing.assert_close(loss, ref_loss)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r)


def test_tuple_output_fused_path_still_engages(tmp_path):
    ref_loss, ref_grads = _run(Trainer, True, tmp_path)
    before = dict(fls.STATS)
    loss, grads = _run(_FusedTrainer, True, tmp_path)
    assert fls.STATS["fused"] == before["fused"] + 1
    torch.testing.assert_close(loss, ref_loss, rtol = 1e-6, atol = 1e-6)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r, rtol = 1e-6, atol = 1e-8)


def test_custom_smoother_is_not_replaced(tmp_path):
    ref_loss, ref_grads = _run(Trainer, True, tmp_path, DoubledLabelSmoother(epsilon = 0.1))
    before = dict(fls.STATS)
    loss, grads = _run(_FusedTrainer, True, tmp_path, DoubledLabelSmoother(epsilon = 0.1))
    assert fls.STATS["fused"] == before["fused"]
    torch.testing.assert_close(loss, ref_loss)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r)
