# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused-CE label smoothing vs stock Trainer LabelSmoother (oracle): same loss and grads, CPU fp64."""

import importlib.util
import os
import pathlib
import re

import pytest
import torch

pytest.importorskip("transformers")
zoo_ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
if not hasattr(zoo_ce, "unsloth_label_smoothing_override"):
    pytest.skip("unsloth_zoo without unsloth_label_smoothing_override", allow_module_level = True)

from transformers import Trainer, TrainingArguments

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_unsloth_fused_label_smoothing", ROOT / "unsloth" / "models" / "fused_label_smoothing.py"
)
fls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fls)

VOCAB, HIDDEN = 37, 8


class _Config:
    model_type = "llama"
    output_router_logits = False


class LlamaForCausalLM(torch.nn.Module):
    """Named like a causal LM so Trainer's LabelSmoother shifts. Labels: fused CE, empty logits (as llama.py)."""

    def __init__(self, fused = True):
        super().__init__()
        torch.manual_seed(0)
        self.config = _Config()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype = torch.float64)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = False, dtype = torch.float64)
        self.fused = fused
        self.logits_built = 0

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
                None,
                hidden,
                self.lm_head.weight,
                None,
                labels,
                n_items = kwargs.get("num_items_in_batch", None),
                torch_compile = False,
            )
            return {"loss": loss, "logits": torch.empty(0)}
        self.logits_built += 1
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB), labels[:, 1:].reshape(-1)
            )
        return out


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


def _trainer(cls, model, eps, tmp_path, **kw):
    args = TrainingArguments(
        output_dir = str(tmp_path), label_smoothing_factor = eps, report_to = [], use_cpu = True, **kw
    )
    return cls(model = model, args = args)


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (3, 9), generator = g)
    labels = ids.clone()
    labels[0, :4] = -100  # uneven prompt mask
    labels[1, :] = -100  # fully ignored row
    labels[2, -2:] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _loss_and_grads(trainer, model, num_items):
    model.train()
    model.zero_grad()
    loss = trainer.compute_loss(model, dict(_batch()), num_items_in_batch = num_items)
    loss.backward()
    return loss.detach().double(), [p.grad.clone() for p in model.parameters()]


@pytest.mark.parametrize("eps", [0.1, 1.0])
@pytest.mark.parametrize("num_items", [None, 11])
def test_matches_stock_label_smoother(eps, num_items, tmp_path):
    ref_model = LlamaForCausalLM()
    ref_loss, ref_grads = _loss_and_grads(
        _trainer(Trainer, ref_model, eps, tmp_path), ref_model, num_items
    )
    model = LlamaForCausalLM()
    trainer = _trainer(_FusedTrainer, model, eps, tmp_path)
    before = dict(fls.STATS)
    loss, grads = _loss_and_grads(trainer, model, num_items)
    assert fls.STATS["fused"] == before["fused"] + 1
    assert fls.STATS["fallback"] == before["fallback"]
    assert model.logits_built == 0 and ref_model.logits_built == 1
    torch.testing.assert_close(loss, ref_loss, rtol = 1e-6, atol = 1e-6)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r, rtol = 1e-6, atol = 1e-8)
    assert type(trainer.label_smoother).__name__ == "LabelSmoother"


def test_fallback_when_forward_skips_fused_ce(tmp_path):
    eps = 0.1
    ref_model = LlamaForCausalLM(fused = False)
    ref_loss, ref_grads = _loss_and_grads(
        _trainer(Trainer, ref_model, eps, tmp_path), ref_model, None
    )
    model = LlamaForCausalLM(fused = False)
    trainer = _trainer(_FusedTrainer, model, eps, tmp_path)
    before = dict(fls.STATS)
    loss, grads = _loss_and_grads(trainer, model, None)
    assert fls.STATS["fallback"] == before["fallback"] + 1 and fls.STATS["fused"] == before["fused"]
    torch.testing.assert_close(loss, ref_loss)
    for g, r in zip(grads, ref_grads):
        torch.testing.assert_close(g, r)
    _loss_and_grads(trainer, model, None)
    assert fls.STATS["fallback"] == before["fallback"] + 1 and fls.STATS["fused"] == before["fused"]


def test_not_engaged_outside_training(tmp_path):
    model = LlamaForCausalLM()
    trainer = _trainer(_FusedTrainer, model, 0.1, tmp_path)
    before = dict(fls.STATS)
    model.eval()
    with torch.no_grad():
        trainer.compute_loss(model, dict(_batch()))
    model.train()
    trainer.compute_loss(model, dict(_batch()), return_outputs = True)
    assert fls.STATS["fused"] == before["fused"] and fls.STATS["fallback"] == before["fallback"]
    assert fls.STATS.get("skip_eval", 0) == before.get("skip_eval", 0) + 2
    assert model.logits_built == 2


def test_opt_out_env_set_after_import_is_honored(tmp_path, monkeypatch):
    # llama.py imports this module with unsloth, so the flag must be read at dispatch, not import.
    model = LlamaForCausalLM()
    trainer = _trainer(_FusedTrainer, model, 0.1, tmp_path)
    monkeypatch.setenv("UNSLOTH_FUSED_LABEL_SMOOTHING", "0")
    before = dict(fls.STATS)
    _loss_and_grads(trainer, model, None)
    assert fls.STATS["fused"] == before["fused"]
    assert fls.STATS.get("skip_off", 0) == before.get("skip_off", 0) + 1
    monkeypatch.setenv("UNSLOTH_FUSED_LABEL_SMOOTHING", "1")
    _loss_and_grads(trainer, model, None)
    assert fls.STATS["fused"] == before["fused"] + 1


def test_sft_compute_loss_routes_through_fused_label_smoothing():
    source = (ROOT / "unsloth" / "models" / "rl_replacements.py").read_text(encoding = "utf-8")
    body = source[source.index("def sft_trainer_compute_loss(") :]
    body = body[: body.index('RL_FUNCTIONS["sft_trainer"].append(sft_trainer_compute_loss)')]
    assert "fused_label_smoothing_compute_loss" in body
    assert not re.search(r"outputs = super\(\)\.compute_loss\(", body)


def test_forward_with_own_labels_is_not_smoothed(tmp_path):
    # A parent compute_loss that never hands the label-free forward to LabelSmoother: its labeled forward keeps
    # the plain loss, and no-grad forwards are never given labels.
    model = LlamaForCausalLM()
    trainer = _trainer(Trainer, model, 0.1, tmp_path)
    batch = _batch()
    before = dict(fls.STATS)

    def parent():
        out = model(**batch)
        with torch.no_grad():
            extra = model(input_ids = batch["input_ids"])
        assert extra["logits"].numel() > 0
        return out["loss"]

    model.train()
    loss = fls.fused_label_smoothing_compute_loss(trainer, model, dict(batch), False, None, parent)
    n = (batch["labels"][:, 1:] != -100).sum()
    ref = (
        torch.nn.functional.cross_entropy(
            model.lm_head(model.embed(batch["input_ids"]))[:, :-1].reshape(-1, VOCAB),
            batch["labels"][:, 1:].reshape(-1),
            reduction = "sum",
        )
        / n
    )
    torch.testing.assert_close(loss.double(), ref)
    assert fls.STATS["fused"] == before["fused"] and fls.STATS["fallback"] == before["fallback"]
    assert type(trainer.label_smoother).__name__ == "LabelSmoother"
