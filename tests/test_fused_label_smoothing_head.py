# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fused-CE label smoothing must keep the stock path when the fused CE cannot reproduce the output head.

The fused CE passes no head bias, so a head bias would drop out of the loss. An unmerged LoRA adapter on lm_head stays
fused only when unsloth_zoo folds it into the kernel inputs ([x, s * A(x)] @ [W, B]^T, as llama.py and the compiled
forwards do); with an older zoo, or an adapter it cannot fold (DoRA), the stock path runs.
"""

import importlib.util
import pathlib

import pytest
import torch

pytest.importorskip("transformers")
peft_lora = pytest.importorskip("peft.tuners.lora.layer")
zoo_ce = pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
if not hasattr(zoo_ce, "unsloth_label_smoothing_override"):
    pytest.skip("unsloth_zoo without unsloth_label_smoothing_override", allow_module_level = True)

from peft import LoraConfig
from transformers import Trainer, TrainingArguments

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_unsloth_fused_label_smoothing_head", ROOT / "unsloth" / "models" / "fused_label_smoothing.py"
)
fls = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fls)

VOCAB, HIDDEN = 29, 8


class _Config:
    model_type = "llama"
    output_router_logits = False


class LlamaForCausalLM(torch.nn.Module):
    """With labels: fused CE with no bias, on lm_head.weight or the folded LoRA inputs, as unsloth/models/llama.py does."""

    def __init__(self, head):
        super().__init__()
        torch.manual_seed(0)
        self.config = _Config()
        self.embed = torch.nn.Embedding(VOCAB, HIDDEN, dtype = torch.float64)
        if head in ("lora", "dora"):
            base = torch.nn.Linear(HIDDEN, VOCAB, bias = False, dtype = torch.float64)
            base.weight.requires_grad_(False)
            self.lm_head = peft_lora.Linear(
                base,
                "default",
                config = LoraConfig(r = 2, lora_alpha = 2, use_dora = head == "dora"),
                r = 2,
                lora_alpha = 2,
                use_dora = head == "dora",
            ).double()
            with torch.no_grad():
                self.lm_head.lora_B["default"].weight.copy_(
                    torch.linspace(-0.5, 0.5, VOCAB * 2, dtype = torch.float64).reshape(VOCAB, 2)
                )
        else:
            self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias = head == "bias", dtype = torch.float64)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask = None,
        labels = None,
        **kwargs,
    ):
        hidden = self.embed(input_ids)
        if labels is not None:
            weight = self.lm_head.weight
            if fls.output_head_folds(self.lm_head):
                hidden, weight = fls.output_head_fused_inputs(self.lm_head, hidden)
            loss = zoo_ce.unsloth_fused_ce_loss(
                None,
                hidden,
                weight,
                None,
                labels,
                n_items = kwargs.get("num_items_in_batch", None),
                torch_compile = False,
            )
            return {"loss": loss, "logits": torch.empty(0)}
        return {"logits": self.lm_head(hidden)}


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


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, VOCAB, (2, 7), generator = g)
    labels = ids.clone()
    labels[0, :3] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def _run(cls, head, tmp_path):
    model = LlamaForCausalLM(head)
    args = TrainingArguments(
        output_dir = str(tmp_path), label_smoothing_factor = 0.1, report_to = [], use_cpu = True
    )
    trainer = cls(model = model, args = args)
    model.train()
    loss = trainer.compute_loss(model, _batch())
    loss.backward()
    grads = {
        n: None if p.grad is None else p.grad.double()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    return loss.detach().double(), grads


@pytest.mark.parametrize("head", ["lora", "lora_older_zoo", "dora", "bias"])
def test_head_adapter_and_bias_keep_stock_loss_and_grads(head, tmp_path, monkeypatch):
    if head == "lora_older_zoo":
        # A zoo without the fold: the LoRA head must take the stock path.
        monkeypatch.setattr(fls, "_zoo_head_needs_module_forward", None)
        head = "lora"
    ref_loss, ref_grads = _run(Trainer, head, tmp_path)
    before = dict(fls.STATS)
    loss, grads = _run(_FusedTrainer, head, tmp_path)
    # Only a LoRA head that unsloth_zoo folds into the kernel inputs stays fused; the rest take the stock path.
    fused = head == "lora" and not fls.output_head_not_fusable(LlamaForCausalLM(head).lm_head)
    assert fls.STATS["fused"] == before["fused"] + fused
    assert fls.STATS.get("skip_head", 0) == before.get("skip_head", 0) + (not fused)
    torch.testing.assert_close(loss, ref_loss)
    for name, ref in ref_grads.items():
        assert ref is not None
        assert grads[name] is not None, name
        torch.testing.assert_close(grads[name], ref)


def test_plain_head_still_fused(tmp_path):
    ref_loss, ref_grads = _run(Trainer, "plain", tmp_path)
    before = dict(fls.STATS)
    loss, grads = _run(_FusedTrainer, "plain", tmp_path)
    assert fls.STATS["fused"] == before["fused"] + 1
    torch.testing.assert_close(loss, ref_loss, rtol = 1e-6, atol = 1e-6)
    for name, ref in ref_grads.items():
        torch.testing.assert_close(grads[name], ref, rtol = 1e-6, atol = 1e-8)
