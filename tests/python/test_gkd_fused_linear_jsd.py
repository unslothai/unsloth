# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""GKD compute_loss reaches the zoo's fused two-head JSD op, matches TRL, and UNSLOTH_FUSED_LINEAR=0 restores TRL."""

from __future__ import annotations

import os
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

pytest.importorskip("trl")
try:
    from trl.experimental.gkd.gkd_trainer import GKDTrainer
except Exception:
    from trl.trainer.gkd_trainer import GKDTrainer

fl = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")
if not hasattr(fl, "fused_linear_jsd"):
    pytest.skip("installed unsloth_zoo predates fused_linear_jsd", allow_module_level = True)


class _TinyLM(torch.nn.Module):
    """Honours UNSLOTH_RETURN_HIDDEN_STATES like Unsloth's compiled forwards."""

    __UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True

    def __init__(
        self,
        vocab,
        hidden,
        softcap,
        seed,
        bias = False,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = torch.nn.Parameter(
            torch.randn(vocab, hidden, generator = g, dtype = torch.float64)
        )
        self.mix = torch.nn.Parameter(
            torch.randn(hidden, hidden, generator = g, dtype = torch.float64) / hidden**0.5
        )
        self.lm_head = torch.nn.Linear(hidden, vocab, bias = bias, dtype = torch.float64)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator = g, dtype = torch.float64))
        self.config = types.SimpleNamespace(final_logit_softcapping = softcap, model_type = "tiny")

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask):
        hidden = torch.tanh(self.embed[input_ids] @ self.mix)
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return types.SimpleNamespace(logits = hidden)
        logits = self.lm_head(hidden)
        if self.config.final_logit_softcapping:
            cap = self.config.final_logit_softcapping
            logits = cap * torch.tanh(logits / cap)
        return types.SimpleNamespace(logits = logits)


def _trainer(
    beta,
    student,
    teacher,
    scaler = None,
):
    trainer = GKDTrainer.__new__(GKDTrainer)
    trainer.__dict__.update(
        beta = beta,
        temperature = 0.9,
        use_liger_gkd_loss = False,
        teacher_model = teacher,
        accelerator = types.SimpleNamespace(unwrap_model = lambda m: m, scaler = scaler),
    )
    return trainer


def _inputs(vocab):
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (3, 21), generator = g)
    labels = input_ids.clone()
    labels[:, :7] = -100
    labels[1, 17:] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prompts": input_ids[:, :7],
    }


def _trl_reference(
    student,
    teacher,
    inputs,
    beta,
    shift,
    num_items = None,
):
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    try:
        s = student(inputs["input_ids"], inputs["attention_mask"]).logits
        with torch.no_grad():
            t = teacher(inputs["input_ids"], inputs["attention_mask"]).logits
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior
    if shift == "prompt":
        s, t, labels = s[:, 6:-1], t[:, 6:-1], inputs["labels"][:, 7:]
    else:
        s, t, labels = s[:, :-1], t[:, :-1], inputs["labels"][:, 1:]
    want = GKDTrainer.generalized_jsd_loss(s, t, labels = labels, beta = beta)
    if num_items is not None:
        want = want * (labels != -100).sum() / num_items
    params = [p for p in student.parameters() if p.requires_grad]
    return want, torch.autograd.grad(want, params)


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("shift", ["prompt", "shift"])
@pytest.mark.parametrize("trainable_head", [False, True])
def test_gkd_reaches_fused_op_and_matches_trl(beta, shift, trainable_head, monkeypatch):
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    monkeypatch.setenv("UNSLOTH_GKD_CHUNK_SIZE", "8")
    vocab = 97
    student = _TinyLM(vocab, 16, 0.0, 1, bias = trainable_head)
    teacher = _TinyLM(vocab, 24, 5.0, 2)
    student.lm_head.requires_grad_(trainable_head)
    trainer, inputs = _trainer(beta, student, teacher), _inputs(vocab)
    num_items = torch.tensor(29) if shift == "shift" else None
    layout = {"shift": shift, "num_items_in_batch": shift == "shift"}
    want, want_grads = _trl_reference(student, teacher, inputs, beta, shift, num_items)

    before = fl.CALLS["jsd"]
    got = rl._unsloth_gkd_chunked_loss(trainer, student, inputs, num_items, layout)
    assert got is not None
    assert fl.CALLS["jsd"] == before + 1, "GKD did not reach the fused JSD op"
    assert trainer._unsloth_gkd_fused_calls == 1
    params = [p for p in student.parameters() if p.requires_grad]
    got_grads = torch.autograd.grad(got, params)
    torch.testing.assert_close(got.double(), want, rtol = 1e-6, atol = 1e-9)
    for g, w in zip(got_grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-6, atol = 1e-9)


def test_kill_switch_restores_trl(monkeypatch):
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    trainer, inputs = _trainer(0.5, student, teacher), _inputs(vocab)
    layout = {"shift": "shift", "num_items_in_batch": False}
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    before = fl.CALLS["jsd"]
    assert rl._unsloth_gkd_chunked_loss(trainer, student, inputs, None, layout) is None
    assert fl.CALLS["jsd"] == before
    assert trainer._unsloth_gkd_chunked_fallbacks == {"UNSLOTH_FUSED_LINEAR=0": 1}
    assert getattr(trainer, "_unsloth_gkd_fused_calls", 0) == 0


@pytest.mark.parametrize("enabled", [True, False])
def test_grad_scaler_scale_reaches_the_op(enabled, monkeypatch):
    class Scaler:
        def is_enabled(self):
            return enabled

        def get_scale(self):
            return 65536.0

    calls = []
    real = rl._unsloth_fused_linear_jsd

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(rl, "_unsloth_fused_linear_jsd", spy)
    vocab = 97
    student, teacher = _TinyLM(vocab, 16, 0.0, 1), _TinyLM(vocab, 24, 0.0, 2)
    trainer, inputs = _trainer(0.5, student, teacher, Scaler()), _inputs(vocab)
    got = rl._unsloth_gkd_chunked_loss(
        trainer, student, inputs, None, {"shift": "prompt", "num_items_in_batch": False}
    )
    assert got is not None and len(calls) == 1
    assert calls[0]["scaling"] == (65536.0 if enabled else None)
    want, want_grads = _trl_reference(student, teacher, inputs, 0.5, "prompt")
    grads = torch.autograd.grad(got, [student.embed, student.mix])
    for g, w in zip(grads, want_grads):
        torch.testing.assert_close(g, w, rtol = 1e-6, atol = 1e-9)


def test_generated_items_import_the_fused_op():
    pre = "\n".join(rl.RL_PRE_ITEMS["gkd_trainer"])
    assert "fused_linear_jsd as _unsloth_fused_linear_jsd" in pre
    assert rl._unsloth_fused_linear_jsd is not None
