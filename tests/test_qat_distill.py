# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT distillation (unsloth/models/_qat_distill.py): the teacher is the student with adapters and fake quantizers
off (the exact 16-bit base, no weight copied), and the mixed CE + KD loss matches an fp64 dense reference, with
gradient accumulation exact. CPU only: a random-init tiny Qwen3 (hidden 128, so int4 group 128 applies)."""

import copy
import math
import os
import types

import pytest
import torch

import unsloth  # noqa: F401  (patches before transformers / peft)
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from unsloth.models import _qat_distill as qd
from unsloth.models._utils import _prepare_model_for_qat, _unsloth_pre_compute_loss

pytest.importorskip("torchao")

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _base(seed = 0, dtype = torch.bfloat16):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 64,
        max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    model = Qwen3ForCausalLM(config).to(dtype).eval()
    model.config._attn_implementation = "eager"
    return model


def _hidden_states_forward(model):
    """Honour UNSLOTH_RETURN_HIDDEN_STATES like Unsloth's CausalLM forward (the head input as .logits)."""

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        use_cache = None,
        **kwargs,
    ):
        hidden = self.model(
            input_ids = input_ids, attention_mask = attention_mask, use_cache = False
        ).last_hidden_state
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return CausalLMOutputWithPast(logits = hidden)
        return CausalLMOutputWithPast(logits = self.lm_head(hidden))

    model.forward = types.MethodType(forward, model)


def _student(
    scheme,
    distill = 0.5,
    seed = 0,
    dtype = torch.bfloat16,
):
    base = _base(seed, dtype)
    reference = copy.deepcopy(base)
    _hidden_states_forward(base)
    torch.manual_seed(seed + 1)
    peft = get_peft_model(
        base,
        LoraConfig(
            r = 8, lora_alpha = 16, target_modules = TARGETS, lora_dropout = 0.0, init_lora_weights = False
        ),
    )
    if distill is not None:
        qd.set_qat_distill(peft, distill)
    peft = _prepare_model_for_qat(peft, scheme)
    return peft, reference


def _quantizers(model):
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase
    return [m for m in model.modules() if isinstance(m, FakeQuantizerBase)]


def _ids(
    rows = 2,
    cols = 12,
    seed = 0,
):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (rows, cols), generator = g)


@pytest.mark.parametrize("scheme", ["int4", "int8"])
def test_teacher_is_the_exact_16bit_base_and_restores(scheme):
    student, reference = _student(scheme)
    assert sum(1 for _ in _quantizers(student)) > 0
    x = _ids()
    with torch.no_grad():
        before = student(input_ids = x).logits
        base_logits = reference.lm_head(reference.model(input_ids = x).last_hidden_state)
        ptrs = {p.data_ptr() for p in student.parameters()}
        enabled = [q.enabled for q in _quantizers(student)]
        with qd.qat_teacher(student):
            teacher = student(input_ids = x).logits
            assert all(not q.enabled for q in _quantizers(student))
        after = student(input_ids = x).logits
    # Adapters and every fake quantizer off: bitwise the 16-bit base model.
    assert torch.equal(teacher, base_logits)
    assert not torch.equal(before, base_logits)
    assert torch.equal(after, before)
    assert [q.enabled for q in _quantizers(student)] == enabled
    # No weight copied or rewritten by the teacher pass.
    assert {p.data_ptr() for p in student.parameters()} == ptrs


def test_distilling_model_is_not_rounded_and_rounded_model_refuses():
    # int8 weight-only: Unsloth rounds frozen bases onto the grid in place unless the model distills.
    student, reference = _student("int8", distill = 0.5)
    assert not qd._rounded_modules(student)
    rounded, _ = _student("int8", distill = None)
    assert qd._rounded_modules(rounded)
    with pytest.raises(RuntimeError, match = "rounded onto"):
        with qd.qat_teacher(rounded):
            pass
    with pytest.raises(RuntimeError, match = "rounded onto"):
        qd.set_qat_distill(rounded, 0.5)


def test_parse():
    assert qd.parse_qat_distill(False) is None
    assert qd.parse_qat_distill(True) == {"coef": 1.0, "beta": 0.0, "temperature": 1.0}
    assert qd.parse_qat_distill(0.25)["coef"] == 0.25
    assert qd.parse_qat_distill({"beta": 0.5, "temperature": 2})["temperature"] == 2.0
    for bad in (1.5, -0.1, {"coef": 0.5, "alpha": 1}, {"temperature": 0}):
        with pytest.raises((ValueError, TypeError)):
            qd.parse_qat_distill(bad)
    os.environ[qd._ENV] = "coef=0.5,beta=0.5"
    try:
        assert qd.parse_qat_distill(None) == {"coef": 0.5, "beta": 0.5, "temperature": 1.0}
    finally:
        os.environ.pop(qd._ENV)
    assert qd.parse_qat_distill(None) is None


def _oracle(student, x, labels, divisor, coef, beta, temperature):
    """fp64 dense CE + generalized JSD, differentiable in the student (TRL GKD convention)."""
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        hs = student(input_ids = x).logits
        with torch.no_grad(), qd.qat_teacher(student):
            ht = student(input_ids = x).logits
    finally:
        os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES")
    head = student.get_output_embeddings()
    w_s = head.weight
    fq = getattr(head, "weight_fake_quantizer", None)
    if fq is not None:
        w_s = fq(w_s)
    shifted = torch.full_like(labels, -100)
    shifted[:, :-1] = labels[:, 1:]
    keep = shifted.reshape(-1) != -100
    s = (hs.reshape(-1, hs.shape[-1]).double() @ w_s.double().t())[keep]
    t = (ht.reshape(-1, ht.shape[-1]).double() @ head.weight.double().t())[keep]
    ce = torch.nn.functional.cross_entropy(s, shifted.reshape(-1)[keep], reduction = "sum") / divisor
    s_lp = torch.log_softmax(s / temperature, -1)
    t_lp = torch.log_softmax(t / temperature, -1)
    if beta == 0:
        kd = (t_lp.exp() * (t_lp - s_lp)).sum()
    elif beta == 1:
        kd = (s_lp.exp() * (s_lp - t_lp)).sum()
    else:
        m = torch.logsumexp(torch.stack([s_lp + math.log(1 - beta), t_lp + math.log(beta)]), 0)
        kd = beta * (t_lp.exp() * (t_lp - m)).sum() + (1 - beta) * (s_lp.exp() * (s_lp - m)).sum()
    kd = kd / divisor
    return (1 - coef) * ce + coef * kd


class _Trainer:
    args = None
    accelerator = None


def _lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _zero(model):
    for p in model.parameters():
        p.grad = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "beta,temperature,coef", [(0.0, 1.0, 0.5), (0.5, 2.0, 1.0), (1.0, 1.0, 0.3)]
)
def test_loss_and_grads_match_fp64_oracle(beta, temperature, coef, dtype):
    student, _ = _student(
        "int4", distill = {"coef": coef, "beta": beta, "temperature": temperature}, dtype = dtype
    )
    student.train()
    x = _ids(3, 12, seed = 1)
    labels = x.clone()
    labels[0, :5] = -100  # prompt mask
    labels[2, :] = -100  # fully masked row
    n = int((labels[:, 1:] != -100).sum())
    loss = qd.qat_distill_compute_loss(_Trainer(), student, {"input_ids": x, "labels": labels})
    assert qd.qat_distill_stats()["fused"] > 0
    loss.backward()
    got = _lora_grads(student)
    _zero(student)
    ref = _oracle(student, x, labels, n, coef, beta, temperature)
    ref.backward()
    want = _lora_grads(student)
    _zero(student)
    # float32: the fused loss is the oracle to fp32 rounding; bfloat16 adds the model's own backward rounding.
    loss_tol, grad_tol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 3e-2)
    assert abs(loss.item() - ref.item()) <= loss_tol * abs(ref.item()) + 1e-7
    num = sum(((got[k].double() - want[k].double()) ** 2).sum() for k in got) ** 0.5
    den = sum((want[k].double() ** 2).sum() for k in want) ** 0.5
    assert den > 0 and num / den < grad_tol, (num / den).item()


def test_gradient_accumulation_is_exact_with_num_items():
    student, _ = _student("int8", distill = 0.5)
    student.train()
    x = _ids(4, 10, seed = 2)
    labels = x.clone()
    labels[1, :6] = -100
    total = int((labels[:, 1:] != -100).sum())
    full = qd.qat_distill_compute_loss(
        _Trainer(), student, {"input_ids": x, "labels": labels}, num_items_in_batch = total
    )
    full.backward()
    g_full = _lora_grads(student)
    _zero(student)
    parts = sum(
        qd.qat_distill_compute_loss(
            _Trainer(),
            student,
            {"input_ids": x[i : i + 2], "labels": labels[i : i + 2]},
            num_items_in_batch = total,
        )
        for i in (0, 2)
    )
    parts.backward()
    g_parts = _lora_grads(student)
    _zero(student)
    assert abs(full.item() - parts.item()) <= 1e-4 * abs(full.item())
    for k in g_full:
        torch.testing.assert_close(g_full[k], g_parts[k], rtol = 2e-2, atol = 1e-5)


def test_identity_gives_zero_kd():
    # Student == teacher (adapter B = 0 and no fake quantizer active): KD is zero, loss is the pure CE.
    student, _ = _student("int8", distill = 1.0)
    student.train()
    for name, p in student.named_parameters():
        if "lora_B" in name:
            torch.nn.init.zeros_(p)
    x = _ids(2, 12, seed = 3)
    labels = x.clone()
    for q in _quantizers(student):
        q.enabled = False
    try:
        kd = qd.qat_distill_compute_loss(_Trainer(), student, {"input_ids": x, "labels": labels})
        kd.backward()
    finally:
        for q in _quantizers(student):
            q.enabled = True
    assert abs(kd.item()) < 1e-6, kd.item()
    grad = max(
        p.grad.abs().max().item() for n, p in student.named_parameters() if p.grad is not None
    )
    assert grad < 1e-5, grad


def test_compute_loss_hook_routes_only_when_distilling():
    calls = []
    trainer = types.SimpleNamespace(
        args = types.SimpleNamespace(gradient_accumulation_steps = 1),
        _old_compute_loss = lambda model, inputs, *a, **k: calls.append("ce") or torch.tensor(1.0),
    )
    student, _ = _student("int8", distill = 0.5)
    student.train()
    x = _ids(2, 8)
    out = _unsloth_pre_compute_loss(trainer, student, {"input_ids": x, "labels": x.clone()})
    assert calls == [] and out.requires_grad
    student._unsloth_qat_distill = None
    _unsloth_pre_compute_loss(trainer, student, {"input_ids": x, "labels": x.clone()})
    student.eval()
    _unsloth_pre_compute_loss(trainer, student, {"input_ids": x, "labels": x.clone()})
    assert calls == ["ce", "ce"]


def test_dense_fallback_matches_fused():
    student, _ = _student("int8", distill = {"coef": 0.4, "beta": 0.5, "temperature": 1.5})
    student.train()
    x = _ids(2, 12, seed = 4)
    labels = x.clone()
    fused = qd.qat_distill_compute_loss(_Trainer(), student, {"input_ids": x, "labels": labels})
    os.environ["UNSLOTH_FUSED_LINEAR"] = "0"
    try:
        dense = qd.qat_distill_compute_loss(_Trainer(), student, {"input_ids": x, "labels": labels})
    finally:
        os.environ.pop("UNSLOTH_FUSED_LINEAR")
    assert abs(fused.item() - dense.item()) <= 2e-3 * abs(dense.item())
