# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Quantization-aware distillation (QAD) for QAT + LoRA, with the student's own weights as the teacher.

`get_peft_model(..., qat_scheme = ..., qat_distill = coef)` (or `UNSLOTH_QAT_DISTILL=coef`) trains on

    loss = (1 - coef) * CE(student) + coef * KD(teacher || student)

where the student is the fake-quantized LoRA model and the teacher is the same model with its adapters off and every
fake quantizer off: the unquantized 16-bit base, scored under `torch.no_grad()`, with no second copy of any weight.
KD is TRL GKD's generalized JSD over `logits / temperature`: beta 0 is forward KL(teacher || student) (the default),
1 reverse KL, in between the JSD with mixture `(1 - beta) p_student + beta p_teacher`. Both terms are summed over the
tokens whose shifted label is not -100 and divided by `num_items_in_batch` when the Trainer passes it (gradient
accumulation exact), else by the local token count. Student and teacher run their decoders once each and return the
head input (UNSLOTH_RETURN_HIDDEN_STATES=1); the head projection, CE and KD then go through the zoo's chunked fused
linear losses, so no full logits tensor is built.

`qat_distill` accepts a coefficient in [0, 1], True (coef 1, pure KD), or a dict with `coef`, `beta`, `temperature`.
Rounding frozen base weights onto the QAT grid in place (`_qat_round`) would turn the teacher into the quantized
base, so a model that distills keeps its 16-bit base weights and is never rounded."""

import contextlib
import math
import os

import torch

__all__ = [
    "qat_teacher",
    "qat_distill_config",
    "parse_qat_distill",
    "set_qat_distill",
    "qat_distill_compute_loss",
    "qat_distill_stats",
]

_ENV = "UNSLOTH_QAT_DISTILL"
_ATTR = "_unsloth_qat_distill"
_TEACHER_FLAG = "_unsloth_qat_teacher"
_ROUNDED_FLAG = "_unsloth_qat_rounded"  # _qat_round._FLAG
_STATS = {"steps": 0, "fused": 0, "dense": 0, "last_ce": None, "last_kd": None}
_DEFAULTS = {"coef": 1.0, "beta": 0.0, "temperature": 1.0}


def qat_distill_stats():
    return dict(_STATS)


def _check(config):
    coef, beta, temperature = (float(config[k]) for k in ("coef", "beta", "temperature"))
    if not 0.0 <= coef <= 1.0:
        raise ValueError(f"Unsloth: qat_distill coef must be in [0, 1], got {coef}.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"Unsloth: qat_distill beta must be in [0, 1], got {beta}.")
    if not (0.0 < temperature < math.inf):
        raise ValueError(
            f"Unsloth: qat_distill temperature must be finite and > 0, got {temperature}."
        )
    return {"coef": coef, "beta": beta, "temperature": temperature}


def parse_qat_distill(value):
    """The QAD config for a `qat_distill` value, or None when off. None reads UNSLOTH_QAT_DISTILL."""
    if value is None:
        value = os.environ.get(_ENV, "").strip()
        if value.lower() in ("", "0", "false", "off", "no"):
            return None
        if value.lower() in ("1", "true", "on", "yes"):
            return dict(_DEFAULTS)
        if "=" in value:
            config = dict(_DEFAULTS)
            for part in value.split(","):
                key, _, number = part.partition("=")
                key = key.strip()
                if key not in config:
                    raise ValueError(
                        f"Unsloth: {_ENV} has an unknown key {key!r}; use coef, beta, temperature."
                    )
                config[key] = float(number)
            return _check(config)
        return _check({**_DEFAULTS, "coef": float(value)})
    if value is False:
        return None
    if value is True:
        return dict(_DEFAULTS)
    if isinstance(value, dict):
        unknown = set(value) - set(_DEFAULTS)
        if unknown:
            raise ValueError(
                f"Unsloth: qat_distill has unknown keys {sorted(unknown)}; use coef, beta, temperature."
            )
        return _check({**_DEFAULTS, **value})
    if isinstance(value, (int, float)):
        return _check({**_DEFAULTS, "coef": float(value)})
    raise TypeError(
        f"Unsloth: qat_distill must be a float, bool or dict, got {type(value).__name__}."
    )


def _unwrap(model):
    for _ in range(8):
        inner = getattr(model, "module", None)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and inner is not None:
            model = inner
        elif getattr(model, "_orig_mod", None) is not None:
            model = model._orig_mod
        else:
            break
    return model


def qat_distill_config(model):
    """The QAD config stored on `model` (or a wrapper of it), else None."""
    return getattr(_unwrap(model), _ATTR, None)


def _rounded_modules(model):
    return [
        name
        for name, module in model.named_modules()
        if getattr(module._modules.get("weight_fake_quantizer"), "__dict__", {}).get(
            _ROUNDED_FLAG, False
        )
    ]


def _refuse_rounded(model):
    rounded = _rounded_modules(model)
    if rounded:
        raise RuntimeError(
            "Unsloth: qat_distill needs the 16-bit base weights as the teacher, but these were already rounded onto "
            f"the QAT grid in place: {', '.join(rounded[:4])}{' ...' if len(rounded) > 4 else ''}. Pass qat_distill "
            "to get_peft_model together with qat_scheme (or set UNSLOTH_QAT_ROUND_INPLACE=0 before it)."
        )


def set_qat_distill(model, value):
    """Store the QAD config for `value` on `model` and its inner models; returns it (None when off)."""
    config = parse_qat_distill(value)
    if config is None:
        return None
    _refuse_rounded(model)
    inner = model
    for _ in range(8):
        try:
            setattr(inner, _ATTR, config)
        except Exception:
            pass
        if not hasattr(inner, "model") or not isinstance(inner.model, torch.nn.Module):
            break
        inner = inner.model
    print(
        f"Unsloth: QAT distillation on: loss = {1 - config['coef']:g} * CE + {config['coef']:g} * KD from the "
        f"16-bit base (beta {config['beta']:g}, temperature {config['temperature']:g})."
    )
    return config


def _forward_fake_quant_linears():
    try:
        from torchao.prototype import qat as prototype_qat
    except Exception:
        return ()
    return tuple(
        cls
        for cls in (
            getattr(prototype_qat, "NVFP4FakeQuantizedLinear", None),
            getattr(prototype_qat, "MXFakeQuantizedLinear", None),
        )
        if isinstance(cls, type)
    )


@contextlib.contextmanager
def qat_teacher(model):
    """Run `model` as its own QAD teacher: adapters off, every torchao fake quantizer off (weight, activation,
    embedding, NVFP4 / MX linears), restored on exit. No weight is copied or written."""
    from ._on_policy_distillation import _adapter_off, _peft_student

    student = _peft_student(model, "qat_distill")
    _refuse_rounded(student)
    try:
        from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase
    except Exception:
        FakeQuantizerBase = None
    forward_linears = _forward_fake_quant_linears()
    quantizers, linears = [], []
    for module in student.modules():
        if FakeQuantizerBase is not None and isinstance(module, FakeQuantizerBase):
            quantizers.append((module, module.enabled))
        elif forward_linears and isinstance(module, forward_linears):
            linears.append((module, getattr(module, _TEACHER_FLAG, False)))
    try:
        for module, _ in quantizers:
            module.enabled = False
        for module, _ in linears:
            setattr(module, _TEACHER_FLAG, True)
        with _adapter_off(student):
            yield student
    finally:
        for module, enabled in quantizers:
            module.enabled = enabled
        for module, flag in linears:
            setattr(module, _TEACHER_FLAG, flag)


def _shifted_labels(inputs):
    shift_labels = inputs.get("shift_labels", None)
    if shift_labels is not None:
        return shift_labels
    labels = inputs["labels"]
    try:
        from ..utils.packing import mask_packed_boundary_labels
        labels = mask_packed_boundary_labels(labels, inputs.get("packed_seq_lengths", None))
    except Exception:
        pass
    shifted = torch.full_like(labels, -100)
    shifted[..., :-1] = labels[..., 1:]
    return shifted


def _head(model):
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    head = get_output_embeddings() if callable(get_output_embeddings) else None
    return head if isinstance(head, torch.nn.Module) else None


def _fusable_head(head):
    """A head the fused losses can read as `weight` (+ the module's own weight / activation fake quantizers)."""
    if (
        head is None
        or hasattr(head, "modules_to_save")
        or hasattr(head, "lora_A")
        or hasattr(head, "base_layer")
    ):
        return False
    from .llama import lm_head_needs_forward

    if lm_head_needs_forward(head):
        return False
    weight = getattr(head, "weight", None)
    return (
        isinstance(weight, torch.Tensor)
        and type(weight) in (torch.Tensor, torch.nn.Parameter)
        and weight.dim() == 2
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        and not weight.is_meta
    )


def _transform(logits, scale, softcap, temperature):
    logits = logits.float()
    if scale != 1.0:
        logits = logits * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits / temperature


def dense_qat_distill_losses(
    student_logits,
    teacher_logits,
    shift_labels,
    divisor,
    beta,
    temperature,
    scale = 1.0,
    softcap = 0.0,
):
    """(CE sum, KD sum) / divisor from full logits: the fallback, and the formula the fused path computes."""
    keep = shift_labels.reshape(-1) != -100
    vocab = student_logits.shape[-1]
    s = _transform(student_logits.reshape(-1, vocab)[keep], scale, softcap, 1.0)
    t = _transform(teacher_logits.detach().reshape(-1, vocab)[keep], scale, softcap, 1.0)
    ce = torch.nn.functional.cross_entropy(s, shift_labels.reshape(-1)[keep], reduction = "sum")
    s_lp = torch.log_softmax(s / temperature, dim = -1)
    t_lp = torch.log_softmax(t / temperature, dim = -1)
    if beta == 0.0:
        kd = (t_lp.exp() * (t_lp - s_lp)).sum()
    elif beta == 1.0:
        kd = (s_lp.exp() * (s_lp - t_lp)).sum()
    else:
        m_lp = torch.logsumexp(
            torch.stack([s_lp + math.log(1 - beta), t_lp + math.log(beta)]), dim = 0
        )
        kd = (
            beta * (t_lp.exp() * (t_lp - m_lp)).sum()
            + (1 - beta) * (s_lp.exp() * (s_lp - m_lp)).sum()
        )
    return ce / divisor, kd / divisor


def _scaler(trainer):
    scaler = getattr(getattr(trainer, "accelerator", None), "scaler", None)
    if scaler is not None and scaler.is_enabled():
        return scaler.get_scale()
    return None


def qat_distill_compute_loss(
    trainer,
    model,
    inputs,
    num_items_in_batch = None,
    return_outputs = False,
    config = None,
):
    """The QAD training loss for one micro batch (see the module docstring), as Trainer.compute_loss returns it."""
    config = config or qat_distill_config(model)
    coef, beta, temperature = config["coef"], config["beta"], config["temperature"]
    unwrapped = _unwrap(model)
    try:
        unwrapped = trainer.accelerator.unwrap_model(model)
    except Exception:
        pass
    shift_labels = _shifted_labels(inputs)
    forward_inputs = {
        k: v for k, v in inputs.items() if k not in ("labels", "shift_labels", "num_items_in_batch")
    }
    forward_inputs.setdefault("use_cache", False)

    from .rl_replacements import _unsloth_gkd_logit_transforms, _unsloth_grpo_returns_hidden_states

    head = _head(unwrapped)
    fused = None
    try:
        from unsloth_zoo.fused_losses.fused_linear import (
            fused_linear_cross_entropy,
            fused_linear_jsd,
        )
        fused = (fused_linear_cross_entropy, fused_linear_jsd)
    except Exception:
        fused = None
    reason = None
    if fused is None:
        reason = "unsloth_zoo has no fused_linear_jsd / fused_linear_cross_entropy"
    elif not _fusable_head(head):
        reason = "the output head is not a plain dense projection"
    elif os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        reason = "UNSLOTH_FUSED_LINEAR=0"

    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    if reason is None:
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        outputs = model(**forward_inputs)
        with torch.no_grad(), qat_teacher(unwrapped):
            teacher_outputs = model(**forward_inputs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior
    student_states, teacher_states = outputs.logits, teacher_outputs.logits
    if reason is None and not (
        _unsloth_grpo_returns_hidden_states(unwrapped, student_states, head.weight)
        and _unsloth_grpo_returns_hidden_states(unwrapped, teacher_states, head.weight)
    ):
        reason = "the forward returned logits, not hidden states"

    scale, softcap = _unsloth_gkd_logit_transforms(unwrapped)
    device = head.weight.device if head is not None else student_states.device
    shift_labels = shift_labels.to(device)
    if num_items_in_batch is None:
        divisor = (shift_labels != -100).sum()
    else:
        divisor = num_items_in_batch
    divisor = (
        torch.as_tensor(divisor, device = device, dtype = torch.float32).reshape(-1)[:1].reshape(())
    )
    divisor = torch.where(divisor == 0, torch.ones_like(divisor), divisor)

    if reason is None:
        from .llama import fake_quantize_lm_head_inputs

        fused_ce, fused_jsd = fused
        student_states = student_states.to(device)
        teacher_states = teacher_states.to(device)
        # The student head runs fake-quantized, as the model's own fused CE does; the teacher reads the 16-bit base.
        s_hidden, s_weight = fake_quantize_lm_head_inputs(head, student_states, head.weight)
        scaling = _scaler(trainer)
        ce = fused_ce(
            s_hidden,
            s_weight,
            head.bias,
            shift_labels,
            divisor = divisor,
            scaling = scaling,
            logit_scale_multiply = scale if scale != 1.0 else 0.0,
            logit_softcapping = softcap,
        )
        kd = fused_jsd(
            s_hidden,
            s_weight,
            teacher_states.detach(),
            head.weight.detach(),
            labels = shift_labels,
            beta = beta,
            temperature = temperature,
            student_bias = head.bias,
            teacher_bias = None if head.bias is None else head.bias.detach(),
            student_logit_scale = scale,
            student_logit_softcapping = softcap,
            teacher_logit_scale = scale,
            teacher_logit_softcapping = softcap,
            num_items_in_batch = divisor,
            scaling = scaling,
        )
        _STATS["fused"] += 1
    else:
        from transformers.utils import logging

        logging.get_logger(__name__).warning_once(
            f"Unsloth: qat_distill computes full logits for both models ({reason}); expect much more memory."
        )
        if student_states.shape[-1] != teacher_states.shape[-1]:
            raise RuntimeError(
                "Unsloth: qat_distill could not obtain logits from the student and the teacher."
            )
        ce, kd = dense_qat_distill_losses(
            student_states.to(device),
            teacher_states.to(device),
            shift_labels,
            divisor,
            beta,
            temperature,
            scale,
            softcap,
        )
        _STATS["dense"] += 1

    loss = (1.0 - coef) * ce + coef * kd if coef < 1.0 else kd
    args = getattr(trainer, "args", None)
    if getattr(args, "average_tokens_across_devices", False) and num_items_in_batch is not None:
        processes = int(getattr(getattr(trainer, "accelerator", None), "num_processes", 1) or 1)
        tp_size = trainer.get_tp_size() if callable(getattr(trainer, "get_tp_size", None)) else 1
        loss = loss * (processes // max(int(tp_size or 1), 1))
    _STATS["steps"] += 1
    # Detached tensors, not floats: reading them here would sync the device every micro batch.
    _STATS["last_ce"], _STATS["last_kd"] = ce.detach(), kd.detach()
    loss = loss.to(inputs["labels"].device if "labels" in inputs else loss.device)
    if not return_outputs:
        return loss
    from ._utils import EMPTY_LOGITS

    outputs.loss = loss
    outputs.logits = EMPTY_LOGITS
    return loss, outputs
