# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""label_smoothing_factor through the fused cross entropy instead of the full logits.

With label_smoothing_factor > 0, Trainer.compute_loss (and TRL >= 1.15 SFTTrainer.compute_loss) calls the model without
labels, so it returns [B, L, V] logits with autograd, and LabelSmoother takes an fp32 log_softmax over all of them
(4 GiB of bf16 logits plus 8 GiB of fp32 log-probs at 8192 rows x 262144 vocab, plus their gradients).

Here, for one training step: the trainer's LabelSmoother is swapped for a proxy, a one-shot forward pre-hook hands the
labels back to that label-free forward, and unsloth_zoo's fused CE computes F.cross_entropy(label_smoothing = eps)
chunk by chunk, divided by the count LabelSmoother would use (this micro-batch's active tokens; num_items_in_batch
on transformers whose LabelSmoother takes it). The proxy returns that loss, so everything the trainer does around the
smoother (gradient accumulation, average_tokens_across_devices, TRL metrics) is unchanged.

A forward that never reaches the fused CE (UNSLOTH_RETURN_LOGITS=1, a head that needs its own forward, other loss
paths) is detected by the engagement count: that step falls back to the stock LabelSmoother and the model keeps the
stock path from then on. UNSLOTH_FUSED_LABEL_SMOOTHING=0 turns all of this off.
"""

import inspect
import os
import weakref

import torch

__all__ = ["fused_label_smoothing_compute_loss"]

_OFF = os.environ.get("UNSLOTH_FUSED_LABEL_SMOOTHING", "1") == "0"
# Models whose forward did not reach the fused CE once; they keep the stock LabelSmoother.
_FALLBACK_MODELS = weakref.WeakSet()
# Engagement counters, read by tests.
STATS = {"fused": 0, "fallback": 0}


def _override():
    try:
        from unsloth_zoo.fused_losses.cross_entropy_loss import (
            unsloth_label_smoothing_override,
        )
    except Exception:
        return None
    return unsloth_label_smoothing_override


def _accepts_num_items(smoother):
    try:
        return "num_items_in_batch" in inspect.signature(smoother.__call__).parameters
    except (TypeError, ValueError):
        return False


def _router_logits_requested(trainer, model, inputs):
    # With router logits the model adds the MoE aux loss to its own loss; LabelSmoother never sees it.
    if getattr(trainer, "aux_loss_enabled", False) or inputs.get("output_router_logits", False):
        return True
    for module in (model, getattr(model, "module", None), getattr(trainer, "model", None)):
        config = getattr(module, "config", None)
        if config is None:
            continue
        configs = [config]
        if hasattr(config, "get_text_config"):
            try:
                configs.append(config.get_text_config())
            except Exception:
                pass
        if any(getattr(c, "output_router_logits", False) for c in configs):
            return True
    return False


def _skip_reason(trainer, model, inputs, return_outputs):
    """None when the fused path applies, else why not (counted in STATS)."""
    smoother = getattr(trainer, "label_smoother", None)
    if _OFF or smoother is None:
        return "off"
    # Evaluation (prediction_step asks for outputs, under no_grad) keeps the stock smoother and its logits.
    if return_outputs or not torch.is_grad_enabled():
        return "eval"
    if getattr(trainer, "compute_loss_func", None) is not None:
        return "compute_loss_func"
    if getattr(trainer.args, "loss_type", None) not in (None, "nll"):
        return "loss_type"
    if (
        not float(getattr(smoother, "epsilon", 0.0) or 0.0) > 0.0
        or getattr(smoother, "ignore_index", -100) != -100
    ):
        return "smoother"
    labels = inputs.get("labels", None)
    if not isinstance(labels, torch.Tensor) or labels.dim() < 2:
        return "labels"
    if inputs.get("shift_labels", None) is not None or "fused_lm_head" in inputs:
        return "shift_labels"
    if model in _FALLBACK_MODELS:
        return "fell_back"
    if _router_logits_requested(trainer, model, inputs):
        return "router_logits"
    if _override() is None:
        return "zoo"
    return None


class _Injection:
    def __init__(self, labels):
        self.labels = labels
        self.seen = False
        self.done = False
        self.state = None
        self.args = None
        self.kwargs = None


class _FusedLabelSmoother:
    """Stands in for the trainer's LabelSmoother for one compute_loss call."""

    def __init__(self, original, state, injection, model):
        self.original = original
        self.epsilon = original.epsilon
        self.ignore_index = original.ignore_index
        self._state = state
        self._injection = injection
        self._model = model

    def _fallback(self, model_output, labels, shift_labels, kwargs):
        STATS["fallback"] += 1
        _FALLBACK_MODELS.add(self._model)
        logits = (
            model_output.get("logits", None)
            if isinstance(model_output, dict)
            else (model_output[1] if len(model_output) > 1 else None)
        )
        if not (
            isinstance(logits, torch.Tensor)
            and logits.dim() == labels.dim() + 1
            and logits.shape[:-1] == labels.shape
        ):
            injection = self._injection
            if injection.args is None:
                return self.original(model_output, labels, shift_labels = shift_labels, **kwargs)
            # No usable logits came back: run the stock label-free forward again.
            del model_output, logits
            model_output = self._model(*injection.args, **injection.kwargs)
        return self.original(model_output, labels, shift_labels = shift_labels, **kwargs)

    def __call__(
        self,
        model_output,
        labels,
        shift_labels = False,
        **kwargs,
    ):
        state = self._state
        engaged = state.engaged
        # Later forwards in this compute_loss (TRL's no-grad metrics forward) keep their plain loss.
        state.active = False
        if not (shift_labels and self._injection.done and engaged == 1):
            return self._fallback(model_output, labels, shift_labels, kwargs)
        loss = model_output.get("loss", None) if isinstance(model_output, dict) else model_output[0]
        if not isinstance(loss, torch.Tensor):
            return self._fallback(model_output, labels, shift_labels, kwargs)
        STATS["fused"] += 1
        return loss


def fused_label_smoothing_compute_loss(
    trainer, model, inputs, return_outputs, num_items_in_batch, call
):
    """Runs `call()` (the parent compute_loss) with label smoothing on the fused CE when it applies; else `call()`."""
    reason = _skip_reason(trainer, model, inputs, return_outputs)
    if reason is not None:
        STATS["skip_" + reason] = STATS.get("skip_" + reason, 0) + 1
        return call()
    original = trainer.label_smoother
    labels = inputs["labels"]
    if _accepts_num_items(original) and num_items_in_batch is not None:
        divisor = num_items_in_batch
    else:
        # LabelSmoother's own count: active shifted labels in this micro-batch.
        divisor = (labels[..., 1:] != original.ignore_index).sum()
    injection = _Injection(labels)

    def _inject(module, args, kwargs):
        # Only the step's first forward, and only the label-free training forward LabelSmoother would read.
        if injection.seen:
            return None
        injection.seen = True
        if (
            kwargs.get("labels", None) is not None
            or kwargs.get("fused_lm_head", False)
            or not torch.is_grad_enabled()
        ):
            return None
        injection.done = True
        injection.args, injection.kwargs = args, dict(kwargs)
        injection.state.active = True
        return args, {**kwargs, "labels": injection.labels}

    handle = model.register_forward_pre_hook(_inject, with_kwargs = True)
    try:
        with _override()(original.epsilon, divisor) as state:
            # Armed by the hook, so no other forward of this step (one with its own labels) is smoothed.
            state.active = False
            injection.state = state
            trainer.label_smoother = _FusedLabelSmoother(original, state, injection, model)
            return call()
    finally:
        trainer.label_smoother = original
        handle.remove()
