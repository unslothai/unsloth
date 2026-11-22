# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""GRPOTrainer subclasses that override TRL's ``_compute_loss`` (TRL's experimental GMPOTrainer, user objectives).

Unsloth's GRPO ``compute_loss`` computes the GRPO objective in unsloth_zoo's fused loss and never calls
``self._compute_loss``, so a subclass objective would be dead code and the run would silently train GRPO. When a
class below Unsloth's fused ``compute_loss`` in the MRO overrides ``_compute_loss``, the fused ``compute_loss`` hands
the batch to it instead. For that call ``_get_per_token_logps_and_entropies`` is routed to a scorer that keeps the
gradient (Unsloth's own scorer returns detached old / reference logprobs): text batches are scored from the final
hidden states by unsloth_zoo's fused linear kernel, so no tokens x vocab logits tensor is materialised; heads the
kernel cannot read fall back to TRL's own math on the logits."""

import contextlib
import os
import types

import torch

__all__ = [
    "grpo_subclass_compute_loss",
    "grpo_run_subclass_compute_loss",
    "grpo_subclass_logps_and_entropies",
]

# Read from the fused compute_loss's code object: its body imports this name, so the class that owns that body is
# found without a marker attribute (which a generated class body cannot set on its own function).
_FUSED_COMPUTE_LOSS_NAME = "grpo_subclass_compute_loss"


def _owns_fused_compute_loss(function):
    seen = 0
    while function is not None and seen < 8:
        code = getattr(function, "__code__", None)
        if code is not None and _FUSED_COMPUTE_LOSS_NAME in code.co_names:
            return True
        function = getattr(function, "__wrapped__", None)
        seen += 1
    return False


def grpo_subclass_compute_loss(trainer):
    """The ``_compute_loss`` function of a class below Unsloth's fused GRPO ``compute_loss`` in the MRO, or None
    when the fused loss is the objective (plain GRPO, or a subclass that only overrides other methods)."""
    mro = type(trainer).__mro__
    owner = None
    for index, klass in enumerate(mro):
        if _owns_fused_compute_loss(klass.__dict__.get("compute_loss")):
            owner = index
            break
    if owner is None:
        return None
    # Only a strict subclass of the fused class can override: the generated class carries TRL's own _compute_loss.
    for klass in mro[:owner]:
        function = klass.__dict__.get("_compute_loss")
        if function is not None:
            return function
    return None


def _returns_aux_loss():
    """TRL >= 1.7.0 unpacks (logps, entropies, aux_loss); older TRL unpacks (logps, entropies)."""
    try:
        from importlib.metadata import version

        from packaging.version import Version
        return Version(version("trl")) >= Version("1.7.0")
    except Exception:
        return True


@contextlib.contextmanager
def _return_hidden_states(enabled):
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior


def _count(trainer, key):
    if torch.compiler.is_compiling():
        return
    counts = trainer.__dict__.setdefault("_unsloth_grpo_subclass_scorer", {})
    counts[key] = counts.get(key, 0) + 1


def grpo_subclass_logps_and_entropies(
    self,
    model,
    input_ids,
    attention_mask,
    logits_to_keep,
    batch_size = None,
    compute_entropy = False,
    *args,
    **kwargs,
):
    """TRL's ``_get_per_token_logps_and_entropies`` contract with the gradient kept: per-token log-probs of the
    completion tokens under ``self.temperature`` and (detached, as TRL computes them under no_grad) entropies."""
    if args:
        raise NotImplementedError(
            "Unsloth: a GRPO subclass _compute_loss passed positional arguments after compute_entropy to "
            "_get_per_token_logps_and_entropies; pass them by keyword."
        )
    if kwargs.pop("compute_aux_loss", False):
        raise NotImplementedError(
            "Unsloth: the MoE router aux loss is not supported for GRPO subclass losses."
        )
    # Vision tensors, token type ids and sampling masks need TRL's per-chunk slicing: refuse rather than mis-slice.
    unsupported = sorted(k for k, v in kwargs.items() if v is not None)
    if unsupported:
        raise NotImplementedError(
            f"Unsloth: {type(self).__name__} overrides _compute_loss, and Unsloth scores such losses for text "
            f"batches only; got {', '.join(unsupported)}."
        )

    from .rl_replacements import _unsloth_grpo_autocast
    from ._fused_rl_logprobs import _unwrap, fused_head, score_hidden_states

    autocast_enabled, autocast_dtype = _unsloth_grpo_autocast(self)
    try:
        unwrapped = self.accelerator.unwrap_model(model)
    except Exception:
        unwrapped = _unwrap(model)
    if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        head_info = (None, None, "FSDP / DeepSpeed")
    else:
        head_info = fused_head(unwrapped)
    use_hidden = head_info[0] is not None
    temperature = float(getattr(self, "temperature", 1.0))
    model_kwarg_keys = getattr(self, "model_kwarg_keys", None)

    batch_size = batch_size or input_ids.size(0)
    all_logps, all_entropies = [], []
    for start in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[start : start + batch_size]
        attention_mask_batch = attention_mask[start : start + batch_size]
        model_inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "use_cache": False,
        }
        if model_kwarg_keys is None or "logits_to_keep" in model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        with (
            _return_hidden_states(use_hidden),
            torch.amp.autocast(
                device_type = input_ids_batch.device.type,
                dtype = autocast_dtype,
                enabled = autocast_enabled,
            ),
        ):
            output = model(**model_inputs).logits
        output = output[:, :-1, :][:, -logits_to_keep:, :]
        completion_ids = input_ids_batch[:, -logits_to_keep:]
        scored = None
        if use_hidden:
            scored = score_hidden_states(
                unwrapped,
                output,
                completion_ids,
                temperature,
                entropy = compute_entropy,
                head_info = head_info,
            )
        if scored is not None:
            logps, entropies = scored
            _count(self, "fused")
        else:
            # The forward returned real logits (no hidden-state support, or a head the kernel cannot read).
            from trl.trainer.utils import entropy_from_logits, selective_log_softmax

            logits = output / temperature
            logps = selective_log_softmax(logits, completion_ids)
            entropies = None
            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
            _count(self, "logits")
        del output
        all_logps.append(logps)
        if compute_entropy:
            all_entropies.append(entropies)
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
    logps = torch.cat(all_logps, dim = 0)
    entropies = torch.cat(all_entropies, dim = 0) if compute_entropy else None
    if _returns_aux_loss():
        return logps, entropies, None
    return logps, entropies


def grpo_run_subclass_compute_loss(trainer, loss_function, model, inputs):
    """Run a subclass ``_compute_loss`` with the gradient-carrying scorer bound on the instance for this call."""
    attribute = "_get_per_token_logps_and_entropies"
    missing = object()
    prior = trainer.__dict__.get(attribute, missing)
    trainer.__dict__[attribute] = types.MethodType(grpo_subclass_logps_and_entropies, trainer)
    try:
        loss = loss_function(trainer, model, inputs)
    finally:
        if prior is missing:
            trainer.__dict__.pop(attribute, None)
        else:
            trainer.__dict__[attribute] = prior
    if not torch.compiler.is_compiling():
        trainer._unsloth_grpo_subclass_loss_calls = (
            getattr(trainer, "_unsloth_grpo_subclass_loss_calls", 0) + 1
        )
    return loss
