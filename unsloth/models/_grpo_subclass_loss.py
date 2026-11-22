# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""GRPOTrainer subclasses that override TRL's ``_compute_loss`` (TRL's experimental GMPOTrainer, user objectives).

Unsloth's GRPO ``compute_loss`` computes the GRPO objective in unsloth_zoo's fused loss and never calls
``self._compute_loss``, so a subclass objective would be dead code and the run would silently train GRPO. When a
class below Unsloth's fused ``compute_loss`` in the MRO overrides ``_compute_loss``, the fused ``compute_loss`` hands
the batch to it instead, through ordinary method lookup.

For such a trainer Unsloth's generated ``_get_per_token_logps_and_entropies`` (and the legacy
``_get_per_token_logps``) delegate every scoring pass to ``grpo_subclass_logps_and_entropies``: the rollout's old /
reference passes and the objective's gradient pass share one scorer, so the three tensors have the completion's
width (Unsloth's own scorer returns a left-packed window that only the fused loss understands) and the importance
ratio starts at exactly 1. Text batches are scored from the final hidden states by unsloth_zoo's fused linear kernel,
so no tokens x vocab logits tensor is materialised; heads the kernel cannot read, a trainable head under DDP with
``find_unused_parameters``, and batches carrying vLLM's sampling candidate sets use TRL's math on the logits."""

import contextlib
import functools
import os

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
    """The ``_compute_loss`` attribute of a class below Unsloth's fused GRPO ``compute_loss`` in the MRO, or None
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


@functools.lru_cache(maxsize = 1)
def _trl_version():
    try:
        from importlib.metadata import version

        from packaging.version import Version
        return Version(version("trl"))
    except Exception:
        return None


def _trl_at_least(minimum):
    from packaging.version import Version
    found = _trl_version()
    return True if found is None else found >= Version(minimum)


def _returns_aux_loss():
    """TRL >= 1.7.0 unpacks (logps, entropies, aux_loss); older TRL unpacks (logps, entropies)."""
    return _trl_at_least("1.7.0")


def _modern_scorer_contract():
    """TRL >= 1.15 scores through its fused lm_head: completion pads score 0 and the entropy keeps its gradient.
    Older TRL leaves pad scores as they come and computes the entropy under no_grad."""
    return _trl_at_least("1.15.0")


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


@contextlib.contextmanager
def _keep_left_padding_mask(model):
    """Unsloth's training forwards may drop the attention mask (they assume right padding); GRPO prompts are left
    padded, so keep it for the scoring forward, as PPO and distillation do."""
    missing = object()
    try:
        states = [
            (module, getattr(module, "_unsloth_keep_padding_mask", missing))
            for module in model.modules()
            if hasattr(module, "embed_tokens")
        ]
    except Exception:
        states = []
    for module, _ in states:
        module._unsloth_keep_padding_mask = True
    try:
        yield
    finally:
        for module, prior in states:
            if prior is missing:
                module.__dict__.pop("_unsloth_keep_padding_mask", None)
            else:
                module._unsloth_keep_padding_mask = prior


def _count(trainer, key):
    if torch.compiler.is_compiling():
        return
    counts = trainer.__dict__.setdefault("_unsloth_grpo_subclass_scorer", {})
    counts[key] = counts.get(key, 0) + 1


def _ddp_finds_unused_parameters(trainer, model):
    """Whether the forward runs inside a DDP wrapper that marks parameters the forward did not reach as ready."""
    from .rl import _unsloth_find_ddp

    for candidate in (model, getattr(trainer, "model_wrapped", None)):
        if getattr(candidate, "find_unused_parameters", False):
            return True
        ddp = _unsloth_find_ddp(candidate)
        if ddp is not None and getattr(ddp, "find_unused_parameters", False):
            return True
    return False


def _temperature(trainer):
    raw = getattr(trainer, "temperature", None)
    if raw is None:
        raw = getattr(getattr(trainer, "args", None), "temperature", None)
    return float(1.0 if raw is None else raw)


def _support_log_softmax(logits, targets, support):
    """log_softmax over each token's sampling candidates (``support``: ids padded with -1), as vLLM normalised the
    sampled token's logprob; positions without candidates use the whole vocabulary."""
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    valid = support >= 0
    gathered = logits.gather(-1, support.clamp(min = 0).to(torch.long))
    # A finite floor, not -inf: rows without candidates take the full vocabulary below, and a logsumexp over -inf
    # alone would send NaN through torch.where's zero gradient.
    gathered = gathered.masked_fill(~valid, torch.finfo(gathered.dtype).min)
    log_z = torch.where(
        valid.any(-1),
        torch.logsumexp(gathered, dim = -1),
        torch.logsumexp(logits, dim = -1),
    )
    return logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - log_z


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
    """TRL's ``_get_per_token_logps_and_entropies`` contract for a subclass objective, for every pass (rollout old /
    reference scores under no_grad, the objective's gradient pass): per-token log-probs of the completion tokens
    under the trainer's temperature, and entropies. With TRL >= 1.15 completion pads score 0 and the entropy keeps
    its gradient, as TRL's fused lm_head returns them; older TRL computes the entropy under no_grad."""
    if args:
        raise NotImplementedError(
            "Unsloth: a GRPO subclass _compute_loss passed positional arguments after compute_entropy to "
            "_get_per_token_logps_and_entropies; pass them by keyword."
        )
    if kwargs.pop("compute_aux_loss", False):
        raise NotImplementedError(
            "Unsloth: the MoE router aux loss is not supported for GRPO subclass losses."
        )
    # vLLM's sampling candidates (TRL >= 1.18): the log-probs are normalised over them, as vLLM's were.
    sampling_mask = kwargs.pop("sampling_mask", None)
    # Vision tensors and token type ids need TRL's per-chunk slicing: refuse rather than mis-slice.
    unsupported = sorted(k for k, v in kwargs.items() if v is not None)
    if unsupported:
        raise NotImplementedError(
            f"Unsloth: {type(self).__name__} overrides _compute_loss, and Unsloth scores such losses for text "
            f"batches only; got {', '.join(unsupported)}."
        )

    from .rl_replacements import _unsloth_grpo_autocast, _unsloth_grpo_refuse_unscored_support
    from ._fused_rl_logprobs import _unwrap, fused_head, score_hidden_states

    # Kept-set scoring rebuilds the sampler's top_k / top_p / min_p set from Unsloth's own scorer, which a subclass
    # objective never runs: refuse it (with a warning) so every pass scores the same distribution.
    _unsloth_grpo_refuse_unscored_support(self)

    modern = _modern_scorer_contract()
    returns_aux = _returns_aux_loss()
    if logits_to_keep == 0:
        # input_ids[:, -0:] is the whole row: return the empty completion window instead.
        empty = torch.zeros((input_ids.size(0), 0), dtype = torch.float32, device = input_ids.device)
        entropies = empty.clone() if compute_entropy else None
        return (empty, entropies, None) if returns_aux else (empty, entropies)

    autocast_enabled, autocast_dtype = _unsloth_grpo_autocast(self)
    try:
        unwrapped = self.accelerator.unwrap_model(model)
    except Exception:
        unwrapped = _unwrap(model)
    if getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        head_info = (None, None, "FSDP / DeepSpeed")
    elif sampling_mask is not None:
        head_info = (None, None, "sampling candidate sets")
    else:
        head_info = fused_head(unwrapped)
    head = head_info[0]
    if (
        head is not None
        and any(p is not None and p.requires_grad for p in (head.weight, head.bias))
        and _ddp_finds_unused_parameters(self, model)
    ):
        # The head would be used outside the DDP forward, so DDP marks it ready twice. Decided for every pass alike,
        # so the old / reference scores come from the same route as the gradient pass.
        head_info = (None, None, "DDP find_unused_parameters with a trainable head")
    use_hidden = head_info[0] is not None
    temperature = _temperature(self)
    model_kwarg_keys = getattr(self, "model_kwarg_keys", None)
    entropy_grad = bool(compute_entropy and modern and torch.is_grad_enabled())

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
        completion_ids = input_ids_batch[:, -logits_to_keep:]
        completion_mask = attention_mask_batch[:, -logits_to_keep:].bool() if modern else None
        with torch.amp.autocast(
            device_type = input_ids_batch.device.type,
            dtype = autocast_dtype,
            enabled = autocast_enabled,
        ):
            with _return_hidden_states(use_hidden), _keep_left_padding_mask(unwrapped):
                output = model(**model_inputs).logits
            output = output[:, :-1, :][:, -logits_to_keep:, :]
            scored = None
            if use_hidden:
                # Inside the forward's autocast, as the old-policy projection runs.
                scored = score_hidden_states(
                    unwrapped,
                    output,
                    completion_ids,
                    temperature,
                    mask = completion_mask,
                    entropy = compute_entropy,
                    head_info = head_info,
                    entropy_requires_grad = None if entropy_grad else False,
                )
        if scored is not None:
            logps, entropies = scored
            _count(self, "fused")
        else:
            # The forward returned real logits (no hidden-state support, or a head the kernel cannot read).
            from trl.trainer.utils import entropy_from_logits, selective_log_softmax

            logits = output / temperature
            if sampling_mask is not None:
                support = sampling_mask[start : start + batch_size].to(logits.device)
                logps = _support_log_softmax(logits, completion_ids, support)
                _count(self, "candidates")
            else:
                logps = selective_log_softmax(logits, completion_ids)
                _count(self, "logits")
            entropies = None
            if compute_entropy:
                with torch.set_grad_enabled(entropy_grad):
                    entropies = entropy_from_logits(logits)
            if completion_mask is not None:
                logps = logps.masked_fill(~completion_mask, 0)
                if entropies is not None:
                    entropies = entropies.masked_fill(~completion_mask, 0)
        del output
        all_logps.append(logps)
        if compute_entropy:
            all_entropies.append(entropies)
    logps = torch.cat(all_logps, dim = 0)
    entropies = torch.cat(all_entropies, dim = 0) if compute_entropy else None
    if returns_aux:
        return logps, entropies, None
    return logps, entropies


def _warn_gmpo_ignores_vllm_correction(trainer):
    if getattr(trainer, "_unsloth_warned_gmpo_vllm_correction", False):
        return
    if not (
        getattr(trainer, "use_vllm", False)
        and getattr(trainer, "vllm_importance_sampling_correction", False)
    ):
        return
    objective = grpo_subclass_compute_loss(trainer)
    module = getattr(objective, "__module__", "") or ""
    if not module.startswith("trl.experimental.gmpo"):
        return
    import warnings

    warnings.warn(
        "Unsloth: TRL's GMPO objective does not apply the vLLM importance-sampling correction "
        "(vllm_importance_sampling_correction=True); its ratio is still computed and logged.",
        stacklevel = 3,
    )
    trainer._unsloth_warned_gmpo_vllm_correction = True


def grpo_run_subclass_compute_loss(trainer, model, inputs):
    """Run the subclass ``_compute_loss`` through ordinary method lookup, so a subclass's own scoring methods (and the
    per-instance context parallel wrappers) stay in place; Unsloth's scorer delegates to the one above."""
    if not torch.compiler.is_compiling():
        _warn_gmpo_ignores_vllm_correction(trainer)
    loss = trainer._compute_loss(model, inputs)
    if not torch.compiler.is_compiling():
        trainer._unsloth_grpo_subclass_loss_calls = (
            getattr(trainer, "_unsloth_grpo_subclass_loss_calls", 0) + 1
        )
    return loss
