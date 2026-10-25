# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""PPO / RLOO log-probs scored from hidden states by unsloth_zoo's fused linear kernel.

The model forward is the stock one, run with ``UNSLOTH_RETURN_HIDDEN_STATES=1`` so ``.logits`` carries the final
hidden states; the kernel then computes ``log_softmax(transform(h @ W.T + b) / T)[target]`` (and the entropy)
without a tokens x vocab tensor. Anything the kernel cannot reproduce keeps TRL's own path."""

import contextlib
import os

import torch

__all__ = [
    "fused_head",
    "hidden_states_forward",
    "score_hidden_states",
    "ppo_route",
    "ppo_batch_generation",
    "ppo_forward",
    "ppo_score",
    "rloo_fused_logps",
    "rloo_stock_traits",
    "ppo_policy_logprobs",
]


def _kernel():
    try:
        from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs
    except Exception:
        return None
    return fused_linear_logprobs


def _note(owner, key, reason):
    try:
        fallbacks = owner.__dict__.setdefault(key, {})
        fallbacks[reason] = fallbacks.get(reason, 0) + 1
    except Exception:
        pass


def _unwrap(model):
    seen = 0
    while seen < 4 and isinstance(
        model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    ):
        model = model.module
        seen += 1
    return model


def fused_head(model):
    """``(head, (multiply, divide, softcap), reason)``; ``head`` is None (with a reason) when the kernel cannot read
    the output head directly, or the model changes the hidden states before it."""
    from .rl_replacements import (
        _unsloth_get_model_config,
        _unsloth_gkd_dense_head,
        _unsloth_pref_logit_transforms,
        _unsloth_text_configs,
    )

    if os.environ.get("UNSLOTH_FUSED_LINEAR", "1") == "0":
        return None, None, "UNSLOTH_FUSED_LINEAR=0"
    if _kernel() is None:
        return None, None, "unsloth_zoo has no fused_linear_logprobs"
    model = _unwrap(model)
    head = _unsloth_gkd_dense_head(model)
    if head is None:
        return None, None, "output head is not a dense nn.Linear"
    # The kernel reads weight / bias directly: a hooked head is not F.linear(h, W, b).
    if (
        hasattr(head, "_hf_hook")
        or "forward" in vars(head)
        or head._forward_hooks
        or head._forward_pre_hooks
        or head._backward_hooks
        or getattr(head, "_backward_pre_hooks", None)
    ):
        return None, None, "output head has hooks or a patched forward"
    if hasattr(head.weight, "ds_id"):
        return None, None, "DeepSpeed ZeRO-3 head"
    for config in _unsloth_text_configs(_unsloth_get_model_config(model)):
        if getattr(config, "model_type", None) == "minicpm3":
            return None, None, "minicpm3 scales hidden states before the head"
    return head, _unsloth_pref_logit_transforms(model), None


@contextlib.contextmanager
def hidden_states_forward(enabled = True):
    """Unsloth's CausalLM forwards return the final hidden states as ``.logits`` while this is active."""
    if not enabled:
        yield
        return
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior


def score_hidden_states(
    model,
    tensor,
    targets,
    temperature,
    mask = None,
    entropy = False,
    head_info = None,
):
    """``(logps, entropy | None)`` from ``tensor`` when it is the final hidden states of ``model``, else None (the
    forward ignored the flag and ``tensor`` is real logits). Entropy is detached, as every caller logs it under
    ``no_grad``; rows outside ``mask`` are 0."""
    from .rl_replacements import _unsloth_pref_returns_hidden_states

    head, transforms, _ = head_info if head_info is not None else fused_head(model)
    if head is None:
        return None
    if not _unsloth_pref_returns_hidden_states(_unwrap(model), tensor, head.weight):
        return None
    multiply, divide, softcap = transforms
    out = _kernel()(
        tensor.to(head.weight.device),
        head.weight,
        targets,
        bias = head.bias,
        mask = mask,
        temperature = float(temperature),
        logit_scale_multiply = multiply,
        logit_scale_divide = divide,
        logit_softcapping = softcap,
        return_entropy = bool(entropy),
        entropy_requires_grad = False,
    )
    if entropy:
        logps, ent = out
        return logps.to(targets.device), ent.detach().to(targets.device)
    return out.to(targets.device), None


class _PPORoute:
    __slots__ = ("trainer",)

    def __init__(self, trainer):
        self.trainer = trainer


def ppo_route(trainer):
    """A route object when PPOTrainer.train can score from hidden states, else None (TRL's logits path runs).
    The policy head is checked here; the reference model is checked per call."""
    reason = None
    if getattr(trainer, "is_fsdp_enabled", False) or getattr(
        trainer, "is_deepspeed_enabled", False
    ):
        reason = "FSDP / DeepSpeed"
    if reason is None:
        try:
            wrapper = trainer.accelerator.unwrap_model(trainer.model)
            policy = wrapper.policy
        except Exception:
            policy = None
            reason = "no PolicyAndValueWrapper.policy"
    if reason is None:
        head, _, reason = fused_head(policy)
    if reason is None and getattr(trainer.model, "find_unused_parameters", False):
        if any(p is not None and p.requires_grad for p in (head.weight, head.bias)):
            reason = "DDP find_unused_parameters with a trainable head"
    if reason is not None:
        _note(trainer, "_unsloth_ppo_fused_fallbacks", reason)
        return None
    return _PPORoute(trainer)


def _right_pad_cat(tensors, value):
    width = max(t.shape[1] for t in tensors)
    return torch.cat(
        [torch.nn.functional.pad(t, (0, width - t.shape[1]), value = value) for t in tensors], dim = 0
    )


@torch.no_grad()
def ppo_batch_generation(
    model, queries, local_rollout_forward_batch_size, pad_token_id, generation_config
):
    """TRL's ``batch_generation`` without ``output_scores``: the rollout log-probs are scored from hidden states, so
    the stacked float32 [batch, response, vocab] scores are never built. Returns ``(query_responses, None)``."""
    query_responses = []
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        query = queries[i : i + local_rollout_forward_batch_size]
        context_length = query.shape[1]
        attention_mask = query != pad_token_id
        input_ids = torch.masked_fill(query, ~attention_mask, 0)
        output = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            generation_config = generation_config,
            return_dict_in_generate = True,
        )
        query_responses.append(torch.cat((query, output.sequences[:, context_length:]), dim = 1))
    return _right_pad_cat(query_responses, pad_token_id), None


def ppo_forward(route, forward, model, query_responses, pad_token_id):
    """TRL's ``forward(model, query_responses, pad_token_id)``, with Unsloth's forwards returning hidden states
    only when ``ppo_score`` will score this model's head (a reference model is not checked by ``ppo_route``)."""
    enabled = route is not None and fused_head(_ppo_policy(route, model))[0] is not None
    with hidden_states_forward(enabled):
        return forward(model, query_responses, pad_token_id)


def _ppo_policy(route, model):
    try:
        model = route.trainer.accelerator.unwrap_model(model)
    except Exception:
        model = _unwrap(model)
    return getattr(model, "policy", model)


def ppo_score(
    route,
    model,
    tensor,
    responses,
    temperature,
    entropy = False,
):
    """``(logps, entropy | None)`` for PPO from ``tensor = output.logits[:, context_length - 1 : -1]``, or None when
    the route is off or the forward returned real logits (TRL's lines then run on them unchanged). ``model`` is the
    PolicyAndValueWrapper (its policy head scores) or a standalone reference policy."""
    if route is None:
        return None
    model = _ppo_policy(route, model)
    head_info = fused_head(model)
    if head_info[0] is None:
        _note(route.trainer, "_unsloth_ppo_fused_fallbacks", head_info[2])
        return None
    out = score_hidden_states(
        model, tensor, responses, temperature, entropy = entropy, head_info = head_info
    )
    if out is None:
        _note(
            route.trainer,
            "_unsloth_ppo_fused_fallbacks",
            "forward returned logits, not hidden states",
        )
        return None
    route.trainer.__dict__["_unsloth_ppo_fused_calls"] = (
        route.trainer.__dict__.get("_unsloth_ppo_fused_calls", 0) + 1
    )
    return out


@torch.no_grad()
def ppo_policy_logprobs(
    route,
    forward,
    selective_log_softmax,
    model,
    query_responses,
    responses,
    context_length,
    pad_token_id,
    temperature,
):
    """Rollout log-probs of the current policy, recomputed by the same forward and scorer as the training pass
    (the stock path reads them off the generation scores, which ``ppo_batch_generation`` no longer keeps). A forward
    that returned real logits is scored with TRL's formula on them."""
    policy = _ppo_policy(route, model)
    output = ppo_forward(route, forward, policy, query_responses, pad_token_id)
    tensor = output.logits[:, context_length - 1 : -1]
    out = ppo_score(route, policy, tensor, responses, temperature)
    if out is not None:
        return out[0]
    return selective_log_softmax(tensor / temperature, responses)


_RLOO_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "num_images",
    "pixel_attention_mask",
    "spatial_shapes",
    "num_tiles",
    "image_sizes",
    "token_type_ids",
    "mm_token_type_ids",
    "image_position_ids",
)


def rloo_stock_traits(source):
    """What the generated dispatcher bakes in from TRL's body: whether it returns the MoE aux loss (a third value)
    and whether it zeroes the log-probs / entropies of padded completion tokens (TRL 1.14's ``row_mask``)."""
    return {
        "aux": "compute_aux_loss" in source.split(")", 1)[0],
        "row_mask": "row_mask=" in source.replace(" ", ""),
    }


def rloo_fused_logps(
    self,
    traits,
    model,
    input_ids,
    attention_mask,
    logits_to_keep,
    batch_size,
    compute_entropy,
    args,
    kwargs,
):
    """RLOOTrainer._get_per_token_logps_and_entropies (TRL < 1.15, which reads ``model(**inputs).logits``) from
    hidden states. None -> the caller runs TRL's body. Text only; vision and MoE aux loss keep TRL's path."""
    reason = None
    if self.__dict__.get("_unsloth_rloo_fused_off"):
        return None
    if args:
        reason = "positional extras"
    elif any(kwargs.get(k) is not None for k in _RLOO_VISION_KEYS):
        reason = "vision inputs"
    elif kwargs.get("compute_aux_loss", False):
        reason = "router aux loss needs the stock forward"
    elif set(kwargs) - set(_RLOO_VISION_KEYS) - {"compute_aux_loss"}:
        reason = "unknown keyword arguments"
    elif getattr(self, "is_fsdp_enabled", False) or getattr(self, "is_deepspeed_enabled", False):
        reason = "FSDP / DeepSpeed"
    head_info = None
    if reason is None:
        try:
            unwrapped = self.accelerator.unwrap_model(model)
        except Exception:
            unwrapped = _unwrap(model)
        head_info = fused_head(unwrapped)
        reason = head_info[2]
    if reason is None and getattr(model, "find_unused_parameters", False):
        head = head_info[0]
        if any(p is not None and p.requires_grad for p in (head.weight, head.bias)):
            reason = "DDP find_unused_parameters with a trainable head"
    if reason is not None:
        _note(self, "_unsloth_rloo_fused_fallbacks", reason)
        return None

    batch_size = batch_size or input_ids.size(0)
    all_logps, all_entropies = [], []
    for start in range(0, input_ids.size(0), batch_size):
        input_ids_batch = input_ids[start : start + batch_size]
        attention_mask_batch = attention_mask[start : start + batch_size]
        model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
        if "logits_to_keep" in getattr(self, "model_kwarg_keys", ()):
            model_inputs["logits_to_keep"] = logits_to_keep + 1
        model_inputs["use_cache"] = False
        with hidden_states_forward():
            hidden = model(**model_inputs).logits
        hidden = hidden[:, :-1, :][:, -logits_to_keep:, :]
        completion_ids = input_ids_batch[:, -logits_to_keep:]
        mask = attention_mask_batch[:, -logits_to_keep:] if traits["row_mask"] else None
        out = score_hidden_states(
            unwrapped,
            hidden,
            completion_ids,
            self.temperature,
            mask = mask,
            entropy = compute_entropy,
            head_info = head_info,
        )
        if out is None:
            # The forward ignored the flag: hand this model to TRL from now on (one wasted forward).
            self.__dict__["_unsloth_rloo_fused_off"] = True
            _note(
                self, "_unsloth_rloo_fused_fallbacks", "forward returned logits, not hidden states"
            )
            return None
        all_logps.append(out[0])
        if compute_entropy:
            all_entropies.append(out[1])
    self.__dict__["_unsloth_rloo_fused_calls"] = (
        self.__dict__.get("_unsloth_rloo_fused_calls", 0) + 1
    )
    logps = torch.cat(all_logps, dim = 0)
    entropies = torch.cat(all_entropies, dim = 0) if compute_entropy else None
    if traits["aux"]:
        return logps, entropies, None
    return logps, entropies
