# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "PatchDPOTrainer",
    "PatchKTOTrainer",
    "fused_preference_loss_enabled",
    "unsloth_fused_dpo_concatenated_forward",
    "unsloth_fused_orpo_concatenated_forward",
]

import inspect
import os
import weakref

import torch


def PatchDPOTrainer():
    return


def PatchKTOTrainer():
    return


# Opt-in: UNSLOTH_FUSED_PREFERENCE_LOSS=1 scores DPO / ORPO / CPO completions from the
# final hidden states in row chunks (unsloth_zoo.fused_losses.preference_loss) instead of
# materialising batch x seq_len x vocab logits. Every helper below returns None when it
# cannot reproduce TRL's concatenated_forward semantics, and the caller then runs TRL's own.
_TEXT_ONLY_KEYS = frozenset(
    (
        "prompt_input_ids",
        "prompt_attention_mask",
        "completion_input_ids",
        "completion_attention_mask",
    )
)
_FLOAT_HEAD_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def fused_preference_loss_enabled():
    return os.environ.get("UNSLOTH_FUSED_PREFERENCE_LOSS", "0") == "1"


def _zoo_kernels():
    try:
        from unsloth_zoo.fused_losses.preference_loss import (
            fused_linear_token_logps,
            linear_mean_logits,
        )
    except Exception:
        return None
    return fused_linear_token_logps, linear_mean_logits


def _wrappers(model):
    seen = 0
    while model is not None and seen < 4:
        yield model
        model = getattr(model, "_orig_mod", None)
        seen += 1


def _uses_fsdp1(trainer, model):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except Exception:
        return False
    cache = vars(trainer).setdefault("_unsloth_fsdp1_models", weakref.WeakKeyDictionary())
    if model not in cache:
        cache[model] = any(isinstance(m, FullyShardedDataParallel) for m in model.modules())
    return cache[model]


def _output_head(trainer, model):
    # FSDP1 frees or reshards the unwrapped head's parameter view outside its own forward.
    if isinstance(model, torch.nn.Module) and _uses_fsdp1(trainer, model):
        return None, None
    try:
        unwrapped = trainer.accelerator.unwrap_model(model)
    except Exception:
        unwrapped = model
    get_head = getattr(unwrapped, "get_output_embeddings", None)
    if get_head is None:
        return None, None
    try:
        head = get_head()
    except Exception:
        return None, None
    # A LoRA / modules_to_save head computes more than weight @ h; leave it to its own forward.
    if head is None or hasattr(head, "base_layer") or hasattr(head, "modules_to_save"):
        return None, None
    # The kernel reads weight / bias directly: only a plain, unhooked nn.Linear is exactly F.linear(h, W, b).
    if (
        type(head) is not torch.nn.Linear
        or hasattr(head, "_hf_hook")
        or "forward" in vars(head)
        or head._forward_hooks
        or head._forward_pre_hooks
    ):
        return None, None
    weight = getattr(head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None, None
    if weight.dtype not in _FLOAT_HEAD_DTYPES or hasattr(weight, "quant_state"):
        return None, None
    if type(weight).__name__ == "DTensor":
        return None, None
    bias = getattr(head, "bias", None)
    bias = bias if isinstance(bias, torch.Tensor) else None
    # DDP(find_unused_parameters = True) marks a trainable head unused once model(...) returns,
    # so applying it afterwards makes backward mark it ready a second time and raise.
    # Accelerate's dynamo integration puts DDP under torch.compile's _orig_mod.
    ddp = next(
        (m for m in _wrappers(model) if isinstance(m, torch.nn.parallel.DistributedDataParallel)),
        None,
    )
    if ddp is not None and getattr(ddp, "find_unused_parameters", False):
        if any(p is not None and p.requires_grad for p in (weight, bias)):
            return None, None
    # vocab_size == hidden_size: the output width cannot tell hidden states from logits, so
    # only a forward that explicitly honours UNSLOTH_RETURN_HIDDEN_STATES is trusted.
    if weight.shape[0] == weight.shape[1]:
        try:
            from .rl_replacements import _unsloth_grpo_hidden_states_signal
            trusted = any(_unsloth_grpo_hidden_states_signal(m) is True for m in (model, unwrapped))
        except Exception:
            trusted = False
        if not trusted:
            return None, None
    return unwrapped, (weight, bias)


def _logit_transforms(unwrapped):
    config = getattr(unwrapped, "config", None)
    try:
        from unsloth_zoo.device_map_planner import detect_logit_transforms
        found = detect_logit_transforms(config)
        return (
            float(found["logit_scale_multiply"] or 0.0),
            float(found["logit_scale_divide"] or 0.0),
            float(found["logit_softcapping"] or 0.0),
        )
    except Exception:
        pass
    try:
        from .rl_replacements import (
            _unsloth_get_final_logit_softcapping,
            _unsloth_resolve_logit_scales,
        )
        multiply, divide = _unsloth_resolve_logit_scales(config)
        softcap = _unsloth_get_final_logit_softcapping(unwrapped)
    except Exception:
        # Unknown transforms would score the wrong logits: let TRL's forward run instead.
        return None
    return float(multiply or 0.0), float(divide or 0.0), float(softcap or 0.0)


def _forward_hidden_states(model, unwrapped, head_weight, args, kwargs):
    """Run the model with UNSLOTH_RETURN_HIDDEN_STATES=1; returns ``(tensor, is_hidden)``."""
    prior = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        outputs = model(*args, **kwargs)
    finally:
        if prior is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prior
    tensor = outputs.logits
    try:
        from .rl_replacements import _unsloth_grpo_returns_hidden_states
        is_hidden = _unsloth_grpo_returns_hidden_states(unwrapped, tensor, head_weight)
    except Exception:
        is_hidden = tensor.shape[-1] == head_weight.shape[1] != head_weight.shape[0]
    return tensor, bool(is_hidden)


def _pad_value(trainer):
    # TRL <= 0.23 names it padding_value, later releases pad_token_id.
    if "pad_token_id" in vars(trainer):
        return trainer.pad_token_id
    return getattr(trainer, "padding_value", None)


def _token_logps(
    hidden_or_logits,
    is_hidden,
    head,
    transforms,
    labels,
    mask,
    want_lssp = False,
):
    """Per-token logps (0 off ``mask``) plus the no-grad vocab-mean of the logits at each position."""
    fused_linear_token_logps, _ = _zoo_kernels()
    if is_hidden:
        out = fused_linear_token_logps(
            hidden_or_logits,
            head[0],
            head[1],
            labels,
            mask,
            logit_scale_multiply = transforms[0],
            logit_scale_divide = transforms[1],
            logit_softcapping = transforms[2],
            return_mean_logits = True,
            return_log_sum_sq_probs = want_lssp,
        )
        return out["per_token_logps"], out["mean_logits"], out.get("log_sum_sq_probs")
    # The forward ignored the flag and returned real logits: score them the way TRL does.
    from trl.trainer.utils import selective_log_softmax

    logits = hidden_or_logits
    per_token_logps = selective_log_softmax(logits, labels.masked_fill(~mask, 0))
    per_token_logps = per_token_logps.masked_fill(~mask, 0)
    with torch.no_grad():
        mean_logits = logits.mean(-1).masked_fill(~mask, 0)
        lssp = None
        if want_lssp:
            lssp = torch.logsumexp(2 * torch.log_softmax(logits, dim = -1), dim = -1)
    return per_token_logps, mean_logits, lssp


def unsloth_fused_dpo_concatenated_forward(
    trainer,
    model,
    batch,
    is_ref_model = False,
):
    """TRL 0.22-0.28 ``DPOTrainer.concatenated_forward`` for text-only decoder models, scored
    from hidden states. Returns ``None`` (run TRL's own) whenever the inputs leave that case."""
    if not fused_preference_loss_enabled() or _zoo_kernels() is None:
        return None
    if getattr(trainer, "is_encoder_decoder", False) or getattr(trainer, "aux_loss_enabled", False):
        return None
    unwrapped, head = _output_head(trainer, model)
    if head is None:
        return None
    transforms = _logit_transforms(unwrapped)
    if transforms is None:
        return None
    from trl.trainer.utils import flush_left, flush_right

    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = trainer.concatenated_inputs(batch, padding_value = _pad_value(trainer))
    if not set(concatenated_batch).issubset(_TEXT_ONLY_KEYS):
        return None

    model_kwargs = {"use_cache": False}
    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim = 1)
    attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim = 1)
    loss_mask = torch.cat(
        (torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim = 1
    )

    max_length = trainer.max_length
    if max_length is not None and max_length < attention_mask.size(1):
        if trainer.truncation_mode == "keep_start":
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            attention_mask = attention_mask[:, :max_length]
            input_ids = input_ids[:, :max_length]
            loss_mask = loss_mask[:, :max_length]
        elif trainer.truncation_mode == "keep_end":
            attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
            loss_mask = loss_mask[:, -max_length:]
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
        else:
            return None
    else:
        attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

    use_logits_to_keep = getattr(trainer, "use_logits_to_keep", False)
    if use_logits_to_keep:
        first_compute_index = loss_mask.nonzero(as_tuple = True)[1].min()
        logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
        model_kwargs["logits_to_keep"] = logits_to_keep

    padding_free = getattr(trainer, "padding_free", False)
    if padding_free:
        input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
        loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
        position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
        model_kwargs["position_ids"] = position_ids
    else:
        model_kwargs["attention_mask"] = attention_mask

    hidden, is_hidden = _forward_hidden_states(
        model, unwrapped, head[0], (input_ids,), model_kwargs
    )

    labels = torch.roll(input_ids, shifts = -1, dims = 1)
    loss_mask = torch.roll(loss_mask, shifts = -1, dims = 1).bool()
    if use_logits_to_keep:
        labels = labels[:, -logits_to_keep:]
        loss_mask = loss_mask[:, -logits_to_keep:]
    if hidden.shape[:2] != labels.shape[:2]:
        hidden = hidden[:, -labels.shape[1] :]

    labels[~loss_mask] = 0
    use_weighting = getattr(trainer, "use_weighting", False)
    per_token_logps, mean_logits, log_sum_sq_probs = _token_logps(
        hidden,
        is_hidden,
        head,
        transforms,
        labels,
        loss_mask,
        want_lssp = use_weighting,
    )
    del hidden
    unrolled_logps = per_token_logps
    per_token_logps = torch.roll(per_token_logps, shifts = 1, dims = 1)

    if padding_free:
        batch_size, seq_len = attention_mask.shape
        per_token_logps_ = per_token_logps.new_zeros(batch_size, seq_len)
        per_token_logps_[attention_mask.bool()] = per_token_logps
        per_token_logps = per_token_logps_

    all_logps = per_token_logps[:, 1:].sum(-1)
    output = {}
    args = trainer.args
    loss_type = trainer.loss_type

    if use_weighting:
        with torch.no_grad():
            per_token_logps_adjusted = per_token_logps - log_sum_sq_probs
            all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
            chosen_weights = all_weights[:num_examples]
            rejected_weights = all_weights[num_examples:]
            output["policy_weights"] = torch.clamp(
                torch.exp(chosen_weights + rejected_weights), max = 1
            )

    if getattr(args, "rpo_alpha", None) is not None or "sft" in loss_type:
        # F.cross_entropy(chosen logits[:, :-1], labels[:, :-1], ignore_index = 0), mean over kept tokens.
        chosen_labels = labels[:num_examples, :-1]
        keep = chosen_labels != 0
        output["nll_loss"] = -(unrolled_logps[:num_examples, :-1] * keep).sum() / keep.sum()

    if "ipo" in loss_type:
        all_logps = all_logps / loss_mask.sum(-1)

    ld_alpha = getattr(args, "ld_alpha", None)
    if ld_alpha is not None and not is_ref_model:
        completion_lengths = loss_mask.sum(dim = 1)
        chosen_lengths = completion_lengths[:num_examples]
        rejected_lengths = completion_lengths[num_examples:]
        public_lengths = torch.min(chosen_lengths, rejected_lengths)
        public_lengths = torch.cat([public_lengths, public_lengths], dim = 0)
        seq_len = per_token_logps.size(1)
        position_ids = torch.arange(seq_len, device = per_token_logps.device).expand_as(
            per_token_logps
        )
        ld_mask = position_ids < public_lengths.unsqueeze(1)
        mask = position_ids < completion_lengths.unsqueeze(1)
        front_mask = (ld_mask & mask).float()
        rear_mask = (~ld_mask & mask).float()
        front_logps = (per_token_logps * front_mask).sum(dim = 1)
        rear_logps = (per_token_logps * rear_mask).sum(dim = 1)
        all_logps = front_logps + ld_alpha * rear_logps

    output["chosen_logps"] = all_logps[:num_examples]
    output["rejected_logps"] = all_logps[num_examples:]

    # logits[rows][loss_mask[rows]].mean() is the mean of the per-position vocab means.
    if padding_free:
        split_idx = (position_ids == 0).nonzero(as_tuple = True)[1][num_examples]
        chosen_sel = loss_mask[0, :split_idx]
        rejected_sel = loss_mask[0, split_idx:]
        mean_chosen_logits = mean_logits[0, :split_idx][chosen_sel].mean()
        mean_rejected_logits = mean_logits[0, split_idx:][rejected_sel].mean()
    else:
        mean_chosen_logits = mean_logits[:num_examples][loss_mask[:num_examples]].mean()
        mean_rejected_logits = mean_logits[num_examples:][loss_mask[num_examples:]].mean()
    output["mean_chosen_logits"] = mean_chosen_logits
    output["mean_rejected_logits"] = mean_rejected_logits
    return output


def unsloth_fused_orpo_concatenated_forward(
    trainer,
    model,
    batch,
    method = "orpo",
):
    """TRL ``ORPOTrainer`` / ``CPOTrainer.concatenated_forward`` (0.22 through the 1.x
    experimental copies) for decoder-only models, scored from hidden states. ``None`` means
    run TRL's own."""
    if not fused_preference_loss_enabled() or _zoo_kernels() is None:
        return None
    if getattr(trainer, "is_encoder_decoder", False) or getattr(trainer, "aux_loss_enabled", False):
        return None
    label_pad_token_id = getattr(trainer, "label_pad_token_id", -100)
    # TRL's NLL is nn.CrossEntropyLoss(), which only ignores -100.
    if label_pad_token_id != -100:
        return None
    unwrapped, head = _output_head(trainer, model)
    if head is None:
        return None
    transforms = _logit_transforms(unwrapped)
    if transforms is None:
        return None

    params = inspect.signature(trainer.concatenated_inputs).parameters
    call_kwargs = {"is_encoder_decoder": False, "padding_value": _pad_value(trainer)}
    if "label_pad_token_id" in params:
        call_kwargs["label_pad_token_id"] = label_pad_token_id
    if "device" in params:
        call_kwargs["device"] = trainer.accelerator.device
    concatenated_batch = trainer.concatenated_inputs(batch, **call_kwargs)
    if set(concatenated_batch) - {
        "concatenated_input_ids",
        "concatenated_attention_mask",
        "concatenated_labels",
    }:
        return None
    len_chosen = batch["chosen_labels"].shape[0]
    input_ids = concatenated_batch["concatenated_input_ids"]
    attention_mask = concatenated_batch["concatenated_attention_mask"]
    concatenated_labels = concatenated_batch["concatenated_labels"]

    # Position t scores token t + 1. The NLL and the preference logps both read input_ids there.
    target_ids = input_ids[:, 1:]
    logps_mask = concatenated_labels[:, 1:] != label_pad_token_id
    if bool((concatenated_labels[:, 1:][logps_mask] != target_ids[logps_mask]).any()):
        return None
    if method == "orpo":
        nll_labels = torch.where(attention_mask == 1, input_ids, -100)[:, 1:]
        average_log_prob = True
    else:
        nll_labels = concatenated_labels[:, 1:]
        average_log_prob = trainer.loss_type in ["ipo", "simpo"]
    nll_mask = nll_labels != -100
    nll_mask[len_chosen:] = False
    need_nll = method == "orpo" or trainer.cpo_alpha != 0

    hidden, is_hidden = _forward_hidden_states(
        model,
        unwrapped,
        head[0],
        (input_ids,),
        {"attention_mask": attention_mask, "use_cache": False},
    )
    score_mask = logps_mask | nll_mask if need_nll else logps_mask
    per_token_logps, _, _ = _token_logps(
        hidden[:, :-1],
        is_hidden,
        head,
        transforms,
        target_ids,
        score_mask,
    )

    if need_nll:
        chosen_nll = nll_mask[:len_chosen]
        nll_loss = -(per_token_logps[:len_chosen] * chosen_nll).sum() / chosen_nll.sum()
    else:
        nll_loss = torch.tensor(0.0).to(trainer.accelerator.device)

    logps = (per_token_logps * logps_mask).sum(-1)
    if average_log_prob:
        logps = logps / logps_mask.sum(-1)
    chosen_logps = logps[:len_chosen]
    rejected_logps = logps[len_chosen:]

    # The trainers only log logits.detach().mean(); per-position vocab means average to the same value.
    if is_hidden:
        _, linear_mean_logits = _zoo_kernels()
        mean_logits = linear_mean_logits(
            hidden.detach(),
            head[0],
            head[1],
            logit_scale_multiply = transforms[0],
            logit_scale_divide = transforms[1],
            logit_softcapping = transforms[2],
        )
    else:
        mean_logits = hidden.detach().mean(-1)
    if method == "orpo":
        mean_logits = mean_logits[:, :-1]
    chosen_logits = mean_logits[:len_chosen]
    rejected_logits = mean_logits[len_chosen:]
    return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, nll_loss)
