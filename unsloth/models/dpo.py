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
    "unsloth_fused_dpo_compute_loss_v1",
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


def _global_module_backward_hooks():
    # torch.nn.modules.module.register_module_full_backward_(pre_)hook applies to every module.
    from torch.nn.modules import module as _module
    return bool(
        getattr(_module, "_global_backward_hooks", None)
        or getattr(_module, "_global_backward_pre_hooks", None)
    )


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
    # The head module is never called, so its module backward hooks (which may rewrite grad_input)
    # would never fire either. Weight tensor / AccumulateGrad hooks still see the kernel's dW.
    if (
        type(head) is not torch.nn.Linear
        or hasattr(head, "_hf_hook")
        or "forward" in vars(head)
        or head._forward_hooks
        or head._forward_pre_hooks
        or head._backward_hooks
        or getattr(head, "_backward_pre_hooks", None)
        or _global_module_backward_hooks()
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


# TRL >= 1.0 DPOTrainer._compute_loss scores [chosen; rejected] from full batch x seq x vocab logits;
# under UNSLOTH_FUSED_PREFERENCE_LOSS=1 the replacement below reproduces it from hidden states.
_DPO_V1_INPUT_KEYS = frozenset(
    (
        "input_ids",
        "attention_mask",
        "completion_mask",
        "ref_chosen_logps",
        "ref_rejected_logps",
    )
)
_DPO_V1_LOSS_TYPES = frozenset(
    (
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_unpaired",
        "apo_zero",
        "apo_down",
        "discopop",
        "sft",
        "sigmoid_norm",
    )
)


def _metric_rows(n_rows, vocab_size):
    # One float32 logits chunk plus its log_softmax and exp: keep each near 256 MB.
    rows = (256 * 1024 * 1024) // max(3 * 4 * vocab_size, 1)
    return max(1, min(int(rows), max(n_rows, 1)))


@torch.no_grad()
def _entropy_and_argmax(hidden_or_logits, is_hidden, head, transforms, mask):
    """Per-position entropy and argmax of the (transformed) logits on ``mask`` rows only,
    one chunk of rows at a time. Positions off ``mask`` are 0 / -1."""
    import torch.nn.functional as F

    flat = hidden_or_logits.reshape(-1, hidden_or_logits.shape[-1])
    flat_mask = mask.reshape(-1).to(device = flat.device, dtype = torch.bool)
    index = flat_mask.nonzero(as_tuple = True)[0]
    # float32 at least, float64 kept (TRL computes its metrics in the logits dtype).
    stat_dtype = torch.promote_types(flat.dtype, torch.float32)
    entropy = torch.zeros(flat_mask.shape[0], dtype = stat_dtype, device = flat.device)
    argmax = torch.full((flat_mask.shape[0],), -1, dtype = torch.long, device = flat.device)
    if index.numel() == 0:
        return entropy.reshape(mask.shape), argmax.reshape(mask.shape)
    weight, bias = head
    vocab_size = weight.shape[0] if is_hidden else flat.shape[-1]
    rows = _metric_rows(index.numel(), vocab_size)
    scale_multiply, scale_divide, softcapping = transforms
    with torch.autocast(device_type = flat.device.type, enabled = False):
        for start in range(0, index.numel(), rows):
            idx = index[start : start + rows]
            if is_hidden:
                chunk = flat[idx].to(device = weight.device, dtype = weight.dtype)
                logits = F.linear(chunk, weight, bias).to(stat_dtype)
                if scale_multiply != 0.0:
                    logits = logits * scale_multiply
                if scale_divide != 0.0:
                    logits = logits / scale_divide
                if softcapping != 0.0:
                    logits = softcapping * torch.tanh(logits / softcapping)
            else:
                logits = flat[idx].to(stat_dtype)
            logps = torch.log_softmax(logits, dim = -1)
            entropy[idx] = (-(torch.exp(logps) * logps).sum(-1)).to(entropy.device)
            argmax[idx] = logits.argmax(dim = -1).to(argmax.device)
            del logits, logps
    return entropy.reshape(mask.shape), argmax.reshape(mask.shape)


def _dpo_v1_supported(trainer, model, inputs, return_outputs, trl_semantics):
    if not isinstance(trl_semantics, dict):
        return False
    if not fused_preference_loss_enabled() or _zoo_kernels() is None:
        return False
    if return_outputs:
        # prediction_step wants the real logits back in `outputs`.
        return False
    if getattr(trainer, "use_liger_kernel", False) or getattr(trainer, "aux_loss_enabled", False):
        return False
    if getattr(trainer, "is_encoder_decoder", False) or getattr(trainer, "is_vision_model", False):
        return False
    args = getattr(trainer, "args", None)
    if getattr(args, "padding_free", False):
        return False
    if not set(inputs).issubset(_DPO_V1_INPUT_KEYS):
        return False
    if "input_ids" not in inputs or "completion_mask" not in inputs:
        return False
    # alpha_divergence clamps by the logps dtype, which is float32 here and half in TRL.
    f_divergence_types = ("reverse_kl", "forward_kl", "js_divergence")
    if trl_semantics.get("legacy_fdiv", True):
        # Before TRL 1.10 these scores drop their constants and replace the log ratios in some
        # losses; only reverse_kl, where the score is the log ratio, is the same everywhere.
        f_divergence_types = ("reverse_kl",)
    if getattr(trainer, "f_divergence_type", None) not in f_divergence_types:
        return False
    loss_types = getattr(trainer, "loss_types", None)
    loss_weights = getattr(trainer, "loss_weights", None)
    if not loss_types or loss_weights is None or len(loss_types) != len(loss_weights):
        return False
    if not set(loss_types).issubset(_DPO_V1_LOSS_TYPES):
        return False
    return True


def _dpo_v1_heads(trainer, model):
    """(unwrapped, head, transforms) for the policy, and the same for a separate reference model
    (``None`` when the reference is the policy with its adapter disabled or precomputed)."""
    unwrapped, head = _output_head(trainer, model)
    if head is None:
        return None
    transforms = _logit_transforms(unwrapped)
    if transforms is None:
        return None
    ref = None
    if not trainer.precompute_ref_logps:
        try:
            from accelerate.utils import is_peft_model
        except Exception:
            return None
        if not (is_peft_model(model) and trainer.ref_model is None):
            if trainer.ref_model is None:
                return None
            ref_unwrapped, ref_head = _output_head(trainer, trainer.ref_model)
            if ref_head is None:
                return None
            ref_transforms = _logit_transforms(ref_unwrapped)
            if ref_transforms is None:
                return None
            ref = (ref_unwrapped, ref_head, ref_transforms)
    return (unwrapped, head, transforms), ref


def unsloth_fused_dpo_compute_loss_v1(
    trainer,
    model,
    inputs,
    return_outputs = False,
    trl_semantics = None,
):
    """TRL 1.x ``DPOTrainer._compute_loss`` for text-only decoder models, scored from hidden
    states in row chunks. Returns ``None`` (run TRL's own) whenever the inputs leave that case.

    ``trl_semantics`` is what rl_replacements read from the installed TRL's source: legacy_fdiv
    (pre-1.10 f-divergence scores), entropy_mean (pre-1.3 entropy metric) and tp_size (1.13+
    divides the token count by the tensor-parallel size)."""
    if not _dpo_v1_supported(trainer, model, inputs, return_outputs, trl_semantics):
        return None
    heads = _dpo_v1_heads(trainer, model)
    if heads is None:
        return None
    (unwrapped, head, transforms), ref = heads

    import math
    import torch.nn.functional as F
    from trl.trainer.dpo_trainer import disable_gradient_checkpointing, use_adapter
    from accelerate.utils import is_peft_model

    self = trainer
    mode = "train" if self.model.training else "eval"
    device = self.accelerator.device

    _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps"}
    model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
    model_kwargs["use_cache"] = False

    input_ids = inputs["input_ids"]
    completion_mask = inputs["completion_mask"]
    shift_labels = input_ids[..., 1:]
    shift_completion_mask = completion_mask[..., 1:]
    score_mask = shift_completion_mask.bool()
    use_weighting = bool(getattr(self, "use_weighting", False))

    hidden, is_hidden = _forward_hidden_states(model, unwrapped, head[0], (), model_kwargs)
    shift_hidden = hidden[..., :-1, :]
    per_token_logps, mean_logits, log_sum_sq_probs = _token_logps(
        shift_hidden,
        is_hidden,
        head,
        transforms,
        shift_labels,
        score_mask,
        want_lssp = use_weighting,
    )
    per_token_entropy, predictions = _entropy_and_argmax(
        shift_hidden.detach(), is_hidden, head, transforms, score_mask
    )
    del hidden, shift_hidden

    if self.ld_alpha is None:
        logps = per_token_logps.sum(dim = 1)
    else:
        comp_pos = shift_completion_mask.cumsum(dim = 1)
        comp_lens = shift_completion_mask.sum(dim = 1).long()
        chosen_lens, rejected_lens = comp_lens.chunk(2, dim = 0)
        shared_lens = torch.minimum(chosen_lens, rejected_lens)
        shared_lens = torch.cat([shared_lens, shared_lens], dim = 0).to(device)
        shared_mask = (comp_pos > 0) & (comp_pos <= shared_lens.unsqueeze(1))
        tail_mask = comp_pos > shared_lens.unsqueeze(1)
        shared_logps = (per_token_logps * shared_mask).sum(dim = 1)
        tail_logps = (per_token_logps * tail_mask).sum(dim = 1)
        logps = shared_logps + self.ld_alpha * tail_logps
    chosen_logps, rejected_logps = logps.chunk(2, dim = 0)

    if self.precompute_ref_logps:
        ref_chosen_logps, ref_rejected_logps = (
            inputs["ref_chosen_logps"],
            inputs["ref_rejected_logps"],
        )
    else:
        ref_model_kwargs = dict(model_kwargs)
        with (
            torch.no_grad(),
            disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs),
        ):
            if ref is None:
                peft_model = self.accelerator.unwrap_model(model)
                with use_adapter(
                    peft_model, adapter_name = "ref" if "ref" in peft_model.peft_config else None
                ):
                    ref_hidden, ref_is_hidden = _forward_hidden_states(
                        self.model, unwrapped, head[0], (), ref_model_kwargs
                    )
                ref_head, ref_transforms = head, transforms
            else:
                ref_unwrapped, ref_head, ref_transforms = ref
                ref_hidden, ref_is_hidden = _forward_hidden_states(
                    self.ref_model, ref_unwrapped, ref_head[0], (), ref_model_kwargs
                )
            ref_per_token_logps, _, _ = _token_logps(
                ref_hidden[..., :-1, :],
                ref_is_hidden,
                ref_head,
                ref_transforms,
                shift_labels,
                score_mask,
            )
            del ref_hidden
        if self.ld_alpha is None:
            ref_logps = ref_per_token_logps.sum(dim = 1)
        else:
            ref_shared_logps = (ref_per_token_logps * shared_mask).sum(dim = 1)
            ref_tail_logps = (ref_per_token_logps * tail_mask).sum(dim = 1)
            ref_logps = ref_shared_logps + self.ld_alpha * ref_tail_logps
        ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim = 0)

    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps

    if self.f_divergence_type == "reverse_kl":
        chosen_scores = chosen_logratios
        rejected_scores = rejected_logratios
    elif self.f_divergence_type == "forward_kl":
        chosen_scores = 1 - torch.exp(-chosen_logratios)
        rejected_scores = 1 - torch.exp(-rejected_logratios)
    else:  # js_divergence
        chosen_scores = math.log(2) + F.logsigmoid(chosen_logratios)
        rejected_scores = math.log(2) + F.logsigmoid(rejected_logratios)

    delta_score = chosen_scores - rejected_scores

    loss = 0.0
    for loss_type, loss_weight in zip(self.loss_types, self.loss_weights):
        if loss_type == "sigmoid":
            per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
        elif loss_type == "hinge":
            per_sequence_loss = torch.relu(1 - self.beta * delta_score)
        elif loss_type == "ipo":
            chosen_mask, rejected_mask = completion_mask.chunk(2, dim = 0)
            chosen_avg_score = chosen_scores / chosen_mask.sum(dim = 1).clamp(min = 1.0)
            rejected_avg_score = rejected_scores / rejected_mask.sum(dim = 1).clamp(min = 1.0)
            ipo_delta = chosen_avg_score - rejected_avg_score
            per_sequence_loss = (ipo_delta - 1 / (2 * self.beta)) ** 2
        elif loss_type == "exo_pair":
            epsilon = torch.tensor(self.label_smoothing, device = device)
            qw = torch.sigmoid(self.beta * delta_score)
            log_qw = F.logsigmoid(self.beta * delta_score)
            log_pw = torch.log1p(-epsilon)
            ql = torch.sigmoid(-self.beta * delta_score)
            log_ql = F.logsigmoid(-self.beta * delta_score)
            log_pl = torch.log(epsilon)
            per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)
        elif loss_type == "nca_pair":
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            per_sequence_loss = (
                -F.logsigmoid(chosen_rewards)
                - 0.5 * F.logsigmoid(-chosen_rewards)
                - 0.5 * F.logsigmoid(-rejected_rewards)
            )
        elif loss_type == "robust":
            clean_loss_term = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
            flipped_loss_term = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
            per_sequence_loss = (clean_loss_term - flipped_loss_term) / (
                1 - 2 * self.label_smoothing
            )
        elif loss_type == "bco_pair":
            chosen_rewards = self.beta * chosen_logratios
            rejected_rewards = self.beta * rejected_logratios
            per_sequence_loss = -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
        elif loss_type == "sppo_hard":
            winner_margin_error = (chosen_logratios - 0.5 / self.beta) ** 2
            loser_margin_error = (rejected_logratios + 0.5 / self.beta) ** 2
            per_sequence_loss = winner_margin_error + loser_margin_error
        elif loss_type == "aot":
            logratios = chosen_logps - rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios_sorted, _ = torch.sort(logratios, dim = 0)
            ref_logratios_sorted, _ = torch.sort(ref_logratios, dim = 0)
            delta = logratios_sorted - ref_logratios_sorted
            per_sequence_loss = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif loss_type == "aot_unpaired":
            chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim = 0)
            rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim = 0)
            delta = chosen_logratios_sorted - rejected_logratios_sorted
            per_sequence_loss = (
                -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * delta) * self.label_smoothing
            )
        elif loss_type == "apo_zero":
            losses_chosen = 1 - torch.sigmoid(self.beta * chosen_logratios)
            losses_rejected = torch.sigmoid(self.beta * rejected_logratios)
            per_sequence_loss = losses_chosen + losses_rejected
        elif loss_type == "apo_down":
            losses_chosen = torch.sigmoid(self.beta * chosen_logratios)
            losses_rejected = 1 - torch.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
            per_sequence_loss = losses_chosen + losses_rejected
        elif loss_type == "discopop":
            logits = delta_score * self.beta
            log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            per_sequence_loss = (
                logistic_component * (1 - log_ratio_modulation)
                + exp_component * log_ratio_modulation
            )
        elif loss_type == "sft":
            # F.cross_entropy over the chosen completion tokens == -mean of their logps.
            chosen_token_logps, _ = per_token_logps.chunk(2, dim = 0)
            chosen_mask, _ = score_mask.chunk(2, dim = 0)
            batch_loss = -chosen_token_logps[chosen_mask].mean()
            per_sequence_loss = batch_loss.expand(chosen_token_logps.size(0))
        else:  # sigmoid_norm
            chosen_mask, rejected_mask = completion_mask.chunk(2, dim = 0)
            chosen_avg_score = chosen_scores / chosen_mask.sum(dim = 1).clamp(min = 1.0)
            rejected_avg_score = rejected_scores / rejected_mask.sum(dim = 1).clamp(min = 1.0)
            delta = chosen_avg_score - rejected_avg_score
            per_sequence_loss = -F.logsigmoid(self.beta * delta)

        if use_weighting:
            completion_lengths = shift_completion_mask.sum(dim = 1).clamp_min(1)
            with torch.no_grad():
                # log_sum_sq_probs = logsumexp(2 l) - 2 logsumexp(l), TRL's log_denom.
                aligned_logps = (per_token_logps - log_sum_sq_probs) * shift_completion_mask
            mean_logps = aligned_logps.sum(dim = 1) / completion_lengths
            weights = torch.exp(mean_logps)
            chosen_weights, rejected_weights = weights.chunk(2, dim = 0)
            per_sequence_loss *= chosen_weights * rejected_weights

        loss += per_sequence_loss.mean() * loss_weight

    # Metrics, as TRL logs them, from the chunked statistics.
    mask = shift_completion_mask
    if trl_semantics.get("entropy_mean", False):
        entropy = per_token_entropy[mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
    else:
        entropy_sum = (per_token_entropy * mask).sum()
        total_tokens = mask.sum()
        entropy_sum = self.accelerator.gather_for_metrics(entropy_sum).sum()
        total_tokens = self.accelerator.gather_for_metrics(total_tokens).sum()
        entropy = (entropy_sum / total_tokens).item() if total_tokens > 0 else 0.0
    self._metrics[mode]["entropy"].append(entropy)

    if mode == "train":
        num_tokens_in_batch = (
            self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
        )
        if trl_semantics.get("tp_size", False):
            num_tokens_in_batch = num_tokens_in_batch // self._tp_size
        self._total_train_tokens += num_tokens_in_batch
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

    chosen_means, rejected_means = mean_logits.detach().chunk(2, dim = 0)
    chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim = 0)
    total_chosen_logits = chosen_means[chosen_mask.bool()].sum()
    total_chosen_tokens = chosen_mask.sum()
    total_rejected_logits = rejected_means[rejected_mask.bool()].sum()
    total_rejected_tokens = rejected_mask.sum()
    total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
    total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
    total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
    total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
    avg_chosen_logits = (
        total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
    )
    avg_rejected_logits = (
        total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
    )
    self._metrics[mode]["logits/chosen"].append(avg_chosen_logits)
    self._metrics[mode]["logits/rejected"].append(avg_rejected_logits)

    half = len(shift_completion_mask) // 2
    chosen_bool = shift_completion_mask[:half].bool()
    chosen_labels = shift_labels[:half]
    correct_predictions = (predictions[:half] == chosen_labels) & chosen_bool
    total_tokens = chosen_bool.sum()
    correct_tokens = correct_predictions.sum()
    correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
    total_tokens = self.accelerator.gather_for_metrics(total_tokens)
    total_sum = total_tokens.sum()
    accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
    self._metrics[mode]["mean_token_accuracy"].append(accuracy)

    chosen_rewards = self.beta * chosen_logratios.detach()
    rejected_rewards = self.beta * rejected_logratios.detach()
    agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
    agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
    self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
    self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())
    reward_accuracies = (chosen_rewards > rejected_rewards).float()
    agg_reward_accuracies = self.accelerator.gather(reward_accuracies)
    self._metrics[mode]["rewards/accuracies"].append(agg_reward_accuracies.mean().item())
    margins = chosen_rewards - rejected_rewards
    agg_margins = self.accelerator.gather(margins)
    self._metrics[mode]["rewards/margins"].append(agg_margins.mean().item())
    self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
    self._metrics[mode]["logps/rejected"].append(
        self.accelerator.gather(rejected_logps).mean().item()
    )
    return loss
