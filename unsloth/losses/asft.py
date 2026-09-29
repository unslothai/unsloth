# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Anchored Supervised Fine-Tuning (ASFT) loss: SFT, DFT, SFT+KL and ASFT modes."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from unsloth.kernels.cross_entropy_loss import Fast_CrossEntropyLoss
from unsloth.utils.packing import mask_packed_sequence_boundaries

__all__ = [
    "ASFTStreamingConfig",
    "fast_cross_entropy_loss_per_token",
    "build_shift_labels",
    "get_reference_forward_callable",
    "compute_asft_loss",
]


_DEFAULT_REF_MICROBATCH_DIVISOR = 2


@dataclass
class ASFTStreamingConfig:
    """Reference-forward streaming: mode "off" (one full forward) or "batch" (microbatched).

    No sequence/KV-cache chunking: Unsloth's patched decoders route any call with
    past_key_values to the single-token decode kernel (asserts q_len == 1).
    """

    mode: Literal["off", "batch"] = "off"
    ref_microbatch_size: Optional[int] = None
    force_fp32_kl: bool = True


def fast_cross_entropy_loss_per_token(
    logits: torch.Tensor,
    labels: torch.Tensor,
    logit_softcapping: float = 0,
    logit_scaling: float = 0,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unreduced per-token CE via Fast_CrossEntropyLoss; returns (losses (B*T,), valid_mask)."""
    logits = logits.reshape(-1, logits.shape[-1])
    labels = labels.reshape(-1)
    valid_mask = labels != ignore_index
    if ignore_index != -100:
        labels = labels.masked_fill(~valid_mask, -100)
    losses = Fast_CrossEntropyLoss.apply(logits, labels, logit_softcapping, logit_scaling)
    return losses, valid_mask


def build_shift_labels(
    labels: torch.Tensor,
    packed_seq_lengths: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Shift labels, not logits (Unsloth CE convention); the last position becomes ignore_index."""
    shift_labels = torch.empty_like(labels)
    shift_labels[..., :-1] = labels[..., 1:]
    shift_labels[..., -1] = ignore_index
    if packed_seq_lengths is not None:
        mask_packed_sequence_boundaries(shift_labels, packed_seq_lengths, ignore_index = ignore_index)
    return shift_labels


@contextmanager
def _inference_eval_context(model: nn.Module):
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        if was_training:
            model.train()


def get_reference_forward_callable(
    model: nn.Module,
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    original_model: Optional[nn.Module] = None,
    return_outputs: bool = False,
) -> Callable[..., torch.Tensor]:
    """Return a reference forward: "disable_adapter" (needs PEFT) or "frozen_copy"."""
    if reference_policy not in ("disable_adapter", "frozen_copy"):
        raise ValueError(f"Unknown reference_policy: {reference_policy}")

    if reference_policy == "disable_adapter" and hasattr(model, "disable_adapter"):

        def ref_forward(**forward_inputs):
            with _inference_eval_context(model):
                disable_adapter = model.disable_adapter
                if not (
                    hasattr(disable_adapter, "__enter__") and hasattr(disable_adapter, "__exit__")
                ):
                    disable_adapter = disable_adapter()
                with disable_adapter:
                    outputs = model(**forward_inputs)
                    return outputs if return_outputs else outputs.logits

        return ref_forward

    if original_model is None:
        original_model = deepcopy(model)
        original_model.eval()
        original_model.requires_grad_(False)

    def ref_forward(**forward_inputs):
        with _inference_eval_context(original_model):
            outputs = original_model(**forward_inputs)
            return outputs if return_outputs else outputs.logits

    return ref_forward


_KL_CHUNK_BYTES = 256 * 1024 * 1024


class _ChunkedKL(torch.autograd.Function):
    """Row-chunked KL that saves only d(KL)/d(cur_logits) in the logits dtype.

    Autograd through log_softmax / softmax / kl_div keeps several fp32 (N, V) tensors alive until
    backward; this keeps one (N, V) tensor and nothing that aliases cur_logits, which
    Fast_CrossEntropyLoss overwrites in place during its backward.
    """

    @staticmethod
    def forward(ctx, cur_logits, ref_logits, reverse, force_fp32):
        vocab = cur_logits.shape[-1]
        cur = cur_logits.reshape(-1, vocab)
        ref = ref_logits.reshape(-1, vocab)
        kl = torch.empty(cur.shape[0], dtype = torch.float32, device = cur.device)
        grad = torch.empty_like(cur)
        step = max(1, _KL_CHUNK_BYTES // (vocab * 4))
        for s in range(0, cur.shape[0], step):
            c, r = cur[s : s + step], ref[s : s + step].to(cur.device)
            if force_fp32:
                c, r = c.float(), r.float()
            log_q, log_p = F.log_softmax(c, dim = -1), F.log_softmax(r, dim = -1)
            if reverse:
                q = log_q.exp()
                k = (q * (log_q - log_p)).sum(-1)
                g = q * (log_q - log_p - k.unsqueeze(-1))
            else:
                p = log_p.exp()
                k = (p * (log_p - log_q)).sum(-1)
                g = log_q.exp() - p
            kl[s : s + step] = k
            grad[s : s + step] = g
        ctx.save_for_backward(grad)
        ctx.shape = cur_logits.shape
        return kl.view(cur_logits.shape[:-1])

    @staticmethod
    def backward(ctx, grad_kl):
        (grad,) = ctx.saved_tensors
        grad.mul_(grad_kl.reshape(-1, 1).to(grad.dtype))
        return grad.view(ctx.shape), None, None, None


def _compute_kl_divergence(
    cur_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
) -> torch.Tensor:
    """Per-token KL over the last dim, forward KL(p_ref || p_cur) by default.

    The ASFT paper says reverse KL, but its code F.kl_div(log(cur), ref) is forward KL;
    we match the code. kl_direction="reverse" gives KL(p_cur || p_ref).
    """
    if kl_direction not in ("forward", "reverse"):
        raise ValueError(f"Unknown kl_direction: {kl_direction}")
    return _ChunkedKL.apply(cur_logits, ref_logits.detach(), kl_direction == "reverse", force_fp32)


def _unwrap_logits(ref_outputs: Any) -> torch.Tensor:
    if hasattr(ref_outputs, "logits"):
        return ref_outputs.logits
    if isinstance(ref_outputs, (tuple, list)):
        return ref_outputs[0]
    return ref_outputs


def _slice_batch_inputs(
    forward_inputs: Dict[str, Any], batch_size: int, b_start: int, b_end: int
) -> Dict[str, Any]:
    return {
        key: value[b_start:b_end]
        if torch.is_tensor(value) and value.shape[:1] == (batch_size,)
        else value
        for key, value in forward_inputs.items()
    }


def _compute_kl(
    cur_logits: torch.Tensor,
    ref_forward: Callable,
    forward_inputs: Dict[str, Any],
    microbatch_size: int,
    force_fp32: bool = True,
    kl_direction: Literal["forward", "reverse"] = "forward",
) -> torch.Tensor:
    """(B, T) KL with the reference forward run microbatch_size rows at a time."""
    batch_size = cur_logits.shape[0]
    microbatch_size = max(1, microbatch_size)
    if microbatch_size >= batch_size:
        ref_logits = _unwrap_logits(ref_forward(**forward_inputs))
        return _compute_kl_divergence(cur_logits, ref_logits, force_fp32, kl_direction)
    kl = []
    for b_start in range(0, batch_size, microbatch_size):
        b_end = min(b_start + microbatch_size, batch_size)
        mb_inputs = _slice_batch_inputs(forward_inputs, batch_size, b_start, b_end)
        ref_logits = _unwrap_logits(ref_forward(**mb_inputs))
        kl.append(
            _compute_kl_divergence(cur_logits[b_start:b_end], ref_logits, force_fp32, kl_direction)
        )
        del ref_logits
    return torch.cat(kl, dim = 0)


def compute_asft_loss(
    model: nn.Module,
    inputs: Dict[str, Any],
    *,
    asft_mode: Literal["sft", "dft", "sft+kl", "asft"] = "asft",
    kl_weight: float = 0.03,
    kl_direction: Literal["forward", "reverse"] = "forward",
    reference_policy: Literal["disable_adapter", "frozen_copy"] = "disable_adapter",
    streaming_config: Optional[ASFTStreamingConfig] = None,
    original_model: Optional[nn.Module] = None,
    normalize_by: Literal["tokens", "weights"] = "tokens",
    return_outputs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
    """Compute the ASFT loss.

    asft_mode: "sft" CE; "dft" CE * detached p(label); "sft+kl" CE + kl_weight * KL;
    "asft" DFT + kl_weight * KL. kl_weight == 0 skips the reference forward.
    normalize_by: "tokens" (matches reference) or "weights".
    """
    if asft_mode not in ("sft", "dft", "sft+kl", "asft"):
        raise ValueError(f"Unknown asft_mode: {asft_mode}")
    if normalize_by not in ("tokens", "weights") or (
        normalize_by == "weights" and asft_mode not in ("dft", "asft")
    ):
        raise ValueError(f"normalize_by={normalize_by!r} is invalid for asft_mode={asft_mode!r}")
    if streaming_config is None:
        streaming_config = ASFTStreamingConfig()
    if streaming_config.mode not in ("off", "batch"):
        raise ValueError(f"Unknown streaming mode: {streaming_config.mode}")

    # Drop labels/num_items so the model returns logits.
    forward_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "num_items_in_batch"}}
    outputs = model(**forward_inputs)
    # Model forwards already apply softcapping / logit scaling, so the CE kernel must not again.
    logits = outputs.logits

    shift_labels = build_shift_labels(inputs["labels"], inputs.get("packed_seq_lengths", None))
    valid_mask = shift_labels != -100
    n_items_tokens = inputs.get("num_items_in_batch", None)
    if n_items_tokens is None:
        n_items_tokens = valid_mask.sum()
    n_items_tokens = max(n_items_tokens, 1)

    if valid_mask.sum() == 0:
        zero_loss = logits.sum() * 0.0
        return (zero_loss, outputs) if return_outputs else zero_loss

    batch_size, seq_len = shift_labels.shape
    ce_losses, _ = fast_cross_entropy_loss_per_token(logits, shift_labels)
    ce_losses = ce_losses.view(batch_size, seq_len)

    dft_weights = None
    token_loss = ce_losses
    if asft_mode in ("dft", "asft"):
        # exp(-CE) is p(label) without a second full-vocab softmax.
        dft_weights = torch.exp(-ce_losses.detach()) * valid_mask
        token_loss = ce_losses * dft_weights

    if asft_mode in ("sft+kl", "asft") and kl_weight != 0:
        ref_model = (
            model.module
            if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel))
            else model
        )
        ref_forward = get_reference_forward_callable(ref_model, reference_policy, original_model)
        microbatch_size = batch_size
        if streaming_config.mode == "batch":
            microbatch_size = streaming_config.ref_microbatch_size or max(
                1, batch_size // _DEFAULT_REF_MICROBATCH_DIVISOR
            )
        kl = _compute_kl(
            logits,
            ref_forward,
            forward_inputs,
            microbatch_size,
            streaming_config.force_fp32_kl,
            kl_direction,
        )
        token_loss = token_loss + kl_weight * kl

    normalizer = n_items_tokens
    if normalize_by == "weights":
        normalizer = dft_weights[valid_mask].sum().clamp_min(1e-8)
    loss = token_loss[valid_mask].sum() / normalizer

    return (loss, outputs) if return_outputs else loss
