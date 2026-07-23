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

"""
Attention-mask helpers compatible with Transformers v5.10+.

`transformers.modeling_attn_mask_utils` is deprecated and scheduled for removal.
Unsloth keeps a local copy of the small API surface it uses so imports keep working
and training runs do not emit deprecation warnings on every forward pass.

Adapted from HuggingFace Transformers `modeling_attn_mask_utils.py` (Apache 2.0).
See: https://github.com/unslothai/unsloth/issues/6860
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from transformers.utils.import_utils import is_torchdynamo_compiling

try:
    # `is_tracing` was added to `transformers.utils.import_utils` in 4.52
    # (commit that introduced `_prepare_4d_attention_mask_for_sdpa` rewrites).
    # Unsloth's declared lower bound is `transformers>=4.51.3`, so import
    # defensively and fall back to a local implementation when the symbol
    # is not exported. The fallback mirrors the tracing-detection expression
    # that upstream `transformers.modeling_attn_mask_utils` used inline before
    # `is_tracing` was added — `torch.jit.is_tracing() or
    # isinstance(inputs_embeds, torch.fx.Proxy) or is_torchdynamo_compiling()`
    # — so the data-dependent `torch.all(...)` branches in the mask helpers
    # continue to be skipped during JIT trace / symbolic trace / Dynamo
    # compilation, preserving the SDPA path selection.
    from transformers.utils.import_utils import is_tracing  # type: ignore[attr-defined]
except ImportError:

    def is_tracing(tensor = None) -> bool:  # type: ignore[no-redef]
        """Local fallback for transformers < 4.52.

        Returns True when the active context is any of: ``torch.jit.trace``,
        ``torch.fx.symbolic_trace``, or Dynamo compilation. Other tracing
        backends that the modern ``transformers.utils.import_utils.is_tracing``
        detects (CUDA stream capture, FakeTensor, JAX via torchax) cannot be
        detected without newer ``import_utils`` helpers; for those we fall
        back to the dynamo check, which matches the conservative pre-4.52
        upstream behavior on the supported lower bound.
        """
        if torch.jit.is_tracing():
            return True
        if tensor is not None and isinstance(tensor, torch.fx.Proxy):
            return True
        return is_torchdynamo_compiling()


@dataclass
class AttentionMaskConverter:
    is_causal: bool
    sliding_window: int | None = None

    def __init__(
        self,
        is_causal: bool,
        sliding_window: int | None = None,
    ):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, str] = "cpu",
    ) -> torch.Tensor | None:
        if not self.is_causal:
            raise ValueError(
                f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True."
            )

        input_shape = (batch_size, query_length)
        past_key_values_length = key_value_length - query_length

        causal_4d_mask = None
        if input_shape[-1] > 1 or self.sliding_window is not None:
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device = device,
                past_key_values_length = past_key_values_length,
                sliding_window = self.sliding_window,
            )

        return causal_4d_mask

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: int | None = None,
    ) -> torch.Tensor:
        input_shape = (attention_mask_2d.shape[0], query_length)

        causal_4d_mask = None
        if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
            if key_value_length is None:
                raise ValueError(
                    "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
                )

            past_key_values_length = key_value_length - query_length
            causal_4d_mask = self._make_causal_mask(
                input_shape,
                dtype,
                device = attention_mask_2d.device,
                past_key_values_length = past_key_values_length,
                sliding_window = self.sliding_window,
            )
        elif self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window is currently only implemented for causal masking"
            )

        expanded_attn_mask = self._expand_mask(
            attention_mask_2d, dtype, tgt_len = input_shape[-1]
        ).to(attention_mask_2d.device)

        if causal_4d_mask is not None:
            expanded_attn_mask = causal_4d_mask.masked_fill(
                expanded_attn_mask.bool(), torch.finfo(dtype).min
            )

        return expanded_attn_mask

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: int | None = None,
    ):
        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device = device)
        mask_cond = torch.arange(mask.size(-1), device = device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat(
                [torch.zeros(tgt_len, past_key_values_length, dtype = dtype, device = device), mask],
                dim = -1,
            )

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(torch.ones_like(mask, dtype = torch.bool), diagonal = diagonal)
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
        tgt_len: int | None = None,
    ):
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(expanded_mask: torch.FloatTensor, min_dtype: float):
        if expanded_mask.dtype == torch.bool:
            raise ValueError(
                "AttentionMaskConverter._unmask_unattended expects a float `expanded_mask`, got a BoolTensor."
            )

        return expanded_mask.mul(~torch.all(expanded_mask == min_dtype, dim = -1, keepdim = True))

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: int | None = None,
        is_training: bool = False,
    ) -> bool:
        _, query_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        key_value_length = query_length + past_key_values_length

        is_tracing_ = is_tracing(inputs_embeds)

        ignore_causal_mask = False

        if attention_mask is None:
            if (
                (is_training or not is_tracing_)
                and (query_length == 1 or key_value_length == query_length)
                and (sliding_window is None or key_value_length < sliding_window)
            ):
                ignore_causal_mask = True
        elif sliding_window is None or key_value_length < sliding_window:
            if len(attention_mask.shape) == 4:
                return False
            elif not is_tracing_ and torch.all(attention_mask == 1):
                if query_length == 1 or key_value_length == query_length:
                    ignore_causal_mask = True

        return ignore_causal_mask


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
):
    attn_mask_converter = AttentionMaskConverter(is_causal = True, sliding_window = sliding_window)

    key_value_length = input_shape[-1] + past_key_values_length

    is_tracing_ = is_tracing(inputs_embeds)

    ignore_causal_mask = AttentionMaskConverter._ignore_causal_mask_sdpa(
        attention_mask = attention_mask,
        inputs_embeds = inputs_embeds,
        past_key_values_length = past_key_values_length,
        sliding_window = sliding_window,
    )

    if ignore_causal_mask:
        expanded_4d_mask = None
    elif attention_mask is None:
        expanded_4d_mask = attn_mask_converter.to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype = inputs_embeds.dtype,
            device = inputs_embeds.device,
        )
    else:
        if attention_mask.dim() == 4:
            expanded_4d_mask = attention_mask
        else:
            expanded_4d_mask = attn_mask_converter.to_4d(
                attention_mask,
                input_shape[-1],
                dtype = inputs_embeds.dtype,
                key_value_length = key_value_length,
            )

    if (
        not is_tracing_
        and expanded_4d_mask is not None
        and expanded_4d_mask.device.type in ["cuda", "xpu"]
    ):
        expanded_4d_mask = AttentionMaskConverter._unmask_unattended(
            expanded_4d_mask, min_dtype = torch.finfo(inputs_embeds.dtype).min
        )

    return expanded_4d_mask


def _prepare_4d_attention_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int | None = None,
):
    return AttentionMaskConverter._expand_mask(mask = mask, dtype = dtype, tgt_len = tgt_len)


def _prepare_4d_attention_mask_for_sdpa(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: int | None = None,
):
    _, key_value_length = mask.shape
    tgt_len = tgt_len if tgt_len is not None else key_value_length

    if not is_tracing(mask) and torch.all(mask == 1):
        return None

    return AttentionMaskConverter._expand_mask(mask = mask, dtype = dtype, tgt_len = tgt_len)
