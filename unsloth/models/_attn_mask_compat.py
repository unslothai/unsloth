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
Attention-mask helpers, verified equivalent across Transformers 4.51.3 - 5.14.1.

`transformers.modeling_attn_mask_utils` warns on every call from 5.5.0 and
announces removal in v5.10. It still ships in 5.14.1, but Transformers itself no
longer imports it (135 internal references at 4.51.3, none from 5.5.0), so it can
be dropped at any release. Keeping a local copy of the surface we use means
imports survive that, and forward passes stop emitting deprecation warnings.

Adapted from HuggingFace Transformers `modeling_attn_mask_utils.py`
(Copyright 2023 The HuggingFace Team, Apache License 2.0).
See: https://github.com/unslothai/unsloth/issues/6860
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch

from transformers.utils.import_utils import is_torchdynamo_compiling

try:
    # `is_tracing` exists only from Transformers 5.0.0, so the fallback below is the live path for all
    # of 4.x, not just the 4.51.3 floor. It mirrors what 4.x modeling_attn_mask_utils computes inline
    # (torch.jit.is_tracing() or isinstance(x, torch.fx.Proxy) or is_torchdynamo_compiling()), so the
    # data-dependent torch.all(...) branches stay skipped under JIT / FX / Dynamo and SDPA path
    # selection is unchanged.
    from transformers.utils.import_utils import is_tracing  # type: ignore[attr-defined]
except ImportError:

    def is_tracing(tensor = None) -> bool:  # type: ignore[no-redef]
        """Local fallback for transformers < 5.0.0.

        True under ``torch.jit.trace``, ``torch.fx.symbolic_trace`` or Dynamo.
        The 5.x helper also detects CUDA stream capture, FakeTensor and JAX via
        torchax; 4.x itself detects none of those, so behaviour on 4.x matches.
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

        # 0-dim tensor, not a Python float: a float literal lowers to an fp32 scalar and breaks ExecuTorch
        # edge-dialect export on fp16/bf16 masks. Matches upstream from 4.53.0
        # (huggingface/transformers#38637).
        inverted_mask = torch.tensor(1.0, dtype = dtype) - expanded_mask

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

        # Attend to all tokens in masked rows (the first rows under left padding), required by
        # F.scaled_dot_product_attention's memory-efficient path (pytorch/pytorch#110213). Scoped to this
        # branch to match upstream: at function scope it also runs on the attention_mask is None path,
        # where it materialises to_causal_4d's stride-0 view into a dense [bsz, 1, q, kv] tensor.
        if not is_tracing_ and expanded_4d_mask.device.type in ["cuda", "xpu"]:
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
