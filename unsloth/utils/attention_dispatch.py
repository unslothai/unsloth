# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Shared helpers for attention backend selection and execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import scaled_dot_product_attention

from ..models._utils import *
from ..utils.packing import (
    build_sdpa_packed_attention_mask,
    build_xformers_block_causal_mask,
)

flash_attn_func = None
flash_attn_varlen_func = None
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
HAS_XFORMERS = xformers is not None


def _xformers_runs_on_device() -> bool:
    """One tiny attention forward; True iff the xformers kernel actually runs here."""
    try:
        # Pre-Ampere GPUs (sm < 80: Turing/Volta) have no bfloat16 attention kernel
        # but run xformers fine in float16, so pick the dtype the device supports.
        dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
        q = torch.zeros((1, 8, 1, 64), device = "cuda", dtype = dtype)
        attn_bias = xformers.attn_bias.BlockDiagonalCausalMask.from_seqlens([8])
        xformers_attention(q, q, q, attn_bias = attn_bias)
        # Launches are async; synchronize so a deferred kernel failure fails the probe here.
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def _xformers_disabled_for_capability(capability, probe = _xformers_runs_on_device) -> bool:
    # At sm_120 (RTX 50-series) xformers' cutlass op is capability-rejected (caps at
    # sm_90) and its flash-2 op runs only if the build ships an sm_120 kernel, so run
    # one real forward to decide. Below sm_120 xformers always works; skip the probe.
    if capability[0] < 12:
        return False
    return not probe()


# FlashAttention always wins in select_attention_backend and nothing downgrades
# flash -> xformers, so when it's installed xformers is never selected: skip the probe.
if HAS_XFORMERS and not HAS_FLASH_ATTENTION and torch.cuda.is_available():
    if _xformers_disabled_for_capability(torch.cuda.get_device_capability()):
        HAS_XFORMERS = False

# On sm_100+ (B200, sm_120) xformers' fp32-capable cutlass op is capability-rejected and
# only its fp16/bf16 flash-2 op runs, so fp32 Q/K/V (DoRA, #1013) must be downcast there;
# below sm_100 cutlass handles fp32 natively. Read once from device 0, like the probe gate.
_XFORMERS_FP32_UNSUPPORTED = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)
SDPA_HAS_GQA = "enable_gqa" in (scaled_dot_product_attention.__doc__ or "")

# PrefixGrouper kernel, resolved once when the env gate is on so PG-off users never load
# torch flex_attention.
_flex_shared_prefix_attention = None
if os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER", "1").lower() not in ("0", "false", "no", "off"):
    try:
        from .prefix_grouper_kernel import (
            flex_shared_prefix_attention as _flex_shared_prefix_attention,
        )
    except Exception:
        _flex_shared_prefix_attention = None

FLASH_VARLEN = "flash_varlen"
FLASH_DENSE = "flash_dense"
XFORMERS = "xformers"
SDPA = "sdpa"


XFORMERS_BLOCK_DIAG_CLS = xformers.attn_bias.BlockDiagonalCausalMask if HAS_XFORMERS else None


@dataclass
class AttentionConfig:
    """
    Per-layer attention metadata.

    NOTE(djsaunde): Constructed on every forward pass (not once per layer) since
        it can be invalid across passes (e.g. switching training/inference). Kept
        separate from AttentionContext to group params.
    """

    backend: str
    n_kv_heads: int
    n_groups: int
    flash_dense_kwargs: Optional[dict[str, Any]] = None
    flash_varlen_kwargs: Optional[dict[str, Any]] = None
    sdpa_kwargs: Optional[dict[str, Any]] = None
    xformers_kwargs: Optional[dict[str, Any]] = None


@dataclass
class AttentionContext:
    """Per-call info required to run attention."""

    bsz: int
    q_len: int
    kv_seq_len: int
    n_heads: int
    head_dim: int
    requires_grad: bool
    seq_info: Optional[Tuple[Tensor, Tensor, int]]
    attention_mask: Optional[Tensor]
    causal_mask: Optional[Any]
    sliding_window: Optional[int] = None
    # PrefixGrouper: non-None routes Q/K/V through the FlexAttention shared-prefix kernel;
    # None leaves every existing construction/behavior unchanged.
    prefix_seg_info: Optional[Any] = None


def select_attention_backend(use_varlen: bool = False) -> str:
    """Return attention backend based on availability / priority order."""

    if HAS_FLASH_ATTENTION:
        if use_varlen:
            return FLASH_VARLEN
        else:
            return FLASH_DENSE
    if HAS_XFORMERS:
        return XFORMERS
    return SDPA


def resolve_prefix_seg_info(kwargs, past_key_value, attention_mask):
    """PrefixGrouper shared-prefix segment table resolver for the arch attention forwards.

    The GRPO PrefixGrouper packed path rides a ``PrefixSegInfo`` in through ``**kwargs``
    (same route as ``packed_seq_lengths``). When present, the forward must route Q/K/V
    through the FlexAttention shared-prefix kernel via ``AttentionContext.prefix_seg_info``.

    Returns the seg table (or ``None`` when PrefixGrouper did not group this batch -- the
    unchanged path). Hardened: the shared-prefix stream is NOT a plain causal sequence, so running
    it under a KV cache or an explicit padding mask would silently produce wrong logprobs.
    That combination can only arise from misuse (PrefixGrouper only rides in via the GRPO
    logprob forward, which is mask-free prefill), so we RAISE loudly instead of degrading
    to a wrong result.

    Factored here so every arch (llama/mistral/qwen3/gemma2/cohere/granite/falcon_h1)
    shares one implementation and cannot drift.
    """
    seg = kwargs.get("prefix_seg_info", None)
    if seg is not None and (past_key_value is not None or attention_mask is not None):
        raise RuntimeError(
            "PrefixGrouper: prefix_seg_info requires prefill with no KV cache and no "
            f"attention_mask (got past_key_value={past_key_value is not None}, "
            f"attention_mask={attention_mask is not None})."
        )
    return seg


def run_attention(
    *, config: AttentionConfig, context: AttentionContext, Q: Tensor, K: Tensor, V: Tensor
) -> Tensor:
    """
    Run attention using config / context info.

    Backend priority (speed): FlashAttention if installed (varlen for packed
    inputs with `seq_info`, else dense), then xFormers, then SDPA as fallback.
    Varlen flash is preferred for packed batches as it avoids padding; xFormers
    and SDPA handle packing via a block-diagonal mask.
    """

    # PrefixGrouper shared-prefix attention (GRPO dedup). Q/K/V here are [bsz, H, T, D];
    # the kernel takes/returns [1, T, H, D], matching the other backends. The field is
    # only set when the env gate is on and grouping succeeded; None keeps every backend
    # byte-identical.
    if context.prefix_seg_info is not None:
        flex_shared_prefix_attention = _flex_shared_prefix_attention
        if flex_shared_prefix_attention is None:
            # gate flipped on after import (or one-time load failed): resolve lazily.
            from ..utils.prefix_grouper_kernel import flex_shared_prefix_attention

        scale = None
        if config.flash_varlen_kwargs:
            scale = config.flash_varlen_kwargs.get("softmax_scale")
        A = flex_shared_prefix_attention(
            Q.transpose(1, 2),
            K.transpose(1, 2),
            V.transpose(1, 2),
            context.prefix_seg_info,
            scale = scale,
        )
        return A  # [1, T, n_heads, head_dim]

    backend = config.backend
    if backend == FLASH_VARLEN and context.seq_info is None:
        backend = FLASH_DENSE if HAS_FLASH_ATTENTION else SDPA

    # [TODO] Flash/xFormers don't support arbitrary attn masks; with a padding
    # mask present (e.g. left-padded generation), fall back to SDPA.
    if context.attention_mask is not None and backend in (
        FLASH_DENSE,
        FLASH_VARLEN,
        XFORMERS,
    ):
        backend = SDPA

    flash_dense_kwargs = config.flash_dense_kwargs or {}
    flash_varlen_kwargs = config.flash_varlen_kwargs or {}
    sdpa_kwargs = config.sdpa_kwargs or {}
    xformers_kwargs = config.xformers_kwargs or {}

    bsz = context.bsz
    n_heads = context.n_heads
    q_len = context.q_len
    head_dim = context.head_dim
    kv_seq_len = context.kv_seq_len
    requires_grad = context.requires_grad
    sliding_window = context.sliding_window

    # DoRA promotes q/k/v_proj outputs to fp32, which FlashAttention rejects (and so does
    # the xformers flash-2 op on sm_100+, see _XFORMERS_FP32_UNSUPPORTED), so downcast any
    # fp32 Q/K/V to a supported dtype (#1013).
    if (
        backend in (FLASH_DENSE, FLASH_VARLEN)
        or (backend == XFORMERS and _XFORMERS_FP32_UNSUPPORTED)
    ) and torch.float32 in (
        Q.dtype,
        K.dtype,
        V.dtype,
    ):
        # Prefer the autocast dtype, else a non-fp32 input's dtype, then clamp.
        if torch.is_autocast_enabled():
            try:
                downcast_dtype = torch.get_autocast_dtype("cuda")
            except (AttributeError, TypeError):
                downcast_dtype = torch.get_autocast_gpu_dtype()
        else:
            downcast_dtype = next(
                (d for d in (Q.dtype, K.dtype, V.dtype) if d != torch.float32), None
            )
        if downcast_dtype not in (torch.float16, torch.bfloat16):
            downcast_dtype = torch.bfloat16 if SUPPORTS_BFLOAT16 else torch.float16
        Q, K, V = Q.to(downcast_dtype), K.to(downcast_dtype), V.to(downcast_dtype)

    if backend == FLASH_VARLEN:
        Q_f = Q.transpose(1, 2).reshape(bsz * q_len, n_heads, head_dim)
        K_f = K.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        V_f = V.transpose(1, 2).reshape(bsz * q_len, config.n_kv_heads, head_dim)
        _, cu_seqlens, max_seqlen = context.seq_info
        return flash_attn_varlen_func(
            Q_f,
            K_f,
            V_f,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            **flash_varlen_kwargs,
        ).view(bsz, q_len, n_heads, head_dim)
    elif backend == FLASH_DENSE:
        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)
        return flash_attn_func(Q_t, K_t, V_t, **flash_dense_kwargs).reshape(
            bsz, q_len, n_heads, head_dim
        )
    elif backend == XFORMERS:
        attn_bias = build_xformers_block_causal_mask(
            context.seq_info,
            sliding_window = sliding_window,
            base_mask = context.causal_mask,
        )

        Q_t = Q.transpose(1, 2)
        K_t = K.transpose(1, 2)
        V_t = V.transpose(1, 2)

        K_mod = K_t
        V_mod = V_t
        Q_mod = Q_t

        if config.n_groups != 1:
            K_mod = K_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            V_mod = V_t.view(bsz, kv_seq_len, config.n_kv_heads, 1, head_dim)
            K_mod = K_mod.expand(bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim)
            V_mod = V_mod.expand(bsz, kv_seq_len, config.n_kv_heads, config.n_groups, head_dim)

            if requires_grad:
                K_mod = K_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q_mod = Q_t.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)

        has_block = XFORMERS_BLOCK_DIAG_CLS is not None and isinstance(
            attn_bias, XFORMERS_BLOCK_DIAG_CLS
        )

        if config.n_groups != 1 and has_block:
            if not requires_grad:
                Q_mod = Q_mod.view(1, bsz * q_len, config.n_kv_heads, config.n_groups, head_dim)
                K_mod = K_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
                V_mod = V_mod.view(
                    1, bsz * kv_seq_len, config.n_kv_heads, config.n_groups, head_dim
                )
            else:
                Q_mod = Q_mod.view(1, bsz * q_len, n_heads, head_dim)
                K_mod = K_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)
                V_mod = V_mod.view(1, bsz * kv_seq_len, n_heads, head_dim)

        out = xformers_attention(
            Q_mod,
            K_mod,
            V_mod,
            attn_bias = attn_bias,
            **xformers_kwargs,
        )

        if config.n_groups != 1 and not requires_grad:
            out = out.view(bsz, q_len, config.n_kv_heads, config.n_groups, head_dim)
            out = out.reshape(bsz, q_len, n_heads, head_dim)
        else:
            out = out.view(bsz, q_len, n_heads, head_dim)
        return out
    else:
        local_mask = context.attention_mask
        is_causal_local = False
        if context.seq_info is not None and local_mask is None:
            local_mask = build_sdpa_packed_attention_mask(
                context.seq_info,
                dtype = Q.dtype,
                device = Q.device,
                sliding_window = sliding_window,
            )
        else:
            q_len_local = Q.shape[-2]
            k_len_local = K.shape[-2]
            # ---- SDPA mask normalization for left padding / 2D masks ----
            if local_mask is not None and isinstance(local_mask, torch.Tensor):
                local_mask = local_mask.to(device = Q.device)

                if local_mask.dim() == 2:
                    # key padding keep mask: (bsz, k_len), 1/True = real token
                    if local_mask.dtype == torch.bool:
                        key_keep = local_mask
                    else:
                        # tokenizer attention_mask is typically int 0/1
                        key_keep = local_mask != 0

                    past_len = k_len_local - q_len_local  # works for prefill (0) and decode
                    q_pos = torch.arange(past_len, past_len + q_len_local, device = Q.device)
                    k_pos = torch.arange(k_len_local, device = Q.device)

                    causal_keep = k_pos[None, :] <= q_pos[:, None]  # True = allowed (SDPA)
                    if sliding_window is not None:
                        causal_keep &= k_pos[None, :] >= (q_pos[:, None] - (sliding_window - 1))

                    # (bsz, 1, q_len, k_len) boolean keep mask
                    local_mask = causal_keep[None, None, :, :] & key_keep[:, None, None, :]

                elif local_mask.dim() == 3:
                    # (bsz, q_len, k_len) -> (bsz, 1, q_len, k_len)
                    local_mask = local_mask[:, None, :, :]

                elif local_mask.dim() == 4:
                    if local_mask.dtype != torch.bool:
                        # Use boolean keep masks for better SDPA stability.
                        local_mask = local_mask.eq(0)
                else:
                    raise ValueError(f"Unsupported SDPA attention_mask rank: {local_mask.dim()}")

                # Avoid NaNs from fully-masked rows (common with left padding).
                if local_mask.dtype == torch.bool:
                    no_allowed = ~local_mask.any(dim = -1, keepdim = True)  # (bsz,1,q_len,1)
                    local_mask = local_mask | no_allowed

            is_causal_local = local_mask is None and q_len_local == k_len_local

        kwargs = dict(sdpa_kwargs)
        kwargs.setdefault("attn_mask", local_mask)
        kwargs.setdefault("is_causal", is_causal_local)

        use_sdpa_gqa = SDPA_HAS_GQA and config.n_groups != 1
        if (
            use_sdpa_gqa
            and (not requires_grad)
            and isinstance(local_mask, torch.Tensor)
            and local_mask.dim() >= 3
            and local_mask.shape[0] > 1
        ):
            # Batched masked inference has shown row-coupled drift with SDPA GQA.
            # Fall back to explicit KV expansion for deterministic row-wise behavior.
            use_sdpa_gqa = False

        if use_sdpa_gqa:
            kwargs.setdefault("enable_gqa", True)
            out = scaled_dot_product_attention(Q, K, V, **kwargs)
            return out.transpose(1, 2)

        K_mod = K
        V_mod = V
        if config.n_groups != 1:
            K_mod = K[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            V_mod = V[:, :, None, :, :].expand(
                bsz, config.n_kv_heads, config.n_groups, kv_seq_len, head_dim
            )
            K_mod = K_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V_mod = V_mod.reshape(bsz, n_heads, kv_seq_len, head_dim)

        out = scaled_dot_product_attention(
            Q.contiguous(),
            K_mod.contiguous(),
            V_mod.contiguous(),
            **kwargs,
        )
        return out.transpose(1, 2).contiguous()


__all__ = [
    "AttentionConfig",
    "AttentionContext",
    "select_attention_backend",
    "resolve_prefix_seg_info",
    "run_attention",
]
