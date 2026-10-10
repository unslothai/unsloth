# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Varlen attention for packed rows on the transformers modeling path.

A padding-free or packed row reaches a transformers attention layer with a dense block-causal
mask built from its position_ids, which SDPA can only run on its masked kernels. Unsloth's own
attention already runs those rows on xformers' block-diagonal mask or flash-attn varlen, so the
transformers "sdpa" entry is wrapped to send a packed row there instead. Anything else, or any
row whose mask is not exactly the block-causal one the packed lengths describe, falls through to
the wrapped function unchanged.
"""

import os
import weakref

import torch

__all__ = ["enable_hf_packed_attention", "HF_PACKED_ATTENTION_STATS"]

_FLAG = "_unsloth_hf_packed_varlen"
_ENV = "UNSLOTH_HF_PACKED_VARLEN"
_ORIG_SDPA = [None]
# fast = varlen calls, fallback = packed rows handed back to the wrapped sdpa.
HF_PACKED_ATTENTION_STATS = {"fast": 0, "fallback": 0, "mask_checks": 0}
# Every layer of one forward (and its checkpoint recompute) gets the same mask object.
_MASK_CHECK = [None, None, None]  # weakref to mask, cu_seqlens it was checked against, verdict


def _disabled() -> bool:
    return os.environ.get(_ENV, "1").lower() in ("0", "false", "no", "off")


def _mask_is_block_causal(attention_mask, cu_seqlens, total) -> bool:
    """Is the dense mask transformers built exactly causal within each packed segment and closed
    across them? Checked once per mask object, so a window, a bidirectional span or a pad tail
    transformers segments differently all keep the masked path."""
    # None is transformers' is_causal skip (one document): the wrapped sdpa already runs it fast.
    if not (
        isinstance(attention_mask, torch.Tensor)
        and attention_mask.dtype == torch.bool
        and attention_mask.dim() == 4
        and attention_mask.shape[0] == 1
        and attention_mask.shape[1] == 1
        and attention_mask.shape[-2:] == (total, total)
    ):
        return False
    # Versions catch an in-place edit of a reused buffer; inference tensors carry none, so are rechecked.
    cacheable = not (torch.is_inference(attention_mask) or torch.is_inference(cu_seqlens))
    key = (attention_mask._version, cu_seqlens._version, total) if cacheable else None
    ref, checked, verdict = _MASK_CHECK
    if cacheable and ref is not None and ref() is attention_mask and checked == (cu_seqlens, key):
        return verdict
    HF_PACKED_ATTENTION_STATS["mask_checks"] += 1
    verdict = False
    if int(cu_seqlens[-1].item()) == total:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        segment = torch.repeat_interleave(
            torch.arange(lengths.numel(), device = attention_mask.device),
            lengths.to(attention_mask.device),
        )
        expected = (segment[:, None] == segment[None, :]).tril_()
        verdict = bool(torch.equal(attention_mask[0, 0], expected))
    if cacheable:
        _MASK_CHECK[:] = [weakref.ref(attention_mask), (cu_seqlens, key), verdict]
    return verdict


def _autocast_dtype_differs(query) -> bool:
    """SDPA under autocast runs in the autocast dtype; the varlen kernels would keep the input's."""
    try:
        enabled = torch.is_autocast_enabled(query.device.type)
        dtype = torch.get_autocast_dtype(query.device.type) if enabled else None
    except (AttributeError, TypeError):
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype() if enabled else None
    return enabled and dtype != query.dtype


def _sliding_or_softcapped(module, kwargs) -> bool:
    if kwargs.get("sliding_window") or getattr(module, "sliding_window", None):
        return True
    if kwargs.get("softcap") or getattr(module, "sinks", None) is not None:
        return True
    config = getattr(module, "config", None)
    return bool(getattr(config, "attn_logit_softcapping", None))


def _sdpa_packed_varlen(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout = 0.0,
    scaling = None,
    is_causal = None,
    **kwargs,
):
    orig = _ORIG_SDPA[0]
    if kwargs.get("packed_seq_lengths") is None or _disabled():
        return orig(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )
    from .attention_dispatch import (
        FLASH_VARLEN,
        XFORMERS,
        AttentionConfig,
        AttentionContext,
        run_attention,
        select_attention_backend,
    )
    from .packing import get_packed_info_from_kwargs

    backend = select_attention_backend(use_varlen = True)
    bsz, n_heads, q_len, head_dim = query.shape
    seq_info = None
    if (
        backend in (FLASH_VARLEN, XFORMERS)
        and not torch.compiler.is_compiling()
        and bsz == 1
        and q_len > 0
        and query.is_cuda
        and query.device == key.device == value.device
        and key.shape[2] == q_len
        # Flash and xformers cover equal Q/K/V widths up to 256 (not MLA's 192 / 128, nor Gemma-4's 512).
        and key.shape[-1] == value.shape[-1] == head_dim
        and head_dim <= 256
        and head_dim % 8 == 0
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.dtype == key.dtype == value.dtype
        and not _autocast_dtype_differs(query)
        and not dropout
        and getattr(module, "is_causal", True)
        and kwargs.get("position_bias") is None
        # A paged / static cache must still be written by the wrapped sdpa.
        and kwargs.get("cache") is None
        and kwargs.get("past_key_values") is None
        and n_heads % key.shape[1] == 0
        and not _sliding_or_softcapped(module, kwargs)
    ):
        seq_info = get_packed_info_from_kwargs(kwargs, query.device)
    if seq_info is None or not _mask_is_block_causal(attention_mask, seq_info[1], q_len):
        HF_PACKED_ATTENTION_STATS["fallback"] += 1
        return orig(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )
    HF_PACKED_ATTENTION_STATS["fast"] += 1
    scale = head_dim**-0.5 if scaling is None else scaling
    n_kv_heads = key.shape[1]
    config = AttentionConfig(
        backend = backend,
        n_kv_heads = n_kv_heads,
        n_groups = n_heads // n_kv_heads,
        flash_varlen_kwargs = {"dropout_p": 0.0, "causal": True, "softmax_scale": scale},
        sdpa_kwargs = {"scale": scale},
        xformers_kwargs = {"scale": scale},
    )
    context = AttentionContext(
        bsz = 1,
        q_len = q_len,
        kv_seq_len = q_len,
        n_heads = n_heads,
        head_dim = head_dim,
        # xformers picks a backward-capable op off the tensors alone, even under no_grad, and that op
        # rejects the grouped 5D layout run_attention uses when nothing needs grad.
        requires_grad = query.requires_grad or key.requires_grad or value.requires_grad,
        seq_info = seq_info,
        attention_mask = None,
        causal_mask = None,
    )
    # [1, T, H, D], the layout transformers' sdpa_attention_forward returns.
    return run_attention(config = config, context = context, Q = query, K = key, V = value), None


def enable_hf_packed_attention() -> bool:
    """Wrap transformers' "sdpa" attention entry once. True when the wrapper is installed."""
    if _disabled():
        return False
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return False
    current = ALL_ATTENTION_FUNCTIONS.get("sdpa", None)
    if current is None:
        return False
    # Once per process: a router installed above this one later (gemma-4's) does not forward the
    # sentinel, and wrapping it again would close a loop through its own boxed original.
    if getattr(current, _FLAG, False) or _ORIG_SDPA[0] is not None:
        return True
    _ORIG_SDPA[0] = current
    # The unsloth_zoo sdpa routers guard re-entry by a sentinel read off the installed entry, and
    # re-run on every model load: forward theirs, or the next load wraps this and loops.
    try:
        for name, value in vars(current).items():
            if name.startswith("_unsloth_"):
                setattr(_sdpa_packed_varlen, name, value)
    except Exception:
        pass
    setattr(_sdpa_packed_varlen, _FLAG, True)
    # Direct assignment: register() does not update the mapping layers read.
    ALL_ATTENTION_FUNCTIONS["sdpa"] = _sdpa_packed_varlen
    return True
