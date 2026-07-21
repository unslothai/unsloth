# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""FlexAttention shared-prefix kernel for PrefixGrouper (GRPO shared-prompt dedup).

In GRPO every prompt spawns ``G = num_generations`` completions that share the same
prompt prefix. The full-row packed path forwards the identical prefix ``G`` times.
PrefixGrouper stores the prefix ONCE and concatenates only the ``G`` suffixes, with an
attention layout where each suffix token attends to ``[the single shared prefix] +
[causal within its own suffix]``. This kernel expresses that one-prefix -> many-suffix
fan-out via a ``torch.nn.attention.flex_attention`` block mask, so the masked-out
cross-suffix / cross-group blocks are never computed and the ``P + G*R`` FLOP saving is
realised (not merely a masked dense ``O(T^2)``).

Mask semantics (identical to the certified SDPA oracle):

    keep(q_idx, kv_idx) = same_group(q, kv) AND
        ( is_prefix[kv_idx]                                  # full prefix visibility
          OR ( suffix_of_kv[kv_idx] == suffix_of_kv[q_idx]   # same suffix ...
               AND kv_idx <= q_idx ) )                       # ... causal within it

This module is self-contained (no dependency on any temp/ scratch dir) so PrefixGrouper
works from the installed source after a fresh compile. It is only imported lazily from
``attention_dispatch.run_attention`` when ``prefix_seg_info`` is present, which itself is
only ever set when ``UNSLOTH_GRPO_PREFIX_GROUPER`` is on and grouping succeeded, so the
default (off) path never touches this file.

Provided entry points:
  * ``PrefixSegInfo``            : per-flat-token segment metadata + cache signature.
  * ``build_seg_info_multigroup``: build PrefixSegInfo for many groups packed flat.
  * ``build_seg_info_from_layout``: build PrefixSegInfo for ONE group (test helper).
  * ``get_block_mask``          : cached create_block_mask keyed on the signature.
  * ``flex_shared_prefix_attention(Q, K, V, prefix_seg_info)``
        Q/K/V of shape [1, T, n_heads, head_dim]; returns [1, T, n_heads, head_dim],
        IDENTICAL semantics to the SDPA oracle.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

# GRPO feeds many distinct segment lengths; at dynamo's default recompile_limit (8) the
# compiled kernel silently reuses a mismatched specialisation (wrong results). Raise it.
torch._dynamo.config.recompile_limit = max(
    getattr(torch._dynamo.config, "recompile_limit", 8), 256
)
torch._dynamo.config.accumulated_recompile_limit = max(
    getattr(torch._dynamo.config, "accumulated_recompile_limit", 256), 2048
)


# Compiled kernels: torch.compile fuses the sparse mask into one kernel. dynamic=True is
# required: T changes almost every GRPO batch and dynamic=False recompiles per T (~14s
# each). T is still padded to a multiple of 128 (_pad_len) for the backward kernel.
_flex_attention_compiled = torch.compile(flex_attention, dynamic = True)
_create_block_mask_compiled = torch.compile(create_block_mask, dynamic = True)

# Flash block sizes by Q dtype (env-overridable). The two disjoint key runs (prefix +
# own-suffix) stress online-softmax accumulation: fp32 needs 32/32 for a ~1e-6 floor;
# bf16 passes parity at 128/64 and is ~5x faster (128/128 OOMs Triton on B200).
_FP32_BLOCK_M = int(os.environ.get("PG_FLEX_BLOCK_M", "32"))
_FP32_BLOCK_N = int(os.environ.get("PG_FLEX_BLOCK_N", "32"))
_BF16_BLOCK_M = int(os.environ.get("PG_FLEX_BF16_BLOCK_M", "128"))
_BF16_BLOCK_N = int(os.environ.get("PG_FLEX_BF16_BLOCK_N", "64"))


def _kernel_options_for_dtype(dtype):
    """Pick the numerically-safe flash block sizes for the Q dtype."""
    if dtype == torch.bfloat16 or dtype == torch.float16:
        return {"BLOCK_M": _BF16_BLOCK_M, "BLOCK_N": _BF16_BLOCK_N}
    return {"BLOCK_M": _FP32_BLOCK_M, "BLOCK_N": _FP32_BLOCK_N}


# Backward-compat constant (fp32 default).
_FLEX_KERNEL_OPTIONS = {"BLOCK_M": _FP32_BLOCK_M, "BLOCK_N": _FP32_BLOCK_N}

# The compiled backward trips an Inductor assertion when T is not a multiple of 128, so
# pad the flat sequence. Pad tokens form a group that attends to / is attended by nothing
# (all-masked rows return 0, not NaN) and are sliced off the output.
_PAD_MULTIPLE = 128
_PAD_GROUP = -99  # sentinel group id / suffix id for pad tokens


def _pad_len(T: int) -> int:
    return ((T + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


# ---------------------------------------------------------------------------
# Segment metadata
# ---------------------------------------------------------------------------


@dataclass
class PrefixSegInfo:
    """Per-flat-token segment metadata driving the shared-prefix block mask.

    The label tensors are 1-D of length ``T_pad`` (>= real ``T``, padded up to a multiple
    of 128 so the backward kernel compiles). Positions ``[T:T_pad)`` are pad tokens
    (group/suffix == _PAD_GROUP) that attend to nothing.

    Attributes
    ----------
    group_of_kv : LongTensor [T_pad]
        Group id per flat token (0..num_groups-1); _PAD_GROUP for pad tokens.
    is_prefix : BoolTensor [T_pad]
        True iff the token is a prefix token of its group (False for pad).
    suffix_of_kv : LongTensor [T_pad]
        Suffix id per flat token; -1 for prefix, _PAD_GROUP for pad. Suffix ids are
        globally unique across groups.
    signature : hashable
        Cache key for the block mask (depends only on the labels + T_pad).
    T : int
        Real flat sequence length (Q/K/V of this length are padded internally).
    T_pad : int
        Padded length (multiple of 128) at which the block mask is built.
    """

    group_of_kv: torch.Tensor
    is_prefix: torch.Tensor
    suffix_of_kv: torch.Tensor
    signature: Tuple
    T: int
    T_pad: int


def _pad_labels(group_of_kv, is_prefix, suffix_of_kv, device):
    """Pad the label tensors up to a multiple of 128 with pad-token sentinels."""
    T = int(group_of_kv.numel())
    T_pad = _pad_len(T)
    if T_pad == T:
        return group_of_kv, is_prefix, suffix_of_kv, T, T_pad
    pad = T_pad - T
    group_of_kv = torch.cat(
        [group_of_kv, torch.full((pad,), _PAD_GROUP, dtype = torch.long, device = device)]
    )
    is_prefix = torch.cat(
        [is_prefix, torch.zeros(pad, dtype = torch.bool, device = device)]
    )
    suffix_of_kv = torch.cat(
        [suffix_of_kv, torch.full((pad,), _PAD_GROUP, dtype = torch.long, device = device)]
    )
    return group_of_kv, is_prefix, suffix_of_kv, T, T_pad


def build_seg_info_from_layout(
    layout, device: Optional[torch.device] = None
) -> PrefixSegInfo:
    """Build PrefixSegInfo for ONE group from an object with ``.flat_ids``, ``.P`` and
    ``.suffix_slices`` (used by the parity test / oracle helpers)."""
    if device is None:
        device = layout.flat_ids.device
    T = int(layout.flat_ids.shape[1])
    P = int(layout.P)

    group_of_kv = torch.zeros(T, dtype = torch.long, device = device)  # single group -> 0
    is_prefix = torch.zeros(T, dtype = torch.bool, device = device)
    is_prefix[:P] = True
    suffix_of_kv = torch.full((T,), -1, dtype = torch.long, device = device)
    for i, (s, e) in enumerate(layout.suffix_slices):
        suffix_of_kv[s:e] = i

    group_of_kv, is_prefix, suffix_of_kv, T, T_pad = _pad_labels(
        group_of_kv, is_prefix, suffix_of_kv, device
    )
    sig = ("single", T_pad, P, tuple((s, e) for (s, e) in layout.suffix_slices))
    return PrefixSegInfo(
        group_of_kv = group_of_kv,
        is_prefix = is_prefix,
        suffix_of_kv = suffix_of_kv,
        signature = sig,
        T = T,
        T_pad = T_pad,
    )


def build_seg_info_multigroup(
    group_specs: List[Tuple[int, List[int]]], device: torch.device
) -> Tuple[PrefixSegInfo, List[dict]]:
    """Build PrefixSegInfo for several shared-prefix groups packed block-diagonally.

    Parameters
    ----------
    group_specs : list of (P_g, [R_{g,0}, R_{g,1}, ...])
        For each group: prefix length and the list of suffix lengths.

    Returns
    -------
    seg : PrefixSegInfo
    group_meta : list of dicts with 'base', 'P', 'prefix_last_index', 'suffix_slices'
        (flat offsets), enough to build the completion index map.
    """
    group_of_list = []
    is_prefix_list = []
    suffix_of_list = []
    group_meta = []

    base = 0
    suffix_counter = 0
    sig_parts = []
    for gid, (P, R_list) in enumerate(group_specs):
        # prefix
        group_of_list.append(torch.full((P,), gid, dtype = torch.long, device = device))
        is_prefix_list.append(torch.ones(P, dtype = torch.bool, device = device))
        suffix_of_list.append(torch.full((P,), -1, dtype = torch.long, device = device))
        prefix_last_index = base + P - 1
        suffix_slices = []
        cursor = base + P
        for r in R_list:
            group_of_list.append(torch.full((r,), gid, dtype = torch.long, device = device))
            is_prefix_list.append(torch.zeros(r, dtype = torch.bool, device = device))
            suffix_of_list.append(
                torch.full((r,), suffix_counter, dtype = torch.long, device = device)
            )
            suffix_slices.append((cursor, cursor + r))
            cursor += r
            suffix_counter += 1
        group_meta.append(
            {
                "base": base,
                "P": P,
                "prefix_last_index": prefix_last_index,
                "suffix_slices": suffix_slices,
            }
        )
        sig_parts.append((P, tuple(R_list)))
        base = cursor

    group_of_kv = torch.cat(group_of_list)
    is_prefix = torch.cat(is_prefix_list)
    suffix_of_kv = torch.cat(suffix_of_list)
    group_of_kv, is_prefix, suffix_of_kv, T, T_pad = _pad_labels(
        group_of_kv, is_prefix, suffix_of_kv, device
    )
    sig = ("multi", T_pad, tuple(sig_parts))
    seg = PrefixSegInfo(
        group_of_kv = group_of_kv,
        is_prefix = is_prefix,
        suffix_of_kv = suffix_of_kv,
        signature = sig,
        T = T,
        T_pad = T_pad,
    )
    return seg, group_meta


# ---------------------------------------------------------------------------
# Block-mask builder + cache, keyed on (signature, device): the mask depends only on the
# per-token labels and T, so it is reused across layers and steps.

_BLOCK_MASK_CACHE: Dict[Tuple, BlockMask] = {}


def _make_mask_mod(group_of_kv, is_prefix, suffix_of_kv):
    """Return a mask_mod closure over the (device) label tensors.

    keep(q, kv) = same_group AND
        ( is_prefix[kv]  AND kv <= q                     # causal within/ into prefix
          OR ( suffix_of_kv[kv] == suffix_of_kv[q]       # same suffix ...
               AND (not is_prefix[q])                    # q is a suffix token ...
               AND kv <= q ) )                           # ... causal within it

    The single ``kv <= q`` guard on the is_prefix branch gives BOTH prefix-causal
    behaviour (a prefix q sees only earlier prefix tokens) AND full-prefix-visibility for
    suffixes (every prefix index < every suffix index in a group, so kv <= q always holds
    for a suffix q vs a prefix kv of its group), matching the SDPA oracle exactly.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        same_group = group_of_kv[q_idx] == group_of_kv[kv_idx]
        kv_is_prefix = is_prefix[kv_idx]
        causal = kv_idx <= q_idx
        same_suffix = (suffix_of_kv[kv_idx] == suffix_of_kv[q_idx]) & (
            ~is_prefix[q_idx]
        )
        keep = same_group & ((kv_is_prefix & causal) | (same_suffix & causal))
        return keep

    return mask_mod


def get_block_mask(
    seg: PrefixSegInfo,
    device: torch.device,
    compile_mask: bool = True,
) -> BlockMask:
    """Return a cached BlockMask for the segment signature (built once, reused).

    CRITICAL: the block mask is cached and shared across BOTH the no-grad old/ref logprob
    forward (which runs under torch.inference_mode) and the grad training forward. If the
    mask were first built under inference_mode, its tensors would be INFERENCE tensors that
    "cannot be saved for backward" when reused in the grad forward. We therefore build the
    mask with inference mode explicitly DISABLED, so the same cached BlockMask is a normal
    tensor usable by autograd. (The mask depends only on integer labels; it needs no grad.)
    """
    key = (seg.signature, str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is not None:
        return bm

    # Move labels to the consumer (Q) device: with a sharded model the seg tensors live on
    # input_ids.device and would index cross-device. Copies once per (signature, device).
    # These copies must also run with inference mode DISABLED (same reason as the mask build):
    # when this entry is first built under the no-grad old/ref forward's inference_mode and
    # device != seg.device, a .to(device) copy would be an inference tensor that mask_mod
    # captures, which then cannot be saved for backward when the grad training forward reuses
    # the cached mask.
    builder = _create_block_mask_compiled if compile_mask else create_block_mask
    with torch.inference_mode(False):
        mask_mod = _make_mask_mod(
            seg.group_of_kv.to(device),
            seg.is_prefix.to(device),
            seg.suffix_of_kv.to(device),
        )
        bm = builder(
            mask_mod,
            B = 1,
            H = None,
            Q_LEN = seg.T_pad,
            KV_LEN = seg.T_pad,
            device = device,
        )
    # FIFO bound: GRPO lengths change nearly every step, so evict the oldest to cap GPU pins.
    if len(_BLOCK_MASK_CACHE) >= 8:
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = bm
    return bm


def clear_block_mask_cache():
    _BLOCK_MASK_CACHE.clear()


def _pad_qkv_seq(x: torch.Tensor, T_pad: int) -> torch.Tensor:
    """Zero-pad a [B, H, T, D] tensor along the sequence dim up to T_pad."""
    T = x.shape[2]
    if T_pad == T:
        return x
    pad = torch.zeros(
        x.shape[0], x.shape[1], T_pad - T, x.shape[3], device = x.device, dtype = x.dtype
    )
    return torch.cat([x, pad], dim = 2)


def _run_flex(q, k, v, block_mask, enable_gqa, scale, compiled, T, T_pad):
    """Pad q/k/v to T_pad, run flex, slice the output back to T. q/k/v: [B,H,T,D]."""
    qp = _pad_qkv_seq(q, T_pad)
    kp = _pad_qkv_seq(k, T_pad)
    vp = _pad_qkv_seq(v, T_pad)
    if compiled:
        out = _flex_attention_compiled(
            qp,
            kp,
            vp,
            block_mask = block_mask,
            enable_gqa = enable_gqa,
            scale = scale,
            kernel_options = _kernel_options_for_dtype(qp.dtype),
        )
    else:
        # eager path (fp64 parity): dense scores, no kernel_options.
        out = flex_attention(
            qp,
            kp,
            vp,
            block_mask = block_mask,
            enable_gqa = enable_gqa,
            scale = scale,
        )
    return out[:, :, :T, :]


# ---------------------------------------------------------------------------
# The kernel entry point
# ---------------------------------------------------------------------------


def flex_shared_prefix_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    prefix_seg_info: PrefixSegInfo,
    scale: Optional[float] = None,
    block_mask: Optional[BlockMask] = None,
    compiled: bool = True,
) -> torch.Tensor:
    """Shared-prefix attention via FlexAttention.

    Parameters
    ----------
    Q, K, V : Tensor [1, T, n_heads, head_dim]
        (Q has n_heads, K/V have n_kv_heads for GQA).
    prefix_seg_info : PrefixSegInfo
    scale : optional float, softmax scale (defaults to 1/sqrt(head_dim)).
    block_mask : optional precomputed BlockMask (else built/cached from seg info).

    Returns
    -------
    Tensor [1, T, n_heads, head_dim], identical semantics to the SDPA oracle branch.
    """
    assert Q.dim() == 4 and Q.shape[0] == 1, f"expected [1,T,H,D], got {tuple(Q.shape)}"
    device = Q.device
    # FlexAttention wants [B, H, T, D].
    q = Q.transpose(1, 2)  # [1, n_heads, T, D]
    k = K.transpose(1, 2)  # [1, n_kv_heads, T, D]
    v = V.transpose(1, 2)

    n_heads = q.shape[1]
    n_kv = k.shape[1]
    enable_gqa = n_heads != n_kv
    T = q.shape[2]
    T_pad = prefix_seg_info.T_pad
    assert T == prefix_seg_info.T, f"Q length {T} != seg.T {prefix_seg_info.T}"

    if block_mask is None:
        block_mask = get_block_mask(prefix_seg_info, device, compile_mask = compiled)

    out = _run_flex(q, k, v, block_mask, enable_gqa, scale, compiled, T, T_pad)
    # back to [1, T, n_heads, D]
    return out.transpose(1, 2).contiguous()


__all__ = [
    "PrefixSegInfo",
    "build_seg_info_multigroup",
    "build_seg_info_from_layout",
    "get_block_mask",
    "clear_block_mask_cache",
    "flex_shared_prefix_attention",
]
