# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""GRPO PrefixGrouper FlexAttention kernel: shared prefix stored once. Imported lazily."""

from __future__ import annotations

import contextlib
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


# torch 2.6 inductor fails on the dynamic mask + flex kernel; suppress_errors then runs eager T x T flex.
PREFIX_GROUPER_MIN_TORCH = (2, 7)


def _strict_dynamo():
    """Dynamo with suppress_errors off; some torch builds (AMD's ROCm torch 2.11) have no such flag."""
    config = torch._dynamo.config
    return (
        config.patch(suppress_errors = False)
        if hasattr(config, "suppress_errors")
        else contextlib.nullcontext()
    )


def torch_supports_prefix_grouper(version = None) -> bool:
    """True unless the torch major.minor is known to be below PREFIX_GROUPER_MIN_TORCH."""
    match = re.match(r"(\d+)\.(\d+)", str(torch.__version__ if version is None else version))
    if match is None:
        return True
    return (int(match.group(1)), int(match.group(2))) >= PREFIX_GROUPER_MIN_TORCH


TORCH_SUPPORTS_PREFIX_GROUPER = torch_supports_prefix_grouper()


# Default recompile_limit (8) silently reuses a mismatched specialisation; old name cache_size_limit.
def _raise_dynamo_limit(names, floor):
    config = torch._dynamo.config
    name = next((n for n in names if hasattr(config, n)), None)
    if name is not None:
        setattr(config, name, max(getattr(config, name), floor))


if TORCH_SUPPORTS_PREFIX_GROUPER:
    _raise_dynamo_limit(("recompile_limit", "cache_size_limit"), 256)
    _raise_dynamo_limit(("accumulated_recompile_limit", "accumulated_cache_size_limit"), 2048)


# dynamic=True: T changes almost every batch; static recompiles per T (~14 s each).
_flex_attention_dynamic = torch.compile(flex_attention, dynamic = True)


def _rocm_static_flex_entry(query, key, value, **kwargs):
    # Its own code object, so static graphs keep their own recompile budget.
    return flex_attention(query, key, value, **kwargs)


# ROCm: flex compiled with a symbolic sequence length returned inf / NaN gradients on gfx1151 (torch 2.11), so
# calls whose backward runs compile per padded length there. UNSLOTH_ROCM_FLEX_STATIC=0 keeps the dynamic kernel.
_ROCM_FLEX_STATIC = bool(getattr(torch.version, "hip", None)) and (
    os.environ.get("UNSLOTH_ROCM_FLEX_STATIC", "1") != "0"
)
_flex_attention_static = (
    torch.compile(_rocm_static_flex_entry, dynamic = False) if _ROCM_FLEX_STATIC else None
)


def _uses_static_flex(query, key, value) -> bool:
    """ROCm GPU call whose backward runs: compiled per padded length (always False on CUDA builds)."""
    return (
        _flex_attention_static is not None
        and query.device.type == "cuda"
        and torch.is_grad_enabled()
        and (query.requires_grad or key.requires_grad or value.requires_grad)
    )


def _flex_attention_compiled(query, key, value, **kwargs):
    if _uses_static_flex(query, key, value):
        return _flex_attention_static(query, key, value, **kwargs)
    return _flex_attention_dynamic(query, key, value, **kwargs)


_create_block_mask_compiled = torch.compile(create_block_mask, dynamic = True)

# fp32 needs 32/32 for a ~1e-6 floor; bf16 128/64 is ~5x faster; 128/128 OOMs Triton on B200.
_FP32_BLOCK_M = int(os.environ.get("PG_FLEX_BLOCK_M", "32"))
_FP32_BLOCK_N = int(os.environ.get("PG_FLEX_BLOCK_N", "32"))
_BF16_BLOCK_M = int(os.environ.get("PG_FLEX_BF16_BLOCK_M", "128"))
_BF16_BLOCK_N = int(os.environ.get("PG_FLEX_BF16_BLOCK_N", "64"))


# The 16-bit 128 / 64 tile needs 98304 B of shared memory at head_dim 128 (one stage too), so it
# cannot compile on 64 KB parts (T4 sm75; AMD parts with 64 KB LDS). Smaller forward and backward
# tiles compile there, picked from the queried limit. Opt-in: Triton emits no tensor-core MMA for
# flex on sm75, so on T4 GRPO steps are 40% slower for Qwen3-0.6B and 3% faster for Qwen3-1.7B.
_SMALL_SMEM_BELOW_BYTES = 98304
_SMALL_SMEM_ENV = os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_SMALL_SMEM", "0")


def _small_smem_opt_in(env, torch_version) -> bool:
    # The fwd_ / bwd_ kernel_options keys are read from torch 2.10 on; older releases keep the opt-in off.
    version = tuple(int(x) for x in re.findall(r"\d+", torch_version.split("+")[0])[:2])
    return env.lower() in ("1", "true", "yes", "on") and version >= (2, 10)


_SMALL_SMEM_OPT_IN = _small_smem_opt_in(_SMALL_SMEM_ENV, torch.__version__)
# T4 sweep (head_dim 128, fp16): forward 64 / 64 needs 65536 B even with one stage; 32 / 32 is the
# fastest forward that fits; backward 32 / 32 with 8 warps beats the 16 / 16 default (37 vs 86 ms).
_SMALL_SMEM_16BIT_OPTIONS = {
    "fwd_BLOCK_M": 32,
    "fwd_BLOCK_N": 32,
    "fwd_num_stages": 1,
    "fwd_num_warps": 4,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
    "bwd_num_stages": 1,
    "bwd_num_warps": 8,
}
_SMEM_LIMIT: Dict[str, Optional[int]] = {}


@torch._dynamo.disable
def _query_smem_limit(device) -> Optional[int]:
    """Per-block shared memory limit the Triton compiler checks against; None off GPU / unknown."""
    limit = None
    if device.type == "cuda":
        try:
            import triton
            props = triton.runtime.driver.active.utils.get_device_properties(device.index or 0)
            limit = int(props["max_shared_mem"])
        except Exception:
            props = torch.cuda.get_device_properties(device)
            for name in ("shared_memory_per_block_optin", "shared_memory_per_block"):
                if getattr(props, name, 0):
                    limit = int(getattr(props, name))
                    break
    _SMEM_LIMIT[str(device)] = limit
    return limit


def _small_smem(device) -> bool:
    if device is None:
        return False
    key = str(device)
    if key not in _SMEM_LIMIT:
        _query_smem_limit(device)
    limit = _SMEM_LIMIT[key]
    return limit is not None and limit < _SMALL_SMEM_BELOW_BYTES


def _kernel_options_for_dtype(
    dtype,
    one_stage = False,
    device = None,
):
    """Pick the numerically-safe flash block sizes for the Q dtype."""
    if dtype == torch.bfloat16 or dtype == torch.float16:
        if _SMALL_SMEM_OPT_IN and _small_smem(device):
            # Stages are already pinned per pass, so the one-stage retry has nothing to change.
            return dict(_SMALL_SMEM_16BIT_OPTIONS)
        options = {"BLOCK_M": _BF16_BLOCK_M, "BLOCK_N": _BF16_BLOCK_N}
    else:
        options = {"BLOCK_M": _FP32_BLOCK_M, "BLOCK_N": _FP32_BLOCK_N}
    if one_stage:
        options["num_stages"] = 1
    return options


# The compiled backward asserts unless T is a multiple of 128; pad tokens are fully masked and sliced off.
_PAD_MULTIPLE = 128
_PAD_GROUP = -99  # sentinel group id / suffix id for pad tokens


def _pad_len(T: int) -> int:
    return ((T + _PAD_MULTIPLE - 1) // _PAD_MULTIPLE) * _PAD_MULTIPLE


@dataclass
class PrefixSegInfo:
    """Per-token labels for the block mask, padded to T_pad (multiple of 128); pad tokens see nothing.

    suffix_of_kv is -1 for prefix tokens and globally unique across groups.
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
    is_prefix = torch.cat([is_prefix, torch.zeros(pad, dtype = torch.bool, device = device)])
    suffix_of_kv = torch.cat(
        [suffix_of_kv, torch.full((pad,), _PAD_GROUP, dtype = torch.long, device = device)]
    )
    return group_of_kv, is_prefix, suffix_of_kv, T, T_pad


def build_seg_info_from_layout(layout, device: Optional[torch.device] = None) -> PrefixSegInfo:
    """PrefixSegInfo for one group (parity test / oracle helper)."""
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
    """group_specs: [(P_g, [R_g0, R_g1, ...])]. Returns (seg, per-group flat offsets)."""
    labels, T, T_pad, group_meta, sig_parts = _seg_labels_host(group_specs)
    group_of_kv, is_prefix, suffix_of_kv = _labels_to_device(labels, T_pad, device)
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


@dataclass
class TreeSegInfo:
    """A prefix trie laid out in DFS pre-order (TRL >= 1.17 AsyncGRPO rows), padded to T_pad.

    Each segment's subtree is one contiguous token range, so kv token j is visible to query q iff
    j <= q < reach_end[j], where reach_end[j] ends the subtree of j's segment; pad tokens hold -1.
    """

    reach_end: torch.Tensor
    signature: Tuple
    T: int
    T_pad: int


def build_tree_seg_info(segment_offsets, segment_parents, device: torch.device) -> TreeSegInfo:
    """Raises ValueError unless segments are in DFS pre-order with every parent before its children."""
    offsets = [int(x) for x in torch.as_tensor(segment_offsets).tolist()]
    parents = [int(x) for x in torch.as_tensor(segment_parents).tolist()]
    n = len(parents)
    if n == 0 or len(offsets) != n + 1 or offsets[0] != 0:
        raise ValueError("tree segments: expected S parents and S + 1 offsets starting at 0")
    if any(b <= a for a, b in zip(offsets[:-1], offsets[1:])):
        raise ValueError("tree segments: every segment needs at least one token")
    end = list(range(1, n + 1))
    for s in range(n - 1, -1, -1):
        p = parents[s]
        if p >= s:
            raise ValueError("tree segments: a parent must precede its children")
        if p >= 0:
            end[p] = max(end[p], end[s])
    # Pre-order check: each segment's parent is the innermost open subtree containing it.
    stack = []
    for s in range(n):
        while stack and end[stack[-1]] <= s:
            stack.pop()
        if parents[s] != (stack[-1] if stack else -1):
            raise ValueError("tree segments are not in DFS pre-order")
        stack.append(s)
    T = offsets[-1]
    T_pad = _pad_len(T)
    lengths = torch.tensor([b - a for a, b in zip(offsets[:-1], offsets[1:])], dtype = torch.long)
    seg_end = torch.tensor([offsets[e] for e in end], dtype = torch.long)
    reach_end = torch.full((T_pad,), -1, dtype = torch.long)
    reach_end[:T] = torch.repeat_interleave(seg_end, lengths, output_size = T)
    signature = ("tree", T_pad, tuple(offsets), tuple(parents))
    return TreeSegInfo(reach_end = reach_end.to(device), signature = signature, T = T, T_pad = T_pad)


def _make_tree_mask_mod(reach_end):
    def mask_mod(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx < reach_end[kv_idx])

    return mask_mod


def _seg_labels_host(group_specs):
    """CPU int64 [3, T_pad] rows (group_of_kv, is_prefix, suffix_of_kv); T, T_pad, group_meta."""
    group_meta = []
    sig_parts = []
    seg_lens = []
    seg_group = []
    seg_prefix = []
    seg_suffix = []
    base = 0
    suffix_counter = 0
    if not group_specs:
        torch.cat([])  # the empty-list error the per-segment build raised
    for gid, (P, R_list) in enumerate(group_specs):
        seg_lens.append(P)
        seg_group.append(gid)
        seg_prefix.append(1)
        seg_suffix.append(-1)
        suffix_slices = []
        cursor = base + P
        for r in R_list:
            seg_lens.append(r)
            seg_group.append(gid)
            seg_prefix.append(0)
            seg_suffix.append(suffix_counter)
            suffix_slices.append((cursor, cursor + r))
            cursor += r
            suffix_counter += 1
        group_meta.append(
            {
                "base": base,
                "P": P,
                "prefix_last_index": base + P - 1,
                "suffix_slices": suffix_slices,
            }
        )
        sig_parts.append((P, tuple(R_list)))
        base = cursor
    T = base
    T_pad = _pad_len(T)
    labels = torch.empty((3, T_pad), dtype = torch.long, device = "cpu")
    lens = torch.tensor(seg_lens, dtype = torch.long, device = "cpu")
    per_seg = torch.tensor([seg_group, seg_prefix, seg_suffix], dtype = torch.long, device = "cpu")
    labels[:, :T] = torch.repeat_interleave(per_seg, lens, dim = 1, output_size = T)
    labels[0, T:] = _PAD_GROUP
    labels[1, T:] = 0
    labels[2, T:] = _PAD_GROUP
    return labels, T, T_pad, group_meta, sig_parts


def _labels_to_device(labels, T_pad, device):
    """One host->device copy; each returned tensor gets its own storage."""
    moved = labels.to(device)
    return moved[0].clone(), moved[1].to(torch.bool), moved[2].clone()


# Block-mask cache, keyed on (signature, device): the mask depends only on per-token labels and T.


_BLOCK_MASK_CACHE: Dict[Tuple, BlockMask] = {}


def _make_mask_mod(group_of_kv, is_prefix, suffix_of_kv):
    """Same group, and kv is a prefix token or in the same suffix as q; one kv <= q guard covers
    both, since every prefix index precedes every suffix index; matches the SDPA oracle."""

    def mask_mod(b, h, q_idx, kv_idx):
        same_group = group_of_kv[q_idx] == group_of_kv[kv_idx]
        kv_is_prefix = is_prefix[kv_idx]
        causal = kv_idx <= q_idx
        same_suffix = (suffix_of_kv[kv_idx] == suffix_of_kv[q_idx]) & (~is_prefix[q_idx])
        keep = same_group & ((kv_is_prefix & causal) | (same_suffix & causal))
        return keep

    return mask_mod


def get_block_mask(
    seg: PrefixSegInfo,
    device: torch.device,
    compile_mask: bool = True,
) -> BlockMask:
    """Cached BlockMask per (signature, device), shared by the no-grad and grad forwards."""
    key = (seg.signature, str(device))
    bm = _BLOCK_MASK_CACHE.get(key)
    if bm is not None:
        return bm

    # Outside inference mode: the no-grad pass builds it first, and backward must save it.
    # torch 2.7 Inductor emits a ZeroDivisionError when the first compiled mask is one 128 block.
    compile_mask = compile_mask and seg.T_pad > _PAD_MULTIPLE
    builder = _create_block_mask_compiled if compile_mask else create_block_mask
    with torch.inference_mode(False):
        if isinstance(seg, TreeSegInfo):
            mask_mod = _make_tree_mask_mod(seg.reach_end.to(device))
        else:
            mask_mod = _make_mask_mod(
                seg.group_of_kv.to(device), seg.is_prefix.to(device), seg.suffix_of_kv.to(device)
            )
        bm = builder(
            mask_mod,
            B = 1,
            H = None,
            Q_LEN = seg.T_pad,
            KV_LEN = seg.T_pad,
            device = device,
        )
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
    pad = torch.zeros(x.shape[0], x.shape[1], T_pad - T, x.shape[3], device = x.device, dtype = x.dtype)
    return torch.cat([x, pad], dim = 2)


def _run_flex(
    q,
    k,
    v,
    block_mask,
    enable_gqa,
    scale,
    compiled,
    T,
    T_pad,
    one_stage = False,
):
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
            kernel_options = _kernel_options_for_dtype(qp.dtype, one_stage, qp.device),
        )
    else:
        out = flex_attention(
            qp,
            kp,
            vp,
            block_mask = block_mask,
            enable_gqa = enable_gqa,
            scale = scale,
        )
    return out[:, :, :T, :]


class PrefixGrouperCompileError(RuntimeError):
    pass


# unsloth_zoo's suppress_errors turns a failed compile into a silent eager flex fallback that
# materialises T x T scores (torch 2.6 codegen bugs, torch <= 2.8 tiles on 101376 B smem cards).
_COMPILE_CHECKED: Dict[Tuple, bool] = {}
# Setups whose 3-stage tiles overflow shared memory (sm86 / sm89 / sm120 on torch 2.7 / 2.8).
_ONE_STAGE_KEYS: set = set()
_COMPILE_FAILURE: Optional[str] = None
_CHECK_ENV = os.environ.get("UNSLOTH_GRPO_PREFIX_GROUPER_COMPILE_CHECK", "1")
_COMPILE_CHECK_ON = _CHECK_ENV.lower() not in ("0", "false", "no", "off")


def compile_check_failed() -> Optional[str]:
    return _COMPILE_FAILURE


def _root_exception(exc: BaseException) -> BaseException:
    root = exc
    for _ in range(8):  # BackendCompilerFailed wraps the inductor / Triton error
        inner = getattr(root, "inner_exception", None) or root.__cause__
        if inner is None or inner is root:
            break
        root = inner
    return root


def _compile_failure_reason(root: BaseException) -> str:
    lines = [
        line.strip()
        for line in str(root).splitlines()
        if line.strip()
        and not line.lstrip().startswith(("Set TORCHDYNAMO_VERBOSE", "For even more"))
    ]
    return f"{type(root).__name__}: {lines[0][:200] if lines else ''}"


def _keep_saved(tensor):
    return tensor


def _detach_saved(tensor):
    # Returning the input saves an op's output on its own node, a cycle gc cannot break: a probe whose
    # backward fails (out of resource retry, compile failure) would keep its Q / K / V copies alive.
    return tensor.detach()


def _record_compile_failure(key, reason):
    global _COMPILE_FAILURE
    _COMPILE_CHECKED[key] = False
    _COMPILE_FAILURE = reason
    print(
        f"Unsloth: GRPO PrefixGrouper is off: its flex attention kernel does not compile with "
        f"torch {torch.__version__} on this GPU ({reason}); using the packed path.",
        flush = True,
    )


def _small_smem_skip_reason(q) -> Optional[str]:
    """NVIDIA small-smem GPU, 16-bit, head_dim >= 128, no opt-in: the default tiles cannot fit, so
    skip the failing compile (about 3 minutes on T4) and keep the packed path."""
    if _SMALL_SMEM_OPT_IN or torch.version.hip is not None:
        return None
    # User tile overrides (PG_FLEX_BF16_BLOCK_M / _N) go through the real compile check instead.
    if (_BF16_BLOCK_M, _BF16_BLOCK_N) != (128, 64):
        return None
    if q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] < 128:
        return None
    if not _small_smem(q.device):
        return None
    return (
        f"head_dim {q.shape[-1]} tiles do not fit {_SMEM_LIMIT[str(q.device)]} B of shared memory; "
        "UNSLOTH_GRPO_PREFIX_GROUPER_SMALL_SMEM=1 uses smaller tiles, faster only for larger models"
    )


@torch._dynamo.disable
def _checked_first_call(q, k, v, seg, block_mask, enable_gqa, scale, T, T_pad, key):
    """Real call plus fwd + bwd on detached copies, strict, so later calls reuse those graphs."""

    def strict_calls(one_stage):
        nonlocal block_mask
        with _strict_dynamo():
            if block_mask is None:
                block_mask = get_block_mask(seg, q.device, compile_mask = True)
            out = _run_flex(q, k, v, block_mask, enable_gqa, scale, True, T, T_pad, one_stage)
            # Else inside a non-reentrant checkpoint the backward unpacks through it and stops recompute.
            own_hooks = torch.autograd.graph.saved_tensors_hooks(_detach_saved, _keep_saved)
            with torch.inference_mode(False), torch.enable_grad(), own_hooks:
                qc, kc, vc = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
                oc = _run_flex(qc, kc, vc, block_mask, enable_gqa, scale, True, T, T_pad, one_stage)
                torch.autograd.grad(oc.float().sum(), (qc, kc, vc))
        return out

    reason = _small_smem_skip_reason(q)
    if reason is not None:
        _record_compile_failure(key, reason)
        raise PrefixGrouperCompileError(reason)
    try:
        try:
            out = strict_calls(False)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            if "out of resource" not in str(_root_exception(exc)):
                raise
            out = strict_calls(True)
            _ONE_STAGE_KEYS.add(key)
    except torch.cuda.OutOfMemoryError:
        raise  # transient: not cached, callers fall back
    except Exception as exc:
        root = _root_exception(exc)
        if isinstance(root, torch.cuda.OutOfMemoryError):
            raise root from exc
        reason = _compile_failure_reason(root)
        _record_compile_failure(key, reason)
        raise PrefixGrouperCompileError(reason) from None
    _COMPILE_CHECKED[key] = True
    return out


def flex_shared_prefix_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    prefix_seg_info: PrefixSegInfo | TreeSegInfo,
    scale: Optional[float] = None,
    block_mask: Optional[BlockMask] = None,
    compiled: bool = True,
) -> torch.Tensor:
    """Q [1, T, n_heads, D], K / V [1, T, n_kv_heads, D] -> [1, T, n_heads, D]."""
    assert Q.dim() == 4 and Q.shape[0] == 1, f"expected [1,T,H,D], got {tuple(Q.shape)}"
    device = Q.device
    q = Q.transpose(1, 2)  # [1, n_heads, T, D]
    k = K.transpose(1, 2)  # [1, n_kv_heads, T, D]
    v = V.transpose(1, 2)

    n_heads = q.shape[1]
    n_kv = k.shape[1]
    enable_gqa = n_heads != n_kv
    T = q.shape[2]
    T_pad = prefix_seg_info.T_pad
    assert T == prefix_seg_info.T, f"Q length {T} != seg.T {prefix_seg_info.T}"

    out = None
    if compiled:
        if _COMPILE_FAILURE is not None:
            raise PrefixGrouperCompileError(_COMPILE_FAILURE)
        key = (str(device), q.dtype, n_heads, n_kv, q.shape[3], scale)
        # ROCm static flex compiles a new kernel per padded length, each of which can fail and fall back to
        # eager T x T flex: check every length. The dynamic kernel (CUDA, no-grad calls) is checked once.
        if _uses_static_flex(q, k, v):
            key = key + (T_pad,)
        if _COMPILE_CHECK_ON and key not in _COMPILE_CHECKED:
            out = _checked_first_call(
                q, k, v, prefix_seg_info, block_mask, enable_gqa, scale, T, T_pad, key
            )
    if out is None:
        if block_mask is None:
            block_mask = get_block_mask(prefix_seg_info, device, compile_mask = compiled)
        one_stage = compiled and key in _ONE_STAGE_KEYS
        out = _run_flex(q, k, v, block_mask, enable_gqa, scale, compiled, T, T_pad, one_stage)
    return out.transpose(1, 2).contiguous()


__all__ = [
    "PREFIX_GROUPER_MIN_TORCH",
    "TORCH_SUPPORTS_PREFIX_GROUPER",
    "torch_supports_prefix_grouper",
    "PrefixSegInfo",
    "TreeSegInfo",
    "build_tree_seg_info",
    "build_seg_info_multigroup",
    "build_seg_info_from_layout",
    "get_block_mask",
    "clear_block_mask_cache",
    "flex_shared_prefix_attention",
    "PrefixGrouperCompileError",
    "compile_check_failed",
]
