# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched multi-LoRA training on one frozen base: tokens (id -1 = no adapter) are stable-sorted by
adapter once per batch and kernels gather through that order. UNSLOTH_MULTI_LORA_TRITON=0, or no
CUDA/XPU, uses a PyTorch loop; LoRA dropout unsupported."""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    _base_quant_state,
    fast_dequantize,
    matmul_lora,
    torch_amp_custom_bwd,
    torch_amp_custom_fwd,
)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False

__all__ = [
    "MultiLoraBatch",
    "MultiLoraLinear",
    "MultiLoraExperts",
    "MultiLoraExpertParam",
    "multi_lora_delta",
    "multi_lora_linear",
    "inject_multi_lora",
    "inject_multi_lora_experts",
    "set_multi_lora_batch",
    "per_adapter_loss",
    "adapter_token_counts",
    "per_adapter_grad_norm",
    "clip_grad_norm_per_adapter_",
    "drop_absent_adapter_grads_",
    "adapter_state_dict",
    "set_multi_lora_triton",
    "MULTI_LORA_MOE_STATS",
]


# Read once: an environment lookup inside a compiled forward would be a guard on every call.
_TRITON_ENABLED = HAS_TRITON and os.environ.get("UNSLOTH_MULTI_LORA_TRITON", "1") != "0"


def set_multi_lora_triton(enabled: bool) -> None:
    """Turn the Triton kernels on or off (off = the PyTorch fallback); ``UNSLOTH_MULTI_LORA_TRITON=0``."""
    global _TRITON_ENABLED
    _TRITON_ENABLED = bool(enabled) and HAS_TRITON


def _use_triton(X: torch.Tensor) -> bool:
    return _TRITON_ENABLED and X.device.type in ("cuda", "xpu")


@dataclass
class MultiLoraBatch:
    """``sorted_idx[seg_start[u]:seg_start[u + 1]]`` are adapter u's rows; id -1 sorts last, in no segment.

    Shapes depend only on the token count, so a new batch never changes a kernel grid or compiled graph.
    """

    sorted_idx: torch.Tensor
    seg_start: torch.Tensor
    num_adapters: int
    token_ids: Optional[torch.Tensor] = None

    @classmethod
    def from_adapter_ids(
        cls,
        adapter_ids: torch.Tensor,
        num_adapters: int,
        seq_len: Optional[int] = None,
    ) -> "MultiLoraBatch":
        """``adapter_ids``: one id per token (any shape), or one per row with ``seq_len`` tokens each."""
        ids = adapter_ids.reshape(-1).to(torch.int64)
        if seq_len is not None:
            ids = ids.repeat_interleave(seq_len)
        if ids.numel() and (ids.max() >= num_adapters or ids.min() < -1):
            raise ValueError(f"Adapter ids must be in [-1, {num_adapters}).")
        keys = torch.where(ids < 0, torch.full_like(ids, num_adapters), ids)
        sorted_idx = torch.argsort(keys, stable = True).to(torch.int32)
        counts = torch.bincount(keys, minlength = num_adapters + 1)[:num_adapters]
        seg_start = torch.zeros(num_adapters + 1, dtype = torch.int32, device = ids.device)
        seg_start[1:] = torch.cumsum(counts, 0)
        return cls(sorted_idx, seg_start, num_adapters, ids)

    @property
    def num_tokens(self) -> int:
        return self.sorted_idx.numel()


if HAS_TRITON:

    @triton.jit
    def _seg_mm_kernel(
        IN,
        W,
        OUT,
        SCALES,
        IDX,
        SEG,
        N,
        K,
        stride_in,
        stride_out,
        stride_wu,
        stride_wk,
        stride_wn,
        IN_SORTED: tl.constexpr,
        OUT_SORTED: tl.constexpr,
        ADD: tl.constexpr,
        SCALE: tl.constexpr,
        IEEE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # OUT[rows] (+)= scale_u * IN[rows] @ W_u, W_u read as (K, N) through strides. A row is either the
        # token's position in the sorted order (H, dH live there) or its original row (X, Y, dY, dX).
        pid = tl.program_id(0)
        u = tl.program_id(1)
        n_tiles = tl.cdiv(N, BLOCK_N)
        pid_m = pid // n_tiles
        pid_n = pid % n_tiles
        start = tl.load(SEG + u)
        end = tl.load(SEG + u + 1)
        m0 = start + pid_m * BLOCK_M
        if m0 >= end:
            return
        offs_p = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_p < end
        rows = tl.load(IDX + offs_p, mask = mask_m, other = 0).to(tl.int64)
        offs_p = offs_p.to(tl.int64)
        if IN_SORTED:
            in_rows = offs_p
        else:
            in_rows = rows
        if OUT_SORTED:
            out_rows = offs_p
        else:
            out_rows = rows
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        offs_k = tl.arange(0, BLOCK_K)
        w_base = W + u.to(tl.int64) * stride_wu
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + offs_k
            mask_k = kk < K
            a = tl.load(
                IN + in_rows[:, None] * stride_in + kk[None, :],
                mask = mask_m[:, None] & mask_k[None, :],
                other = 0.0,
            )
            b = tl.load(
                w_base + kk[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask = mask_k[:, None] & mask_n[None, :],
                other = 0.0,
            )
            if IEEE:
                acc += tl.dot(a, b, input_precision = "ieee")
            else:
                acc += tl.dot(a, b)
        if SCALE:
            acc *= tl.load(SCALES + u).to(tl.float32)
        ptrs = OUT + out_rows[:, None] * stride_out + offs_n[None, :]
        mask = mask_m[:, None] & mask_n[None, :]
        if ADD:
            acc += tl.load(ptrs, mask = mask, other = 0.0).to(tl.float32)
        tl.store(ptrs, acc.to(OUT.dtype.element_ty), mask = mask)

    @triton.jit
    def _seg_wgrad_kernel(
        P,
        Q,
        OUT,
        SCALES,
        IDX,
        SEG,
        I,
        J,
        stride_p,
        stride_q,
        stride_ou,
        stride_oi,
        stride_oj,
        P_SORTED: tl.constexpr,
        Q_SORTED: tl.constexpr,
        SCALE: tl.constexpr,
        IEEE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_I: tl.constexpr,
        BLOCK_J: tl.constexpr,
    ):
        # OUT_u = scale_u * P[seg_u].T @ Q[seg_u]; an adapter with no tokens stores zeros.
        pid = tl.program_id(0)
        u = tl.program_id(1)
        j_tiles = tl.cdiv(J, BLOCK_J)
        pid_i = pid // j_tiles
        pid_j = pid % j_tiles
        start = tl.load(SEG + u)
        end = tl.load(SEG + u + 1)
        offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        offs_j = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        mask_i = offs_i < I
        mask_j = offs_j < J
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype = tl.float32)
        for t0 in range(start, end, BLOCK_T):
            offs_p = t0 + tl.arange(0, BLOCK_T)
            mask_t = offs_p < end
            rows = tl.load(IDX + offs_p, mask = mask_t, other = 0).to(tl.int64)
            offs_p = offs_p.to(tl.int64)
            if P_SORTED:
                p_rows = offs_p
            else:
                p_rows = rows
            if Q_SORTED:
                q_rows = offs_p
            else:
                q_rows = rows
            p = tl.load(
                P + p_rows[None, :] * stride_p + offs_i[:, None],
                mask = mask_i[:, None] & mask_t[None, :],
                other = 0.0,
            )
            q = tl.load(
                Q + q_rows[:, None] * stride_q + offs_j[None, :],
                mask = mask_t[:, None] & mask_j[None, :],
                other = 0.0,
            )
            if IEEE:
                acc += tl.dot(p, q, input_precision = "ieee")
            else:
                acc += tl.dot(p, q)
        if SCALE:
            acc *= tl.load(SCALES + u).to(tl.float32)
        ptrs = (
            OUT
            + u.to(tl.int64) * stride_ou
            + offs_i[:, None] * stride_oi
            + offs_j[None, :] * stride_oj
        )
        tl.store(ptrs, acc.to(OUT.dtype.element_ty), mask = mask_i[:, None] & mask_j[None, :])


def _block(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, triton.next_power_of_2(n)))


def _launch_seg_mm(
    inp: torch.Tensor,
    W: torch.Tensor,
    out: torch.Tensor,
    scales: Optional[torch.Tensor],
    idx: torch.Tensor,
    seg: torch.Tensor,
    N: int,
    stride_wu: int,
    stride_wk: int,
    stride_wn: int,
    in_sorted: bool,
    out_sorted: bool,
    add: bool,
) -> None:
    M, K = inp.shape
    BLOCK_M = 64
    BLOCK_N = _block(N, 16, 128)
    BLOCK_K = _block(K, 16, 64 if BLOCK_N > 32 else 128)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), seg.shape[0] - 1)
    _wrap(_seg_mm_kernel)[grid](
        inp,
        W,
        out,
        scales if scales is not None else inp,
        idx,
        seg,
        N,
        K,
        inp.stride(0),
        out.stride(0),
        stride_wu,
        stride_wk,
        stride_wn,
        IN_SORTED = in_sorted,
        OUT_SORTED = out_sorted,
        ADD = add,
        SCALE = scales is not None,
        IEEE = inp.dtype == torch.float32,
        BLOCK_M = BLOCK_M,
        BLOCK_N = BLOCK_N,
        BLOCK_K = BLOCK_K,
        num_warps = 4,
        enable_fp_fusion = True,
    )


def _launch_seg_wgrad(
    P: torch.Tensor,
    Q: torch.Tensor,
    out: torch.Tensor,
    scales: Optional[torch.Tensor],
    idx: torch.Tensor,
    seg: torch.Tensor,
    p_sorted: bool,
    q_sorted: bool,
) -> None:
    _, I, J = out.shape
    BLOCK_I = _block(I, 16, 64)
    BLOCK_J = _block(J, 16, 64)
    grid = (triton.cdiv(I, BLOCK_I) * triton.cdiv(J, BLOCK_J), seg.shape[0] - 1)
    _wrap(_seg_wgrad_kernel)[grid](
        P,
        Q,
        out,
        scales if scales is not None else P,
        idx,
        seg,
        I,
        J,
        P.stride(0),
        Q.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        P_SORTED = p_sorted,
        Q_SORTED = q_sorted,
        SCALE = scales is not None,
        IEEE = P.dtype == torch.float32,
        BLOCK_T = 64,
        BLOCK_I = BLOCK_I,
        BLOCK_J = BLOCK_J,
        num_warps = 4,
        enable_fp_fusion = True,
    )


# triton_op lets Dynamo trace the launches without a graph break; wrap_triton rejects interpreter kernels.
_INTERPRETED = os.environ.get("TRITON_INTERPRET", "0") == "1"
if HAS_TRITON and hasattr(torch.library, "triton_op") and not _INTERPRETED:
    _wrap = torch.library.wrap_triton
    _seg_mm_op = torch.library.triton_op("unsloth::multi_lora_seg_mm", mutates_args = {"out"})(
        _launch_seg_mm
    )
    _seg_wgrad_op = torch.library.triton_op("unsloth::multi_lora_seg_wgrad", mutates_args = {"out"})(
        _launch_seg_wgrad
    )
else:
    _wrap = lambda kernel: kernel  # noqa: E731
    _seg_mm_op, _seg_wgrad_op = _launch_seg_mm, _launch_seg_wgrad


def _eager_guard(device: torch.device):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    return _device_guard(device)


def _seg_mm(
    inp: torch.Tensor,
    W: torch.Tensor,
    w_strides: tuple[int, int, int],
    N: int,
    out: torch.Tensor,
    batch: MultiLoraBatch,
    scales: Optional[torch.Tensor],
    in_sorted: bool,
    out_sorted: bool,
    add: bool,
) -> torch.Tensor:
    with _eager_guard(inp.device):
        _seg_mm_op(
            inp,
            W,
            out,
            scales,
            batch.sorted_idx,
            batch.seg_start,
            N,
            *w_strides,
            in_sorted,
            out_sorted,
            add,
        )
    return out


def _seg_wgrad(
    P: torch.Tensor,
    Q: torch.Tensor,
    out: torch.Tensor,
    batch: MultiLoraBatch,
    scales: Optional[torch.Tensor],
    p_sorted: bool,
    q_sorted: bool,
) -> torch.Tensor:
    with _eager_guard(P.device):
        _seg_wgrad_op(P, Q, out, scales, batch.sorted_idx, batch.seg_start, p_sorted, q_sorted)
    return out


def _device_guard(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.device(device)
    if device.type == "xpu":
        return torch.xpu.device(device)
    return contextlib.nullcontext()


def _shrink(
    X,
    A,
    batch,
    H = None,
):
    if H is None:
        H = torch.empty(X.shape[0], A.shape[1], dtype = X.dtype, device = X.device)
    return _seg_mm(
        X,
        A,
        (A.stride(0), A.stride(2), A.stride(1)),
        A.shape[1],
        H,
        batch,
        None,
        False,
        True,
        False,
    )


def _expand_add(H, B, scales, Y, batch):
    return _seg_mm(
        H,
        B,
        (B.stride(0), B.stride(2), B.stride(1)),
        B.shape[1],
        Y,
        batch,
        scales,
        True,
        False,
        True,
    )


class MultiLoRA_W(torch.autograd.Function):
    """Y = X @ W.T + per-token-adapter LoRA, fused forward and backward over sorted adapter segments."""

    @staticmethod
    @torch_amp_custom_fwd
    def forward(ctx, X, W, W_quant, A, B, scales, sorted_idx, seg_start):
        batch = MultiLoraBatch(sorted_idx, seg_start, A.shape[0])
        shape = X.shape
        X = X.reshape(-1, shape[-1])
        if not X.is_contiguous():
            X = X.contiguous()
        dtype = X.dtype
        A_dtype, B_dtype = A.dtype, B.dtype
        A, B = A.to(dtype), B.to(dtype)
        Y = matmul_lora(X, W, W_quant, None, None, None)
        if not Y.is_contiguous():
            Y = Y.contiguous()
        H = _shrink(X, A, batch)
        _expand_add(H, B, scales, Y, batch)
        ctx.custom_saved_tensors = (W, W_quant)
        ctx.save_for_backward(X, H, A, B, scales, sorted_idx, seg_start)
        ctx.shape = shape
        ctx.ab_dtypes = (A_dtype, B_dtype)
        return Y.view(*shape[:-1], Y.shape[-1])

    @staticmethod
    @torch_amp_custom_bwd
    def backward(ctx, dY):
        W, W_quant = ctx.custom_saved_tensors
        X, H, A, B, scales, sorted_idx, seg_start = ctx.saved_tensors
        batch = MultiLoraBatch(sorted_idx, seg_start, A.shape[0])
        dY = dY.reshape(-1, dY.shape[-1])
        if not dY.is_contiguous():
            dY = dY.contiguous()
        dtype = X.dtype
        if dY.dtype != dtype:
            dY = dY.to(dtype)
        U, R, _ = A.shape
        dH = torch.empty(X.shape[0], R, dtype = dtype, device = X.device)
        _seg_mm(
            dY, B, (B.stride(0), B.stride(1), B.stride(2)), R, dH, batch, scales, False, True, False
        )
        dA = torch.empty(A.shape, dtype = torch.float32, device = X.device)
        dB = torch.empty(B.shape, dtype = torch.float32, device = X.device)
        _seg_wgrad(dH, X, dA, batch, None, True, False)
        _seg_wgrad(dY, H, dB, batch, scales, False, True)
        W = fast_dequantize(W.t(), W_quant, dtype = dtype).to(dtype)
        dX = dY @ W.t()
        del W
        _seg_mm(
            dH,
            A,
            (A.stride(0), A.stride(1), A.stride(2)),
            A.shape[2],
            dX,
            batch,
            None,
            True,
            False,
            True,
        )
        A_dtype, B_dtype = ctx.ab_dtypes
        return dX.view(ctx.shape), None, None, dA.to(A_dtype), dB.to(B_dtype), None, None, None


def multi_lora_delta(X, A, B, scales, batch: MultiLoraBatch):
    """Reference LoRA delta in plain PyTorch: every adapter on every token, masked; no host sync."""
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    U = A.shape[0]
    ids = batch.token_ids
    if ids is None:
        ids = torch.full((X.shape[0],), -1, dtype = torch.int64, device = X.device)
        seg = batch.seg_start.long()
        pos = torch.arange(X.shape[0], device = X.device)
        ids = ids.scatter(0, batch.sorted_idx.long(), torch.searchsorted(seg[1:], pos, right = True))
        ids = torch.where(ids >= U, torch.full_like(ids, -1), ids)
    # torch.where, not a 0/1 multiply (0 * inf = nan); X is masked too so dH @ A_u cannot leak into dX.
    scales = scales.to(X.dtype)
    out = X.new_zeros(X.shape[0], B.shape[1])
    for u in range(U):
        own = (ids == u).unsqueeze(1)
        H = torch.where(own, torch.where(own, X, 0) @ A[u].to(X.dtype).t(), 0)
        out = torch.where(own, (H @ B[u].to(X.dtype).t()) * scales[u], out)
    return out.view(*shape[:-1], out.shape[-1])


def multi_lora_linear(X, W, W_quant, bias, A, B, scales, batch: Optional[MultiLoraBatch]):
    """``F.linear(X, W, bias)`` plus each token's adapter delta. ``W`` may be a 4-bit / quantized weight."""
    if batch is None or X.shape[:-1].numel() != batch.num_tokens:
        raise ValueError(
            "Call set_multi_lora_batch(model, ...) with this batch's adapter ids first."
        )
    if _use_triton(X):
        Y = MultiLoRA_W.apply(X, W, W_quant, A, B, scales, batch.sorted_idx, batch.seg_start)
    else:
        Wd = fast_dequantize(W, W_quant, dtype = X.dtype) if W_quant is not None else W.to(X.dtype)
        Y = F.linear(X, Wd) + multi_lora_delta(X, A, B, scales, batch)
    if bias is not None:
        Y = Y + bias.to(Y.dtype)
    return Y


class MultiLoraLinear(nn.Module):
    """Frozen base linear with ``num_adapters`` LoRA adapters, separate (A, B) params each (PEFT init).

    Separate params keep optimizer state per adapter, so an absent adapter's AdamW state is untouched.
    """

    def __init__(
        self,
        base: nn.Module,
        num_adapters: int,
        r: int | Iterable[int] = 16,
        lora_alpha: float | Iterable[float] = 16,
        use_rslora: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.base_layer = base
        for p in base.parameters():
            p.requires_grad_(False)
        ranks = [int(r)] * num_adapters if isinstance(r, int) else [int(x) for x in r]
        alphas = (
            [float(lora_alpha)] * num_adapters
            if isinstance(lora_alpha, (int, float))
            else [float(x) for x in lora_alpha]
        )
        if len(ranks) != num_adapters or len(alphas) != num_adapters:
            raise ValueError("Need one rank and one alpha per adapter.")
        self.ranks = ranks
        out_features, in_features = base.out_features, base.in_features
        device = base.weight.device
        kw = dict(dtype = dtype, device = device)
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros(k, in_features, **kw)) for k in ranks]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(out_features, k, **kw)) for k in ranks]
        )
        scales = [a / (math.sqrt(k) if use_rslora else k) for a, k in zip(alphas, ranks)]
        self.register_buffer("scales", torch.tensor(scales, dtype = torch.float32, device = device))
        self.reset_parameters()
        # Not a buffer: DDP broadcasts buffers from rank 0, giving every rank rank 0's routing.
        self.batch: Optional[MultiLoraBatch] = None
        self.seen: Optional[torch.Tensor] = None
        self.trained: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset_parameters(self):
        for A, B in zip(self.lora_A, self.lora_B):
            nn.init.kaiming_uniform_(A, a = math.sqrt(5))
            B.zero_()

    @property
    def num_adapters(self) -> int:
        return len(self.lora_A)

    def stacked(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(U, R, in) and (U, out, R), zero padded to the largest rank."""
        R = max(self.ranks)
        A = torch.stack([F.pad(a, (0, 0, 0, R - a.shape[0])) for a in self.lora_A])
        B = torch.stack([F.pad(b, (0, R - b.shape[1])) for b in self.lora_B])
        return A, B

    def forward(self, x):
        W = self.base_layer.weight
        A, B = self.stacked()
        return multi_lora_linear(
            x,
            W,
            _base_quant_state(self.base_layer, W),
            getattr(self.base_layer, "bias", None),
            A,
            B,
            self.scales,
            self.batch,
        )


def inject_multi_lora(
    model: nn.Module,
    target_modules: str | Iterable[str],
    num_adapters: int,
    r: int | Iterable[int] = 16,
    lora_alpha: float | Iterable[float] = 16,
    use_rslora: bool = False,
) -> list[str]:
    """Freeze ``model`` and wrap every linear layer named, or dotted-suffixed by, a target, as PEFT does."""
    targets = (target_modules,) if isinstance(target_modules, str) else tuple(target_modules)
    found = [
        (name, module)
        for name, module in model.named_modules()
        if name
        and any(name == t or name.endswith("." + t) for t in targets)
        and hasattr(module, "in_features")
        and hasattr(module, "weight")
    ]
    if not found:
        raise ValueError(f"No linear layers named {targets} found.")
    _freeze_all_but_adapters(model)
    for name, module in found:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, MultiLoraLinear(module, num_adapters, r, lora_alpha, use_rslora))
    return [name for name, _ in found]


class MultiLoraExpertParam(nn.Module):
    """LoRA on one fused expert stack: per adapter, A (E, r, in) and B (E, out, r), each expert as PEFT inits it."""

    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        num_adapters: int,
        r: int | Iterable[int] = 16,
        lora_alpha: float | Iterable[float] = 16,
        use_rslora: bool = False,
        device = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        ranks = [int(r)] * num_adapters if isinstance(r, int) else [int(x) for x in r]
        alphas = (
            [float(lora_alpha)] * num_adapters
            if isinstance(lora_alpha, (int, float))
            else [float(x) for x in lora_alpha]
        )
        if len(ranks) != num_adapters or len(alphas) != num_adapters:
            raise ValueError("Need one rank and one alpha per adapter.")
        self.ranks = ranks
        self.num_experts, self.in_features, self.out_features = (
            num_experts,
            in_features,
            out_features,
        )
        kw = dict(dtype = dtype, device = device)
        self.lora_A = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_experts, k, in_features, **kw)) for k in ranks]
        )
        self.lora_B = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_experts, out_features, k, **kw)) for k in ranks]
        )
        self.scales = [a / (math.sqrt(k) if use_rslora else k) for a, k in zip(alphas, ranks)]
        self.reset_parameters()
        self.batch: Optional[MultiLoraBatch] = None
        self.seen: Optional[torch.Tensor] = None
        self.trained: Optional[torch.Tensor] = None

    @torch.no_grad()
    def reset_parameters(self):
        # PEFT inits A as one (E*r, in) matrix, so fan_in is `in` for every expert slice.
        for A, B in zip(self.lora_A, self.lora_B):
            for e in range(A.shape[0]):
                nn.init.kaiming_uniform_(A[e], a = math.sqrt(5))
            B.zero_()

    @property
    def num_adapters(self) -> int:
        return len(self.lora_A)

    def grouped_mm_factors(
        self,
        adapter: int,
        reversed_io: bool = False,
    ) -> tuple:
        """Zoo's stash (first (E, in, r), second (E, r, out), scaling, E); ``reversed_io`` swaps the factors' roles
        as Zoo's _reversed_lora_weights_for_grouped_mm does."""
        A, B = self.lora_A[adapter], self.lora_B[adapter]
        if reversed_io:
            return B.contiguous(), A.contiguous(), self.scales[adapter], self.num_experts
        return (
            A.transpose(1, 2).contiguous(),
            B.transpose(1, 2).contiguous(),
            self.scales[adapter],
            self.num_experts,
        )

    def grouped_stacks(self, reversed_io: bool, dtype: torch.dtype) -> tuple:
        """Every adapter's factors as (E*(U+1), in, R) and (E*(U+1), R, out), group e*(U+1) + u, R the largest rank
        rounded up to 8 (zero padding is exact), scale folded into the second; group e*(U+1) + U is zero."""
        R = -(-max(self.ranks) // 8) * 8
        firsts, seconds = [], []
        for u in range(self.num_adapters):
            first, second, scale, _ = self.grouped_mm_factors(u, reversed_io)
            r = first.shape[-1]
            firsts.append(F.pad(first.to(dtype), (0, R - r)))
            seconds.append(F.pad(second.to(dtype) * scale, (0, 0, 0, R - r)))
        firsts.append(torch.zeros_like(firsts[0]))
        seconds.append(torch.zeros_like(seconds[0]))
        return torch.stack(firsts, 1).flatten(0, 1), torch.stack(seconds, 1).flatten(0, 1)

    def folded(self, W: torch.Tensor, adapter: int, transposed: bool) -> torch.Tensor:
        """W + delta in W's dtype, as PEFT's ParamWrapper folds it (one baddbmm, autocast off)."""
        A, B = self.lora_A[adapter].to(W.dtype), self.lora_B[adapter].to(W.dtype)
        lhs, rhs = (A.transpose(1, 2), B.transpose(1, 2)) if transposed else (B, A)
        with torch.autocast(device_type = W.device.type, enabled = False):
            return torch.baddbmm(W, lhs, rhs, alpha = self.scales[adapter])

    def peft_tensors(self, adapter: int, b_layout: str) -> tuple[torch.Tensor, torch.Tensor]:
        """PEFT ParamWrapper's packing: lora_A (E*r, in) expert-major, lora_B (out, E*r) in ``b_layout``."""
        A, B = _full(self.lora_A[adapter]), _full(self.lora_B[adapter])
        E, out, k = B.shape
        flat_A = A.reshape(E * k, A.shape[-1])
        if b_layout == "rank_major":
            flat_B = B.permute(1, 2, 0).reshape(out, k * E)
        else:
            flat_B = B.permute(1, 0, 2).reshape(out, E * k)
        return flat_A.detach().clone(), flat_B.detach().clone()


@functools.lru_cache(maxsize = None)
def _peft_expert_in_axis(is_transposed: bool) -> int:
    """Which axis of an (E, d1, d2) stack the installed PEFT's ParamWrapper calls in_features (it moved in 0.19)."""
    import peft

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(2, 3, 5))
            self.is_transposed = is_transposed

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()

    model = peft.get_peft_model(Block(), peft.LoraConfig(r = 1, target_parameters = ["experts.w"]))
    A = next(p for n, p in model.named_parameters() if "lora_A" in n)
    return 1 if A.shape[-1] == 3 else 2


def _expert_stack_shape(param) -> tuple[int, int, int]:
    shape = getattr(param, "_original_shape", None) or tuple(param.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a fused (experts, d1, d2) parameter, got shape {tuple(shape)}.")
    return tuple(int(x) for x in shape)


def _stash_name(parameter_name: str) -> str:
    return f"_unsloth_lora_{parameter_name}"


def _zoo_stash_read_store(experts: nn.Module):
    try:
        from unsloth_zoo.temporary_patches.moe_utils import _MOE_LORA_STASH_READ_ATTR
    except Exception:  # noqa: BLE001 - without Unsloth Zoo's MoE forwards nothing reads a stash
        return None
    store = experts.__dict__.get(_MOE_LORA_STASH_READ_ATTR)
    if store is None:
        store = {}
        setattr(experts, _MOE_LORA_STASH_READ_ATTR, store)
    return store


def _adapter_groups(batch: MultiLoraBatch) -> list:
    """(adapter or None, token rows) per adapter present; one host sync per batch, shared by every MoE layer."""
    groups = batch.__dict__.get("_expert_groups")
    if groups is None:
        seg = batch.seg_start.tolist()
        groups = [
            (u, batch.sorted_idx[seg[u] : seg[u + 1]].long())
            for u in range(batch.num_adapters)
            if seg[u + 1] > seg[u]
        ]
        if seg[-1] < batch.num_tokens:
            groups.append((None, batch.sorted_idx[seg[-1] :].long()))
        batch.__dict__["_expert_groups"] = groups
    return groups


def _token_adapters(batch: MultiLoraBatch) -> torch.Tensor:
    """Each token's adapter id (-1 = none)."""
    ids = batch.token_ids
    if ids is None:
        ids = batch.__dict__.get("_token_adapters")
    if ids is None:
        n, U = batch.num_tokens, batch.num_adapters
        seg = batch.seg_start.long()
        pos = torch.arange(n, device = seg.device)
        ids = torch.full((n,), -1, dtype = torch.int64, device = seg.device)
        ids = ids.scatter(0, batch.sorted_idx.long(), torch.searchsorted(seg[1:], pos, right = True))
        ids = torch.where(ids >= U, torch.full_like(ids, -1), ids)
        batch.__dict__["_token_adapters"] = ids
    return ids


MULTI_LORA_MOE_STATS = {"grouped_calls": 0, "per_adapter_calls": 0, "grouped_fallbacks": 0}
_HALF = (torch.float16, torch.bfloat16)
_FLOAT = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
_ZOO_GROUPED = (
    "_base_grouped_mm",
    "_get_base_weight",
    "preprocess_weight",
    "_moe_recompute_enabled",
    "_selective_dequant_plan",
    "_grouped_mm_with_backward_fix",
    "_grouped_mm_supported_on",
    "weighted_unpermute",
    "combine_permuted_moe_outputs",
)


def _zoo_moe_utils():
    """Unsloth Zoo's MoE module when it has the grouped GEMMs its single-job forward runs on, else None."""
    try:
        from unsloth_zoo.temporary_patches import moe_utils
    except Exception:  # noqa: BLE001
        return None
    return moe_utils if all(hasattr(moe_utils, n) for n in _ZOO_GROUPED) else None


def _grouped_supported(device: torch.device, dtype: torch.dtype) -> bool:
    """Zoo's grouped GEMM runs as a kernel here: its Triton grouped GEMM (CUDA, half) or torch's grouped GEMM.
    Elsewhere it loops over groups, which at E * (U + 1) groups is slower than one adapter at a time."""
    moe_utils = _zoo_moe_utils()
    if moe_utils is None:
        return False
    if device.type != "cuda":
        return True
    try:
        if dtype in _HALF:
            from unsloth_zoo.temporary_patches import moe_grouped_fp16
            if moe_grouped_fp16.triton_grouped_available(device):
                return True
        return bool(moe_utils._grouped_mm_supported_on(device))
    except Exception:  # noqa: BLE001
        return False


def _is_quantized(param) -> bool:
    return getattr(param, "quant_state", None) is not None or param.dtype not in _FLOAT


def _grouped_disabled() -> bool:
    return os.environ.get("UNSLOTH_MULTI_LORA_MOE_GROUPED", "1") == "0"


_WARNED: set = set()


def _warn_once(message: str, level: int = logging.WARNING) -> None:
    if message in _WARNED:
        return
    _WARNED.add(message)
    try:
        from unsloth_zoo.log import logger
    except Exception:  # noqa: BLE001
        logger = logging.getLogger(__name__)
    logger.log(level, message)


class MultiLoraExperts(nn.Module):
    """Frozen fused experts. Mode "grouped": one pass for every adapter, rows sorted by (expert, adapter), the frozen
    stacks in E-group GEMMs and each adapter's LoRA in E*(U+1)-group GEMMs over the same rows. Otherwise each
    adapter's tokens run the experts' own forward with only its factors, as a lone PEFT job does: through Zoo's
    stash when its MoE forward reads one ("stash"), else with the delta folded in ("fold")."""

    def __init__(
        self,
        base: nn.Module,
        parameter_names: Iterable[str],
        num_adapters: int,
        r: int | Iterable[int] = 16,
        lora_alpha: float | Iterable[float] = 16,
        use_rslora: bool = False,
    ):
        super().__init__()
        self.base_layer = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.parameter_names = tuple(parameter_names)
        is_transposed = bool(getattr(base, "is_transposed", False))
        # Shapes and fold follow the installed PEFT; the stash follows Zoo's extractor (stored input axis).
        peft_in_axis = _peft_expert_in_axis(is_transposed)
        zoo_in_axis = (
            1 if is_transposed or getattr(base, "_unsloth_grouped_mm_format", False) else 2
        )
        self.transposed = peft_in_axis == 1
        self.reversed_io = peft_in_axis != zoo_in_axis
        self.adapters = nn.ModuleDict()
        for name in self.parameter_names:
            param = getattr(base, name)
            E, d1, d2 = _expert_stack_shape(param)
            out_features, in_features = (d2, d1) if self.transposed else (d1, d2)
            device = param.device if param.device.type != "meta" else None
            self.adapters[name] = MultiLoraExpertParam(
                E, in_features, out_features, num_adapters, r, lora_alpha, use_rslora, device = device
            )
        # How a lone PEFT run of these experts applies LoRA ("stash" or "fold"), and how this module runs
        # ("grouped", or the same as peft_mode); both decided on the first call with an adapter.
        self.peft_mode: Optional[str] = None
        self.mode: Optional[str] = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base_layer"], name)

    @property
    def num_adapters(self) -> int:
        return next(iter(self.adapters.values())).num_adapters

    def _run_stash(self, adapter: int, args) -> Optional[torch.Tensor]:
        base = self.base_layer
        store = _zoo_stash_read_store(base)
        if store is None:
            return None
        for name in self.parameter_names:
            store.pop(name, None)
            setattr(
                base,
                _stash_name(name),
                self.adapters[name].grouped_mm_factors(adapter, self.reversed_io),
            )
        try:
            out = base(*args)
        finally:
            for name in self.parameter_names:
                if _stash_name(name) in base.__dict__:
                    delattr(base, _stash_name(name))
        if all(store.get(name) for name in self.parameter_names):
            return out
        return None

    def _run_fold(self, adapter: int, args) -> torch.Tensor:
        base = self.base_layer
        weights = {}
        for name in self.parameter_names:
            W = getattr(base, name)
            if W.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise RuntimeError(
                    f"These experts' forward ignores the LoRA stash and {name} is quantized, so its delta cannot be folded."
                )
            weights[name] = self.adapters[name].folded(W, adapter, self.transposed)
        return torch.func.functional_call(base, weights, args)

    def _run(self, adapter: Optional[int], args) -> torch.Tensor:
        if adapter is None:
            return self.base_layer(*args)
        if self.peft_mode == "fold":
            return self._run_fold(adapter, args)
        out = self._run_stash(adapter, args)
        if out is None:
            raise RuntimeError("The experts' forward stopped reading the LoRA stash mid-run.")
        return out

    def _per_adapter(self, hidden_states: torch.Tensor, args, groups) -> torch.Tensor:
        if len(groups) == 1:
            return self._run(groups[0][0], (hidden_states, *args))
        out = None
        for u, rows in groups:
            y = self._run(u, (hidden_states[rows], *(a[rows] for a in args)))
            if out is None:
                out = y.new_zeros((hidden_states.shape[0], *y.shape[1:]))
            out = out.index_copy(0, rows, y)
        return out

    def _choose_mode(self, hidden_states: torch.Tensor, args, groups) -> None:
        adapter, rows = next(((u, rows) for u, rows in groups if u is not None), (None, None))
        if adapter is None:
            return
        # Probes run without autograd, so a checkpointed forward and its recompute run the same ops.
        with torch.no_grad():
            probe = self._run_stash(
                adapter, (hidden_states[rows[:1]], *(a[rows[:1]] for a in args))
            )
        self.peft_mode = "fold" if probe is None else "stash"
        self.mode = self.peft_mode
        if _grouped_disabled():
            return
        try:
            reason = self._grouped_unsupported(hidden_states, args) or self._grouped_parity(
                hidden_states, args, groups
            )
        except Exception as error:  # noqa: BLE001 - anything unexpected keeps the per-adapter path
            reason = f"{type(error).__name__}: {error}"
        if reason is None:
            self.mode = "grouped"
            _warn_once(
                f"Unsloth: batched MoE LoRA on {type(self.base_layer).__name__} runs every adapter in one grouped pass.",
                logging.INFO,
            )
            return
        if not torch.compiler.is_compiling():
            MULTI_LORA_MOE_STATS["grouped_fallbacks"] += 1
        _warn_once(
            f"Unsloth: batched MoE LoRA on {type(self.base_layer).__name__} runs one adapter at a time ({reason})."
        )

    def _in_out(self, name: str) -> tuple[int, int, int]:
        """(E, in, out) of a fused stack, from its stored (E, d1, d2) and layout."""
        base = self.base_layer
        E, d1, d2 = _expert_stack_shape(getattr(base, name))
        input_major = getattr(base, "is_transposed", False) or getattr(
            base, "_unsloth_grouped_mm_format", False
        )
        return (E, d1, d2) if input_major else (E, d2, d1)

    def _compute_dtype(self, hidden_states: torch.Tensor) -> torch.dtype:
        """As Unsloth Zoo's MoE forward: a half activation runs as it is, half stacks set the dtype, quantized ones
        take Zoo's ``moe_compute_dtype``; float32 stacks stay float32 even under autocast (grouped GEMM is not
        autocast-cast)."""
        if hidden_states.dtype in _HALF:
            return hidden_states.dtype
        param = self.base_layer.gate_up_proj
        if param.dtype in _HALF:
            return param.dtype
        if _is_quantized(param):
            from unsloth_zoo.temporary_patches.moe_utils import moe_compute_dtype
            return moe_compute_dtype(hidden_states)
        return hidden_states.dtype

    def _grouped_unsupported(self, hidden_states: torch.Tensor, args) -> Optional[str]:
        """Why the grouped path cannot run these experts, or None."""
        base = self.base_layer
        if not set(self.parameter_names) <= {"gate_up_proj", "down_proj"}:
            return f"LoRA on {self.parameter_names}"
        if "GptOss" in type(base).__name__ or getattr(base, "is_concatenated", True) is False:
            return "interleaved gate_up_proj"
        if base.__dict__.get("_unsloth_own_apply_gate") or getattr(
            type(base), "_unsloth_own_apply_gate", False
        ):
            return "a custom gate"
        if not callable(getattr(base, "act_fn", None)):
            return "no act_fn"
        names = [n for n, _ in base.named_parameters()] + [n for n, _ in base.named_buffers()]
        if any("bias" in n for n in names):
            return "expert biases"
        params = [getattr(base, n, None) for n in ("gate_up_proj", "down_proj")]
        if not all(isinstance(p, torch.Tensor) for p in params):
            return "no fused gate_up_proj / down_proj"
        if len(args) != 2 or hidden_states.dim() != 2:
            return "an experts forward not called as (hidden_states, top_k_index, top_k_weights)"
        top_k_index, top_k_weights = args
        if (
            top_k_index.dtype.is_floating_point
            or not top_k_weights.dtype.is_floating_point
            or top_k_index.shape != top_k_weights.shape
            or top_k_index.dim() != 2
        ):
            return "an experts forward not called as (hidden_states, top_k_index, top_k_weights)"
        moe_utils = _zoo_moe_utils()
        if moe_utils is None:
            return "an Unsloth Zoo without its grouped MoE GEMMs"
        layout = getattr(moe_utils, "_mxfp4_expert_layout", None)
        model_type = getattr(base, "_unsloth_model_type", None)
        H = hidden_states.shape[-1]
        if layout is not None and any(
            layout(getattr(base, n), proj, H, model_type, base) is not None
            for n, proj in (("gate_up_proj", "gate_up"), ("down_proj", "down"))
        ):
            return "MXFP4 experts"
        dtype = self._compute_dtype(hidden_states)
        (E, gate_in, gate_out), (E_down, down_in, down_out) = (
            self._in_out("gate_up_proj"),
            self._in_out("down_proj"),
        )
        if E != E_down or gate_in != H or down_out != H or gate_out != 2 * down_in:
            return "unexpected expert shapes"
        # Grouped GEMM needs 16-byte rows (the LoRA rank is padded to 8).
        step = 16 // torch.empty((), dtype = dtype).element_size()
        if any(d % step for d in (H, down_in)):
            return "expert dims not a multiple of 16 bytes"
        if not _grouped_supported(hidden_states.device, dtype):
            return f"no grouped GEMM kernel in {dtype} on {hidden_states.device.type}"
        return None

    def _grouped_parity(self, hidden_states: torch.Tensor, args, groups) -> Optional[str]:
        """A few rows per adapter through both paths; None when they agree."""
        ids = _token_adapters(next(iter(self.adapters.values())).batch)
        picks = [rows[:4] for _, rows in groups]
        sel = torch.cat(picks)
        with torch.no_grad():
            got = self._grouped(hidden_states[sel], *(a[sel] for a in args), ids[sel]).float()
            ref = torch.cat(
                [
                    self._run(u, (hidden_states[r], *(a[r] for a in args))).float()
                    for (u, _), r in zip(groups, picks)
                ]
            )
        err = (got - ref).abs().max().item()
        # Under autocast the per-adapter path may itself run in half precision (Zoo's CPU fallback does).
        half = self._compute_dtype(hidden_states) in _HALF or torch.is_autocast_enabled(
            hidden_states.device.type
        )
        tol = 5e-2 if half else 1e-3
        if err <= tol * max(ref.abs().max().item(), 1e-6):
            return None
        return f"the grouped probe differs by {err:.3g}"

    def _project(
        self, moe_utils, name: str, x, offs_expert, offs_pair, counts, plan, dtype
    ) -> torch.Tensor:
        """Frozen base over E expert groups through Zoo's base GEMM (its weight provider, recompute and selective
        dequant, as the single-job forward), plus every adapter's LoRA over the E * (U + 1) pair groups."""
        base = self.base_layer
        src, proj = getattr(base, name), name.removesuffix("_proj")
        hidden, model_type = (
            self._in_out("gate_up_proj")[1],
            getattr(base, "_unsloth_model_type", None),
        )

        def provider(experts = None):
            weight = moe_utils._get_base_weight(src, dtype, counts, experts)
            return moe_utils.preprocess_weight(
                weight, proj, hidden, model_type, experts_module = base
            )

        recompute = moe_utils._moe_recompute_enabled(src, dtype = dtype)
        y = moe_utils._base_grouped_mm(
            x, offs_expert, provider, recompute, plan if recompute else None
        )
        if name in self.adapters:
            first, second = self.adapters[name].grouped_stacks(self.reversed_io, dtype)
            h = moe_utils._grouped_mm_with_backward_fix(x, first, offs_pair).contiguous()
            y = y + moe_utils._grouped_mm_with_backward_fix(h, second, offs_pair)
        return y

    def _grouped(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        adapter_ids: torch.Tensor,
    ) -> torch.Tensor:
        T, k = top_k_index.shape
        if T * k == 0:
            # Nothing routed: zeros still tied to the hidden states and every adapter (empty slices, so exact 0 even
            # for an inf / NaN adapter), as backward and default DDP need.
            zero = hidden_states.reshape(-1)[:0].sum() + sum(
                p.reshape(-1)[:0].sum()
                for adapter in self.adapters.values()
                for p in (*adapter.lora_A, *adapter.lora_B)
            )
            return hidden_states.new_zeros(hidden_states.shape) + zero.to(hidden_states.dtype)
        E = _expert_stack_shape(self.base_layer.gate_up_proj)[0]
        U = self.num_adapters
        Ug = U + 1
        dtype = self._compute_dtype(hidden_states)
        # Route = expert * Ug + adapter, slot U for rows of no adapter (zero factors) and for dropped routes
        # (expert id outside [0, E): they read a zero row and get weight 0, so they add exactly 0 and nothing can
        # overflow or turn a gradient NaN); sorted, each expert's rows stay contiguous.
        index = top_k_index.long()
        valid = (index >= 0) & (index < E)
        slot = torch.where(adapter_ids < 0, U, adapter_ids).to(index.device).unsqueeze(1)
        key = (torch.where(valid, index, 0) * Ug + torch.where(valid, slot, U)).reshape(-1)
        order = torch.argsort(key, stable = True)
        # Fixed-size counts: bincount's output size depends on the data, so it syncs on CUDA and breaks a compiled graph.
        counts = torch.zeros(E * Ug, dtype = torch.int32, device = key.device)
        counts.scatter_add_(0, key, torch.ones_like(key, dtype = torch.int32))
        offs_pair = torch.cumsum(counts, 0, dtype = torch.int32)
        offs_expert = offs_pair.view(E, Ug)[:, -1].contiguous()
        counts = counts.view(E, Ug).sum(1, dtype = torch.int32)
        moe_utils = _zoo_moe_utils()
        base = self.base_layer
        plan = None
        if moe_utils._moe_recompute_enabled(base.gate_up_proj, dtype = dtype):
            plan = moe_utils._selective_dequant_plan(
                (base.gate_up_proj, base.down_proj), counts, T * k
            )
        with torch.autocast(device_type = hidden_states.device.type, enabled = False):
            rows = torch.where(valid.reshape(-1)[order], order // k, T)
            x = torch.cat(
                [
                    hidden_states.to(dtype),
                    hidden_states.new_zeros(1, hidden_states.shape[1], dtype = dtype),
                ]
            )
            x = x[rows]
            h = self._project(
                moe_utils, "gate_up_proj", x, offs_expert, offs_pair, counts, plan, dtype
            )
            gate, up = h.chunk(2, dim = -1)
            inter = base.act_fn(gate) * up
            # From here on as Unsloth Zoo's MoE forward combines routes: its gate-gradient identity when enabled,
            # then its fused weighted unpermute (deterministic fallback without Triton).
            weights = torch.where(valid, top_k_weights, 0).reshape(-1)[order]
            gategrad = getattr(moe_utils, "_moe_gategrad_enabled", lambda: False)()
            gategrad = gategrad and hasattr(moe_utils, "_MoEGateGradIdentity")
            gategrad = gategrad and top_k_weights.requires_grad and inter.dtype != torch.float16
            if gategrad:
                raw = weights.float() if weights.dtype == torch.float16 else weights
                floored = torch.where(raw >= 0, raw.clamp(min = 1e-12), raw.clamp(max = -1e-12))
                weights = raw + (floored - raw).detach()
                inter = moe_utils._MoEGateGradIdentity.apply(inter, weights)
                weights = weights.detach()
            y = self._project(
                moe_utils, "down_proj", inter, offs_expert, offs_pair, counts, plan, dtype
            )
            out = moe_utils.weighted_unpermute(
                y, order, weights, T, k, out_dtype = hidden_states.dtype
            )
            if out is None:
                out = moe_utils.combine_permuted_moe_outputs(
                    y * weights.unsqueeze(-1), order, T, k, out_dtype = hidden_states.dtype
                )
            return out

    def forward(self, hidden_states: torch.Tensor, *args):
        batch = next(iter(self.adapters.values())).batch
        if batch is None or hidden_states.shape[0] != batch.num_tokens:
            raise ValueError(
                "Call set_multi_lora_batch(model, ...) with this batch's adapter ids first."
            )
        if self.mode is None:
            self._choose_mode(hidden_states, args, _adapter_groups(batch))
        grouped = self.mode == "grouped"
        if grouped and torch.compiler.is_compiling():
            # Grouped GEMM only traces in bfloat16 (its meta checks the dtype on every device).
            grouped = self._compute_dtype(hidden_states) == torch.bfloat16
        elif grouped:
            # With one adapter's tokens there is nothing to fuse: its own pass is the same work in fewer groups.
            grouped = sum(u is not None for u, _ in _adapter_groups(batch)) >= 2
        if grouped:
            if not torch.compiler.is_compiling():
                MULTI_LORA_MOE_STATS["grouped_calls"] += 1
            # Every adapter's factors are in the graph, so absent adapters get exact zero grads.
            return self._grouped(hidden_states, *args, _token_adapters(batch))
        if not torch.compiler.is_compiling():
            MULTI_LORA_MOE_STATS["per_adapter_calls"] += 1
        groups = _adapter_groups(batch)
        out = self._per_adapter(hidden_states, args, groups)
        present = {u for u, _ in groups}
        absent = [u for u in range(self.num_adapters) if u not in present]
        if absent and torch.is_grad_enabled():
            # Default DDP needs every trainable parameter used; empty slices stay zero even for an inf / NaN adapter.
            zero = sum(
                A[u].reshape(-1)[:0].sum() + B[u].reshape(-1)[:0].sum()
                for adapter in self.adapters.values()
                for A, B in ((adapter.lora_A, adapter.lora_B),)
                for u in absent
            )
            out = out + zero.to(out.dtype)
        return out


def inject_multi_lora_experts(
    model: nn.Module,
    target_parameters: str | Iterable[str],
    num_adapters: int,
    r: int | Iterable[int] = 16,
    lora_alpha: float | Iterable[float] = 16,
    use_rslora: bool = False,
) -> list[str]:
    """Wrap every fused experts module holding a ``target_parameters`` entry, matched on path suffix as PEFT does."""
    targets = (
        (target_parameters,) if isinstance(target_parameters, str) else tuple(target_parameters)
    )
    by_module: dict[str, list[str]] = {}
    for name, module in model.named_modules():
        if isinstance(module, MultiLoraExperts):
            continue
        for param, value in module._parameters.items():
            if not isinstance(value, torch.Tensor):
                continue
            key = f"{name}.{param}"
            if any(key == t or key.endswith("." + t) for t in targets):
                by_module.setdefault(name, []).append(param)
    if not by_module:
        raise ValueError(f"No fused expert parameters named {targets} found.")
    for name, params in by_module.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        bypass = _reads_experts_directly(parent, child, params)
        if bypass:
            raise ValueError(
                f"{type(parent).__name__}.forward reads {child}.{bypass} itself instead of calling {child}(...), "
                "so per-adapter expert LoRA would never run there."
            )
    _freeze_all_but_adapters(model)
    for name, params in by_module.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        module = parent.get_submodule(child)
        setattr(
            parent, child, MultiLoraExperts(module, params, num_adapters, r, lora_alpha, use_rslora)
        )
    return [f"{name}.{param}" for name, params in by_module.items() for param in params]


def _reads_experts_directly(parent: nn.Module, child: str, params: list) -> Optional[str]:
    """The targeted parameter a parent's forward reads off its experts (e.g. a grouped-GEMM fast path), if any."""
    try:
        source = inspect.getsource(type(parent).forward)
    except (OSError, TypeError):
        return None
    for param in params:
        if re.search(rf"\bself\.{re.escape(child)}\.{re.escape(param)}\b", source):
            return param
    return None


def _freeze_all_but_adapters(model: nn.Module) -> None:
    # Dense and expert adapters are injected one after the other: neither call may freeze the other's.
    keep = {id(p) for _, layer in _layers(model) for p in (*layer.lora_A, *layer.lora_B)}
    for p in model.parameters():
        if id(p) not in keep:
            p.requires_grad_(False)


def _layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (n, m)
        for n, m in model.named_modules()
        if isinstance(m, (MultiLoraLinear, MultiLoraExpertParam))
    ]


def set_multi_lora_batch(
    model: nn.Module,
    adapter_ids: torch.Tensor,
    seq_len: Optional[int] = None,
) -> MultiLoraBatch:
    """Route this batch's tokens to their adapters (one id per token, or one per row with ``seq_len``)."""
    layers = _layers(model)
    if not layers:
        raise ValueError("The model has no multi-LoRA layers.")
    counts = {layer.num_adapters for _, layer in layers}
    if len(counts) > 1:
        raise ValueError(
            f"Every multi-LoRA layer must have the same number of adapters, got {sorted(counts)}."
        )
    # One routing table per device: a model split over GPUs keeps each layer's indices local.
    by_device: dict[torch.device, MultiLoraBatch] = {}
    for _, layer in layers:
        device = layer.lora_A[0].device
        if device not in by_device:
            by_device[device] = MultiLoraBatch.from_adapter_ids(
                adapter_ids.to(device), layer.num_adapters, seq_len
            )
        layer.batch = by_device[device]
        seg = layer.batch.seg_start
        present = (seg[1:] - seg[:-1]) > 0
        layer.seen = present if layer.seen is None else layer.seen | present
    return next(iter(by_device.values()))


def _distributed() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


@torch.no_grad()
def drop_absent_adapter_grads_(model: nn.Module, process_group = None) -> list[int]:
    """Set grads to None for adapters with no valid tokens since the last call on any rank; call before ``step()``."""
    layers = _layers(model)
    first = layers[0][1] if layers else None
    present = (
        None if first is None else (first.trained if first.trained is not None else first.seen)
    )
    if present is None:
        return []
    present = present.to(torch.int32)
    if _distributed():
        torch.distributed.all_reduce(present, group = process_group)
    absent = [u for u, n in enumerate(present.tolist()) if n == 0]
    for _, layer in layers:
        layer.seen = layer.trained = None
        for u in absent:
            layer.lora_A[u].grad = None
            layer.lora_B[u].grad = None
    return absent


def _label_adapter_ids(ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # A shifted label belongs to its own token's adapter: (B, T) ids for packed rows, (B,) per row.
    ids = ids.to(labels.device).long()
    return ids[:, 1:] if ids.dim() == 2 else ids[:, None].expand(-1, labels.shape[1] - 1)


def adapter_token_counts(
    labels: torch.Tensor,
    row_adapter_ids: torch.Tensor,
    num_adapters: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Valid (shifted) label tokens per adapter in one microbatch; ids of -1 count for none."""
    valid = (labels[:, 1:] != ignore_index).float()
    ids = _label_adapter_ids(row_adapter_ids, labels)
    keep = (ids >= 0).to(valid.dtype)
    return torch.zeros(num_adapters, device = labels.device).index_add(
        0, ids.clamp_min(0).reshape(-1), (valid * keep).reshape(-1)
    )


def per_adapter_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_adapter_ids: torch.Tensor,
    num_adapters: int,
    ignore_index: int = -100,
    process_group = None,
    model: Optional[nn.Module] = None,
    num_items_in_batch: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM loss averaged within each adapter, summed over adapters.

    ``row_adapter_ids``: (B,) or per token (B, T). Pass ``model`` for ``drop_absent_adapter_grads_``, and under
    gradient accumulation ``num_items_in_batch`` = per-adapter ``adapter_token_counts`` summed over the window.
    Returns (total, this rank's per-adapter losses).
    """
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index = ignore_index,
        reduction = "none",
    ).view(shift_labels.shape)
    valid = (shift_labels != ignore_index).float()
    ids = _label_adapter_ids(row_adapter_ids, labels)
    # Tokens with no adapter add zero to slot 0 instead of being masked out: no host sync under compile.
    keep = (ids >= 0).to(token_loss.dtype)
    ids = ids.clamp_min(0).reshape(-1)
    sums = torch.zeros(num_adapters, device = logits.device).index_add(
        0, ids, (token_loss * keep).reshape(-1)
    )
    counts = torch.zeros(num_adapters, device = logits.device).index_add(
        0, ids, (valid * keep).reshape(-1)
    )
    denom = counts if num_items_in_batch is None else num_items_in_batch.to(counts)
    losses = sums / denom.clamp_min(1)
    if model is not None:
        # Adapters with valid label tokens: routed rows whose labels are all ignored do not train.
        trained = counts > 0
        for _, layer in _layers(model):
            mark = trained.to(layer.lora_A[0].device)
            layer.trained = mark if layer.trained is None else layer.trained | mark
    if not _distributed():
        return losses.sum(), losses.detach()
    # Undo DDP / FSDP's rank average so each adapter's gradient is its mean over the global batch.
    total = denom.detach().clone()
    torch.distributed.all_reduce(total, group = process_group)
    world = torch.distributed.get_world_size(process_group)
    return (sums / total.clamp_min(1)).sum() * world, losses.detach()


def _adapter_grads(model: nn.Module) -> list[list[torch.Tensor]]:
    layers = _layers(model)
    grads = [[] for _ in range(layers[0][1].num_adapters)] if layers else []
    for _, layer in layers:
        for u in range(layer.num_adapters):
            for p in (layer.lora_A[u], layer.lora_B[u]):
                if p.grad is not None:
                    grads[u].append(p.grad)
    return grads


def _local(t: torch.Tensor) -> torch.Tensor:
    # FSDP2 parameters and grads are DTensors: work on this rank's shard.
    return t.to_local() if hasattr(t, "to_local") else t


def _grad_sq(model: nn.Module, process_group = None) -> torch.Tensor:
    layers = _layers(model)
    if not layers:
        raise ValueError("The model has no multi-LoRA layers.")
    device = _local(layers[0][1].lora_A[0]).device
    out = torch.zeros(layers[0][1].num_adapters, device = device)
    sharded = False
    for u, gs in enumerate(_adapter_grads(model)):
        for g in gs:
            sharded = sharded or hasattr(g, "to_local")
            out[u] += _local(g).float().pow(2).sum().to(device)
    if sharded and _distributed():
        torch.distributed.all_reduce(out, group = process_group)
    return out


def per_adapter_grad_norm(model: nn.Module, process_group = None) -> torch.Tensor:
    """(U,) gradient L2 norm of each adapter over every multi-LoRA layer (0 for an adapter without grads)."""
    return _grad_sq(model, process_group).sqrt()


@torch.no_grad()
def clip_grad_norm_per_adapter_(
    model: nn.Module,
    max_norm: float,
    eps: float = 1e-6,
    process_group = None,
) -> torch.Tensor:
    """Clip each adapter's gradient to ``max_norm`` on its own, so one adapter never rescales another."""
    norms = per_adapter_grad_norm(model, process_group)
    coef = (max_norm / (norms + eps)).clamp(max = 1.0)
    for u, gs in enumerate(_adapter_grads(model)):
        for g in gs:
            local = _local(g)
            local.mul_(coef[u].to(device = local.device, dtype = local.dtype))
    return norms


def _full(p: torch.Tensor) -> torch.Tensor:
    return p.full_tensor() if hasattr(p, "full_tensor") else p


def adapter_state_dict(
    model: nn.Module,
    adapter: int,
    prefix: str = "base_model.model.",
) -> dict:
    """Adapter ``adapter`` as a PEFT LoRA state dict (``<name>.lora_A.weight`` / ``lora_B.weight``)."""
    out = {}
    for name, layer in _layers(model):
        if isinstance(layer, MultiLoraLinear):
            out[f"{prefix}{name}.lora_A.weight"] = _full(layer.lora_A[adapter]).detach().clone()
            out[f"{prefix}{name}.lora_B.weight"] = _full(layer.lora_B[adapter]).detach().clone()
    b_layout = _peft_moe_lora_b_layout()
    for name, experts in model.named_modules():
        if not isinstance(experts, MultiLoraExperts):
            continue
        # PEFT nests one ParamWrapper per target parameter: the first listed is the innermost.
        depth = len(experts.parameter_names)
        for i, param in enumerate(experts.parameter_names):
            key = f"{prefix}{name}" + ".base_layer" * (depth - 1 - i)
            # Experts whose forward ignores the stash load through PEFT's own ParamWrapper delta, which is
            # rank-major whatever Zoo uses; how they trained here (grouped or not) does not matter.
            layout = "rank_major" if experts.peft_mode == "fold" else b_layout
            A, B = experts.adapters[param].peft_tensors(adapter, layout)
            out[f"{key}.lora_A.weight"] = A
            out[f"{key}.lora_B.weight"] = B
    return out


def _peft_moe_lora_b_layout() -> str:
    try:
        from unsloth_zoo.temporary_patches.moe_utils import moe_lora_b_layout
        return moe_lora_b_layout()
    except Exception:  # noqa: BLE001 - PEFT's own packing: weight_B.reshape(out, r, E)
        return "rank_major"
