# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched multi-LoRA training on one frozen base: tokens (id -1 = no adapter) are stable-sorted by
adapter once per batch and kernels gather through that order. UNSLOTH_MULTI_LORA_TRITON=0, or no
CUDA/XPU, uses a PyTorch loop; LoRA dropout unsupported."""

from __future__ import annotations

import contextlib
import math
import os
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
    "multi_lora_delta",
    "multi_lora_linear",
    "inject_multi_lora",
    "set_multi_lora_batch",
    "per_adapter_loss",
    "adapter_token_counts",
    "per_adapter_grad_norm",
    "clip_grad_norm_per_adapter_",
    "drop_absent_adapter_grads_",
    "adapter_state_dict",
    "set_multi_lora_triton",
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
    for p in model.parameters():
        p.requires_grad_(False)
    for name, module in found:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, MultiLoraLinear(module, num_adapters, r, lora_alpha, use_rslora))
    return [name for name, _ in found]


def _layers(model: nn.Module) -> list[tuple[str, MultiLoraLinear]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MultiLoraLinear)]


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
        out[f"{prefix}{name}.lora_A.weight"] = _full(layer.lora_A[adapter]).detach().clone()
        out[f"{prefix}{name}.lora_B.weight"] = _full(layer.lora_B[adapter]).detach().clone()
    return out
