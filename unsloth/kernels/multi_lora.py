# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Batched multi-LoRA training: many LoRA adapters on one frozen base model, trained in one batch.

Each token belongs to one adapter (or none, id -1). Tokens are stable-sorted by adapter once per batch
(``MultiLoraBatch``) and every kernel gathers rows through that order, like Punica's SGMV and vLLM's
``lora_shrink`` / ``lora_expand``, which are inference only. Training adds dX and per-adapter dA / dB
reduced over each adapter's own tokens (``_seg_wgrad_kernel``), so with ``per_adapter_loss``,
``drop_absent_adapter_grads_``, ``clip_grad_norm_per_adapter_`` and an elementwise optimizer each adapter
trains as it would alone.

``UNSLOTH_MULTI_LORA_TRITON=0`` (or no CUDA / XPU) uses a PyTorch loop. LoRA dropout is unsupported.
"""

from __future__ import annotations

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
    "per_adapter_grad_norm",
    "clip_grad_norm_per_adapter_",
    "drop_absent_adapter_grads_",
    "adapter_state_dict",
]


def _use_triton(X: torch.Tensor) -> bool:
    return (
        HAS_TRITON
        and X.device.type in ("cuda", "xpu")
        and os.environ.get("UNSLOTH_MULTI_LORA_TRITON", "1") != "0"
    )


@dataclass
class MultiLoraBatch:
    """Token order shared by every multi-LoRA layer for one batch.

    ``sorted_idx[seg_start[u]:seg_start[u + 1]]`` are adapter u's token rows, in their original order.
    Tokens with adapter id -1 sort last and belong to no segment. Shapes depend only on the token count,
    so a new batch never changes a kernel grid or a compiled graph.
    """

    sorted_idx: torch.Tensor
    seg_start: torch.Tensor
    num_adapters: int

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
        return cls(sorted_idx, seg_start, num_adapters)

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
    M, K = inp.shape
    BLOCK_M = 64
    BLOCK_N = _block(N, 16, 128)
    BLOCK_K = _block(K, 16, 64 if BLOCK_N > 32 else 128)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), batch.num_adapters)
    with torch.cuda.device(inp.device) if inp.device.type == "cuda" else _nullcontext():
        _seg_mm_kernel[grid](
            inp,
            W,
            out,
            scales if scales is not None else inp,
            batch.sorted_idx,
            batch.seg_start,
            N,
            K,
            inp.stride(0),
            out.stride(0),
            *w_strides,
            IN_SORTED = in_sorted,
            OUT_SORTED = out_sorted,
            ADD = add,
            SCALE = scales is not None,
            IEEE = inp.dtype == torch.float32,
            BLOCK_M = BLOCK_M,
            BLOCK_N = BLOCK_N,
            BLOCK_K = BLOCK_K,
            num_warps = 4,
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
    _, I, J = out.shape
    BLOCK_I = _block(I, 16, 64)
    BLOCK_J = _block(J, 16, 64)
    grid = (triton.cdiv(I, BLOCK_I) * triton.cdiv(J, BLOCK_J), batch.num_adapters)
    with torch.cuda.device(P.device) if P.device.type == "cuda" else _nullcontext():
        _seg_wgrad_kernel[grid](
            P,
            Q,
            out,
            scales if scales is not None else P,
            batch.sorted_idx,
            batch.seg_start,
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
        )
    return out


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


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
    """Reference LoRA delta in plain PyTorch (any device, any dtype, autograd through ordinary ops)."""
    shape = X.shape
    X = X.reshape(-1, shape[-1])
    out = X.new_zeros(X.shape[0], B.shape[1])
    starts = batch.seg_start.tolist()
    idx = batch.sorted_idx.long()
    for u in range(batch.num_adapters):
        if starts[u] == starts[u + 1]:
            continue
        rows = idx[starts[u] : starts[u + 1]]
        h = X.index_select(0, rows) @ A[u].to(X.dtype).t()
        out = out.index_add(0, rows, (h @ B[u].to(X.dtype).t()) * scales[u].to(X.dtype))
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
    """A frozen base linear layer with ``num_adapters`` LoRA adapters, one (A, B) parameter pair each.

    Separate parameters per adapter keep optimizer state per adapter: an adapter absent from a batch gets
    no gradient (``drop_absent_adapter_grads_``), so AdamW leaves it, its moments and its step count alone.
    ``r`` / ``lora_alpha`` may differ per adapter. Initialisation matches PEFT (kaiming uniform A, zero B).
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
        self.batch: Optional[MultiLoraBatch] = None

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
    for p in model.parameters():
        p.requires_grad_(False)
    replaced = []
    for name, module in list(model.named_modules()):
        if not name or not any(name == t or name.endswith("." + t) for t in targets):
            continue
        if not hasattr(module, "in_features") or not hasattr(module, "weight"):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, MultiLoraLinear(module, num_adapters, r, lora_alpha, use_rslora))
        replaced.append(name)
    if not replaced:
        raise ValueError(f"No linear layers named {targets} found.")
    return replaced


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
    # One routing table per device: a model split over GPUs keeps each layer's indices local.
    by_device: dict[torch.device, MultiLoraBatch] = {}
    for _, layer in layers:
        device = layer.lora_A[0].device
        if device not in by_device:
            by_device[device] = MultiLoraBatch.from_adapter_ids(
                adapter_ids.to(device), layer.num_adapters, seq_len
            )
        layer.batch = by_device[device]
    return next(iter(by_device.values()))


@torch.no_grad()
def drop_absent_adapter_grads_(model: nn.Module) -> list[int]:
    """Clear the gradients of adapters with no tokens in the current batch; call before ``optimizer.step()``.

    PyTorch optimizers skip a parameter whose ``grad`` is None, so an absent adapter keeps its weights,
    moments and step count, exactly as if it had not trained this step. Returns the absent adapters.
    """
    layers = _layers(model)
    if not layers or layers[0][1].batch is None:
        return []
    seg = layers[0][1].batch.seg_start
    absent = [u for u, n in enumerate((seg[1:] - seg[:-1]).tolist()) if n == 0]
    for _, layer in layers:
        for u in absent:
            layer.lora_A[u].grad = None
            layer.lora_B[u].grad = None
    return absent


def per_adapter_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    row_adapter_ids: torch.Tensor,
    num_adapters: int,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal LM loss averaged within each adapter, summed over adapters.

    Each adapter's parameters only see its own rows, so the gradient of the sum is, per adapter, the
    gradient of its own mean loss: the same as training it alone. Returns (total, per-adapter losses).
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
    ids = row_adapter_ids.to(logits.device).long()
    keep = ids >= 0
    sums = torch.zeros(num_adapters, device = logits.device).index_add(
        0, ids[keep], token_loss.sum(1)[keep]
    )
    counts = torch.zeros(num_adapters, device = logits.device).index_add(
        0, ids[keep], valid.sum(1)[keep]
    )
    losses = sums / counts.clamp_min(1)
    return losses.sum(), losses.detach()


def _adapter_grads(model: nn.Module) -> list[list[torch.Tensor]]:
    layers = _layers(model)
    grads = [[] for _ in range(layers[0][1].num_adapters)] if layers else []
    for _, layer in layers:
        for u in range(layer.num_adapters):
            for p in (layer.lora_A[u], layer.lora_B[u]):
                if p.grad is not None:
                    grads[u].append(p.grad)
    return grads


def per_adapter_grad_norm(model: nn.Module) -> torch.Tensor:
    """(U,) gradient L2 norm of each adapter over every multi-LoRA layer (0 for an adapter without grads)."""
    grads = _adapter_grads(model)
    if not any(grads):
        raise ValueError("No multi-LoRA gradients.")
    device = next(g for gs in grads for g in gs).device
    return torch.stack(
        [
            torch.linalg.vector_norm(torch.stack([g.float().norm().to(device) for g in gs]))
            if gs
            else torch.zeros((), device = device)
            for gs in grads
        ]
    )


@torch.no_grad()
def clip_grad_norm_per_adapter_(
    model: nn.Module,
    max_norm: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Clip each adapter's gradient to ``max_norm`` on its own, so one adapter never rescales another."""
    norms = per_adapter_grad_norm(model)
    coef = (max_norm / (norms + eps)).clamp(max = 1.0)
    for u, gs in enumerate(_adapter_grads(model)):
        for g in gs:
            g.mul_(coef[u].to(device = g.device, dtype = g.dtype))
    return norms


def _full(p: torch.Tensor) -> torch.Tensor:
    # An FSDP2-sharded parameter is a DTensor: export the whole tensor, not this rank's shard.
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
