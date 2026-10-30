# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused int8 weight-only GEMV for small-M Linears (the per-block AdaLN modulations, M = batch).

torchao's weight-only ``Int8Tensor`` linear casts the whole int8 weight to bf16 on every call (``qdata.t().to(bf16)``,
then ``mm``, then ``* scale``, then ``+ bias``, each rounding to bf16). On a 3072 x 18432 modulation that is a
113 MB bf16 temporary per call for a GEMV that only needs to read the 57 MB of int8 codes once. This kernel reads the
codes and the scales directly, accumulates in float32 and rounds once at the end, so it is never slower than that
path and slightly more accurate. Per-row (``block_size [1, K]``) and per-group (``block_size [1, G]``) symmetric
scales are both handled; anything else (zero points, activation quant, odd K) keeps torchao's own forward.

The op is opaque to Inductor (``torch.library.custom_op`` with a fake impl), and M is the batch, independent of the
prompt length, so it adds no recompiles and no CUDA-graph re-captures.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from typing import Any, Optional

INT8_GEMV_ENV = "UNSLOTH_DIFFUSION_INT8_GEMV"
# Largest flattened row count the GEMV takes; above it the stock dequant + GEMM path is the faster one.
MAX_ROWS = 4
# Input channels kept dense beside the int8 codes (``OUTLIER_*`` buffers); the kernel pads them to 16.
MAX_OUTLIERS = 16
OUTLIER_IDX = "outlier_idx"
OUTLIER_WEIGHT = "outlier_weight"
_OP_NAMESPACE = "unsloth_studio"
_OP_NAME = "int8_wo_gemv"
# (BLOCK_N, BLOCK_K, warps, stages), first that launches on a card wins. 128 x 128 x 3 stages needs ~40 KB of shared
# memory (RTX PRO 6000: 15 us per 3072 x 18432 call at M = 2, torchao's weight-only path 208 us, bf16 33 us); the
# second fits any CUDA card Triton supports.
_CONFIGS = ((128, 128, 8, 3), (64, 64, 4, 2))
_DEVICE_CFG: dict = {}
_LOCK = threading.Lock()
# Kernel launches since import (read by benches as engagement evidence).
COUNTS = {"gemv": 0, "fallback": 0}
# Resolved once by ``install`` (outside any trace): the op, or None when the kernel is off / unavailable.
_OP_HANDLE: Any = None


def gemv_enabled() -> bool:
    raw = (os.environ.get(INT8_GEMV_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


@lru_cache(maxsize = 1)
def _kernel() -> Any:
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001 - no Triton: stock path
        return None

    @triton.jit
    def i8_wo_gemv(
        x_ptr,
        q_ptr,
        s_ptr,
        b_ptr,
        oi_ptr,
        ow_ptr,
        y_ptr,
        M,
        N,
        K,
        KO,
        stride_xm,
        stride_qn,
        stride_sn,
        stride_on,
        stride_ym,
        HAS_BIAS: tl.constexpr,
        HAS_OUT: tl.constexpr,
        GROUPED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rm = tl.arange(0, BLOCK_M)
        n_ok = rn < N
        m_ok = rm < M
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)
        for k0 in range(0, K, BLOCK_K):
            rk = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(q_ptr + rn[:, None] * stride_qn + rk[None, :], mask = n_ok[:, None], other = 0)
            x = tl.load(
                x_ptr + rm[:, None] * stride_xm + rk[None, :], mask = m_ok[:, None], other = 0.0
            )
            # int8 -> bf16 is exact and bf16 x bf16 products are exact in the float32 accumulator
            part = tl.dot(x, tl.trans(q.to(tl.bfloat16)), out_dtype = tl.float32)
            if GROUPED:
                s = tl.load(s_ptr + rn * stride_sn + k0 // BLOCK_K, mask = n_ok, other = 0.0)
                acc += part * s.to(tl.float32)[None, :]
            else:
                acc += part
        if not GROUPED:
            s = tl.load(s_ptr + rn * stride_sn, mask = n_ok, other = 0.0)
            acc = acc * s.to(tl.float32)[None, :]
        if HAS_OUT:
            # bf16 outlier columns (their int8 codes are zero): x[:, idx] @ W_out.T, same accumulator
            ro = tl.arange(0, 16)
            o_ok = ro < KO
            idx = tl.load(oi_ptr + ro, mask = o_ok, other = 0)
            xo = tl.load(
                x_ptr + rm[:, None] * stride_xm + idx[None, :],
                mask = m_ok[:, None] & o_ok[None, :],
                other = 0.0,
            )
            wo = tl.load(
                ow_ptr + rn[:, None] * stride_on + ro[None, :],
                mask = n_ok[:, None] & o_ok[None, :],
                other = 0.0,
            )
            acc += tl.dot(xo, tl.trans(wo), out_dtype = tl.float32)
        if HAS_BIAS:
            b = tl.load(b_ptr + rn, mask = n_ok, other = 0.0)
            acc = acc + b.to(tl.float32)[None, :]
        tl.store(
            y_ptr + rm[:, None] * stride_ym + rn[None, :],
            acc.to(tl.bfloat16),
            mask = m_ok[:, None] & n_ok[None, :],
        )

    return triton, i8_wo_gemv


def reference(
    x: Any,
    q: Any,
    s: Any,
    bias: Any,
    oidx: Any = None,
    ow: Any = None,
) -> Any:
    """Same math in torch (float32 accumulate, one bf16 rounding): the CPU / no-Triton body of the op."""
    import torch

    n, k = q.shape
    groups = s.numel() // n
    xf = x.float()
    if groups == 1:
        y = (xf @ q.float().t()) * s.reshape(1, n).float()
    else:
        g = k // groups
        parts = torch.einsum(
            "mgk,ngk->mng", xf.reshape(-1, groups, g), q.float().reshape(n, groups, g)
        )
        y = (parts * s.reshape(1, n, groups).float()).sum(-1)
    if oidx is not None:
        y = y + xf.index_select(-1, oidx.long()) @ ow.float().t()
    if bias is not None:
        y = y + bias.float()
    return y.to(torch.bfloat16)


def _launch(x: Any, q: Any, s2: Any, bias: Any, oidx: Any, ow: Any, cfg: tuple) -> Any:
    import torch

    triton, fn = _kernel()
    block_n, block_k, warps, stages = cfg
    m, k = x.shape
    n, groups = s2.shape
    if groups > 1:
        block_k = k // groups
    ko = 0 if oidx is None else int(oidx.numel())
    y = torch.empty((m, n), device = x.device, dtype = torch.bfloat16)
    fn[(triton.cdiv(n, block_n),)](
        x,
        q,
        s2,
        bias if bias is not None else y,
        oidx if ko else y,
        ow if ko else y,
        y,
        m,
        n,
        k,
        ko,
        x.stride(0),
        q.stride(0),
        s2.stride(0),
        ow.stride(0) if ko else 0,
        y.stride(0),
        HAS_BIAS = bias is not None,
        HAS_OUT = ko > 0,
        GROUPED = groups > 1,
        # tl.dot wants >= 16 rows: the masked pad rows load zeros and are never stored
        BLOCK_M = 16,
        BLOCK_N = block_n,
        BLOCK_K = block_k,
        num_warps = warps,
        num_stages = stages,
    )
    return y


def _run(
    x: Any,
    q: Any,
    s: Any,
    bias: Any,
    oidx: Any = None,
    ow: Any = None,
) -> Any:
    m, k = x.shape
    n = q.shape[0]
    groups = s.numel() // n
    block_k = k // groups if groups > 1 else _CONFIGS[-1][1]
    ko = 0 if oidx is None else int(oidx.numel())
    index = x.device.index if x.is_cuda else None
    start = _DEVICE_CFG.get(index, 0)
    if (
        index is None
        or _kernel() is None
        or start >= len(_CONFIGS)
        or k % block_k
        or (groups == 1 and k % _CONFIGS[start][1])
        or q.stride(-1) != 1
        or block_k & (block_k - 1)
        or block_k < 16
        or ko > MAX_OUTLIERS
        or (ko and ow.stride(-1) != 1)
    ):
        COUNTS["fallback"] += 1
        return reference(x, q, s, bias, oidx, ow)
    x = x if x.stride(-1) == 1 else x.contiguous()
    s2 = s.reshape(n, groups)
    if s2.stride(-1) != 1:
        s2 = s2.contiguous()
    for pos in range(start, len(_CONFIGS)):
        try:
            y = _launch(x, q, s2, bias, oidx, ow, _CONFIGS[pos])
        except Exception:  # noqa: BLE001 - a tile this card cannot hold (shared memory): the next, smaller one
            continue
        _DEVICE_CFG[index] = pos
        COUNTS["gemv"] += 1
        return y
    _DEVICE_CFG[index] = len(_CONFIGS)
    COUNTS["fallback"] += 1
    return reference(x, q, s, bias, oidx, ow)


@lru_cache(maxsize = 1)
def _op() -> Any:
    """``unsloth_studio::int8_wo_gemv`` (opaque to Inductor), or None."""
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    ns = getattr(torch.ops, _OP_NAMESPACE, None)
    if ns is not None and hasattr(ns, _OP_NAME):
        return getattr(ns, _OP_NAME)
    custom_op = getattr(getattr(torch, "library", None), "custom_op", None)
    if custom_op is None:
        return None
    try:

        @custom_op(
            f"{_OP_NAMESPACE}::{_OP_NAME}",
            mutates_args = (),
            schema = "(Tensor x, Tensor q, Tensor s, Tensor? bias, Tensor? oidx, Tensor? ow) -> Tensor",
        )
        def _int8_wo_gemv(x, q, s, bias, oidx, ow):
            return _run(x, q, s, bias, oidx, ow)

        @_int8_wo_gemv.register_fake
        def _(x, q, s, bias, oidx, ow):
            return x.new_empty((x.shape[0], q.shape[0]), dtype = torch.bfloat16)
    except Exception:  # noqa: BLE001
        return None
    return getattr(torch.ops, _OP_NAMESPACE).int8_wo_gemv


def weight_only_parts(w: Any) -> Optional[tuple]:
    """(int8 [N, K], scale [N, K / G]) of a symmetric weight-only torchao ``Int8Tensor`` (per row or per group)."""
    if type(w).__name__ != "Int8Tensor":
        return None
    if (
        getattr(w, "act_quant_kwargs", None) is not None
        or getattr(w, "act_pre_scale", None) is not None
    ):
        return None
    q, s = getattr(w, "qdata", None), getattr(w, "scale", None)
    block = list(getattr(w, "block_size", ()) or ())
    if q is None or s is None or q.dim() != 2 or len(block) != 2 or block[0] != 1:
        return None
    n, k = q.shape
    if k % block[1] or s.numel() != n * (k // block[1]):
        return None
    return q, s


def _outliers(module: Any) -> tuple:
    return getattr(module, OUTLIER_IDX, None), getattr(module, OUTLIER_WEIGHT, None)


def _forward(self: Any, x: Any) -> Any:
    import torch
    import torch.nn.functional as F

    w = self.weight
    oidx, ow = _outliers(self)
    parts = weight_only_parts(w)
    flat_rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
    op = _OP_HANDLE
    if (
        op is None
        or parts is None
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or flat_rows == 0
        or flat_rows > MAX_ROWS
        or parts[0].device != x.device
        or (oidx is not None and (oidx.device != x.device or ow.device != x.device))
    ):
        # stock torchao weight-only forward, plus the dense outlier columns (exact: their codes are zero)
        y = F.linear(x, w, self.bias)
        if oidx is not None:
            y = y + F.linear(
                x.index_select(-1, oidx.to(x.device)), ow.to(device = x.device, dtype = x.dtype)
            )
        return y
    lead = x.shape[:-1]
    y = op(x.reshape(-1, x.shape[-1]), parts[0], parts[1], self.bias, oidx, ow)
    return y.reshape(*lead, y.shape[-1])


@lru_cache(maxsize = None)
def gemv_linear_class() -> Any:
    """Built once: a fresh class per call would make torch.compile retrace every block."""
    from torch import nn

    class Int8GemvLinear(nn.Linear):
        forward = _forward

    return Int8GemvLinear


def is_gemv_linear(module: Any) -> bool:
    return type(module).__name__ == "Int8GemvLinear"


def split_outliers_(module: Any, channels: Any) -> int:
    """Before quantize_: move the weight columns of ``channels`` into a dense ``outlier_weight`` buffer and zero them
    in the weight, so ``x @ W.T == x @ W_rest.T + x[:, idx] @ W_out.T`` exactly while the int8 scales only see the
    rest. Returns how many columns moved (0 leaves the module untouched)."""
    import torch

    if channels is None or not len(channels) or len(channels) > MAX_OUTLIERS:
        return 0
    w = module.weight.data
    idx = torch.as_tensor(sorted(int(c) for c in channels), dtype = torch.int64, device = w.device)
    if int(idx.max()) >= w.shape[1]:
        return 0
    module.register_buffer(OUTLIER_WEIGHT, w.index_select(1, idx).contiguous())
    module.register_buffer(OUTLIER_IDX, idx)
    rest = w.clone()
    rest[:, idx] = 0
    module.weight.data = rest
    # swapped now, not at install: from here on only this forward adds the columns back (dense or quantized weight)
    module.__class__ = gemv_linear_class()
    return int(idx.numel())


def prepare_outlier_buffers(transformer: Any, state_dict: Any) -> int:
    """Before ``load_state_dict(strict = True, assign = True)`` of an artifact built with outlier columns: give each
    named Linear placeholder buffers of the stored shapes and the GEMV forward, so the strict load assigns them and the
    forward adds them back. A key naming no Linear is left alone (the strict load then refuses the artifact)."""
    import torch
    from torch import nn

    suffix = "." + OUTLIER_WEIGHT
    count = 0
    for key in [k for k in state_dict if isinstance(k, str) and k.endswith(suffix)]:
        fqn = key[: -len(suffix)]
        idx = state_dict.get(f"{fqn}.{OUTLIER_IDX}")
        try:
            module = transformer.get_submodule(fqn)
        except AttributeError:
            continue
        if idx is None or not isinstance(module, nn.Linear):
            continue
        weight = state_dict[key]
        module.register_buffer(
            OUTLIER_WEIGHT, torch.empty(tuple(weight.shape), dtype = weight.dtype, device = "meta")
        )
        module.register_buffer(
            OUTLIER_IDX, torch.empty(tuple(idx.shape), dtype = idx.dtype, device = "meta")
        )
        module.__class__ = gemv_linear_class()
        count += 1
    return count


def timestep_outlier_channels(transformer: Any, count: int) -> tuple:
    """The ``count`` input channels with the largest |SiLU(temb)| over the whole timestep range, from the model's own
    ``time_text_embed`` (Qwen-Image: the modulations' only input is SiLU of it, a function of the timestep alone, so
    this needs no prompts). () when the transformer has no such embedder or anything fails."""
    if count <= 0:
        return ()
    try:
        import torch

        tte = getattr(transformer, "time_text_embed", None)
        proj = getattr(tte, "time_proj", None)
        embedder = getattr(tte, "timestep_embedder", None)
        if proj is None or embedder is None:
            return ()
        weight = embedder.linear_1.weight
        if getattr(weight, "is_meta", False):
            return ()
        with torch.no_grad():
            sigma = torch.linspace(0.0, 1.0, 1001, device = weight.device)
            emb = embedder(proj(sigma).to(weight.dtype))
            extra = getattr(tte, "addition_t_embedding", None)
            if extra is not None:
                rows = extra.weight.to(emb.dtype)
                emb = torch.cat([emb + r for r in rows], dim = 0)
            score = torch.nn.functional.silu(emb.float()).abs().amax(dim = 0)
        if not bool(torch.isfinite(score).all()) or count > score.numel():
            return ()
        return tuple(sorted(int(i) for i in torch.topk(score, count).indices.tolist()))
    except Exception:  # noqa: BLE001 - calibration is optional: no outliers, plain int8 codes
        return ()


def install(
    transformer: Any,
    fqns: Any,
    *,
    logger: Any = None,
) -> int:
    """Swap the named weight-only Linears onto the GEMV forward (class swap, no hook); returns how many. A Linear
    carrying outlier columns is swapped even with the kernel off: its forward is what adds them back."""
    from torch import nn

    global _OP_HANDLE
    kernel = gemv_enabled() and _kernel() is not None and _op() is not None
    _OP_HANDLE = _op() if kernel else None
    modules = dict(transformer.named_modules())
    count = 0
    with _LOCK:
        for fqn in fqns:
            module = modules.get(fqn)
            if type(module) is not nn.Linear and not is_gemv_linear(module):
                continue
            if weight_only_parts(module.weight) is None or (
                not kernel and _outliers(module)[0] is None
            ):
                continue
            module.__class__ = gemv_linear_class()
            count += 1
    if logger is not None and count:
        logger.info(
            "diffusion.int8_gemv: %s on %d weight-only Linear(s)",
            "fused GEMV" if kernel else "outlier columns (stock GEMM)",
            count,
        )
    return count


def call_count() -> int:
    return COUNTS["gemv"]
