# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""INT8 attention for DiT inference (backend ``unsloth_int8``): Q.K^T and P.V on the int8 tensor cores.

One call is four small Triton passes (per-channel statistics of K and V, then the Q, K and V quantisers) and one CUDA
kernel (``diffusion_int8_attention_cuda``). It takes the strided ``[B, S, H, D]`` views the strided processors already
hold and returns ``[B, S, H, D]``; nothing is copied to a heads-first layout.

How the values are quantised (ideas from the cited papers, written from scratch here):

* Q and K are rotated by one fixed orthogonal map (signs, then a 16-point Hadamard transform per 16 channels) before
  rounding, which leaves Q.K^T unchanged and evens out each row's use of the int8 range (QuaRot, arXiv 2404.00456).
* K is first shifted by a per-channel mean over a sub-sample of the keys: the softmax does not see a shift common to
  all keys (SageAttention, arXiv 2410.02367). Q has one scale per row; K has one scale per group of keys that share
  a tensor-core accumulator lane (SageAttention2, arXiv 2411.10958), so scores are rescaled without a per-key load.
* The probabilities are 8-bit integers too, rescaled per key tile so every tile uses the full range, and the softmax
  denominator is summed over the rounded values (INT-FlashAttention, arXiv 2409.16997).
* V is centred and divided by a per-channel gain (both undone on the output), with one scale per key tile.

The kernel never replaces attention on its own: a strided processor runs it only after ``install_int8_attention``
found the kernel accurate on this card and, unless it was asked for by name, more than 2% faster than what the
processor ran before. Calls it cannot take (masks, other head dims, other devices) run native attention.

``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL``: ``0`` never engages it, ``1`` engages it wherever it is accurate without
the race, unset races it where INT8 attention is already wanted (an INT8 checkpoint) or it was requested by name.
"""

from __future__ import annotations

import ctypes
import math
import os
import threading
import types
from functools import lru_cache
from typing import Any, Callable, Optional

INT8_ATTENTION_ENV = "UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL"
BACKEND = "unsloth_int8"
_OP = "unsloth_studio::int8_attention_nhd"
HEAD_DIMS = (64, 128)
# engaged only when it beats the incumbent by more than this margin on the timed probe (a tie keeps the incumbent)
RACE_MARGIN = 0.98
RACE_TOKENS = (4096, 1024)
# the accuracy every card must reach against fp32 attention on the probe before the kernel may run at all
MIN_COSINE = 0.999
MAX_REL_L1 = 0.05
# the same probe with q and k scaled up, as the sharp attention of late DiT blocks is: every output finite and close
SHARP_GAIN = 5.0
MIN_SHARP_COSINE = 0.99
LOG2E = 1.4426950408889634

# Per CUDA capability and head dim: keys per tile (bn), warps per block (wpc), 16-row query tiles per warp (nq),
# shared-memory tile buffers (stages) and, where the assembler's own register choice costs a block per SM, a register
# cap. Measured per card (see the PR); a card missing here keeps native attention.
KERNEL_CONFIGS: dict = {
    (8, 0): {
        128: {"bn": 64, "wpc": 4, "nq": 1, "stages": 1, "max_registers": 168},
        64: {"bn": 64, "wpc": 4, "nq": 1, "stages": 2},
    },
    (8, 6): {
        128: {"bn": 64, "wpc": 4, "nq": 1, "stages": 1, "max_registers": 168},
        64: {"bn": 64, "wpc": 4, "nq": 1, "stages": 2},
    },
    (8, 9): {
        128: {"bn": 64, "wpc": 6, "nq": 1, "stages": 1, "max_registers": 168},
        64: {"bn": 64, "wpc": 4, "nq": 1, "stages": 1},
    },
    (12, 0): {
        128: {"bn": 64, "wpc": 12, "nq": 1, "stages": 2},
        64: {"bn": 64, "wpc": 8, "nq": 1, "stages": 2},
    },
}
# ROCm runs the Triton form of the kernel (``diffusion_int8_attention_triton``): per gfx arch and head dim, its block
# shape and how P.V is computed. Measured per arch on the real card (see the PR); an arch missing here keeps native
# attention.
ROCM_KERNEL_CONFIGS: dict = {
    "gfx1151": {
        128: {
            "bm": 64, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile",
            "long_keys": 8000, "long": {"bm": 128, "warps": 8},
        },
        64: {"bm": 128, "bn": 32, "warps": 4, "stages": 1, "pv": "half_strided", "kscale": "tile"},
    },
}
QUANT_ROWS = 64  # rows per quantiser block
STATS_STEP = 8  # the K mean and the V range are taken over every 8th key

_LOCK = threading.Lock()
_VERDICTS: dict = {}
_BROKEN: set = set()
_VERIFIED: set = set()
STATS = {"int8": 0, "native": 0}
_OUT_KIND = {"torch.float16": 0, "torch.bfloat16": 1, "torch.float32": 2}


def mode() -> str:
    """``off`` | ``auto`` | ``force`` from ``UNSLOTH_DIFFUSION_INT8_ATTENTION_KERNEL``."""
    raw = (os.environ.get(INT8_ATTENTION_ENV) or "auto").strip().lower()
    if raw in ("0", "off", "false", "no"):
        return "off"
    if raw in ("1", "on", "true", "yes", "force"):
        return "force"
    return "auto"


def requested(value: Any) -> bool:
    """Whether a load's attention request names this backend."""
    return isinstance(value, str) and value.strip().lower() == BACKEND


def kernel_config(capability: Any, head_dim: int) -> Optional[dict]:
    try:
        table = KERNEL_CONFIGS.get((int(capability[0]), int(capability[1])))
    except Exception:  # noqa: BLE001
        return None
    found = (table or {}).get(int(head_dim))
    return dict(found) if found else None


def _is_rocm() -> bool:
    import torch

    from .rocm_bf16 import is_rocm_torch

    return bool(is_rocm_torch(torch))


def _gfx_arch(index: int) -> str:
    import torch

    from .rocm_bf16 import normalize_gfx_arch

    return normalize_gfx_arch(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))


def rocm_kernel_config(arch: Any, head_dim: int) -> Optional[dict]:
    found = (ROCM_KERNEL_CONFIGS.get(str(arch)) or {}).get(int(head_dim))
    return dict(found) if found else None


def device_config(index: int, head_dim: int) -> Optional[dict]:
    """The kernel config for this device and head dim: the CUDA table by capability, the ROCm table by gfx arch."""
    import torch

    if _is_rocm():
        return rocm_kernel_config(_gfx_arch(index), head_dim)
    return kernel_config(torch.cuda.get_device_capability(index), head_dim)


def kernel_defines(head_dim: int, config: dict) -> dict:
    return {
        "HD": int(head_dim),
        "BN": int(config["bn"]),
        "NQ": int(config["nq"]),
        "WPC": int(config["wpc"]),
        "STAGES": int(config["stages"]),
    }


def shared_bytes(head_dim: int, config: dict) -> int:
    bn, wpc, nq = int(config["bn"]), int(config["wpc"]), int(config["nq"])
    stage = bn * (head_dim + 16) + head_dim * (bn + 16)
    return int(config["stages"]) * stage + wpc * 16 * nq * (head_dim + 16)


def _pad(n: int, multiple: int) -> int:
    return (n + multiple - 1) // multiple * multiple


@lru_cache(maxsize = None)
def _quantisers() -> Any:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 -- older Triton layout
        from triton.language.extra.cuda import libdevice

    @triton.jit
    def _rint(x):
        return libdevice.float2int_rn(x)

    @triton.jit
    def _rotate(x, R: tl.constexpr, D: tl.constexpr):
        # [R, D]: fixed signs, then an unnormalised 16-point Hadamard transform inside every 16 channels (a factor 4
        # in norm, taken out of the Q scale)
        cols = tl.arange(0, D)
        flip = (((cols * 7919 + 13) * (cols + 3)) >> 7) & 1
        x = x * (1.0 - 2.0 * flip.to(tl.float32))[None, :]
        for _ in tl.static_range(4):
            a, b = tl.split(tl.reshape(x, (R, D // 2, 2)))
            z = tl.reshape(tl.join(a + b, a - b), (R, D // 16, 8, 2))
            x = tl.reshape(tl.permute(z, (0, 1, 3, 2)), (R, D))
        return x

    @triton.jit
    def stats_kernel(
        x_ptr,
        c_ptr,
        g_ptr,
        n_rows,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        STEP: tl.constexpr,
        R: tl.constexpr,
        D: tl.constexpr,
        MODE: tl.constexpr,
    ):
        # per (batch, head, channel) over every STEP-th token: MODE 0 = mean -> c; MODE 1 = mid-range -> c and half
        # range (>= 1e-12) -> g
        bh = tl.program_id(0)
        b = bh // n_heads
        h = bh % n_heads
        cols = tl.arange(0, D)
        n_sub = (n_rows + STEP - 1) // STEP
        if MODE == 0:
            acc = tl.zeros([D], tl.float32)
        else:
            lo = tl.full([D], 3.0e38, tl.float32)
            hi = tl.full([D], -3.0e38, tl.float32)
        for blk in range(0, tl.cdiv(n_sub, R)):
            rows = (blk * R + tl.arange(0, R)) * STEP
            live = rows < n_rows
            x = tl.load(
                x_ptr
                + b * x_stride_b
                + h * x_stride_h
                + rows[:, None].to(tl.int64) * x_stride_s
                + cols[None, :],
                mask = live[:, None],
                other = 0.0,
            ).to(tl.float32)
            if MODE == 0:
                acc += tl.sum(x, axis = 0)
            else:
                lo = tl.minimum(lo, tl.min(tl.where(live[:, None], x, 3.0e38), axis = 0))
                hi = tl.maximum(hi, tl.max(tl.where(live[:, None], x, -3.0e38), axis = 0))
        if MODE == 0:
            tl.store(c_ptr + bh * D + cols, acc / n_sub)
        else:
            tl.store(c_ptr + bh * D + cols, 0.5 * (hi + lo))
            tl.store(g_ptr + bh * D + cols, tl.maximum(0.5 * (hi - lo), 1e-12))

    @triton.jit
    def quant_q_kernel(
        x_ptr,
        q_ptr,
        s_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        mul,
        R: tl.constexpr,
        D: tl.constexpr,
    ):
        # R rows of one (batch, head): int8 = rint(rotated x * 127 / amax(row)), scale[row] = amax / 127 * mul;
        # padded rows are zeros with scale 0
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
        rows = blk * R + tl.arange(0, R)
        cols = tl.arange(0, D)
        live = rows < n_rows
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = live[:, None],
            other = 0.0,
        ).to(tl.float32)
        x = _rotate(x, R, D)
        amax = tl.maximum(tl.max(tl.abs(x), axis = 1), 1e-20)
        base = bh.to(tl.int64) * n_pad
        tl.store(s_ptr + base + rows, tl.where(live, amax * (mul / 127.0), 0.0))
        tl.store(
            q_ptr + (base + rows[:, None]) * D + cols[None, :],
            _rint(x * (127.0 / amax)[:, None]).to(tl.int8),
        )

    @triton.jit
    def quant_k_kernel(
        x_ptr,
        c_ptr,
        q_ptr,
        s_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
    ):
        # one tile of T keys of one (batch, head), centred and rotated: the keys with the same (key % 8) // 2 share a
        # scale (the lanes of one tensor-core accumulator register). scale [BH, n_pad / T, 4]
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
        rows = blk * T + tl.arange(0, T)
        cols = tl.arange(0, D)
        live = rows < n_rows
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = live[:, None],
            other = 0.0,
        ).to(tl.float32)
        c = tl.load(c_ptr + bh * D + cols)
        x = _rotate(tl.where(live[:, None], x - c[None, :], 0.0), T, D)
        amax = tl.max(tl.abs(x), axis = 1)
        grp = tl.maximum(tl.max(tl.max(tl.reshape(amax, (T // 8, 4, 2)), axis = 2), axis = 0), 1e-20)
        per_row = tl.reshape(tl.broadcast_to(grp[None, :, None], (T // 8, 4, 2)), (T,))
        base = bh.to(tl.int64) * n_pad
        tl.store(s_ptr + (bh.to(tl.int64) * (n_pad // T) + blk) * 4 + tl.arange(0, 4), grp / 127.0)
        tl.store(
            q_ptr + (base + rows[:, None]) * D + cols[None, :],
            _rint(x * (127.0 / per_row)[:, None]).to(tl.int8),
        )

    @triton.jit
    def quant_v_kernel(
        x_ptr,
        c_ptr,
        g_ptr,
        q_ptr,
        s_ptr,
        n_rows,
        n_pad,
        n_heads,
        x_stride_b,
        x_stride_s,
        x_stride_h,
        T: tl.constexpr,
        D: tl.constexpr,
    ):
        # one tile of T keys of one (batch, head): (x - centre) / gain per channel, one scale for the tile, stored
        # channel-major [BH, n_pad / T, D, T] in the key order the kernel's operand registers want: position
        # 4 * t + c of every 16 keys holds key 8 * (c >> 1) + 2 * t + (c & 1)
        bh = tl.program_id(0)
        blk = tl.program_id(1)
        b = bh // n_heads
        h = bh % n_heads
        pos = tl.arange(0, T)
        rows = blk * T + (pos & -16) + 8 * ((pos & 3) >> 1) + 2 * ((pos >> 2) & 3) + (pos & 1)
        cols = tl.arange(0, D)
        live = rows < n_rows
        x = tl.load(
            x_ptr
            + b * x_stride_b
            + h * x_stride_h
            + rows[:, None].to(tl.int64) * x_stride_s
            + cols[None, :],
            mask = live[:, None],
            other = 0.0,
        ).to(tl.float32)
        c = tl.load(c_ptr + bh * D + cols)
        g = tl.load(g_ptr + bh * D + cols)
        x = tl.where(live[:, None], (x - c[None, :]) / g[None, :], 0.0)
        amax = tl.maximum(tl.max(tl.max(tl.abs(x), axis = 1), axis = 0), 1e-20)
        tile = bh.to(tl.int64) * (n_pad // T) + blk
        tl.store(s_ptr + tile, amax / 127.0)
        tl.store(
            q_ptr + (tile * D + cols[:, None]) * T + pos[None, :],
            tl.trans(_rint(x * (127.0 / amax)).to(tl.int8)),
        )

    return types.SimpleNamespace(
        stats = stats_kernel, quant_q = quant_q_kernel, quant_k = quant_k_kernel, quant_v = quant_v_kernel
    )


def quantise(
    q: Any,
    k: Any,
    v: Any,
    config: dict,
    scale: Optional[float] = None,
) -> dict:
    """The kernel's inputs from ``[B, S, H, D]`` q / k / v (any strides on the first three dims)."""
    import torch

    kernels = _quantisers()
    B, Sq, H, D = q.shape
    Sk = k.shape[1]
    bn, wpc, nq = int(config["bn"]), int(config["wpc"]), int(config["nq"])
    dev = q.device
    # log2(e) turns the kernel's 2^x into e^x; 16 is the norm the rotation adds to Q and to K
    mul = (1.0 / math.sqrt(D) if scale is None else float(scale)) * LOG2E / 16.0
    n_qpad = _pad(Sq, math.lcm(QUANT_ROWS, 16 * nq * wpc))
    n_kpad = _pad(Sk, bn)
    tiles = n_kpad // bn
    BH = B * H

    def strides(x):
        return x.stride(0), x.stride(1), x.stride(2)

    qi = torch.empty((BH, n_qpad, D), dtype = torch.int8, device = dev)
    qs = torch.empty((BH, n_qpad), dtype = torch.float32, device = dev)
    kernels.quant_q[(BH, n_qpad // QUANT_ROWS)](
        q, qi, qs, Sq, n_qpad, H, *strides(q), mul, R = QUANT_ROWS, D = D
    )
    kc = torch.empty((BH, D), dtype = torch.float32, device = dev)
    kernels.stats[(BH,)](k, kc, kc, Sk, H, *strides(k), STEP = STATS_STEP, R = 32, D = D, MODE = 0)
    ki = torch.empty((BH, n_kpad, D), dtype = torch.int8, device = dev)
    ks = torch.empty((BH, tiles, 4), dtype = torch.float32, device = dev)
    kernels.quant_k[(BH, tiles)](k, kc, ki, ks, Sk, n_kpad, H, *strides(k), T = bn, D = D)
    vc = torch.empty((BH, D), dtype = torch.float32, device = dev)
    vg = torch.empty((BH, D), dtype = torch.float32, device = dev)
    kernels.stats[(BH,)](v, vc, vg, Sk, H, *strides(v), STEP = STATS_STEP, R = 32, D = D, MODE = 1)
    vt = torch.empty((BH, tiles, D, bn), dtype = torch.int8, device = dev)
    vs = torch.empty((BH, tiles), dtype = torch.float32, device = dev)
    kernels.quant_v[(BH, tiles)](v, vc, vg, vt, vs, Sk, n_kpad, H, *strides(v), T = bn, D = D)
    return {
        "qi": qi,
        "qs": qs,
        "ki": ki,
        "ks": ks,
        "kc": kc,
        "vt": vt,
        "vs": vs,
        "vc": vc,
        "vg": vg,
        "shape": (B, Sq, Sk, H, D),
        "dtype": v.dtype,
        "config": config,
    }


def kernel_function(device: Any, head_dim: int, config: dict) -> Any:
    """The loaded CUDA kernel for this device and config (compiled on first use)."""
    import torch

    from . import diffusion_int8_attention_cuda as cuda_kernel

    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    return cuda_kernel.load(
        index,
        tuple(torch.cuda.get_device_capability(index)),
        kernel_defines(head_dim, config),
        config.get("max_registers"),
        shared_bytes(head_dim, config),
    )


def attend(p: dict) -> Any:
    """Launch the kernel on ``quantise``'s output; ``[B, Sq, H, D]`` in the dtype of V."""
    import torch

    config = p["config"]
    B, Sq, Sk, H, D = p["shape"]
    dev = p["qi"].device
    out = torch.empty((B, Sq, H, D), dtype = p["dtype"], device = dev)
    fn = kernel_function(dev, D, config)
    n_qpad, n_kpad = p["qi"].shape[1], p["ki"].shape[1]
    wpc, nq = int(config["wpc"]), int(config["nq"])
    P, I, L = ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong
    args = [
        P(p["qi"].data_ptr()),
        P(p["ki"].data_ptr()),
        P(p["vt"].data_ptr()),
        P(p["qs"].data_ptr()),
        P(p["ks"].data_ptr()),
        P(p["vs"].data_ptr()),
        P(p["vg"].data_ptr()),
        P(p["vc"].data_ptr()),
        P(out.data_ptr()),
        I(Sq),
        I(n_qpad),
        I(Sk),
        I(n_kpad),
        I(H),
        L(out.stride(0)),
        L(out.stride(1)),
        L(out.stride(2)),
        I(_OUT_KIND[str(out.dtype)]),
    ]
    with torch.cuda.device(dev):
        fn(
            (n_qpad // (16 * nq * wpc), B * H),
            32 * wpc,
            args,
            shared_bytes(D, config),
            torch.cuda.current_stream().cuda_stream,
        )
    return out


def unsupported_reason(q: Any, k: Any, v: Any) -> Optional[str]:
    """Why this call cannot run the kernel (it then runs native attention), or None."""
    import torch

    if not (q.is_cuda and k.is_cuda and v.is_cuda) or not (q.device == k.device == v.device):
        return "not on one CUDA device"
    if q.dim() != 4 or k.dim() != 4 or v.shape != k.shape:
        return "not [B, S, H, D]"
    B, Sq, H, D = q.shape
    if k.shape[0] != B or k.shape[2] != H or k.shape[3] != D or Sq < 1 or k.shape[1] < 1:
        return "q / k / v shapes differ"
    if D not in HEAD_DIMS:
        return f"head dim {D}"
    if (
        str(v.dtype) not in _OUT_KIND
        or str(q.dtype) not in _OUT_KIND
        or str(k.dtype) not in _OUT_KIND
    ):
        return "dtype"
    if B * H > 65535:
        return "batch * heads"
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        return "needs gradients"
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    if device_config(index, D) is None:
        return "no kernel config for this card"
    if (index, D) in _BROKEN:
        return "kernel failed on this card"
    return None


def _verified(q: Any) -> bool:
    """Whether the accuracy check passed for this call's card and head dim (the op runs the kernel only then)."""
    import torch

    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    return (index, int(q.shape[-1])) in _VERIFIED


def int8_attention(
    q: Any,
    k: Any,
    v: Any,
    scale: Optional[float] = None,
    config: Optional[dict] = None,
) -> Any:
    """``softmax(q k^T * scale) v`` with int8 products. ``[B, Sq, H, D]``, ``[B, Sk, H, D]`` x 2 -> ``[B, Sq, H, D]``
    in the dtype of V. Raises where ``unsupported_reason`` says so; ``int8_attention_impl`` is the guarded form."""
    import torch

    reason = unsupported_reason(q, k, v)
    if reason is not None:
        raise ValueError(f"INT8 attention cannot take this call: {reason}")
    index = q.device.index if q.device.index is not None else torch.cuda.current_device()
    config = config or device_config(index, q.shape[-1])
    # the quantisers address the last dim with stride 1
    q, k, v = (x if x.stride(3) == 1 else x.contiguous() for x in (q, k, v))
    if _is_rocm():
        from . import diffusion_int8_attention_triton as triton_kernel

        with torch.no_grad(), torch.cuda.device(q.device):
            return triton_kernel.int8_attention(q, k, v, config, scale = scale)
    with torch.no_grad():
        return attend(quantise(q, k, v, config, scale))


def native_attention(q: Any, k: Any, v: Any) -> Any:
    """torch's attention on the same ``[B, S, H, D]`` views."""
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(
        q.to(v.dtype).permute(0, 2, 1, 3), k.to(v.dtype).permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    )
    return out.permute(0, 2, 1, 3)


def int8_attention_impl(q: Any, k: Any, v: Any) -> Any:
    """The kernel where it can run, native attention elsewhere and after any failure."""
    import torch

    if unsupported_reason(q, k, v) is None and _verified(q):
        try:
            out = int8_attention(q, k, v)
            STATS["int8"] += 1
            return out
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 -- a JIT / launch failure keeps native attention on this card
            index = q.device.index if q.device.index is not None else torch.cuda.current_device()
            _BROKEN.add((index, int(q.shape[-1])))
            _log().warning(
                "diffusion.int8_attention: kernel failed, native attention from here on: %s", exc
            )
    STATS["native"] += 1
    return native_attention(q, k, v)


def _log() -> Any:
    import logging
    return logging.getLogger(__name__)


@lru_cache(maxsize = None)
def int8_attention_op() -> Any:
    """Registered once per process: opaque to torch.compile, with a fake for shape propagation."""
    import torch

    # explicit schema: the annotations here are strings
    @torch.library.custom_op(
        _OP, mutates_args = (), schema = "(Tensor q, Tensor k, Tensor v) -> Tensor"
    )
    def int8_attention_nhd(q, k, v):
        return int8_attention_impl(q, k, v)

    @int8_attention_nhd.register_fake
    def _(q, k, v):
        return v.new_empty(q.shape)

    return int8_attention_nhd


def reference_attention(q: Any, k: Any, v: Any) -> Any:
    """fp32 attention on ``[B, S, H, D]``: what the kernel is checked against."""
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(
        q.float().permute(0, 2, 1, 3), k.float().permute(0, 2, 1, 3), v.float().permute(0, 2, 1, 3)
    )
    return out.permute(0, 2, 1, 3)


def probe_inputs(
    device: Any,
    dtype: Any,
    tokens: int,
    heads: int,
    head_dim: int,
    seed: int = 0,
) -> tuple:
    """q / k like RMS-normalised projections with a learned gain, v with per-channel spread and offset."""
    import torch

    g = torch.Generator(device = "cpu").manual_seed(1000 + seed)
    shape = (1, tokens, heads, head_dim)
    q = torch.randn(shape, generator = g) * 1.6
    k = torch.randn(shape, generator = g) * 1.6 + 0.4 * torch.randn(
        (1, 1, heads, head_dim), generator = g
    )
    v = torch.randn(shape, generator = g) * (0.5 + torch.rand((1, 1, heads, head_dim), generator = g))
    v = v + 0.5 * torch.randn((1, 1, heads, head_dim), generator = g)
    return tuple(x.to(device = device, dtype = dtype) for x in (q, k, v))


def accuracy(got: Any, ref: Any) -> dict:
    import torch.nn.functional as F
    got, ref = got.float(), ref.float()
    return {
        "cos": float(F.cosine_similarity(got.flatten(), ref.flatten(), dim = 0)),
        "rel_l1": float((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-30)),
        "max_abs": float((got - ref).abs().max()),
        "mean_abs": float((got - ref).abs().mean()),
    }


def _median_ms(fns: dict, rounds: int = 5) -> dict:
    """Median milliseconds per callable, timed with CUDA events in interleaved rounds after two warm calls each."""
    import torch

    for fn in fns.values():
        fn()
        fn()
    times: dict = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            a.record()
            fn()
            b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b))
    return {name: sorted(ts)[len(ts) // 2] for name, ts in times.items()}


def _check(
    device: Any, dtype: Any, heads: int, head_dim: int, incumbent: Optional[Callable], timed: bool
) -> dict:
    """``ok``: the kernel runs here and matches fp32 attention on the probe, also with q and k scaled up. ``ms``: int8 and incumbent milliseconds
    per call at each RACE_TOKENS size (when ``timed``). ``min_tokens``: the sequence length from which the kernel
    is more than 2% faster than the incumbent, or None when it is not at the larger size."""
    import torch

    out: dict = {"ok": False, "ms": {}, "min_tokens": None}
    try:
        with torch.no_grad():
            q, k, v = probe_inputs(device, dtype, 1000, min(int(heads), 4), head_dim)
            acc = accuracy(int8_attention(q, k, v), reference_attention(q, k, v))
            q, k = q * SHARP_GAIN, k * SHARP_GAIN
            sharp = int8_attention(q, k, v)
            finite = bool(torch.isfinite(sharp).all())
            sharp_cos = accuracy(sharp, reference_attention(q, k, v))["cos"] if finite else 0.0
    except torch.OutOfMemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 -- no kernel on this card
        out["error"] = repr(exc)[:300]
        return out
    out.update(acc)
    out["sharp_cos"] = sharp_cos
    out["ok"] = bool(
        acc["cos"] >= MIN_COSINE and acc["rel_l1"] <= MAX_REL_L1 and sharp_cos >= MIN_SHARP_COSINE
    )
    if out["ok"]:
        dev = torch.device(device)
        _VERIFIED.add(
            (dev.index if dev.index is not None else torch.cuda.current_device(), int(head_dim))
        )
    if not out["ok"] or not timed:
        return out
    wins = {}
    for tokens in RACE_TOKENS:
        with torch.no_grad():
            q, k, v = probe_inputs(device, dtype, tokens, heads, head_dim, seed = tokens)
            fns = {
                "incumbent": (lambda: incumbent(q, k, v))
                if incumbent
                else (lambda: native_attention(q, k, v)),
                "int8": lambda: int8_attention(q, k, v),
            }
            try:
                ms = _median_ms(fns)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 -- an incumbent that cannot be timed is kept
                out["error"] = repr(exc)[:300]
                return out
        out["ms"][tokens] = {name: round(value, 4) for name, value in ms.items()}
        wins[tokens] = ms["int8"] < RACE_MARGIN * ms["incumbent"]
    large, small = RACE_TOKENS
    if wins[large]:
        out["min_tokens"] = 0 if wins[small] else (large + small) // 2
    return out


def verdict(
    device: Any,
    dtype: Any,
    heads: int,
    head_dim: int,
    incumbent: Optional[Callable] = None,
    incumbent_name: str = "native",
    timed: bool = True,
) -> dict:
    """``_check`` cached per (device, dtype, heads, head dim, incumbent, timed)."""
    import torch

    dev = torch.device(device)
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    key = (index, str(dtype), int(heads), int(head_dim), incumbent_name, bool(timed))
    found = _VERDICTS.get(key)
    if found is not None:
        return found
    with _LOCK:
        if key not in _VERDICTS:
            with torch.cuda.device(index):
                _VERDICTS[key] = _check(
                    torch.device("cuda", index), dtype, int(heads), int(head_dim), incumbent, timed
                )
    return _VERDICTS[key]


def _head_dim(attn: Any) -> Optional[int]:
    value = getattr(attn, "head_dim", None)
    if isinstance(value, int):
        return value
    try:
        return int(attn.to_out[0].in_features) // int(attn.heads)
    except Exception:  # noqa: BLE001
        return None


def int8_processors(transformer: Any) -> list:
    """The attention modules whose processor can route to the kernel (the strided processors)."""
    return [
        m
        for m in transformer.modules()
        if hasattr(type(getattr(m, "processor", None)), "_unsloth_int8") and hasattr(m, "heads")
    ]


def _sdpa_incumbent(backend: Any) -> tuple:
    """(name, callable) for a processor's native backend: torch attention pinned the way the processor pins it."""
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    value = str(getattr(backend, "value", backend)) if backend is not None else "native"
    kernel = {
        "_native_cudnn": SDPBackend.CUDNN_ATTENTION,
        "_native_flash": SDPBackend.FLASH_ATTENTION,
    }.get(value)
    if kernel is None:
        return "native", native_attention

    def run(q: Any, k: Any, v: Any) -> Any:
        with sdpa_kernel(kernel):
            out = F.scaled_dot_product_attention(
                q.to(v.dtype).permute(0, 2, 1, 3),
                k.to(v.dtype).permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3),
            )
        return out.permute(0, 2, 1, 3)

    return value, run


def install_int8_attention(
    transformer: Any,
    *,
    device: Any,
    dtype: Any,
    modules: Optional[list] = None,
    incumbent: Optional[tuple] = None,
    force: bool = False,
    logger: Any = None,
) -> int:
    """Route strided attention processors to the INT8 kernel. Returns how many now run it.

    ``modules`` limits the attention modules considered (default: every one with a strided processor).
    ``incumbent`` is ``(name, callable(q, k, v))`` for what those modules run now (default: their native backend).
    The kernel must match fp32 attention on this card; unless ``force`` (or the env switch is ``1``) it must also be
    more than 2% faster than the incumbent on the timed probe. Never raises."""
    try:
        return _install(transformer, device, dtype, modules, incumbent, force, logger)
    except Exception as exc:  # noqa: BLE001 -- an optimisation: the processors stay as they were
        if logger is not None:
            logger.info("diffusion.int8_attention: skipped (%s)", exc)
        return 0


def _install(
    transformer: Any,
    device: Any,
    dtype: Any,
    modules: Optional[list],
    incumbent: Optional[tuple],
    force: bool,
    logger: Any,
) -> int:
    import torch

    setting = mode()
    if transformer is None or setting == "off":
        return 0
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return 0
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    if str(dtype) not in _OUT_KIND:
        return 0
    candidates = [
        m for m in int8_processors(transformer) if modules is None or any(m is x for x in modules)
    ]
    if not candidates:
        return 0
    force = force or setting == "force"
    int8_attention_op()
    engaged = 0
    report: dict = {}
    for m in candidates:
        proc = m.processor
        head_dim = _head_dim(m)
        if head_dim is None or device_config(index, head_dim) is None:
            continue
        if getattr(proc, "_parallel_config", None) is not None:
            continue
        name, fn = (
            incumbent
            if incumbent is not None
            else _sdpa_incumbent(getattr(proc, "_attention_backend", None))
        )
        found = verdict(dev, dtype, int(m.heads), head_dim, fn, name, timed = not force)
        report[(int(m.heads), head_dim, name)] = found
        if not found.get("ok"):
            continue
        if force:
            minimum = 0
        elif found.get("min_tokens") is None:
            continue
        else:
            minimum = int(found["min_tokens"])
        proc._unsloth_int8 = True
        proc._unsloth_int8_min_tokens = minimum
        engaged += 1
    if logger is not None:
        for (heads, head_dim, name), found in report.items():
            logger.info(
                "diffusion.int8_attention: heads=%d head_dim=%d vs %s: ok=%s cos=%s ms=%s min_tokens=%s%s",
                heads,
                head_dim,
                name,
                found.get("ok"),
                found.get("cos"),
                found.get("ms"),
                found.get("min_tokens"),
                " (forced)" if force else "",
            )
        if engaged:
            logger.info("diffusion.int8_attention: %s on %d attention modules", BACKEND, engaged)
    return engaged


def install_requested(
    pipe: Any,
    request: Any,
    target: Any,
    logger: Any = None,
) -> int:
    """For ``apply_attention_backend``: engage on the pipeline's DiT when the load asked for ``unsloth_int8`` by name
    or the env switch forces it (no race either way: the accuracy check still gates it). Returns the count."""
    import torch

    setting = mode()
    if setting == "off" or not (setting == "force" or requested(request)):
        return 0
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return 0
    device = str(getattr(target, "torch_device", None) or getattr(target, "device", "") or "")
    if not device.startswith("cuda"):
        return 0
    dtype = getattr(target, "dtype", None)
    if str(dtype) not in _OUT_KIND:
        dtype = torch.bfloat16
    try:
        from .diffusion_flux3_attn import ensure_strided_processors
        ensure_strided_processors(pipe, target, logger)
    except Exception:  # noqa: BLE001 -- other families bring their own strided processors
        pass
    return install_int8_attention(
        transformer, device = device, dtype = dtype, force = True, logger = logger
    )


def remove_int8_attention(transformer: Any) -> int:
    """Every processor back to the attention it ran before ``install_int8_attention``."""
    count = 0
    for m in int8_processors(transformer) if transformer is not None else ():
        if getattr(m.processor, "_unsloth_int8", False):
            m.processor._unsloth_int8 = False
            count += 1
    return count


def int8_attention_count(transformer: Any) -> int:
    """Engagement census: attention modules whose processor routes to the kernel."""
    if transformer is None:
        return 0
    return sum(
        1 for m in int8_processors(transformer) if getattr(m.processor, "_unsloth_int8", False)
    )


def status(transformer: Any = None) -> dict:
    """What the status panel and the logs report."""
    return {
        "backend": BACKEND,
        "mode": mode(),
        "modules": int8_attention_count(transformer),
        "calls": dict(STATS),
        "verdicts": {
            f"cuda:{k[0]} {k[1]} heads={k[2]} head_dim={k[3]} vs {k[4]}": {
                x: v.get(x)
                for x in ("ok", "cos", "rel_l1", "sharp_cos", "ms", "min_tokens", "error")
                if x in v
            }
            for k, v in _VERDICTS.items()
        },
    }
