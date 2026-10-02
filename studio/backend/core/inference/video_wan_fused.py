# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan transformer block modulation as fused Triton kernels on fp16 GPUs (T4 and other pre-Ampere cards).

diffusers' ``WanTransformerBlock.forward`` keeps the norm, AdaLN modulation and gated residual math in float32:
``scale_shift_table + temb.float()`` materialises a float32 ``[B, L, 6, D]`` tensor per block (Wan2.2-TI2V-5B
modulates per token: 454 MB per block at 1280x704x25), and every norm / modulation / residual step round-trips through
float32 tensors with casts on both sides. On a T4 that is ~2.3 s of a ~11 s denoise step and ~3,700 kernel launches.

This runs the same arithmetic per element without the float32 intermediates in memory:

* LayerNorm statistics come from PyTorch's own kernel run on the fp16 input. Its vectorised kernel reads 4 elements per
  thread for every dtype ("that would lead to different results between float and low-p types"), so the per-row mean
  and rstd are bit-identical to the stock ``layer_norm(x.float())``.
* One Triton kernel then applies ``rstd * (x - mean)``, the ``1 + (table + temb)`` scale and ``table + temb`` shift
  (read straight from the fp16 ``temb`` and the fp32 table), each product and sum rounded in float32 exactly where the
  stock op chain rounds (FMA contraction disabled), and rounds to fp16 once, where ``type_as`` does.
* The gated residuals ``x.float() + branch * (table + temb)`` and the affine ``norm2`` are fused the same way.

So the fp16 block output is bit-identical to stock; a self-check against the stock forward runs once per device and
shape class before the fused path is trusted. Scope: only ``WanTransformerBlock`` (class-level forward, fingerprinted),
only fp16 CUDA (non-ROCm) tensors without grad, outside ``torch.compile``; everything else calls the stock forward.
Installed only for an fp16 Wan load (bf16 GPUs keep stock). Kill switch: ``UNSLOTH_DIFFUSION_WAN_FUSED_ADALN=0``.
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable, Optional

WAN_FUSED_ENV = "UNSLOTH_DIFFUSION_WAN_FUSED_ADALN"
_MODULE = "diffusers.models.transformers.transformer_wan"
_CLASS = "WanTransformerBlock"
# ``_digest(WanTransformerBlock.forward)``: identical in diffusers 0.36.0, 0.37.0, 0.40.0 and main (80c7ed26).
_FINGERPRINTS = frozenset({"69aa0565fb00eeae"})

_LOCK = threading.Lock()
_STATE: dict = {}
# (device index, temb rank, has norm2 affine) -> True once the fused block matched the stock block bit for bit
_VERIFIED: dict = {}
# fused block calls since install (engagement evidence for tests and the A/B harness)
_COUNTS = {"fused": 0, "stock": 0}


def disabled() -> bool:
    return (os.environ.get(WAN_FUSED_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def wanted(dtype: Any, device: Any = "cuda") -> bool:
    """Only an fp16 CUDA (non-ROCm) load with Triton available: bf16 / fp32 loads keep the stock forward."""
    if disabled():
        return False
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if dtype is not torch.float16 or not str(device).startswith("cuda"):
        return False
    if getattr(getattr(torch, "version", None), "hip", None):
        return False
    return _kernels() is not None


@functools.lru_cache(maxsize = 1)
def _kernels() -> Optional[dict]:
    """Build the Triton kernels once; None when Triton is unavailable (the stock forward then stays)."""
    try:
        import triton
        import triton.language as tl
    except Exception:  # noqa: BLE001
        return None

    @triton.jit
    def _mul_rn(a, b):
        # A product rounded on its own. ``enable_fp_fusion=False`` alone is not enough: ptxas still contracts
        # ``mul.rn.f32x2`` + ``add.rn.f32x2`` into FFMA2 on sm_100, so the multiply is opaque inline PTX.
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype = tl.float32, is_pure = True, pack = 1
        )

    @triton.jit
    def _modnorm(
        X,
        MEAN,
        RSTD,
        T,
        TBL,
        OUT,
        D,
        L,
        stb,
        stl,
        stk,
        sbk,
        K_SHIFT: tl.constexpr,
        K_SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # out = fp16((rstd * (x - mean)) * (1 + (tbl[scale] + temb[scale])) + (tbl[shift] + temb[shift]))
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        b = row // L
        l = row - b * L
        x = tl.load(X + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        mean = tl.load(MEAN + row)
        rstd = tl.load(RSTD + row)
        trow = T + b * stb + l * stl + col
        t_sh = tl.load(trow + K_SHIFT * stk, mask = mask, other = 0.0).to(tl.float32)
        t_sc = tl.load(trow + K_SCALE * stk, mask = mask, other = 0.0).to(tl.float32)
        g_sh = tl.load(TBL + K_SHIFT * sbk + col, mask = mask, other = 0.0).to(tl.float32)
        g_sc = tl.load(TBL + K_SCALE * sbk + col, mask = mask, other = 0.0).to(tl.float32)
        ln = _mul_rn(rstd, x - mean)
        scale = 1.0 + (g_sc + t_sc)
        shift = g_sh + t_sh
        y = _mul_rn(ln, scale)
        y = y + shift
        tl.store(OUT + row * D + col, y.to(OUT.dtype.element_ty), mask = mask)

    @triton.jit
    def _gate_residual(
        X,
        A,
        T,
        TBL,
        OUT,
        D,
        L,
        stb,
        stl,
        stk,
        sbk,
        K_GATE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        # out = fp16(x.float() + a.float() * (tbl[gate] + temb[gate]))
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        b = row // L
        l = row - b * L
        x = tl.load(X + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        a = tl.load(A + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        t_g = tl.load(T + b * stb + l * stl + K_GATE * stk + col, mask = mask, other = 0.0).to(tl.float32)
        g_g = tl.load(TBL + K_GATE * sbk + col, mask = mask, other = 0.0).to(tl.float32)
        gate = g_g + t_g
        y = _mul_rn(a, gate)
        y = x + y
        tl.store(OUT + row * D + col, y.to(OUT.dtype.element_ty), mask = mask)

    @triton.jit
    def _affine_norm(
        X,
        MEAN,
        RSTD,
        W,
        BIAS,
        OUT,
        D,
        BLOCK: tl.constexpr,
    ):
        # out = fp16(fma(w, rstd * (x - mean), b)): PyTorch's vectorised kernel writes ``gamma * (...) + beta``, which
        # nvcc contracts to one fma; checked bit for bit by the self-check.
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        x = tl.load(X + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        mean = tl.load(MEAN + row)
        rstd = tl.load(RSTD + row)
        w = tl.load(W + col, mask = mask, other = 0.0).to(tl.float32)
        bb = tl.load(BIAS + col, mask = mask, other = 0.0).to(tl.float32)
        y = tl.fma(w, _mul_rn(rstd, x - mean), bb)
        tl.store(OUT + row * D + col, y.to(OUT.dtype.element_ty), mask = mask)

    block = 1024

    def _grid(rows: int, d: int) -> tuple:
        return (rows, triton.cdiv(d, block))

    def _temb_strides(temb: Any) -> tuple:
        # [B, L, 6, D] (Wan2.2-TI2V per-token) or [B, 6, D] (one modulation per sample: L stride 0)
        if temb.dim() == 4:
            return temb.stride(0), temb.stride(1), temb.stride(2)
        return temb.stride(0), 0, temb.stride(1)

    def modnorm(x, mean, rstd, temb, table, k_shift: int, k_scale: int):
        import torch

        B, L, D = x.shape
        out = torch.empty_like(x)
        stb, stl, stk = _temb_strides(temb)
        _modnorm[_grid(B * L, D)](
            x,
            mean,
            rstd,
            temb,
            table,
            out,
            D,
            L,
            stb,
            stl,
            stk,
            table.stride(-2),
            K_SHIFT = k_shift,
            K_SCALE = k_scale,
            BLOCK = block,
            num_warps = 4,
            enable_fp_fusion = False,
        )
        return out

    def gate_residual(x, a, temb, table, k_gate: int):
        import torch

        B, L, D = x.shape
        out = torch.empty_like(x)
        stb, stl, stk = _temb_strides(temb)
        _gate_residual[_grid(B * L, D)](
            x,
            a,
            temb,
            table,
            out,
            D,
            L,
            stb,
            stl,
            stk,
            table.stride(-2),
            K_GATE = k_gate,
            BLOCK = block,
            num_warps = 4,
            enable_fp_fusion = False,
        )
        return out

    def affine_norm(x, mean, rstd, weight, bias):
        import torch

        B, L, D = x.shape
        out = torch.empty_like(x)
        _affine_norm[_grid(B * L, D)](
            x,
            mean,
            rstd,
            weight,
            bias,
            out,
            D,
            BLOCK = block,
            num_warps = 4,
            enable_fp_fusion = False,
        )
        return out

    return {"modnorm": modnorm, "gate_residual": gate_residual, "affine_norm": affine_norm}


def _stats(x: Any, eps: float) -> tuple:
    """Per-row (mean, rstd) from PyTorch's own layer norm on the fp16 rows: bit-identical to ``layer_norm(x.float())``."""
    import torch

    D = x.shape[-1]
    _, mean, rstd = torch.native_layer_norm(x, (D,), None, None, eps)
    return mean.reshape(-1).contiguous(), rstd.reshape(-1).contiguous()


def _plain_fp32_norm(norm: Any) -> bool:
    """diffusers' FP32LayerNorm over the last dim without affine params (norm1 / norm3)."""
    return (
        type(norm).__name__ == "FP32LayerNorm"
        and getattr(norm, "weight", None) is None
        and getattr(norm, "bias", None) is None
        and len(tuple(getattr(norm, "normalized_shape", ()))) == 1
    )


def _affine_fp32_norm(norm: Any, dim: int) -> bool:
    import torch

    w, b = getattr(norm, "weight", None), getattr(norm, "bias", None)
    return (
        type(norm).__name__ == "FP32LayerNorm"
        and len(tuple(getattr(norm, "normalized_shape", ()))) == 1
        and torch.is_tensor(w)
        and torch.is_tensor(b)
        and w.dtype in (torch.float32, torch.float16)
        and b.dtype in (torch.float32, torch.float16)
        and tuple(w.shape) == (dim,)
        and tuple(b.shape) == (dim,)
        and w.is_contiguous()
        and b.is_contiguous()
    )


def _eligible(block: Any, x: Any, enc: Any, temb: Any) -> bool:
    import torch

    if not torch.is_tensor(x) or x.dtype is not torch.float16 or not x.is_cuda or x.dim() != 3:
        return False
    if torch.is_grad_enabled() and (x.requires_grad or any(p.requires_grad for p in block.parameters(recurse = False))):
        return False
    if torch.compiler.is_compiling():
        return False
    B, L, D = x.shape
    if D % 4 or not x.is_contiguous() or x.numel() == 0:
        return False
    table = getattr(block, "scale_shift_table", None)
    # float32 as diffusers keeps it (_keep_in_fp32_modules); an fp16 table upcasts exactly in both paths
    if not torch.is_tensor(table) or table.dtype not in (torch.float32, torch.float16) or tuple(table.shape) != (1, 6, D):
        return False
    if table.device != x.device or table.stride(-1) != 1:
        return False
    if not torch.is_tensor(temb) or temb.device != x.device or temb.stride(-1) != 1:
        return False
    if temb.dtype not in (torch.float16, torch.float32):
        return False
    if temb.dim() == 4:
        if tuple(temb.shape) != (B, L, 6, D):
            return False
    elif temb.dim() == 3:
        if tuple(temb.shape) != (B, 6, D):
            return False
    else:
        return False
    if not (_plain_fp32_norm(block.norm1) and _plain_fp32_norm(block.norm3)):
        return False
    norm2 = block.norm2
    if not (isinstance(norm2, torch.nn.Identity) or _affine_fp32_norm(norm2, D)):
        return False
    return True


def _fused_forward(block: Any, hidden_states: Any, encoder_hidden_states: Any, temb: Any, rotary_emb: Any) -> Any:
    """The stock block, step for step, with the float32 elementwise chains replaced by the kernels above."""
    import torch

    k = _kernels()
    table = block.scale_shift_table
    x = hidden_states
    # 1. self-attention: norm1 + (shift, scale) = chunks 0, 1; gate = chunk 2
    mean, rstd = _stats(x, block.norm1.eps)
    n = k["modnorm"](x, mean, rstd, temb, table, 0, 1)
    attn = block.attn1(n, None, None, rotary_emb)
    attn = attn if attn.dtype is torch.float16 and attn.is_contiguous() else None
    if attn is None:
        raise _Fallback()
    x = k["gate_residual"](x, attn, temb, table, 2)
    # 2. cross-attention: norm2 (affine FP32LayerNorm, or Identity), plain fp16 residual add
    norm2 = block.norm2
    if isinstance(norm2, torch.nn.Identity):
        n = x
    else:
        mean, rstd = _stats(x, norm2.eps)
        n = k["affine_norm"](x, mean, rstd, norm2.weight, norm2.bias)
    attn = block.attn2(n, encoder_hidden_states, None, None)
    x = x + attn
    if x.dtype is not torch.float16 or not x.is_contiguous():
        raise _Fallback()
    # 3. feed-forward: norm3 + (c_shift, c_scale) = chunks 3, 4; c_gate = chunk 5
    mean, rstd = _stats(x, block.norm3.eps)
    n = k["modnorm"](x, mean, rstd, temb, table, 3, 4)
    ff = block.ffn(n)
    if ff.dtype is not torch.float16 or not ff.is_contiguous() or ff.shape != x.shape:
        raise _Fallback()
    return k["gate_residual"](x, ff, temb, table, 5)


class _Fallback(Exception):
    """A sub-module returned a layout the kernels do not take; recompute the block on the stock path."""


def _verify(block: Any, args: tuple, stock: Callable) -> bool:
    """Run the fused and stock block once on the same inputs; trust the fused path only on a bit-for-bit match."""
    import torch

    try:
        with torch.no_grad():
            ref = stock(block, *args)
            got = _fused_forward(block, *args)
        return bool(torch.equal(ref, got))
    except Exception:  # noqa: BLE001
        return False


def _make_forward(stock: Callable) -> Callable:
    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        if _kernels() is None or not _eligible(self, hidden_states, encoder_hidden_states, temb):
            _COUNTS["stock"] += 1
            return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
        key = (
            hidden_states.device.index,
            temb.dim(),
            temb.dtype,
            not hasattr(self.norm2, "weight") or self.norm2.weight is None,
        )
        ok = _VERIFIED.get(key)
        if ok is None:
            ok = _verify(self, (hidden_states, encoder_hidden_states, temb, rotary_emb), stock)
            _VERIFIED[key] = ok
            logger = _STATE.get("logger")
            if logger is not None:
                try:
                    if ok:
                        logger.info("video.wan_fused: fused fp16 modulation matched the stock block bit for bit")
                    else:
                        logger.warning(
                            "video.wan_fused: fused fp16 modulation did not match the stock block; keeping stock"
                        )
                except Exception:  # noqa: BLE001
                    pass
        if not ok:
            _COUNTS["stock"] += 1
            return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
        try:
            out = _fused_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
        except _Fallback:
            _COUNTS["stock"] += 1
            return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
        _COUNTS["fused"] += 1
        return out

    forward._unsloth_wan_fused = True  # type: ignore[attr-defined]
    forward.__wrapped__ = stock  # type: ignore[attr-defined]
    return forward


def _stock_supported(cls: Any) -> bool:
    try:
        from .diffusion_qwenimage21_rope import _digest
    except Exception:  # noqa: BLE001
        return False
    return _digest(cls.forward) in _FINGERPRINTS


def install(dtype: Any, device: Any = "cuda", logger: Any = None) -> bool:
    """Point ``WanTransformerBlock.forward`` at the fused version for an fp16 load. Idempotent; False (and nothing
    patched) when not wanted, disabled, or the installed diffusers block is not the one this was built for. Must run
    before group-offload hooks are attached: they capture the block's bound forward at registration."""
    if not wanted(dtype, device):
        uninstall()
        return False
    import importlib

    with _LOCK:
        if "forward" in _STATE:
            if logger is not None:
                _STATE["logger"] = logger
            return True
        try:
            mod = importlib.import_module(_MODULE)
        except Exception:  # noqa: BLE001
            return False
        cls = getattr(mod, _CLASS, None)
        if cls is None or getattr(cls.forward, "_unsloth_wan_fused", False):
            return False
        if not _stock_supported(cls):
            if logger is not None:
                try:
                    logger.info("video.wan_fused: diffusers' WanTransformerBlock changed; keeping the stock forward")
                except Exception:  # noqa: BLE001
                    pass
            return False
        stock = cls.forward
        _STATE["forward"] = stock
        _STATE["cls"] = cls
        _STATE["logger"] = logger
        _VERIFIED.clear()
        _COUNTS["fused"] = _COUNTS["stock"] = 0
        cls.forward = _make_forward(stock)
    if logger is not None:
        try:
            logger.info("video.wan_fused: fused fp16 norm / modulation / gated-residual kernels installed")
        except Exception:  # noqa: BLE001
            pass
    return True


def install_for_pipe(pipe: Any, dtype: Any, device: Any = "cuda", logger: Any = None) -> bool:
    """Per-load entry: install for an fp16 Wan denoiser, otherwise make sure the stock forward is back."""
    dits = [getattr(pipe, name, None) for name in ("transformer", "transformer_2")]
    if not any(d is not None and type(d).__module__ == _MODULE for d in dits):
        uninstall()
        return False
    return install(dtype, device, logger)


def uninstall() -> None:
    """Restore the stock forward (idempotent)."""
    with _LOCK:
        stock = _STATE.pop("forward", None)
        cls = _STATE.pop("cls", None)
        _STATE.pop("logger", None)
        _VERIFIED.clear()
        if stock is None or cls is None:
            return
        if getattr(cls.forward, "_unsloth_wan_fused", False):
            cls.forward = stock


def is_installed() -> bool:
    return "forward" in _STATE


def counts() -> dict:
    return dict(_COUNTS)
