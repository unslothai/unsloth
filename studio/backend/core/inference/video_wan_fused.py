# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fused fp16 Triton kernels for diffusers' ``WanTransformerBlock`` (T4 and other fp16-only GPUs).

Stock keeps norm / AdaLN modulation / gated residuals / rotary in float32 tensors (a float32 ``[B, L, 6, D]`` table per
block for TI2V-5B). The kernels do the same per-element math, rounding in float32 exactly where the stock op chain does
and to fp16 once, so the output is bit-identical (self-checked once per device before use). LayerNorm stats come from
``native_layer_norm`` on the fp16 rows, bit-identical to ``layer_norm(x.float())`` (4-wide vectorised loads either way).
Scope: fingerprinted block, fp16 CUDA (non-ROCm), no grad, not compiling. Kill switches
``UNSLOTH_DIFFUSION_WAN_FUSED_ADALN=0`` (all), ``UNSLOTH_DIFFUSION_WAN_FUSED_ROPE=0`` (rotary only).
"""

from __future__ import annotations

import functools
import os
import threading
from typing import Any, Callable, Optional

WAN_FUSED_ENV = "UNSLOTH_DIFFUSION_WAN_FUSED_ADALN"
# rotary only (the block fusion stays): UNSLOTH_DIFFUSION_WAN_FUSED_ROPE=0
WAN_FUSED_ROPE_ENV = "UNSLOTH_DIFFUSION_WAN_FUSED_ROPE"
_MODULE = "diffusers.models.transformers.transformer_wan"
_CLASS = "WanTransformerBlock"
# _digest(WanTransformerBlock.forward): same in diffusers 0.36.0, 0.37.0, 0.40.0 and main.
_FINGERPRINTS = frozenset({"69aa0565fb00eeae"})
# _digest(WanAttnProcessor.__call__): diffusers 0.37.0+; 0.36.0 keeps stock attention.
_ATTN_FINGERPRINTS = frozenset({"6c5a5095338dfe67"})

_LOCK = threading.Lock()
_STATE: dict = {}
_VERIFIED: dict = {}
_VERIFY_OOM: set = set()
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
        # enable_fp_fusion=False is not enough (ptxas forms FFMA2 on sm_100), hence inline PTX.
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
        # int64: l * 6 * D passes 2**31 on long clips; int32 wraps to an out-of-bounds temb read.
        row = tl.program_id(0).to(tl.int64)
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
        X, A, T, TBL, OUT, D, L, stb, stl, stk, sbk, K_GATE: tl.constexpr, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)  # int64 temb offset, as in _modnorm
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        b = row // L
        l = row - b * L
        x = tl.load(X + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        a = tl.load(A + row * D + col, mask = mask, other = 0.0).to(tl.float32)
        t_g = tl.load(T + b * stb + l * stl + K_GATE * stk + col, mask = mask, other = 0.0).to(
            tl.float32
        )
        g_g = tl.load(TBL + K_GATE * sbk + col, mask = mask, other = 0.0).to(tl.float32)
        gate = g_g + t_g
        y = _mul_rn(a, gate)
        y = x + y
        tl.store(OUT + row * D + col, y.to(OUT.dtype.element_ty), mask = mask)

    @triton.jit
    def _affine_norm(X, MEAN, RSTD, W, BIAS, OUT, D, BLOCK: tl.constexpr):
        # Matches PyTorch's gamma * (...) + beta, which nvcc contracts to one fma.
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

    @triton.jit
    def _rope(
        X, COS, SIN, OUT, L, scl, ssl, PAIRS: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr
    ):
        # Each product a separately rounded fp32 multiply, then one fp16 rounding, matching stock.
        row = tl.program_id(0)
        l = row % L
        idx = tl.arange(0, BLOCK)
        mask = idx < PAIRS
        i = idx % HALF
        xo = X + row * (2 * PAIRS) + 2 * idx
        x1 = tl.load(xo, mask = mask, other = 0.0).to(tl.float32)
        x2 = tl.load(xo + 1, mask = mask, other = 0.0).to(tl.float32)
        c = tl.load(COS + l * scl + 2 * i, mask = mask, other = 0.0).to(tl.float32)
        sn = tl.load(SIN + l * ssl + 2 * i + 1, mask = mask, other = 0.0).to(tl.float32)
        even = _mul_rn(x1, c) - _mul_rn(x2, sn)
        odd = _mul_rn(x1, sn) + _mul_rn(x2, c)
        oo = OUT + row * (2 * PAIRS) + 2 * idx
        tl.store(oo, even.to(OUT.dtype.element_ty), mask = mask)
        tl.store(oo + 1, odd.to(OUT.dtype.element_ty), mask = mask)

    block = 1024

    def _grid(rows: int, d: int) -> tuple:
        return (rows, triton.cdiv(d, block))

    def _temb_strides(temb: Any) -> tuple:
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

    def rope(x, cos, sin):
        import torch

        B, L, H, D = x.shape
        out = torch.empty_like(x)
        pairs = H * (D // 2)
        _rope[(B * L,)](
            x,
            cos,
            sin,
            out,
            L,
            cos.stride(1),
            sin.stride(1),
            PAIRS = pairs,
            HALF = D // 2,
            BLOCK = triton.next_power_of_2(pairs),
            num_warps = 4 if pairs <= 2048 else 8,
            enable_fp_fusion = False,
        )
        return out

    return {
        "modnorm": modnorm,
        "gate_residual": gate_residual,
        "affine_norm": affine_norm,
        "rope": rope,
    }


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
    if torch.is_grad_enabled() and (
        x.requires_grad or any(p.requires_grad for p in block.parameters(recurse = False))
    ):
        return False
    if torch.compiler.is_compiling():
        return False
    B, L, D = x.shape
    if D % 4 or not x.is_contiguous() or x.numel() == 0:
        return False
    table = getattr(block, "scale_shift_table", None)
    # float32 as diffusers keeps it (_keep_in_fp32_modules); an fp16 table upcasts exactly in both paths
    if (
        not torch.is_tensor(table)
        or table.dtype not in (torch.float32, torch.float16)
        or tuple(table.shape) != (1, 6, D)
    ):
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


def _rope_ok(x: Any, freqs: Any) -> bool:
    import torch
    B, L, H, D = x.shape
    return (
        torch.is_tensor(freqs)
        and freqs.dtype is torch.float32
        and freqs.device == x.device
        and freqs.dim() == 4
        and tuple(freqs.shape) == (1, L, 1, D)
        and freqs.stride(-1) == 1
    )


def _self_attention(attn: Any, hidden_states: Any, rotary_emb: Any) -> Any:
    """``WanAttnProcessor.__call__`` for self-attention (no mask, no image branch), step for step, with the rotary as
    one kernel per q / k. None when the processor or the inputs are not the ones this reproduces; the caller then runs
    the stock module."""
    import torch

    if (os.environ.get(WAN_FUSED_ROPE_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    stock_call = _STATE.get("attn_call")
    proc = getattr(attn, "processor", None)
    if stock_call is None or type(proc).__call__ is not stock_call:
        return None
    if (
        getattr(attn, "add_k_proj", None) is not None
        or not isinstance(rotary_emb, (tuple, list))
        or len(rotary_emb) != 2
    ):
        return None
    cos, sin = rotary_emb
    L = hidden_states.shape[1]
    for f in (cos, sin):
        if (
            not torch.is_tensor(f)
            or f.dtype is not torch.float32
            or f.dim() != 4
            or f.shape[1] != L
        ):
            return None
    mod = _STATE.get("module")
    query, key, value = mod._get_qkv_projections(attn, hidden_states, None)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))
    if (
        query.dtype is not torch.float16
        or key.dtype is not torch.float16
        or not query.is_contiguous()
        or not key.is_contiguous()
        or query.shape[-1] % 2
        or not _rope_ok(query, cos)
        or not _rope_ok(query, sin)
    ):
        return None
    k = _kernels()
    query = k["rope"](query, cos, sin)
    key = k["rope"](key, cos, sin)
    out = mod.dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask = None,
        dropout_p = 0.0,
        is_causal = False,
        backend = proc._attention_backend,
        parallel_config = proc._parallel_config,
    )
    out = out.flatten(2, 3)
    out = out.type_as(query)
    out = attn.to_out[0](out)
    out = attn.to_out[1](out)
    return out


def _fused_forward(
    block: Any, hidden_states: Any, encoder_hidden_states: Any, temb: Any, rotary_emb: Any
) -> Any:
    import torch

    # Triton launches on the current device's stream; a load on another GPU (cuda:1 of a T4x2) must switch to it
    with torch.cuda.device(hidden_states.device):
        return _fused_block(block, hidden_states, encoder_hidden_states, temb, rotary_emb)


def _fused_block(
    block: Any, hidden_states: Any, encoder_hidden_states: Any, temb: Any, rotary_emb: Any
) -> Any:
    """The stock block, step for step, with the float32 elementwise chains replaced by the kernels above."""
    import torch

    k = _kernels()
    table = block.scale_shift_table
    x = hidden_states
    mean, rstd = _stats(x, block.norm1.eps)
    n = k["modnorm"](x, mean, rstd, temb, table, 0, 1)
    attn = _self_attention(block.attn1, n, rotary_emb) if rotary_emb is not None else None
    if attn is None:
        attn = block.attn1(n, None, None, rotary_emb)
    attn = attn if attn.dtype is torch.float16 and attn.is_contiguous() else None
    if attn is None:
        raise _Fallback()
    x = k["gate_residual"](x, attn, temb, table, 2)
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
    mean, rstd = _stats(x, block.norm3.eps)
    n = k["modnorm"](x, mean, rstd, temb, table, 3, 4)
    ff = block.ffn(n)
    if ff.dtype is not torch.float16 or not ff.is_contiguous() or ff.shape != x.shape:
        raise _Fallback()
    return k["gate_residual"](x, ff, temb, table, 5)


class _Fallback(Exception):
    """A sub-module returned a layout the kernels do not take; recompute the block on the stock path."""


def _verify(block: Any, args: tuple, stock: Callable) -> Optional[bool]:
    """Run the fused and stock block once on the same inputs; trust the fused path only on a bit-for-bit match.
    None when the check itself ran out of memory (retried once on the next call)."""
    import torch
    try:
        # a streamed group's onload can still be in flight on the copy stream: a half-copied reference fails the check
        torch.cuda.synchronize(args[0].device)
        with torch.no_grad():
            ref = stock(block, *args)
            got = _fused_forward(block, *args)
        return bool(torch.equal(ref, got))
    except torch.OutOfMemoryError:
        return None
    except Exception:  # noqa: BLE001
        return False


def _make_forward(stock: Callable) -> Callable:
    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        import torch

        # Under compile, trace stock and skip counters: dynamo guards globals, so bumps recompile.
        if torch.compiler.is_compiling():
            return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
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
            if ok is None and key not in _VERIFY_OOM:
                _VERIFY_OOM.add(key)
                _COUNTS["stock"] += 1
                return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
            ok = bool(ok)
            _VERIFIED[key] = ok
            logger = _STATE.get("logger")
            if logger is not None:
                try:
                    if ok:
                        logger.info(
                            "video.wan_fused: fused fp16 modulation matched the stock block bit for bit"
                        )
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
        except torch.OutOfMemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            _VERIFIED[key] = False
            logger = _STATE.get("logger")
            if logger is not None:
                try:
                    logger.warning("video.wan_fused: fused block failed (%s); keeping stock", exc)
                except Exception:  # noqa: BLE001
                    pass
            _COUNTS["stock"] += 1
            return stock(self, hidden_states, encoder_hidden_states, temb, rotary_emb)
        _COUNTS["fused"] += 1
        return out

    forward._unsloth_wan_fused = True  # type: ignore[attr-defined]
    forward.__wrapped__ = stock  # type: ignore[attr-defined]
    return forward


def _attn_supported(cls: Any) -> bool:
    try:
        from .diffusion_qwenimage21_rope import _digest
    except Exception:  # noqa: BLE001
        return False
    return _digest(cls.__call__) in _ATTN_FINGERPRINTS


def _stock_supported(cls: Any) -> bool:
    try:
        from .diffusion_qwenimage21_rope import _digest
    except Exception:  # noqa: BLE001
        return False
    return _digest(cls.forward) in _FINGERPRINTS


def install(
    dtype: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> bool:
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
                    logger.info(
                        "video.wan_fused: diffusers' WanTransformerBlock changed; keeping the stock forward"
                    )
                except Exception:  # noqa: BLE001
                    pass
            return False
        stock = cls.forward
        _STATE["forward"] = stock
        _STATE["cls"] = cls
        _STATE["module"] = mod
        proc_cls = getattr(mod, "WanAttnProcessor", None)
        if (
            proc_cls is not None
            and _attn_supported(proc_cls)
            and hasattr(mod, "_get_qkv_projections")
        ):
            _STATE["attn_call"] = proc_cls.__call__
        _STATE["logger"] = logger
        _VERIFIED.clear()
        _VERIFY_OOM.clear()
        _COUNTS["fused"] = _COUNTS["stock"] = 0
        cls.forward = _make_forward(stock)
    if logger is not None:
        try:
            logger.info(
                "video.wan_fused: fused fp16 norm / modulation / gated-residual kernels installed"
            )
        except Exception:  # noqa: BLE001
            pass
    return True


def install_for_pipe(
    pipe: Any,
    dtype: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> bool:
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
        for key in ("logger", "module", "attn_call"):
            _STATE.pop(key, None)
        _VERIFIED.clear()
        _VERIFY_OOM.clear()
        if stock is None or cls is None:
            return
        if getattr(cls.forward, "_unsloth_wan_fused", False):
            cls.forward = stock


def is_installed() -> bool:
    return "forward" in _STATE


def counts() -> dict:
    return dict(_COUNTS)
