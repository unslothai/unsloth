# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Q / K prologue (float32 RMSNorm, RoPE, downcast) in one Triton kernel, bit-identical to the stock chain.

The norm mirrors torch's CUDA ``_fused_rms_norm`` reduction order (fp16 / bf16 squares are exact in float32, so order
alone fixes the sum); RoPE rounds each product and the sum to float32 (no FMA). Engaged per key only after a bitwise
probe match and a timed win; anything else runs the stock lines. Off with ``UNSLOTH_DIFFUSION_FLUX3_QK_NORM_ROPE=0``.
"""

from __future__ import annotations

import os
import threading
import weakref
from functools import lru_cache
from typing import Any, Optional

from .rocm_bf16 import is_rocm_torch

QK_NORM_ROPE_ENV = "UNSLOTH_DIFFUSION_FLUX3_QK_NORM_ROPE"
# registry switch; strided and ConvRot paths call the helper without the swap
PATCH_ENV = "UNSLOTH_DIFFUSION_PATCH_FLUX3_QK_ROPE"
_MODULE = "diffusers.models.transformers.transformer_flux3"
_LOCK = threading.Lock()
_VERDICTS: dict = {}
STATS = {"fused": 0, "stock": 0}
_FAILED = [False]
_CLS_REF: list = [None]
_ROWS = 16
_WARPS = 4


def enabled() -> bool:
    off = ("0", "off", "false", "no")
    return all(
        (os.environ.get(env) or "1").strip().lower() not in off
        for env in (QK_NORM_ROPE_ENV, PATCH_ENV)
    )


def reference_rope(x: Any, cos: Any, sin: Any) -> Any:
    import torch

    cos = cos[None, :, None, :].to(x.device)
    sin = sin[None, :, None, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim = -1).flatten(3)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def reference_qk(x: Any, weight: Any, eps: float, cos: Any, sin: Any, out_dtype: Any) -> Any:
    import torch.nn.functional as F
    normed = F.rms_norm(x.float(), (x.shape[-1],), weight, eps)
    return reference_rope(normed, cos, sin).to(out_dtype)


@lru_cache(maxsize = None)
def _kernel() -> Any:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice
    except Exception:  # noqa: BLE001 -- older Triton layout
        from triton.language.extra.cuda import libdevice

    # torch's row kernel: ::rsqrtf, an IEEE division by N, an fp32 add of eps
    _rsqrt = libdevice.rsqrt
    _div = libdevice.div_rn
    _add = libdevice.add_rn

    @triton.jit(do_not_specialize = ["n_rows", "n_heads", "seq"])
    def _f3_qk_norm_rope(
        x_ptr,
        w_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_rows,
        n_heads,
        seq,
        stride_b,
        stride_s,
        stride_h,
        cos_stride,
        sin_stride,
        eps,
        D: tl.constexpr,
        ROWS: tl.constexpr,
        G: tl.constexpr,  # warps of torch's row kernel that hold data: max(1, D // 128)
    ):
        odt = out_ptr.dtype.element_ty
        pid = tl.program_id(0).to(tl.int64)
        rows = pid * ROWS + tl.arange(0, ROWS).to(tl.int64)
        row_ok = rows < n_rows
        h = rows % n_heads
        t = rows // n_heads
        s = t % seq
        b = t // seq
        base = b * stride_b + s * stride_s + h * stride_h
        cols = tl.arange(0, D)
        partner = cols ^ 1
        mask = row_ok[:, None]
        x = tl.load(x_ptr + base[:, None] + cols[None, :], mask = mask, other = 0.0).to(tl.float32)
        xp = tl.load(x_ptr + base[:, None] + partner[None, :], mask = mask, other = 0.0).to(tl.float32)
        # torch's order: lane l of warp g sums elements 4(32g+l)..+3, then lanes, then warps (probe-proven)
        lane = tl.arange(0, 32)
        for g in tl.static_range(G):
            vec = 32 * g + lane
            vmask = row_ok[:, None] & (vec < D // 4)[None, :]
            vptr = x_ptr + base[:, None] + 4 * vec[None, :]
            acc = tl.zeros([ROWS, 32], dtype = tl.float32)
            for j in tl.static_range(4):
                v = tl.load(vptr + j, mask = vmask, other = 0.0).to(tl.float32)
                acc = acc + v * v
            if g == 0:
                total = tl.sum(acc, axis = 1)
            else:
                total = total + tl.sum(acc, axis = 1)
        rstd = _rsqrt(_add(_div(total, D * 1.0), eps))
        w = tl.load(w_ptr + cols).to(tl.float32)
        wp = tl.load(w_ptr + partner).to(tl.float32)
        xn = w[None, :] * (rstd[:, None] * x)
        pn = wp[None, :] * (rstd[:, None] * xp)
        c = tl.load(cos_ptr + s[:, None] * cos_stride + cols[None, :], mask = mask, other = 0.0)
        sn = tl.load(sin_ptr + s[:, None] * sin_stride + cols[None, :], mask = mask, other = 0.0)
        sign = tl.where((cols % 2) == 0, -1.0, 1.0)
        y = xn * c.to(tl.float32) + (pn * sign[None, :]) * sn.to(tl.float32)
        tl.store(out_ptr + rows[:, None] * D + cols[None, :], y.to(odt), mask = mask)

    return _f3_qk_norm_rope


def launch(
    x: Any,
    weight: Any,
    eps: float,
    cos: Any,
    sin: Any,
    out_dtype: Any = None,
) -> Any:
    """``x``: [B, S, H, D] fp16 / bf16, last dim dense. Returns a contiguous [B, S, H, D] in ``out_dtype``."""
    import torch

    B, S, H, D = x.shape
    out = torch.empty((B, S, H, D), device = x.device, dtype = out_dtype or x.dtype)
    rows = B * S * H
    grid = ((rows + _ROWS - 1) // _ROWS,)
    with torch.cuda.device(x.device):
        _kernel()[grid](
            x,
            weight,
            cos,
            sin,
            out,
            rows,
            H,
            S,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            cos.stride(0),
            sin.stride(0),
            float(eps),
            D = D,
            ROWS = _ROWS,
            G = max(1, D // 128),
            num_warps = _WARPS,
            enable_fp_fusion = False,
        )
    return out


def eligible(x: Any, weight: Any, cos: Any, sin: Any, out_dtype: Any) -> bool:
    import torch

    if not torch.is_tensor(x) or not x.is_cuda or x.dim() != 4 or x.numel() == 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16) or out_dtype != x.dtype:
        return False
    if x.requires_grad and torch.is_grad_enabled():
        return False
    B, S, H, D = x.shape
    if D not in (64, 128, 256) or x.stride(3) != 1:
        return False
    # torch's vectorized row kernel needs 4-element alignment; the probe compiled that specialization
    if x.data_ptr() % 16 or any(st % 8 for st in x.stride()[:3]):
        return False
    if not torch.is_tensor(weight) or tuple(weight.shape) != (D,) or not weight.is_contiguous():
        return False
    if weight.device != x.device or weight.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return False
    for f in (cos, sin):
        if not torch.is_tensor(f) or f.dim() != 2 or tuple(f.shape) != (S, D):
            return False
        if f.device != x.device or f.stride(1) != 1 or f.dtype != torch.float32:
            return False
    return x.numel() < 2**31


def _probe_inputs(device: Any, dtype: Any, wdtype: Any, dim: int, eps: float, seed: int) -> list:
    import torch

    g = torch.Generator(device = "cpu").manual_seed(seed)
    out = []
    for seq, heads, strided in ((257, 6, True), (96, 3, False)):
        base = torch.randn(1, seq, heads, dim, generator = g)
        scale = 10.0 ** torch.randint(-3, 4, (1, seq, heads, 1), generator = g).float()
        base.mul_(scale).add_(torch.randn(1, seq, heads, 1, generator = g) * scale * 0.5)
        base[..., :3] *= 40
        cap = torch.finfo(dtype).max / 8
        base.clamp_(-cap, cap)
        if strided:
            wide = torch.zeros(1, seq, 3 * heads * dim)
            wide[..., heads * dim : 2 * heads * dim] = base.flatten(2)
            x = (
                wide.to(dtype)
                .to(device)[..., heads * dim : 2 * heads * dim]
                .unflatten(-1, (heads, dim))
            )
        else:
            x = base.to(dtype).to(device)
        w = (1 + 2.0 * torch.rand(dim, generator = g)).to(wdtype).to(device)
        ang = torch.rand(seq, dim // 2, generator = g, dtype = torch.float64) * 60
        ang = ang.repeat_interleave(2, dim = -1)
        out.append((x, w, ang.cos().float().to(device), ang.sin().float().to(device)))
    return out


def _race_ms(fn: Any, reps: int = 7) -> float:
    import torch

    for _ in range(2):
        fn()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def _check(device: Any, dtype: Any, wdtype: Any, dim: int, eps: float) -> dict:
    import torch

    exact = True
    for seed in (11, 12, 13):
        for x, w, cos, sin in _probe_inputs(device, dtype, wdtype, dim, eps, seed):
            ref = reference_qk(x, w, eps, cos, sin, dtype)
            if not eligible(x, w, cos, sin, dtype):
                return {
                    "exact": False,
                    "faster": False,
                    "ms": {},
                    "why": "probe layout not eligible",
                }
            try:
                got = launch(x, w, eps, cos, sin, dtype)
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 -- a JIT / launch failure keeps the stock chain
                return {
                    "exact": False,
                    "faster": False,
                    "ms": {},
                    "why": f"{type(exc).__name__}: {exc}"[:200],
                }
            if not torch.equal(got, ref):
                return {"exact": False, "faster": False, "ms": {}, "why": "mismatch"}
    g = torch.Generator(device = "cpu").manual_seed(29)
    x = torch.randn(1, 2048, 24, dim, generator = g).to(dtype).to(device)
    w = (1 + torch.rand(dim, generator = g)).to(wdtype).to(device)
    ang = (torch.rand(2048, dim // 2, generator = g) * 60).repeat_interleave(2, -1).to(device)
    cos, sin = ang.cos(), ang.sin()
    ms = {
        "stock": round(_race_ms(lambda: reference_qk(x, w, eps, cos, sin, dtype)), 5),
        "fused": round(_race_ms(lambda: launch(x, w, eps, cos, sin, dtype)), 5),
    }
    return {"exact": exact, "faster": ms["fused"] < 0.98 * ms["stock"], "ms": ms}


def verdict(x: Any, weight: Any, eps: float) -> Optional[dict]:
    """Cached ``_check``; None while capturing a CUDA graph before it ran."""
    import torch

    index = x.device.index if x.device.index is not None else torch.cuda.current_device()
    key = (index, x.dtype, weight.dtype, int(x.shape[-1]), float(eps))
    found = _VERDICTS.get(key)
    if found is not None:
        return found
    if torch.cuda.is_current_stream_capturing():
        return None
    with _LOCK:
        found = _VERDICTS.get(key)
        if found is None:
            try:
                with torch.no_grad(), torch.cuda.device(x.device):
                    found = _check(x.device, x.dtype, weight.dtype, int(x.shape[-1]), float(eps))
            except torch.OutOfMemoryError:
                return None
            except Exception as exc:  # noqa: BLE001
                found = {"exact": False, "faster": False, "ms": {}, "why": f"{type(exc).__name__}"}
            _VERDICTS[key] = found
    return found


def _capturing(x: Any) -> bool:
    import torch
    try:
        return bool(x.is_cuda and torch.cuda.is_current_stream_capturing())
    except Exception:  # noqa: BLE001 -- a build without CUDA graphs is never capturing
        return False


def _norm_params(norm: Any) -> Optional[tuple]:
    import torch

    if type(norm) is not torch.nn.RMSNorm or not norm.elementwise_affine or norm.weight is None:
        return None
    if tuple(norm.normalized_shape) != (int(norm.weight.shape[0]),):
        return None
    eps = norm.eps
    if eps is None:
        return None  # torch would use finfo(float32).eps; not worth a second probe key
    if getattr(norm, "_forward_hooks", None) or getattr(norm, "_forward_pre_hooks", None):
        return None
    # the stock chain is torch's own forward: a patched class or instance forward keeps it
    if (
        "forward" in vars(norm)
        or getattr(type(norm).forward, "__module__", "") != "torch.nn.modules.normalization"
    ):
        return None
    return norm.weight, float(eps)


def _fused_one(norm: Any, x: Any, rotary: Any, out_dtype: Any) -> Optional[Any]:
    import torch

    if _FAILED[0] or rotary is None or not isinstance(rotary, (tuple, list)) or len(rotary) != 2:
        return None
    params = _norm_params(norm)
    if params is None:
        return None
    weight, eps = params
    cos, sin = rotary
    if not eligible(x, weight, cos, sin, out_dtype):
        return None
    found = verdict(x, weight, eps)
    if not found or not found.get("exact") or not found.get("faster"):
        return None
    try:
        out = launch(x, weight, eps, cos, sin, out_dtype)
    except torch.OutOfMemoryError:
        raise
    except Exception:  # noqa: BLE001 -- never fail a render over the fast path
        _FAILED[0] = True
        return None
    if not found.get("real_checked"):
        if _capturing(x):
            return None
        with _LOCK:
            if not found.get("real_checked"):
                ref = reference_qk(x, weight, eps, cos, sin, out_dtype)
                found["real_exact"] = bool(torch.equal(out, ref))
                found["real_checked"] = True
                if not found["real_exact"]:
                    found["exact"] = False
                    found["why"] = "first real call differed"
                    return None
    return out


def qk_norm_rope(
    attn: Any, query: Any, key: Any, value_dtype: Any, image_rotary_emb: Any, rope_fn: Any
) -> tuple:
    import torch

    if not torch.compiler.is_compiling():
        if enabled() and image_rotary_emb is not None:
            q = _fused_one(attn.norm_q, query, image_rotary_emb, value_dtype)
            if q is not None:
                k = _fused_one(attn.norm_k, key, image_rotary_emb, value_dtype)
                if k is not None:
                    STATS["fused"] += 1
                    return q, k
        STATS["stock"] += 1
    query = attn.norm_q(query.float())
    key = attn.norm_k(key.float())
    if image_rotary_emb is not None:
        query = rope_fn(query, image_rotary_emb, sequence_dim = 1)
        key = rope_fn(key, image_rotary_emb, sequence_dim = 1)
    return query.to(value_dtype), key.to(value_dtype)


@lru_cache(maxsize = None)
def processor_class() -> Any:
    """Built once: dynamo guards on the class."""
    from diffusers.models.transformers import transformer_flux3 as tf3

    class UnslothFlux3QkProcessor(tf3.Flux3AttnProcessor):
        def __call__(
            self,
            attn: Any,
            hidden_states: Any,
            attention_mask: Any = None,
            image_rotary_emb: Any = None,
        ) -> Any:
            if attention_mask is not None or self._parallel_config is not None:
                return super().__call__(attn, hidden_states, attention_mask, image_rotary_emb)
            query, key, value = tf3._get_qkv_projections(attn, hidden_states)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            # module attributes at call time: the fused RoPE and fp16 attention installs patch them there
            query, key = qk_norm_rope(
                attn, query, key, value.dtype, image_rotary_emb, tf3.apply_rotary_emb
            )
            hidden_states = tf3.dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask = attention_mask,
                backend = self._attention_backend,
                parallel_config = self._parallel_config,
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.to(query.dtype)
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            return hidden_states

    _CLS_REF[0] = UnslothFlux3QkProcessor
    return UnslothFlux3QkProcessor


def is_qk_processor(proc: Any) -> bool:
    """Plain reads only (traced inside compiled blocks); no diffusers import."""
    cls = _CLS_REF[0]
    return cls is not None and type(proc) is cls


_SWAPPED: list = []


def install_for_pipe(
    pipe: Any,
    device: Any = "cuda",
    logger: Any = None,
) -> int:
    """Swap stock FLUX.3 processors; returns the count, never raises. Math changes only after ``verdict``."""
    try:
        return _install(pipe, device, logger)
    except Exception as exc:  # noqa: BLE001 - stock processors stay
        if logger is not None:
            logger.info("diffusion.flux3_qk_rope: skipped (%s)", exc)
        return 0


def _install(pipe: Any, device: Any, logger: Any) -> int:
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or type(transformer).__module__ != _MODULE or not wanted(device):
        return 0
    from diffusers.models.transformers import transformer_flux3 as tf3

    from .diffusion_autoopt.patches import source_digest
    from .diffusion_flux3_attn import _PROCESSOR_DIGESTS

    if source_digest(tf3.Flux3AttnProcessor.__call__) not in _PROCESSOR_DIGESTS or not callable(
        getattr(tf3, "_get_qkv_projections", None)
    ):
        if logger is not None:
            logger.info(
                "diffusion.flux3_qk_rope: diffusers' FLUX.3 attention changed; stock processor kept"
            )
        return 0
    cls = processor_class()
    swapped = 0
    for attn in transformer.modules():
        if not isinstance(attn, tf3.Flux3Attention):
            continue
        proc = getattr(attn, "processor", None)
        if type(proc) is not tf3.Flux3AttnProcessor:
            continue
        new = cls()
        new._attention_backend = proc._attention_backend
        new._parallel_config = proc._parallel_config
        attn.set_processor(new)
        _SWAPPED.append(weakref.ref(attn))
        swapped += 1
    if swapped and logger is not None:
        logger.info(
            "diffusion.flux3_qk_rope: %d FLUX.3 attention calls build Q / K in one kernel when exact on the card",
            swapped,
        )
    return swapped


def uninstall() -> None:
    try:
        from diffusers.models.transformers import transformer_flux3 as tf3
    except Exception:  # noqa: BLE001
        _SWAPPED.clear()
        return
    while _SWAPPED:
        attn = _SWAPPED.pop()()
        if attn is None or not is_qk_processor(getattr(attn, "processor", None)):
            continue
        stock = tf3.Flux3AttnProcessor()
        stock._attention_backend = attn.processor._attention_backend
        stock._parallel_config = attn.processor._parallel_config
        attn.set_processor(stock)


def wanted(device: Any = "cuda") -> bool:
    if not enabled():
        return False
    try:
        import torch
        import triton  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return str(device).startswith("cuda") and torch.cuda.is_available() and not is_rocm_torch(torch)


def reset() -> None:
    with _LOCK:
        _VERDICTS.clear()
        _FAILED[0] = False


__all__ = [
    "QK_NORM_ROPE_ENV",
    "STATS",
    "enabled",
    "eligible",
    "launch",
    "install_for_pipe",
    "is_qk_processor",
    "processor_class",
    "qk_norm_rope",
    "reference_qk",
    "uninstall",
    "verdict",
    "wanted",
]
