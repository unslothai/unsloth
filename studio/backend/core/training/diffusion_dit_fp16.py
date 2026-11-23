# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""fp16 DiT LoRA training on non-bf16 GPUs: nf4 matmuls in fp16, fp32 residual stream / LoRA / overflowing modules."""

from __future__ import annotations

import contextlib
import re
from typing import Any, Iterable, Optional

# Per-family module-name patterns run in fp32; overflowing entries measured with overflow_probe on a T4.
FP16_FP32_MODULES: dict[str, tuple[str, ...]] = {
    "flux.2-klein": (
        r"^time_guidance_embed$",
        r"^double_stream_modulation_img$",
        r"^double_stream_modulation_txt$",
        r"^single_stream_modulation$",
        r"^norm_out$",
        r"^proj_out$",
    ),
    "z-image": (
        r"^t_embedder$",
        r"^all_final_layer\.",
        # to_out / w2 feed post-norms, so nothing bounds them: measured overflowing in every block family on a T4.
        r"^(noise_refiner|context_refiner|layers)\.\d+\.attention\.to_out\.0$",
        r"^(noise_refiner|context_refiner|layers)\.\d+\.feed_forward\.w2$",
        r"^(noise_refiner|layers)\.\d+\.adaLN_modulation$",
    ),
    "wan2.2-ti2v-5b": (
        r"^condition_embedder\.time_embedder$",
        r"^condition_embedder\.time_proj$",
        r"^norm_out$",
        r"^proj_out$",
    ),
}

# Islands whose nf4 matmul runs on fp16 tensor cores with fp32 accumulate and output instead of an fp32 sgemm.
# Inputs are scaled by a power of two (exact), so the fp16 cast cannot overflow. The forward splits the input
# into fp16 high + low parts (about 22 mantissa bits): one fp16 rounding there moved the LoRA grads 2x further
# from an fp32 reference on a T4. The input-grad GEMM takes one fp16 term: rounding it changed nothing measurable.
FP16_SCALED_MM_MODULES: dict[str, tuple[str, ...]] = {
    "z-image": (
        r"\.attention\.to_out\.0$",
        r"\.feed_forward\.w2$",
    ),
}
SCALED_MM_ENV = "UNSLOTH_STUDIO_DIT_FP16_SCALED_MM"
# Per-tensor amax after the scale lands in (2^14, 2^15]: 2x under fp16 max, subnormals start 2^29 below.
_SCALE_TARGET = 2.0**15
# Scale exponent bound: fp32 max is just under 2^128, so 2^-113 still takes it to 2^15 (2^-112 would not).
_SCALE_EXP_MAX = 113
# Scaled-GEMM forwards actually taken (``scaled_mm_stats``); the trainer reports the per-run delta.
SCALED_MM_STATS: dict[str, int] = {"calls": 0}

# This many skips in a row means the forward itself overflows (loss scaling cannot fix it): stop the run.
MAX_CONSECUTIVE_SKIPS = 30


def _is_bnb_4bit(module) -> bool:
    return type(module).__name__ in ("Linear4bit", "LinearNF4", "LinearFP4") and hasattr(
        module, "compute_dtype"
    )


def _cast_floats(obj, dtype):
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(dtype) if obj.is_floating_point() and obj.dtype != dtype else obj
    if isinstance(obj, (list, tuple)):
        cast = [_cast_floats(o, dtype) for o in obj]
        return type(obj)(cast) if isinstance(obj, tuple) and not hasattr(obj, "_fields") else cast
    if isinstance(obj, dict):
        return {k: _cast_floats(v, dtype) for k, v in obj.items()}
    return obj


def make_fp32_island(module, device_type: str = "cuda") -> None:
    """Run ``module`` with autocast off and floating inputs in fp32; nf4 matmuls inside compute in fp32. Idempotent."""
    import torch

    if getattr(module, "_unsloth_fp32_island", False):
        return
    for sub in module.modules():
        if _is_bnb_4bit(sub):
            sub.compute_dtype = torch.float32
    inner = module.forward

    def forward(*args, **kwargs):
        with torch.autocast(device_type = device_type, enabled = False):
            return inner(*_cast_floats(args, torch.float32), **_cast_floats(kwargs, torch.float32))

    module.forward = forward
    module._unsloth_fp32_island = True


# Probe verdict per device, keyed (type, index): one process can drive cards of different generations.
_MM_FP32_OUT_OK: dict[tuple, bool] = {}


def _mm_fp32_out(a, b):
    import torch
    return torch.mm(a, b, out_dtype = torch.float32)


def scaled_mm_supported(device) -> bool:
    """fp16 x fp16 -> fp32 cuBLAS GEMM (``torch.mm(out_dtype=)``) runs on ``device`` (``"cuda"``, ``"cuda:1"``,
    a ``torch.device``) and this torch, on tensor cores."""
    import os

    import torch

    if os.environ.get(SCALED_MM_ENV, "").strip().lower() in ("0", "false", "off", "no"):
        return False
    try:
        dev = torch.device(device)
    except (TypeError, RuntimeError):
        return False
    if dev.type != "cuda":
        return False
    if dev.index is None:
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
        except Exception:  # noqa: BLE001 -- no usable CUDA device
            return False
    key = (dev.type, dev.index)
    if key not in _MM_FP32_OUT_OK:
        try:
            # Tensor cores (sm70+) only: on Pascal an fp16 GEMM is no faster than the fp32 one it replaces.
            if tuple(torch.cuda.get_device_capability(dev)) < (7, 0):
                _MM_FP32_OUT_OK[key] = False
                return False
            a = torch.ones(16, 16, device = dev, dtype = torch.float16)
            y = _mm_fp32_out(a, a)
            ok = bool(y.dtype == torch.float32 and float(y[0, 0]) == 16.0)
        except torch.cuda.OutOfMemoryError:
            # A full card says nothing about the GEMM: never cache it as unsupported.
            raise
        except Exception:  # noqa: BLE001 -- older torch: no out_dtype
            ok = False
        _MM_FP32_OUT_OK[key] = ok
    return _MM_FP32_OUT_OK[key]


def scaled_mm_stats() -> dict:
    return dict(SCALED_MM_STATS)


def _dequant_4bit(layer):
    import bitsandbytes.functional as bnbf
    return bnbf.dequantize_4bit(layer.weight.data, layer.weight.quant_state)


def _pow2_scale(t):
    """Power of two taking ``t``'s amax into (2^14, 2^15]; nan / inf stay non-finite, so GradScaler still skips."""
    import torch

    amax = torch.linalg.vector_norm(t, float("inf")).float()
    e = torch.floor(torch.log2(_SCALE_TARGET / amax)).clamp(-_SCALE_EXP_MAX, _SCALE_EXP_MAX)
    return torch.exp2(e)


def _scaled_fp32_mm(
    x2,
    w16t,
    split: bool = False,
):
    """``x2 @ w16t`` in fp32 from fp16 operands; ``split`` adds the input's fp16 rounding residual as a second term."""
    import torch

    s = _pow2_scale(x2)
    m = x2.shape[0]
    x16 = torch.empty((2 * m if split else m, x2.shape[1]), dtype = torch.float16, device = x2.device)
    if not split:
        torch.mul(x2, s, out = x16)
        return _mm_fp32_out(x16, w16t).div_(s)
    # hi = fp16(x2 * s); lo = fp16(x2 * s - hi), with x2 * s exact (power of two) and formed inside each kernel:
    # no fp32 copy of the input. -hi + x2 * s rounds like x2 * s - hi, so +0 lands where hi is exact.
    hi, lo = x16[:m], x16[m:]
    torch.mul(x2, s, out = hi)
    hi.neg_()
    torch.addcmul(hi, x2, s, out = lo)
    hi.neg_()
    y = _mm_fp32_out(x16, w16t)
    return torch.add(y[:m], y[m:]).div_(s)


def _make_scaled_mm_fn():
    import torch
    class ScaledFp16MM(torch.autograd.Function):
        """y = x @ W^T for a frozen nf4 W: fp16 operands (input split hi + lo), fp32 accumulate and output; dx one term."""

        @staticmethod
        def forward(ctx, x, layer):
            ctx.layer = layer
            ctx.x_shape = x.shape
            w16 = _dequant_4bit(layer).to(torch.float16)
            y = _scaled_fp32_mm(x.reshape(-1, x.shape[-1]).float(), w16.t(), split = True)
            return y.reshape(*x.shape[:-1], w16.shape[0])

        @staticmethod
        def backward(ctx, g):
            w16 = _dequant_4bit(ctx.layer).to(torch.float16)
            dx = _scaled_fp32_mm(g.reshape(-1, g.shape[-1]).float(), w16)
            return dx.reshape(ctx.x_shape), None

    return ScaledFp16MM


_SCALED_MM_FN = None


def _scaled_mm_forward(self, x, *args, **kwargs):
    global _SCALED_MM_FN
    import torch

    if args or kwargs or not x.is_floating_point() or self.weight.requires_grad:
        return self.__dict__["_unsloth_scaled_mm_orig"](x, *args, **kwargs)
    if _SCALED_MM_FN is None:
        _SCALED_MM_FN = _make_scaled_mm_fn()
    SCALED_MM_STATS["calls"] += 1
    y = _SCALED_MM_FN.apply(x, self)
    if self.bias is not None:
        y = y + self.bias.to(torch.float32)
    return y.to(x.dtype)


def install_scaled_mm(module) -> int:
    """Route ``module``'s nf4 linears through ``ScaledFp16MM``; a weight past fp16 range keeps the fp32 path."""
    import types

    import torch

    n = 0
    for sub in module.modules():
        if not _is_bnb_4bit(sub) or "_unsloth_scaled_mm_orig" in sub.__dict__:
            continue
        try:
            if not bool(torch.isfinite(_dequant_4bit(sub).to(torch.float16)).all()):
                continue
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception:  # noqa: BLE001 -- not a dequantisable 4-bit weight: leave it fp32
            continue
        sub.__dict__["_unsloth_scaled_mm_orig"] = sub.forward
        sub.forward = types.MethodType(_scaled_mm_forward, sub)
        n += 1
    return n


def island_names(
    model,
    family: str,
    extra: Iterable[str] = (),
) -> list[str]:
    """Module names matching the family's patterns; a PEFT wrapper's inner ``base_layer`` is never matched on its own."""
    patterns = [re.compile(p) for p in (*FP16_FP32_MODULES.get(family, ()), *extra)]
    names = [n for n, _ in model.named_modules() if n and any(p.search(n) for p in patterns)]
    return [n for n in names if not any(n.startswith(o + ".") for o in names if o != n)]


def prepare_fp16_base(model) -> dict[str, Any]:
    """nf4 matmuls in fp16, every other floating param / buffer in fp32. Call before the LoRA attaches."""
    import torch

    n4 = 0
    for mod in model.modules():
        if _is_bnb_4bit(mod):
            mod.compute_dtype = torch.float16
            n4 += 1
    up = 0
    for mod in model.modules():
        for _pname, p in list(mod.named_parameters(recurse = False)):
            if p.is_floating_point() and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
                up += 1
        for bname, b in list(mod.named_buffers(recurse = False)):
            if (
                b is not None
                and b.is_floating_point()
                and b.dtype in (torch.float16, torch.bfloat16)
            ):
                mod._buffers[bname] = b.to(torch.float32)
    return {"linear4bit_fp16": n4, "params_to_fp32": up}


def _weight_device(modules, device_type):
    """The card the routed nf4 weights live on (``cuda:N``), else ``device_type``: the probe must run there."""
    import torch

    want = torch.device(device_type).type
    for mod in modules:
        for sub in mod.modules():
            w = getattr(sub, "weight", None) if _is_bnb_4bit(sub) else None
            if w is not None and w.device.type == want:
                return w.device
    return device_type


def install_fp32_islands(
    model,
    family: str,
    device_type: str = "cuda",
    extra: Iterable[str] = (),
    report: Optional[dict] = None,
) -> list[str]:
    """Call AFTER the LoRA attaches, so the island wraps the PEFT wrapper and its adapter branch runs fp32 too."""
    import torch

    names = island_names(model, family, extra)
    mods = dict(model.named_modules())
    scaled = [re.compile(p) for p in FP16_SCALED_MM_MODULES.get(family, ())]
    targets = {n for n in names if any(p.search(n) for p in scaled)}
    use_scaled = bool(targets) and scaled_mm_supported(
        _weight_device([mods[n] for n in names if n in targets], device_type)
    )
    autocast_type = torch.device(device_type).type
    n_scaled = 0
    for n in names:
        make_fp32_island(mods[n], autocast_type)
        if use_scaled and n in targets:
            n_scaled += install_scaled_mm(mods[n])
    if report is not None:
        report["fp16_scaled_mm"] = n_scaled
    return names


@contextlib.contextmanager
def overflow_probe(model, record: Optional[dict] = None):
    """Record leaf modules whose output goes non-finite from finite inputs (``first_overflow``, ``overflows``, ``max_abs``)."""
    import torch

    rec = record if record is not None else {}
    rec.setdefault("overflows", {})
    rec.setdefault("max_abs", {})
    rec.setdefault("first_overflow", None)
    handles = []

    def _tensors(x):
        if isinstance(x, torch.Tensor):
            yield x
        elif isinstance(x, (list, tuple)):
            for o in x:
                yield from _tensors(o)
        elif isinstance(x, dict):
            for o in x.values():
                yield from _tensors(o)

    def hook(name):
        def fn(mod, args, kwargs, out):
            outs = [t for t in _tensors(out) if t.is_floating_point()]
            if not outs:
                return
            with torch.no_grad():
                bad = any(not bool(torch.isfinite(t).all()) for t in outs)
                m = max(
                    float(t.detach().float().abs().nan_to_num(0.0, 0.0, 0.0).max())
                    for t in outs
                    if t.numel()
                )
                rec["max_abs"][name] = max(rec["max_abs"].get(name, 0.0), m)
                if bad:
                    ins = [t for t in _tensors((args, kwargs)) if t.is_floating_point()]
                    if all(bool(torch.isfinite(t).all()) for t in ins):
                        rec["overflows"][name] = rec["overflows"].get(name, 0) + 1
                        if rec["first_overflow"] is None:
                            rec["first_overflow"] = name

        return fn

    for name, mod in model.named_modules():
        if name and not any(True for _ in mod.children()):
            handles.append(mod.register_forward_hook(hook(name), with_kwargs = True))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()
