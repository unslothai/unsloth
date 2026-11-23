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

# Islands whose nf4 matmul runs fp16 in, fp32 accumulate and out, inputs scaled by a power of two (exact),
# instead of an fp32 sgemm. Only the fp16 cast of the input can overflow, and the scale keeps it in range.
FP16_SCALED_MM_MODULES: dict[str, tuple[str, ...]] = {
    "z-image": (
        r"\.attention\.to_out\.0$",
        r"\.feed_forward\.w2$",
    ),
}
SCALED_MM_ENV = "UNSLOTH_STUDIO_DIT_FP16_SCALED_MM"
# Per-tensor amax after the scale lands in (2^14, 2^15]: 2x under fp16 max, subnormals start 2^29 below.
_SCALE_TARGET = 2.0**15

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


_MM_FP32_OUT_OK: dict[str, bool] = {}


def _mm_fp32_out(a, b):
    import torch
    return torch.mm(a, b, out_dtype = torch.float32)


def scaled_mm_supported(device_type: str) -> bool:
    """fp16 x fp16 -> fp32 cuBLAS GEMM (``torch.mm(out_dtype=)``) runs on this device and torch."""
    import os

    import torch

    if os.environ.get(SCALED_MM_ENV, "").strip().lower() in ("0", "false", "off", "no"):
        return False
    if device_type != "cuda":
        return False
    if device_type not in _MM_FP32_OUT_OK:
        try:
            a = torch.ones(16, 16, device = device_type, dtype = torch.float16)
            y = _mm_fp32_out(a, a)
            _MM_FP32_OUT_OK[device_type] = bool(y.dtype == torch.float32 and float(y[0, 0]) == 16.0)
        except Exception:  # noqa: BLE001 -- older torch: no out_dtype
            _MM_FP32_OUT_OK[device_type] = False
    return _MM_FP32_OUT_OK[device_type]


def _dequant_4bit(layer):
    import bitsandbytes.functional as bnbf
    return bnbf.dequantize_4bit(layer.weight.data, layer.weight.quant_state)


def _pow2_scale(t):
    """Power of two taking ``t``'s amax into (2^14, 2^15]; nan / inf stay non-finite, so GradScaler still skips."""
    import torch

    amax = torch.linalg.vector_norm(t, float("inf")).float()
    return torch.exp2(torch.floor(torch.log2(_SCALE_TARGET / amax)).clamp(-64, 64))


def _scaled_fp32_mm(x2, w16t):
    import torch

    s = _pow2_scale(x2)
    x16 = torch.empty(x2.shape, dtype = torch.float16, device = x2.device)
    torch.mul(x2, s, out = x16)
    return _mm_fp32_out(x16, w16t).div_(s)


def _make_scaled_mm_fn():
    import torch
    class ScaledFp16MM(torch.autograd.Function):
        """y = x @ W^T for a frozen nf4 W: fp16 operands, fp32 accumulate and output; dx the same way."""

        @staticmethod
        def forward(ctx, x, layer):
            ctx.layer = layer
            ctx.x_shape = x.shape
            w16 = _dequant_4bit(layer).to(torch.float16)
            y = _scaled_fp32_mm(x.reshape(-1, x.shape[-1]).float(), w16.t())
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


def install_fp32_islands(
    model,
    family: str,
    device_type: str = "cuda",
    extra: Iterable[str] = (),
    report: Optional[dict] = None,
) -> list[str]:
    """Call AFTER the LoRA attaches, so the island wraps the PEFT wrapper and its adapter branch runs fp32 too."""
    names = island_names(model, family, extra)
    mods = dict(model.named_modules())
    scaled = [re.compile(p) for p in FP16_SCALED_MM_MODULES.get(family, ())]
    use_scaled = bool(scaled) and scaled_mm_supported(device_type)
    n_scaled = 0
    for n in names:
        make_fp32_island(mods[n], device_type)
        if use_scaled and any(p.search(n) for p in scaled):
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
