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
) -> list[str]:
    """Call AFTER the LoRA attaches, so the island wraps the PEFT wrapper and its adapter branch runs fp32 too."""
    names = island_names(model, family, extra)
    mods = dict(model.named_modules())
    for n in names:
        make_fp32_island(mods[n], device_type)
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
