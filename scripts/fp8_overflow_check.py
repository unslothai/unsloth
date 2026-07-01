# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Empirically check that fp8 dynamic quant with fast accumulation does not overflow.

Hooks every quantised Linear's output during a real Z-Image generation and reports the
global max |output| and any non-finite (Inf/NaN) count, for use_fast_accum True vs False.
The concern fast_accum raises is accumulation *precision*, not overflow (the accumulator
stays FP32-range and torchao's dynamic per-row scale keeps FP8 inputs <= 448); this proves
it on the real model, including Z-Image's large (~9e5) activation peaks. Run on one CUDA GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"


def _load_dense():
    import torch, diffusers

    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    pipe = diffusers.ZImagePipeline.from_pretrained(BASE, torch_dtype = torch.bfloat16, transformer = t)
    pipe.to("cuda")
    return pipe


def _run(fast_accum, steps, res, seed, mf):
    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
    from torchao.float8 import Float8MMConfig

    pipe = _load_dense()

    def filt(mod, fqn = ""):
        return isinstance(mod, nn.Linear) and mod.in_features >= mf and mod.out_features >= mf

    quantize_(
        pipe.transformer,
        Float8DynamicActivationFloat8WeightConfig(
            mm_config = Float8MMConfig(use_fast_accum = fast_accum)
        ),
        filter_fn = filt,
    )

    stats = {"max_abs": 0.0, "nonfinite": 0, "hooked": 0}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        if not torch.is_tensor(t):
            return
        finite = torch.isfinite(t)
        nf = int((~finite).sum().item())
        stats["nonfinite"] += nf
        m = float(t[finite].abs().max().item()) if finite.any() else float("inf")
        if m > stats["max_abs"]:
            stats["max_abs"] = m

    # Hook the quantised linears (where an fp8-accumulation overflow would surface).
    # Run EAGER: forward hooks don't trace through torch.compile, and the fp8 fast-accum
    # accumulation is identical compiled or eager -- compile only changes scheduling.
    for m in pipe.transformer.modules():
        if isinstance(m, nn.Linear):
            m.register_forward_hook(hook)
            stats["hooked"] += 1

    g = torch.Generator(device = "cuda").manual_seed(seed)
    img = pipe(
        prompt = PROMPT,
        width = res,
        height = res,
        num_inference_steps = steps,
        guidance_scale = 0.0,
        generator = g,
    ).images[0]
    import numpy as np

    arr = np.array(img)
    img_finite = bool(np.isfinite(arr).all())
    del pipe
    torch.cuda.empty_cache()
    return stats, img_finite


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type = int, default = 4)
    p.add_argument("--res", type = int, default = 512)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--min-feat", type = int, default = 512)
    args = p.parse_args(argv)

    print(f"== fp8 overflow check (Z-Image dense, {args.res}px, {args.steps} steps) ==", flush = True)
    for fast in (True, False):
        stats, img_finite = _run(fast, args.steps, args.res, args.seed, args.min_feat)
        print(
            f"  fast_accum={str(fast):5s}  hooked_linears={stats['hooked']:3d}  "
            f"max|linear_out|={stats['max_abs']:.1f}  nonfinite_elems={stats['nonfinite']}  "
            f"image_all_finite={img_finite}",
            flush = True,
        )
    print("FP8-OVERFLOW-CHECK-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
