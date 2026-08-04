# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe two candidate levers from the optimization research, on the real GGUF path:

  * `coordinate_descent_tuning` (Inductor) -- lossless extra kernel autotuning.
  * FirstBlockCache (diffusers `apply_first_block_cache`) -- step-skip cache, lossy,
    evaluated at a low (8) step count where its ceiling is lower.

Each config is a fresh pipeline load (so Inductor config / compile artifacts don't
cross-contaminate). Reports latency + PSNR vs the eager reference. Run on one CUDA GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = "unsloth/Z-Image-Turbo-GGUF"
GGUF = "z-image-turbo-Q4_K_M.gguf"
BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


def _load():
    import torch
    import diffusers
    from huggingface_hub import hf_hub_download

    t = diffusers.ZImageTransformer2DModel.from_single_file(
        hf_hub_download(REPO, GGUF),
        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = torch.bfloat16),
        torch_dtype = torch.bfloat16,
        config = BASE,
        subfolder = "transformer",
    )
    pipe = diffusers.ZImagePipeline.from_pretrained(BASE, torch_dtype = torch.bfloat16, transformer = t)
    pipe.to("cuda")
    return pipe


def _gen(pipe, steps, seed, res):
    import torch

    g = torch.Generator(device = "cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    img = pipe(
        prompt = PROMPT,
        width = res,
        height = res,
        num_inference_steps = steps,
        guidance_scale = 0.0,
        generator = g,
    ).images[0]
    torch.cuda.synchronize()
    return img, time.time() - t0


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    args = p.parse_args(argv)
    steps, res, seed = args.steps, args.res, args.seed

    import torch

    def compile_blocks(pipe, *, cdt = False):
        if cdt:
            import torch._inductor.config as ic
            ic.coordinate_descent_tuning = True
        pipe.transformer.compile_repeated_blocks(fullgraph = True, dynamic = True)

    def run(
        tag,
        *,
        compile = False,
        cdt = False,
        fbc = None,
    ):
        # reset inductor config between runs
        import torch._inductor.config as ic

        ic.coordinate_descent_tuning = False
        torch.compiler.reset()
        pipe = _load()
        if fbc is not None:
            from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
            apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold = fbc))
        if compile:
            compile_blocks(pipe, cdt = cdt)
            _gen(pipe, steps, seed, res)  # warmup / compilation
        else:
            _gen(pipe, steps, seed, res)  # allocator warmup
        img, dt = _gen(pipe, steps, seed, res)
        del pipe
        torch.cuda.empty_cache()
        return tag, np.array(img), dt

    results = []
    print(f"== leverage probe (Z-Image Q4_K_M, {res}px, {steps} steps) ==", flush = True)
    _, eager, eager_t = run("eager")
    print(f"  eager: {eager_t:.3f}s", flush = True)
    results.append(("eager", eager_t, 0.0))

    for tag, kw in [
        ("compile(default)", dict(compile = True)),
        ("compile+coord_desc", dict(compile = True, cdt = True)),
        ("fbc0.12+compile", dict(compile = True, fbc = 0.12)),
        ("fbc0.20+compile", dict(compile = True, fbc = 0.20)),
        ("fbc0.20(no compile)", dict(fbc = 0.20)),
    ]:
        try:
            t, img, dt = run(tag, **kw)
            ps = _psnr(eager, img)
            results.append((tag, dt, ps))
            print(
                f"  {tag:22s} {dt:.3f}s  ({(eager_t-dt)/eager_t*100:+.0f}% vs eager)  PSNR={ps:.1f} dB",
                flush = True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {tag:22s} FAILED: {type(exc).__name__}: {str(exc)[:140]}", flush = True)

    print("\n==== SUMMARY ====", flush = True)
    for tag, dt, ps in results:
        sp = f"{(results[0][1]-dt)/results[0][1]*100:+.0f}%" if tag != "eager" else "ref"
        pss = f"{ps:.1f}dB" if ps else "ref"
        print(f"  {tag:24s} {dt:.3f}s  {sp:>6s}  {pss:>8s}", flush = True)
    print("LEVERAGE-PROBE-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
