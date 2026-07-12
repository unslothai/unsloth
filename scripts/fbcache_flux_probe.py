# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validate First-Block-Cache (FBCache) on a MANY-step DiT (Flux.1-dev), vs the compiled
baseline. FBCache reuses the transformer tail across denoise steps when the first block's
residual barely changes -- a real speedup only when there are enough steps (it is why it is
gated OFF for few-step distilled models like Z-Image-Turbo). Reports median latency,
speedup, peak VRAM, and LPIPS vs the no-cache baseline. One CUDA GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "black-forest-labs/FLUX.1-dev"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research" / "fbcache_flux_images"


_LP = {"fn": None}


def _lpips(ref, arr):
    try:
        import lpips
        import torch

        if _LP["fn"] is None:
            _LP["fn"] = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()

        def t(x):
            return (torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).cuda()

        with torch.no_grad():
            return float(_LP["fn"](t(ref), t(arr)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"    (lpips: {type(exc).__name__})", flush = True)
        return None


def _load():
    import os
    import diffusers
    import torch

    pipe = diffusers.FluxPipeline.from_pretrained(
        BASE, torch_dtype = torch.bfloat16, token = os.environ.get("HF_TOKEN")
    )
    pipe.to("cuda")
    return pipe


def _gen(pipe, steps, seed, res, guidance):
    import torch

    g = torch.Generator(device = "cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    img = pipe(
        prompt = PROMPT,
        width = res,
        height = res,
        num_inference_steps = steps,
        guidance_scale = guidance,
        generator = g,
    ).images[0]
    torch.cuda.synchronize()
    return img, time.time() - t0


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def run(
    tag,
    steps,
    seed,
    res,
    guidance,
    iters,
    *,
    threshold = None,
    compile_ = True,
):
    import torch

    torch.compiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pipe = _load()
    if threshold is not None:
        from diffusers import FirstBlockCacheConfig
        try:
            pipe.transformer.enable_cache(FirstBlockCacheConfig(threshold = threshold))
        except Exception as exc:  # noqa: BLE001
            from diffusers.hooks import apply_first_block_cache
            apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold = threshold))
    if compile_:
        # FBCache's per-step decision is a graph break, so a cached run must compile with
        # fullgraph=False (mirroring production); fullgraph=True would fail the warmup compile and
        # the row would fall back to an eager cached run, producing misleading speedups.
        fullgraph = threshold is None
        try:
            pipe.transformer.compile_repeated_blocks(fullgraph = fullgraph, dynamic = True)
        except Exception as exc:  # noqa: BLE001
            print(f"    [{tag}] compile {type(exc).__name__}: {str(exc)[:80]}", flush = True)
    try:
        _gen(pipe, steps, seed, res, guidance)  # warmup / compile
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"    [{tag}] FAILED: {type(exc).__name__}: {str(exc)[:100]}", flush = True)
        del pipe
        torch.cuda.empty_cache()
        return None
    dts, img = [], None
    for _ in range(iters):
        img, dt = _gen(pipe, steps, seed, res, guidance)
        dts.append(dt)
    peak = torch.cuda.max_memory_allocated() / 1e9
    arr = np.array(img)
    OUT.mkdir(parents = True, exist_ok = True)
    img.save(OUT / f"{tag}.png")
    del pipe
    torch.cuda.empty_cache()
    return _median(dts), arr, peak


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type = int, default = 28)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--guidance", type = float, default = 3.5)
    p.add_argument("--iters", type = int, default = 2)
    args = p.parse_args(argv)
    s, r, seed, gd, it = args.steps, args.res, args.seed, args.guidance, args.iters

    print(f"== FBCache on Flux.1-dev ({r}px, {s} steps, guidance {gd}) ==", flush = True)
    base = run("baseline", s, seed, r, gd, it)
    if base is None:
        print("baseline FAILED", flush = True)
        return 1
    bmed, ref, bpeak = base
    print(f"  baseline {bmed:.3f}s peak={bpeak:.1f}G", flush = True)
    rows = [("baseline", bmed, bpeak, 0.0)]
    for thr in (0.08, 0.12, 0.20):
        out = run(f"fbcache_{thr}", s, seed, r, gd, it, threshold = thr)
        if out is None:
            rows.append((f"fbcache_{thr}", None, None, None))
            continue
        med, arr, peak = out
        lp = _lpips(ref, arr)
        rows.append((f"fbcache_{thr}", med, peak, lp))
        print(
            f"  fbcache_{thr}: {med:.3f}s ({bmed/med:.2f}x) peak={peak:.1f}G LPIPS={lp}", flush = True
        )

    print("\n==== SUMMARY (Flux.1-dev, ref = no-cache compile) ====", flush = True)
    for tag, med, peak, lp in rows:
        if med is None:
            print(f"  {tag:16s} FAILED")
            continue
        spd = f"{bmed/med:.2f}x"
        lpv = "ref" if tag == "baseline" else (f"{lp:.3f}" if lp is not None else "n/a")
        print(f"  {tag:16s} {med:.3f}s  {spd:>6s}  peak={peak:.1f}G  LPIPS={lpv:>6s}", flush = True)
    print("FBCACHE-FLUX-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
