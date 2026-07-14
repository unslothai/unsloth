# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure the next-phase diffusion levers on the real model, vs today's compiled baseline.

Variants (Z-Image dense bf16, regional compile = the shipped "default" speed profile):
  baseline            -- channels_last + compile_repeated_blocks (reference image)
  inductor_flags      -- + the lossless inductor autotune flags (conv_1x1_as_mm,
                         coordinate_descent_tuning(+all_dirs), epilogue_fusion=False)
  attn_cudnn          -- + set_attention_backend("_native_cudnn") (exact)
  attn_flash4         -- + set_attention_backend("flash_4_hub")   (exact, SM100)
  attn_sage           -- + set_attention_backend("sage")          (INT8 QK, quantized)
  fbcache             -- + First-Block-Cache (threshold 0.12)      (few-step headroom test)

Reports median latency, vs-baseline speedup, peak VRAM, and LPIPS vs baseline. One CUDA GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research" / "perf_levers_images"


_LP = {"fn": None}


def _lpips(ref, arr):
    try:
        import lpips
        import torch

        # Keep the metric model on CPU: caching it on CUDA leaves it resident across variants, and
        # each run resets peak-memory stats, so its VRAM would be charged to (and reduce headroom
        # for) every later variant's measurement.
        if _LP["fn"] is None:
            _LP["fn"] = lpips.LPIPS(net = "alex", verbose = False).eval()

        def t(x):
            return torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return float(_LP["fn"](t(ref), t(arr)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"    (lpips: {type(exc).__name__})", flush = True)
        return None


def _set_inductor_flags():
    import torch._inductor.config as ic

    ic.conv_1x1_as_mm = True
    ic.coordinate_descent_tuning = True
    ic.coordinate_descent_check_all_directions = True
    ic.epilogue_fusion = False
    try:
        ic.force_fuse_int_mm_with_mul = True
    except Exception:  # noqa: BLE001
        pass


def _reset_inductor_flags():
    import torch._inductor.config as ic

    ic.conv_1x1_as_mm = False
    ic.coordinate_descent_tuning = False
    ic.coordinate_descent_check_all_directions = False
    ic.epilogue_fusion = True
    # Reset the int-mm fusion flag too, or it leaks from the inductor_flags variant into
    # every later compiled row and the attention/fbcache measurements stop being isolated.
    try:
        ic.force_fuse_int_mm_with_mul = False
    except Exception:  # noqa: BLE001
        pass


def _load():
    import diffusers
    import torch

    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    pipe = diffusers.ZImagePipeline.from_pretrained(BASE, torch_dtype = torch.bfloat16, transformer = t)
    pipe.to("cuda")
    try:
        pipe.vae.to(memory_format = torch.channels_last)
    except Exception:  # noqa: BLE001
        pass
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


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def run(
    tag,
    steps,
    seed,
    res,
    iters,
    *,
    attn = None,
    fbcache = None,
    inductor = False,
):
    import torch

    torch.compiler.reset()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _reset_inductor_flags()
    if inductor:
        _set_inductor_flags()
    pipe = _load()
    note = ""
    if attn is not None:
        try:
            pipe.transformer.set_attention_backend(attn)
        except Exception as exc:  # noqa: BLE001
            note = f"attn({attn})={type(exc).__name__}:{str(exc)[:60]}"
            print(f"    [{tag}] {note}", flush = True)
            del pipe  # free the resident pipe so a skipped variant doesn't leak VRAM
            torch.cuda.empty_cache()
            return None
    else:
        # set_attention_backend pins diffusers' PROCESS-WIDE active backend, and a fresh
        # transformer's processors (backend None) inherit it. Force native for the no-attn variants
        # so they aren't measured under a prior variant's kernel (e.g. fbcache with a leftover sage
        # backend).
        try:
            pipe.transformer.set_attention_backend("native")
        except Exception as exc:  # noqa: BLE001 — best-effort isolation
            print(
                f"    [{tag}] attn(native-reset)={type(exc).__name__}:{str(exc)[:60]}", flush = True
            )
    if fbcache is not None:
        try:
            from diffusers.hooks import FirstBlockCacheConfig, apply_first_block_cache
            apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold = fbcache))
        except Exception as exc:  # noqa: BLE001
            print(f"    [{tag}] fbcache={type(exc).__name__}:{str(exc)[:60]}", flush = True)
            del pipe
            torch.cuda.empty_cache()
            return None
    try:
        pipe.transformer.compile_repeated_blocks(fullgraph = True, dynamic = True)
    except Exception as exc:  # noqa: BLE001
        print(f"    [{tag}] compile={type(exc).__name__}:{str(exc)[:60]}", flush = True)
    try:
        _gen(pipe, steps, seed, res)  # warmup / compile
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"    [{tag}] FAILED first gen: {type(exc).__name__}:{str(exc)[:80]}", flush = True)
        del pipe
        torch.cuda.empty_cache()
        return None
    dts, img = [], None
    for _ in range(iters):
        img, dt = _gen(pipe, steps, seed, res)
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
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--iters", type = int, default = 3)
    args = p.parse_args(argv)
    s, r, seed, it = args.steps, args.res, args.seed, args.iters

    print(f"== perf levers (Z-Image dense, {r}px, {s} steps) ==", flush = True)
    base = run("baseline", s, seed, r, it)
    if base is None:
        print("baseline FAILED", flush = True)
        return 1
    bmed, ref, bpeak = base
    print(f"  baseline {bmed:.3f}s  peak={bpeak:.1f}G", flush = True)
    rows = [("baseline", bmed, bpeak, 0.0)]

    variants = [
        ("inductor_flags", dict(inductor = True)),
        ("attn_cudnn", dict(attn = "_native_cudnn")),
        ("attn_flash4", dict(attn = "flash_4_hub")),
        ("attn_sage", dict(attn = "sage")),
        ("attn_sage_inductor", dict(attn = "sage", inductor = True)),
        ("fbcache_0p12", dict(fbcache = 0.12)),
    ]
    for tag, kw in variants:
        out = run(tag, s, seed, r, it, **kw)
        if out is None:
            rows.append((tag, None, None, None))
            continue
        med, arr, peak = out
        lp = _lpips(ref, arr)
        rows.append((tag, med, peak, lp))
        spd = f"{bmed/med:.2f}x" if med else "-"
        print(f"  {tag:20s} {med:.3f}s ({spd} vs base) peak={peak:.1f}G LPIPS={lp}", flush = True)

    print("\n==== SUMMARY (ref = baseline compile) ====", flush = True)
    for tag, med, peak, lp in rows:
        if med is None:
            print(f"  {tag:20s} FAILED")
            continue
        spd = f"{bmed/med:.2f}x" if med else "-"
        lpv = "ref" if (tag == "baseline") else (f"{lp:.3f}" if lp is not None else "n/a")
        print(f"  {tag:20s} {med:.3f}s  {spd:>6s}  peak={peak:.1f}G  LPIPS={lpv:>6s}", flush = True)
    print("PERF-LEVERS-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
