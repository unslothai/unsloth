# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe NVFP4 via torchao with use_triton_kernel=False (no MSLK) on the real dense
Z-Image transformer: is it a genuine FP4-tensor-core speedup over fp8, and is quality
in-bar? Reference for LPIPS is dense bf16 eager. Run on one CUDA (Blackwell) GPU."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research" / "nvfp4_images"


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


_LP = {"fn": None}


def _lpips(ref, arr):
    try:
        import torch, lpips

        if _LP["fn"] is None:
            _LP["fn"] = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()

        def t(x):
            return (torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).cuda()

        with torch.no_grad():
            return float(_LP["fn"](t(ref), t(arr)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"    (lpips: {type(exc).__name__})", flush = True)
        return None


def _load_dense():
    import torch, diffusers

    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
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


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--iters", type = int, default = 3)
    p.add_argument("--min-feat", type = int, default = 512)
    p.add_argument("--out-dir", default = None, help = "image output dir (default: repo outputs/)")
    args = p.parse_args(argv)
    steps, res, seed, mf = args.steps, args.res, args.seed, args.min_feat
    import torch
    import torch.nn as nn

    global OUT
    if args.out_dir:
        OUT = Path(args.out_dir).expanduser()
    OUT.mkdir(parents = True, exist_ok = True)

    def filt(mod, fqn = ""):
        return isinstance(mod, nn.Linear) and mod.in_features >= mf and mod.out_features >= mf

    def run(
        tag,
        *,
        cfg = None,
        compile = True,
    ):
        torch.compiler.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pipe = _load_dense()
        if cfg is not None:
            from torchao.quantization import quantize_
            quantize_(pipe.transformer, cfg, filter_fn = filt)
        if compile:
            try:
                pipe.transformer.compile_repeated_blocks(fullgraph = True, dynamic = True)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    [{tag}] compile failed: {type(exc).__name__}: {str(exc)[:90]}", flush = True
                )
        _gen(pipe, steps, seed, res)  # warmup / compile
        dts, img = [], None
        for _ in range(args.iters):
            img, dt = _gen(pipe, steps, seed, res)
            dts.append(dt)
        gp = torch.cuda.max_memory_allocated() / 1e9
        arr = np.array(img)
        img.save(OUT / f"{tag}.png")
        del pipe
        torch.cuda.empty_cache()
        return _median(dts), arr, gp

    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as FP8
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as NV

    print(f"== nvfp4 probe (Z-Image dense, {res}px, {steps} steps, min_feat={mf}) ==", flush = True)
    bref, ref, _ = run("bf16_eager", cfg = None, compile = False)
    print(f"  bf16 eager ref: {bref:.3f}s", flush = True)
    rows = [("bf16_eager", bref, float("inf"), 0.0, None)]

    specs = [
        ("bf16_compile", None, True),
        ("fp8_compile", FP8(), True),
        ("nvfp4_notriton_compile", NV(use_triton_kernel = False), True),
        ("nvfp4_notriton_eager", NV(use_triton_kernel = False), False),
    ]
    for tag, cfg, comp in specs:
        try:
            med, arr, gp = run(tag, cfg = cfg, compile = comp)
            ps, lp = _psnr(ref, arr), _lpips(ref, arr)
            rows.append((tag, med, ps, lp, gp))
            print(
                f"  {tag:24s} {med:.3f}s ({bref/med:.2f}x vs eager) PSNR={ps:.1f} LPIPS={lp} VRAM={gp:.1f}G",
                flush = True,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  {tag:24s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush = True)
            rows.append((tag, None, None, None, None))

    fp8 = next((r[1] for r in rows if r[0] == "fp8_compile" and r[1]), None)
    print("\n==== SUMMARY (ref = bf16 dense eager) ====", flush = True)
    for tag, med, ps, lp, gp in rows:
        if med is None:
            print(f"  {tag:24s} FAILED")
            continue
        vs_fp8 = f"{fp8/med:.2f}x" if fp8 else "-"
        psv = "inf" if ps == float("inf") else f"{ps:.1f}"
        lpv = (
            "ref"
            if (lp == 0.0 and tag == "bf16_eager")
            else (f"{lp:.3f}" if lp is not None else "n/a")
        )
        print(
            f"  {tag:24s} {med:.3f}s  vs_fp8:{vs_fp8:>6s}  PSNR={psv:>5s}  LPIPS={lpv:>6s}",
            flush = True,
        )
    print("NVFP4-PROBE-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
