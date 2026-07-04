# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""torch>=2.11 NVFP4 probe. Three parts:

  A. diagnostics  -- torch/torchao versions, cpp-extension load state, device.
  B. GEMM micro   -- isolated per-linear forward latency (bf16 / fp8 / nvfp4-cutlass /
                     nvfp4-triton) at Z-Image-like shapes, to measure raw FP4
                     tensor-core throughput free of pipeline overhead.
  C. end-to-end   -- real dense Z-Image transformer, latency + LPIPS + PSNR + VRAM,
                     reference = dense bf16 eager.

Run on one CUDA (Blackwell) GPU. This is the experiment that decides whether NVFP4
becomes a genuine speedup once torch>=2.11 + torchao's CUTLASS FP4 GEMM is present."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research" / "nvfp4_t211_images"


# ----------------------------------------------------------------------------- diag
def diagnostics() -> None:
    import torch
    import torchao

    print("== A. diagnostics ==", flush = True)
    print(f"  torch    {torch.__version__}", flush = True)
    print(f"  torchao  {torchao.__version__}", flush = True)
    print(f"  cuda     {torch.version.cuda}", flush = True)
    if torch.cuda.is_available():
        print(
            f"  device   {torch.cuda.get_device_name(0)} sm{torch.cuda.get_device_capability(0)}",
            flush = True,
        )
    print(f"  torch.ops.torchao present: {hasattr(torch.ops, 'torchao')}", flush = True)
    print(
        f"  fp4 primitives: e2m1={hasattr(torch, 'float4_e2m1fn_x2')} "
        f"e8m0={hasattr(torch, 'float8_e8m0fnu')} _scaled_mm={hasattr(torch, '_scaled_mm')}",
        flush = True,
    )
    # torchao prints "Skipping import of cpp extensions ..." to stderr at import on torch<2.11.
    # On 2.11 that line is absent -> the CUTLASS FP4 GEMM extension is live.
    print(
        "  (no 'Skipping import of cpp extensions' line above => cpp/CUTLASS ext loaded)",
        flush = True,
    )


# ----------------------------------------------------------------------------- micro
def _configs():
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as FP8
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as NV
    return {
        "bf16": None,
        "fp8": FP8(),
        "nvfp4_cutlass": NV(use_triton_kernel = False),
        "nvfp4_triton": NV(use_triton_kernel = True),
    }


def _bench_linear(K, N, M, cfg, iters, compile_):
    import torch
    import torch.nn as nn
    from torchao.quantization import quantize_

    torch.compiler.reset()
    torch.cuda.empty_cache()
    m = nn.Sequential(nn.Linear(K, N, bias = False)).cuda().to(torch.bfloat16)
    if cfg is not None:
        quantize_(m, cfg)
    fn = torch.compile(m, fullgraph = True, dynamic = False) if compile_ else m
    x = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        for _ in range(3):  # warmup / compile
            fn(x)
        torch.cuda.synchronize()
        dts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            torch.cuda.synchronize()
            dts.append(time.perf_counter() - t0)
    del m, fn, x
    torch.cuda.empty_cache()
    med = sorted(dts)[len(dts) // 2]
    tflops = 2.0 * M * K * N / med / 1e12
    return med, tflops


def micro(M, iters, compile_):
    print(f"\n== B. GEMM micro (M={M}, compile={compile_}, iters={iters}) ==", flush = True)
    # (K, N): qkv-ish, mlp-up, mlp-down for a ~3072-dim DiT
    shapes = [(3072, 3072), (3072, 12288), (12288, 3072)]
    cfgs = _configs()
    for K, N in shapes:
        print(f"  shape K={K} N={N}:", flush = True)
        base_ms = None
        fp8_ms = None
        for name, cfg in cfgs.items():
            try:
                med, tfl = _bench_linear(K, N, M, cfg, iters, compile_)
                ms = med * 1e3
                if name == "bf16":
                    base_ms = ms
                if name == "fp8":
                    fp8_ms = ms
                vs_bf16 = f"{base_ms/ms:.2f}x" if base_ms else "-"
                vs_fp8 = f"{fp8_ms/ms:.2f}x" if fp8_ms else "-"
                print(
                    f"    {name:16s} {ms:7.3f} ms  {tfl:7.1f} TFLOPS  vs_bf16={vs_bf16:>6s}  vs_fp8={vs_fp8:>6s}",
                    flush = True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    {name:16s} FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush = True)


# ----------------------------------------------------------------------------- e2e
def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


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


def _load_dense():
    import diffusers
    import torch

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


def e2e(steps, res, seed, iters, mf):
    import torch
    import torch.nn as nn

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
        for _ in range(iters):
            img, dt = _gen(pipe, steps, seed, res)
            dts.append(dt)
        gp = torch.cuda.max_memory_allocated() / 1e9
        arr = np.array(img)
        img.save(OUT / f"{tag}.png")
        del pipe
        torch.cuda.empty_cache()
        return _median(dts), arr, gp

    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig as NV
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig as FP8

    print(
        f"\n== C. end-to-end (Z-Image dense, {res}px, {steps} steps, min_feat={mf}) ==", flush = True
    )
    bref, ref, _ = run("bf16_eager", cfg = None, compile = False)
    print(f"  bf16 eager ref: {bref:.3f}s", flush = True)
    rows = [("bf16_eager", bref, float("inf"), 0.0, None)]

    specs = [
        ("bf16_compile", None, True),
        ("fp8_compile", FP8(), True),
        ("nvfp4_cutlass_compile", NV(use_triton_kernel = False), True),
        ("nvfp4_triton_compile", NV(use_triton_kernel = True), True),
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


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--iters", type = int, default = 3)
    p.add_argument("--micro-M", type = int, default = 4096)
    p.add_argument("--min-feat", type = int, default = 512)
    p.add_argument("--only", choices = ["diag", "micro", "e2e", "all"], default = "all")
    args = p.parse_args(argv)

    diagnostics()
    if args.only in ("micro", "all"):
        micro(args.micro_M, args.iters, compile_ = True)
    if args.only in ("e2e", "all"):
        e2e(args.steps, args.res, args.seed, args.iters, args.min_feat)
    print("NVFP4-T211-PROBE-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
