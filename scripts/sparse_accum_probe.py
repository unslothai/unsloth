# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe two consumer-GPU-motivated levers on the real dense Z-Image transformer:

  * fp8 fast_accum on/off -- on consumer Blackwell, fp8 with FP16 accumulate is ~2x
    fp8 with FP32 accumulate (838 vs 419 TFLOPS). torchao defaults use_fast_accum=True,
    so this confirms we are already on the fast path and quantifies it (muted on a B200,
    which is not nerfed, but the knob still moves latency).
  * 2:4 semi-structured sparsity -- doubles tensor-core rate in theory. Two blockers to
    test empirically: (a) QUALITY -- inference-only 2:4 magnitude-pruning drops 50% of
    weights with no fine-tune; (b) it does NOT compose with torch.compile, so the real
    sparse path runs eager. We measure sparse-no-compile speed vs our fp8+compile
    baseline (the bar it must beat) and the LPIPS of 2:4 pruning.

Reference for quality is the dense bf16 eager image. Run on one CUDA GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path("/mnt/disks/unslothai/ubuntu/workspace_81/outputs/quant_research/sparse_images")


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


def _big_linears(transformer, min_feat = 512):
    import torch.nn as nn
    return [
        m
        for m in transformer.modules()
        if isinstance(m, nn.Linear) and m.in_features >= min_feat and m.out_features >= min_feat
    ]


def _prune_24_(transformer, min_feat = 512):
    """In-place 2:4 magnitude prune (zero the 2 smallest of every 4 along in_features)
    of the FLOP-heavy linears. Dense format -> measures the QUALITY of 2:4 with no kernel."""
    import torch

    n = 0
    for lin in _big_linears(transformer, min_feat):
        w = lin.weight.data
        o, i = w.shape
        if i % 4:
            continue
        g = w.view(o, i // 4, 4)
        idx = g.abs().argsort(dim = -1)[..., :2]
        g.scatter_(-1, idx, 0.0)
        n += 1
    return n


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
    args = p.parse_args(argv)
    steps, res, seed, mf = args.steps, args.res, args.seed, args.min_feat
    import torch

    OUT.mkdir(parents = True, exist_ok = True)

    def filt(mod, fqn = ""):
        import torch.nn as nn
        return isinstance(mod, nn.Linear) and mod.in_features >= mf and mod.out_features >= mf

    def run(
        tag,
        *,
        quant = None,
        fast_accum = True,
        prune = False,
        real_sparse = False,
        compile = True,
    ):
        torch.compiler.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pipe = _load_dense()
        note = ""
        if prune or real_sparse:
            n = _prune_24_(pipe.transformer, mf)
            note += f" pruned24={n}"
        if real_sparse:
            from torchao.sparsity import sparsify_, semi_sparse_weight
            sparsify_(pipe.transformer, semi_sparse_weight(), filter_fn = filt)
            note += " +semi_sparse"
        if quant == "fp8":
            from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
            from torchao.float8 import Float8MMConfig

            cfg = Float8DynamicActivationFloat8WeightConfig(
                mm_config = Float8MMConfig(use_fast_accum = fast_accum)
            )
            quantize_(pipe.transformer, cfg, filter_fn = filt)
            note += f" fp8(fast_accum={fast_accum})"
        if compile:
            try:
                pipe.transformer.compile_repeated_blocks(fullgraph = True, dynamic = True)
            except Exception as exc:  # noqa: BLE001
                note += f" [compile FAILED {type(exc).__name__}]"
        print(f"  [{tag}]{note}", flush = True)
        _gen(pipe, steps, seed, res)  # warmup / compile
        dts = []
        img = None
        for _ in range(args.iters):
            img, dt = _gen(pipe, steps, seed, res)
            dts.append(dt)
        gp = torch.cuda.max_memory_allocated() / 1e9
        arr = np.array(img)
        img.save(OUT / f"{tag}.png")
        del pipe
        torch.cuda.empty_cache()
        return _median(dts), arr, gp

    print(
        f"== sparse/accum probe (Z-Image dense, {res}px, {steps} steps, min_feat={mf}) ==",
        flush = True,
    )
    rows = []
    # quality reference: dense bf16 eager (no compile, no quant)
    bref, ref, _ = run("bf16_eager", compile = False)
    rows.append(("bf16_eager", bref, float("inf"), 0.0, None))
    print(f"  bf16 eager ref: {bref:.3f}s", flush = True)

    specs = [
        ("bf16_compile", dict()),
        ("fp8_fastT_c", dict(quant = "fp8", fast_accum = True)),
        ("fp8_fastF_c", dict(quant = "fp8", fast_accum = False)),
        (
            "fake24_fp8_c",
            dict(quant = "fp8", fast_accum = True, prune = True),
        ),  # quality of 2:4+fp8 (fake=no kernel)
        (
            "real24_nocompile",
            dict(real_sparse = True, compile = False),
        ),  # sparse SPEED (no quant, no compile)
        (
            "real24_compile_try",
            dict(real_sparse = True, compile = True),
        ),  # does sparse survive compile?
    ]
    for tag, kw in specs:
        try:
            med, arr, gp = run(tag, **kw)
            ps, lp = _psnr(ref, arr), _lpips(ref, arr)
            rows.append((tag, med, ps, lp, gp))
            print(
                f"  {tag:18s} {med:.3f}s ({bref/med:.2f}x vs eager) PSNR={ps:.1f} LPIPS={lp} VRAM={gp:.1f}G",
                flush = True,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"  {tag:18s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush = True)
            rows.append((tag, None, None, None, None))

    print("\n==== SUMMARY (ref = bf16 dense eager) ====", flush = True)
    base = next((r[1] for r in rows if r[0] == "fp8_fastT_c" and r[1]), None)
    for tag, med, ps, lp, gp in rows:
        if med is None:
            print(f"  {tag:18s} FAILED")
            continue
        vs_eager = f"{bref/med:.2f}x"
        vs_fp8 = f"{base/med:.2f}x" if base else "-"
        psv = "inf" if ps == float("inf") else f"{ps:.1f}"
        lpv = (
            "ref"
            if (lp == 0.0 and tag == "bf16_eager")
            else (f"{lp:.3f}" if lp is not None else "n/a")
        )
        print(
            f"  {tag:18s} {med:.3f}s  eager:{vs_eager:>6s}  fp8:{vs_fp8:>6s}  PSNR={psv:>5s}  LPIPS={lpv:>6s}",
            flush = True,
        )
    print("SPARSE-ACCUM-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
