# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Empirical quant probe: torchao int8/fp8/fp4 dynamic quant vs GGUF+compile.

Question this answers: GGUF stores the Z-Image DiT at 4-bit but dequantizes to bf16
per matmul, so it runs at bf16 tensor-core rate. Can a low-precision *tensor-core*
path (int8dq on any Ampere+, fp8dq on Ada+, NVFP4/MXFP8 on Blackwell), loaded from
the dense bf16 transformer, beat GGUF+compile on speed while staying inside the
quality bar -- and how does its quality compare to GGUF's own 4-bit loss?

Reference for all quality numbers is the DENSE bf16 EAGER image (the best this model
can do). Each config is a fresh pipeline (no compile/quant cross-contamination).
Reports median latency, PSNR + LPIPS vs reference, and peak VRAM. Run on one CUDA GPU.
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
OUT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research" / "probe_images"


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else float(10 * np.log10(255.0**2 / mse))


_LPIPS = {"fn": None}


def _lpips(ref_arr, arr):
    """Perceptual LPIPS (alexnet) vs reference; lower is closer. None if unavailable.

    Runs on CPU so the scorer never holds CUDA memory: each row resets peak VRAM, so a
    resident GPU LPIPS module would inflate the reported load/gen VRAM and could even OOM."""
    try:
        import torch
        import lpips

        if _LPIPS["fn"] is None:
            _LPIPS["fn"] = lpips.LPIPS(net = "alex", verbose = False).eval()

        def t(x):
            return torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return float(_LPIPS["fn"](t(ref_arr), t(arr)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"    (lpips unavailable: {type(exc).__name__}: {str(exc)[:80]})", flush = True)
        return None


def _load_dense():
    import torch
    import diffusers

    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
    )
    pipe = diffusers.ZImagePipeline.from_pretrained(BASE, torch_dtype = torch.bfloat16, transformer = t)
    pipe.to("cuda")
    return pipe


def _load_gguf():
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


def _quant_config(name):
    """Return a torchao config instance for `name`, or raise to mark FAILED."""
    from torchao.quantization import (
        Int8WeightOnlyConfig,
        Int8DynamicActivationInt8WeightConfig,
        Float8DynamicActivationFloat8WeightConfig,
    )

    if name == "int8wo":
        return Int8WeightOnlyConfig()
    if name == "int8dq":
        return Int8DynamicActivationInt8WeightConfig()
    if name == "fp8dq":
        return Float8DynamicActivationFloat8WeightConfig()
    if name == "nvfp4":
        from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
        return NVFP4DynamicActivationNVFP4WeightConfig()
    if name == "mxfp8":
        from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
        try:
            import torch
            return MXDynamicActivationMXWeightConfig(
                activation_dtype = torch.float8_e4m3fn, weight_dtype = torch.float8_e4m3fn
            )
        except (TypeError, AttributeError):
            return MXDynamicActivationMXWeightConfig()
    raise ValueError(name)


def _make_filter_fn(min_features):
    """Keep only the FLOP-heavy linears: nn.Linear with both in/out >= min_features.
    The int8 dynamic path uses torch._int_mm (needs activation M>16), and the tiny
    timestep/pooled projections (in_features=256) run at M=1 and crash it -- skip them."""
    import torch.nn as nn

    def filter_fn(module, fqn = ""):
        return (
            isinstance(module, nn.Linear)
            and getattr(module, "in_features", 0) >= min_features
            and getattr(module, "out_features", 0) >= min_features
        )

    return filter_fn


def _apply_quant(pipe, name, log, min_features):
    import torch.nn as nn
    from torchao.quantization import quantize_

    cfg = _quant_config(name)
    total = sum(1 for m in pipe.transformer.modules() if isinstance(m, nn.Linear))
    filt = _make_filter_fn(min_features)
    q = sum(1 for n, m in pipe.transformer.named_modules() if filt(m, n))
    quantize_(pipe.transformer, cfg, filter_fn = filt)
    log(f"    quantized transformer with {name} ({q}/{total} linears >= {min_features} feat)")


def _compile(pipe, log):
    fn = getattr(pipe.transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        return False
    for kw in ({"fullgraph": True, "dynamic": True}, {"dynamic": True}, {}):
        try:
            fn(**kw)
            log(f"    compiled repeated blocks {kw}")
            return True
        except Exception as exc:  # noqa: BLE001
            log(f"    compile {kw} failed: {type(exc).__name__}: {str(exc)[:90]}")
    return False


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
    p.add_argument(
        "--min-feat",
        type = int,
        default = 512,
        help = "only quantize Linear with in&out features >= this (int8 _int_mm needs M>16)",
    )
    p.add_argument(
        "--configs",
        default = "bf16,bf16_c,gguf_c,int8dq_c,fp8dq_c,nvfp4_c,mxfp8_c,int8wo_c",
        help = "comma list; suffix _c = +compile",
    )
    args = p.parse_args(argv)
    steps, res, seed, iters = args.steps, args.res, args.seed, args.iters

    import torch

    OUT.mkdir(parents = True, exist_ok = True)

    def run(
        tag,
        *,
        source,
        quant = None,
        compile = False,
    ):
        torch.compiler.reset()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        pipe = _load_dense() if source == "dense" else _load_gguf()
        load_peak = torch.cuda.max_memory_allocated() / 1e9
        if quant is not None:
            _apply_quant(pipe, quant, print_, args.min_feat)
        if compile:
            _compile(pipe, print_)
            _gen(pipe, steps, seed, res)  # warmup / compilation
        else:
            _gen(pipe, steps, seed, res)  # allocator warmup
        torch.cuda.reset_peak_memory_stats()
        dts, img = [], None
        for _ in range(iters):
            img, dt = _gen(pipe, steps, seed, res)
            dts.append(dt)
        gen_peak = torch.cuda.max_memory_allocated() / 1e9
        arr = np.array(img)
        img.save(OUT / f"{tag}.png")
        del pipe
        torch.cuda.empty_cache()
        return tag, _median(dts), arr, load_peak, gen_peak

    print_ = lambda s: print(s, flush = True)  # noqa: E731

    # config table: tag -> (source, quant, compile)
    table = {
        "bf16": ("dense", None, False),
        "bf16_c": ("dense", None, True),
        "gguf_c": ("gguf", None, True),
        "int8wo_c": ("dense", "int8wo", True),
        "int8dq_c": ("dense", "int8dq", True),
        "fp8dq_c": ("dense", "fp8dq", True),
        "nvfp4_c": ("dense", "nvfp4", True),
        "mxfp8_c": ("dense", "mxfp8", True),
    }
    want = [c.strip() for c in args.configs.split(",") if c.strip()]

    print(f"== quant probe (Z-Image-Turbo, {res}px, {steps} steps, seed {seed}) ==", flush = True)
    ref_arr = None
    rows = []
    for tag in want:
        if tag not in table:
            print(f"  {tag}: unknown config, skipping", flush = True)
            continue
        source, quant, compile = table[tag]
        print(f"-- {tag} (source={source} quant={quant} compile={compile}) --", flush = True)
        try:
            _, med, arr, lp, gp = run(tag, source = source, quant = quant, compile = compile)
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"  {tag:10s} FAILED: {type(exc).__name__}: {str(exc)[:160]}", flush = True)
            traceback.print_exc()
            rows.append((tag, None, None, None, None, None))
            continue
        if ref_arr is None and tag == "bf16":
            ref_arr = arr
        psnr = _psnr(ref_arr, arr) if ref_arr is not None else None
        lpips_v = (
            _lpips(ref_arr, arr)
            if (ref_arr is not None and tag != "bf16")
            else (0.0 if tag == "bf16" else None)
        )
        rows.append((tag, med, psnr, lpips_v, lp, gp))
        ps = f"{psnr:.1f}dB" if psnr is not None else "n/a"
        lps = f"{lpips_v:.3f}" if lpips_v is not None else "n/a"
        print(
            f"  {tag:10s} {med:.3f}s  PSNR={ps:>7s}  LPIPS={lps:>6s}  loadVRAM={lp:.1f}G genVRAM={gp:.1f}G",
            flush = True,
        )

    base = next((r[1] for r in rows if r[0] == "bf16" and r[1]), None)
    gguf = next((r[1] for r in rows if r[0] == "gguf_c" and r[1]), None)
    print("\n==== SUMMARY (ref = bf16 dense eager) ====", flush = True)
    print(
        f"{'config':10s} {'sec':>7s} {'vs_bf16':>8s} {'vs_gguf':>8s} {'PSNR':>8s} {'LPIPS':>7s} {'loadG':>6s} {'genG':>6s}",
        flush = True,
    )
    for tag, med, psnr, lpips_v, lp, gp in rows:
        if med is None:
            print(f"{tag:10s} {'FAILED':>7s}", flush = True)
            continue
        vb = f"{base/med:.2f}x" if base else "-"
        vg = f"{gguf/med:.2f}x" if gguf else "-"
        ps = (
            f"{psnr:.1f}"
            if psnr is not None and psnr != float("inf")
            else ("inf" if psnr == float("inf") else "n/a")
        )
        lps = f"{lpips_v:.3f}" if lpips_v is not None else "n/a"
        print(
            f"{tag:10s} {med:>7.3f} {vb:>8s} {vg:>8s} {ps:>8s} {lps:>7s} {lp:>6.1f} {gp:>6.1f}",
            flush = True,
        )
    print("QUANT-PROBE-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    sys.exit(main())
