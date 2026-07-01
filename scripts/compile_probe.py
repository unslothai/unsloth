# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe: does regional ``torch.compile`` work on the GGUF diffusion transformer?

The speed layer gates ``compile_repeated_blocks`` OFF for GGUF (it dequantises
per-op). Since the backend is GGUF-only, that makes regional compile dead on
every shipping model. This probe loads a GGUF transformer exactly as
``diffusion.py`` does, runs an eager generation, then compiles the repeated
denoiser block and runs the same seed again, reporting: whether compile raised,
per-generation latency eager vs compiled, and PSNR(compiled vs eager). If compile
is clean and PSNR is high, the gate can be relaxed for this family.

Run on one CUDA GPU. Read-only w.r.t. the backend (does not import the gate).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _psnr(a: "np.ndarray", b: "np.ndarray") -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0**2) / mse))


def _gen(pipe, prompt, *, steps, seed, width, height, guidance):
    import torch

    gen = torch.Generator(device = "cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.time()
    image = pipe(
        prompt = prompt,
        width = width,
        height = height,
        num_inference_steps = steps,
        guidance_scale = guidance,
        generator = gen,
    ).images[0]
    torch.cuda.synchronize()
    return image, time.time() - t0


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default = "unsloth/Z-Image-Turbo-GGUF")
    p.add_argument("--gguf", default = "z-image-turbo-Q4_K_M.gguf")
    p.add_argument("--base-repo", default = "Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--transformer-class", default = "ZImageTransformer2DModel")
    p.add_argument("--pipeline-class", default = "ZImagePipeline")
    p.add_argument(
        "--prompt",
        default = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed",
    )
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--width", type = int, default = 1024)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--guidance", type = float, default = 0.0)
    p.add_argument(
        "--mode", default = "default", help = "compile mode: default | max-autotune-no-cudagraphs"
    )
    p.add_argument(
        "--dynamic", action = "store_true", help = "dynamic=True (default False here for speed)"
    )
    p.add_argument("--out-dir", default = "outputs/compile_probe")
    args = p.parse_args(argv)

    import torch
    import diffusers
    from huggingface_hub import hf_hub_download

    out = Path(args.out_dir)
    out.mkdir(parents = True, exist_ok = True)
    dtype = torch.bfloat16

    gguf_path = hf_hub_download(args.repo, args.gguf)
    print(f"gguf: {gguf_path}", flush = True)

    transformer_cls = getattr(diffusers, args.transformer_class)
    transformer = transformer_cls.from_single_file(
        gguf_path,
        quantization_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype),
        torch_dtype = dtype,
        config = args.base_repo,
        subfolder = "transformer",
    )
    pipeline_cls = getattr(diffusers, args.pipeline_class)
    pipe = pipeline_cls.from_pretrained(args.base_repo, torch_dtype = dtype, transformer = transformer)
    pipe.to("cuda")
    print("pipeline loaded on cuda", flush = True)

    # warm the eager path once (allocator / cudnn), then time eager.
    _gen(
        pipe,
        args.prompt,
        steps = args.steps,
        seed = args.seed,
        width = args.width,
        height = args.height,
        guidance = args.guidance,
    )
    eager_img, eager_t = _gen(
        pipe,
        args.prompt,
        steps = args.steps,
        seed = args.seed,
        width = args.width,
        height = args.height,
        guidance = args.guidance,
    )
    eager_img.save(out / "eager.png")
    eager_arr = np.array(eager_img)
    print(f"EAGER: {eager_t:.2f}s/gen", flush = True)

    # compile the repeated denoiser block.
    fn = getattr(pipe.transformer, "compile_repeated_blocks", None)
    if not callable(fn):
        print("RESULT: transformer has no compile_repeated_blocks -> N/A", flush = True)
        return 3
    compile_kwargs = {"fullgraph": True, "dynamic": bool(args.dynamic)}
    if args.mode and args.mode != "default":
        compile_kwargs["mode"] = args.mode
    print(f"compiling repeated blocks: {compile_kwargs} ...", flush = True)
    try:
        t0 = time.time()
        fn(**compile_kwargs)
        print(
            f"  compile_repeated_blocks() returned in {time.time()-t0:.1f}s (compilation is lazy)",
            flush = True,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT: compile_repeated_blocks RAISED: {type(exc).__name__}: {exc}", flush = True)
        return 1

    # first compiled gen triggers the actual compilation (untimed warmup).
    try:
        t0 = time.time()
        _gen(
            pipe,
            args.prompt,
            steps = args.steps,
            seed = args.seed,
            width = args.width,
            height = args.height,
            guidance = args.guidance,
        )
        print(f"  first compiled gen (compilation) took {time.time()-t0:.1f}s", flush = True)
    except Exception as exc:  # noqa: BLE001
        print(f"RESULT: first compiled generation RAISED: {type(exc).__name__}: {exc}", flush = True)
        return 2

    comp_img, comp_t = _gen(
        pipe,
        args.prompt,
        steps = args.steps,
        seed = args.seed,
        width = args.width,
        height = args.height,
        guidance = args.guidance,
    )
    comp_img.save(out / "compiled.png")
    psnr = _psnr(eager_arr, np.array(comp_img))

    speedup = (eager_t - comp_t) / eager_t * 100.0
    print("\n==== COMPILE PROBE RESULT ====", flush = True)
    print(f"  eager:    {eager_t:.2f}s/gen", flush = True)
    print(f"  compiled: {comp_t:.2f}s/gen   ({speedup:+.1f}% vs eager)", flush = True)
    print(f"  PSNR(compiled vs eager): {psnr:.1f} dB", flush = True)
    print(
        f"  verdict: {'COMPILE-WORKS' if psnr >= 30 else 'COMPILE-DIVERGES'} "
        f"{'FASTER' if comp_t < eager_t else 'NOT-FASTER'}",
        flush = True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
