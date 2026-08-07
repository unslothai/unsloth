# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does a *pre-quantized* checkpoint fix the dense-quant load-VRAM spike?

The current fast-transformer path materialises the dense bf16 transformer on the GPU
and quantises it in place -> ~2x the GGUF load peak. This probe checks the fix: quantise
once, ``torch.save`` the quantized state dict, then load it onto an empty (meta) model
with ``load_state_dict(assign=True)`` so the bf16 never touches the GPU.

Modes (run each in its own process so peak VRAM is clean):
  build     -- load dense bf16, quantize_ fp8, torch.save the state dict + on-disk size.
  baseline  -- current path: from_pretrained bf16 -> quantize_ on GPU. Report load peak + gen.
  prequant  -- meta-init -> load_state_dict(saved, assign=True) -> cuda. Report load peak + gen.

Run on one CUDA (Blackwell) GPU. Reference image for LPIPS is the baseline path."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

BASE = "Tongyi-MAI/Z-Image-Turbo"
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
ROOT = Path(__file__).resolve().parent.parent / "outputs" / "quant_research"
CKPT = ROOT / "prequant_fp8" / "transformer_fp8_state.pt"
OUT = ROOT / "prequant_images"
MIN_FEAT = 512


def _filt(mod, fqn = ""):
    import torch.nn as nn
    return (
        isinstance(mod, nn.Linear) and mod.in_features >= MIN_FEAT and mod.out_features >= MIN_FEAT
    )


def _fp8_cfg():
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
    return Float8DynamicActivationFloat8WeightConfig()


def _build():
    import torch
    import diffusers
    from torchao.quantization import quantize_

    torch.cuda.reset_peak_memory_stats()
    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
    ).to("cuda")
    quantize_(t, _fp8_cfg(), filter_fn = _filt)
    CKPT.parent.mkdir(parents = True, exist_ok = True)
    sd = t.state_dict()
    # move to cpu for a portable, gpu-free checkpoint
    sd = {k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in sd.items()}
    torch.save(sd, CKPT)
    sz = CKPT.stat().st_size / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[build] saved {CKPT.name}  on-disk={sz:.2f} GB  build_gpu_peak={peak:.1f} GB", flush = True
    )
    return 0


def _make_pipe_from_transformer(t):
    import diffusers
    import torch

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


def _baseline(steps, seed, res):
    import torch
    import diffusers
    from torchao.quantization import quantize_

    torch.cuda.reset_peak_memory_stats()
    t = diffusers.ZImageTransformer2DModel.from_pretrained(
        BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
    ).to("cuda")
    quantize_(t, _fp8_cfg(), filter_fn = _filt)
    load_peak = torch.cuda.max_memory_allocated() / 1e9
    pipe = _make_pipe_from_transformer(t)
    img, dt = _gen(pipe, steps, seed, res)  # warmup
    img, dt = _gen(pipe, steps, seed, res)
    OUT.mkdir(parents = True, exist_ok = True)
    img.save(OUT / "baseline.png")
    print(f"[baseline] transformer_load_gpu_peak={load_peak:.1f} GB  gen={dt:.3f}s", flush = True)
    return 0


def _prequant(steps, seed, res):
    import torch
    import diffusers
    from accelerate import init_empty_weights

    if not CKPT.exists():
        print(f"[prequant] missing checkpoint {CKPT}; run --mode build first", flush = True)
        return 1
    torch.cuda.reset_peak_memory_stats()
    cfg = diffusers.ZImageTransformer2DModel.load_config(BASE, subfolder = "transformer")
    with init_empty_weights():
        t = diffusers.ZImageTransformer2DModel.from_config(cfg)
    sd = torch.load(CKPT, weights_only = False, map_location = "cpu")
    missing, unexpected = t.load_state_dict(sd, strict = False, assign = True)
    # any param/buffer still on meta (e.g. non-persistent buffers) -> materialise on cuda
    leftover = [n for n, p in t.named_parameters() if p.is_meta] + [
        n for n, b in t.named_buffers() if b.is_meta
    ]
    if leftover:
        print(
            f"[prequant] {len(leftover)} meta leftovers (non-persistent buffers): {leftover[:4]}",
            flush = True,
        )
        t = t.to_empty(device = "cuda")  # fallback path; re-loads sd below
        t.load_state_dict(sd, strict = False, assign = True)
    t = t.to(torch.bfloat16).to("cuda")
    load_peak = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[prequant] missing={len(missing)} unexpected={len(unexpected)} "
        f"transformer_load_gpu_peak={load_peak:.1f} GB",
        flush = True,
    )
    pipe = _make_pipe_from_transformer(t)
    img, dt = _gen(pipe, steps, seed, res)  # warmup
    img, dt = _gen(pipe, steps, seed, res)
    OUT.mkdir(parents = True, exist_ok = True)
    img.save(OUT / "prequant.png")
    # LPIPS vs baseline if present
    bpath = OUT / "baseline.png"
    lp = None
    if bpath.exists():
        try:
            import lpips
            from PIL import Image

            fn = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()

            def tt(p):
                a = np.array(Image.open(p).convert("RGB"))
                return (
                    torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
                ).cuda()

            with torch.no_grad():
                lp = float(fn(tt(bpath), tt(OUT / "prequant.png")).item())
        except Exception as exc:  # noqa: BLE001
            print(f"  (lpips: {type(exc).__name__})", flush = True)
    print(f"[prequant] gen={dt:.3f}s  LPIPS_vs_baseline={lp}", flush = True)
    return 0


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices = ["build", "baseline", "prequant"], required = True)
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    args = p.parse_args(argv)
    if args.mode == "build":
        return _build()
    if args.mode == "baseline":
        return _baseline(args.steps, args.seed, args.res)
    return _prequant(args.steps, args.seed, args.res)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "studio" / "backend"))
    rc = main()
    print("PREQUANT-PROBE-DONE", flush = True)
    sys.exit(rc)
