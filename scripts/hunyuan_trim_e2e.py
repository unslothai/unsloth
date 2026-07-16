#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""End-to-end pixel validation of the HunyuanVideo-1.5 attention trim: same seed, stock vs trim,
per-frame LPIPS + mean luma (black-frame guard), plus a full-resolution trim gen for real
wall-clock. Stock is only run at MODEST settings (a full 121-frame stock gen is ~19 min).

Run: CUDA_VISIBLE_DEVICES=3 python scripts/hunyuan_trim_e2e.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))
OUT = _REPO_ROOT / "outputs" / "video_speedmem"
OUT.mkdir(parents = True, exist_ok = True)


def _import_diffusers():
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


def _gen(pipe, seed, frames, steps, w, h):
    g = torch.Generator(device = "cuda").manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(
        prompt = "a cat playing piano on a stage, cinematic",
        num_frames = frames,
        width = w,
        height = h,
        num_inference_steps = steps,
        generator = g,
        output_type = "np",
    )
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) * 1e3
    frames_np = out.frames[0]  # [F,H,W,C] in [0,1]
    return np.asarray(frames_np), wall


def _luma(frames):
    # BT.601 luma over [F,H,W,C] in [0,1]
    r, gg, b = frames[..., 0], frames[..., 1], frames[..., 2]
    return float((0.299 * r + 0.587 * gg + 0.114 * b).mean())


def _lpips_mean(
    loss_fn,
    a,
    b,
    stride = 4,
):
    vals = []
    for i in range(0, len(a), stride):
        ta = torch.from_numpy(a[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        tb = torch.from_numpy(b[i]).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
        with torch.no_grad():
            vals.append(loss_fn(ta.cuda(), tb.cuda()).item())
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    ap.add_argument("--cmp-frames", type = int, default = 25)
    ap.add_argument("--cmp-steps", type = int, default = 50)
    ap.add_argument("--cmp-w", type = int, default = 832)
    ap.add_argument("--cmp-h", type = int, default = 480)
    ap.add_argument("--full-frames", type = int, default = 121)
    ap.add_argument("--full-steps", type = int, default = 50)
    ap.add_argument("--seed", type = int, default = 1234)
    args = ap.parse_args()

    diffusers = _import_diffusers()
    from core.inference.diffusion_attention import install_hunyuan_attention_trim
    from core.inference.video_families import detect_video_family
    import lpips

    print(f"loading {args.repo} ...", flush = True)
    pipe = diffusers.DiffusionPipeline.from_pretrained(args.repo, torch_dtype = torch.bfloat16).to(
        "cuda"
    )
    fam = detect_video_family(args.repo) or detect_video_family("hunyuanvideo-1.5")
    loss_fn = lpips.LPIPS(net = "alex").cuda()

    fr, st, w, hh = args.cmp_frames, args.cmp_steps, args.cmp_w, args.cmp_h
    # ---- STOCK x2 (nondeterminism floor) ----
    print(f"\n[stock#1] gen {fr}f/{st}steps {w}x{hh} ...", flush = True)
    stock1, w_stock = _gen(pipe, args.seed, fr, st, w, hh)
    print(
        f"[stock#1] wall={w_stock:.0f} ms  luma={_luma(stock1):.4f}  frames={stock1.shape}",
        flush = True,
    )
    print(f"[stock#2] gen (same seed, measures run-to-run nondeterminism) ...", flush = True)
    stock2, _ = _gen(pipe, args.seed, fr, st, w, hh)

    # ---- TRIM (same seed) ----
    engaged = install_hunyuan_attention_trim(pipe, fam, logger = None)
    print(f"\ninstall_hunyuan_attention_trim engaged={engaged}", flush = True)
    trim, w_trim = _gen(pipe, args.seed, fr, st, w, hh)
    print(
        f"[trim ] wall={w_trim:.0f} ms  luma={_luma(trim):.4f}  ({w_stock/w_trim:.2f}x vs stock)",
        flush = True,
    )

    floor = _lpips_mean(loss_fn, stock1, stock2)
    lp = _lpips_mean(loss_fn, stock1, trim)
    print(f"\nLPIPS(stock#1, stock#2) = {floor:.5f}   <- nondeterminism floor", flush = True)
    print(f"LPIPS(stock#1, trim  ) = {lp:.5f}   <- trim vs stock", flush = True)
    print(
        f"  => trim is {'WITHIN' if lp <= floor * 1.5 + 0.002 else 'ABOVE'} the nondeterminism floor",
        flush = True,
    )

    # ---- FULL-RES TRIM (wall-clock + black-frame guard; stock at full-res is ~19 min, skipped) ----
    print(
        f"\n[trim-full] gen {args.full_frames}f/{args.full_steps}steps {args.cmp_w}x{args.cmp_h} ...",
        flush = True,
    )
    full, w_full = _gen(pipe, args.seed, args.full_frames, args.full_steps, args.cmp_w, args.cmp_h)
    print(
        f"[trim-full] wall={w_full/1000:.1f} s  luma={_luma(full):.4f}  frames={full.shape}",
        flush = True,
    )

    try:
        from PIL import Image

        Image.fromarray((trim[0] * 255).astype("uint8")).save(OUT / "vid_hunyuan_trim_cmp.png")
        Image.fromarray((full[0] * 255).astype("uint8")).save(OUT / "vid_hunyuan_trim_full.png")
        print(f"\nsaved sample frames to {OUT}", flush = True)
    except Exception as exc:  # noqa: BLE001
        print(f"(png save skipped: {exc})", flush = True)


if __name__ == "__main__":
    main()
