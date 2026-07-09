#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Is the Hunyuan attention trim LESS accurate, or just DIFFERENT? Neither bf16-masked (stock)
nor bf16-trim is ground truth. Compare BOTH against an fp32 reference (same seed): if the trim is
as close to fp32 as stock is, the 0.14 LPIPS stock-vs-trim is a benign bf16-kernel resample, not a
quality loss. If trim is clearly farther from fp32 than stock, it is a real regression.

Run: CUDA_VISIBLE_DEVICES=1 python scripts/hunyuan_trim_fp32ref.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))


def _import_diffusers():
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


def _gen(pipe, seed, fr, st, w, h):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(prompt="a cat playing piano on a stage, cinematic",
               num_frames=fr, width=w, height=h, num_inference_steps=st,
               generator=g, output_type="np")
    return np.asarray(out.frames[0])


def _lpips_mean(loss_fn, a, b, stride=3):
    vals = []
    for i in range(0, len(a), stride):
        ta = torch.from_numpy(a[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
        tb = torch.from_numpy(b[i]).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
        with torch.no_grad():
            vals.append(loss_fn(ta, tb).item())
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--w", type=int, default=832)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    diffusers = _import_diffusers()
    from core.inference.diffusion_attention import install_hunyuan_attention_trim
    from core.inference.video_families import detect_video_family
    import lpips

    fam = detect_video_family(args.repo) or detect_video_family("hunyuanvideo-1.5")
    loss_fn = lpips.LPIPS(net="alex").cuda()
    fr, st, w, h, seed = args.frames, args.steps, args.w, args.h, args.seed

    # ---- bf16 stock + bf16 trim on one pipe ----
    print(f"loading bf16 {args.repo} ...", flush=True)
    pipe = diffusers.DiffusionPipeline.from_pretrained(args.repo, torch_dtype=torch.bfloat16).to("cuda")
    print(f"[bf16 stock] gen {fr}f/{st}steps ...", flush=True)
    stock = _gen(pipe, seed, fr, st, w, h)
    install_hunyuan_attention_trim(pipe, fam, logger=None)
    print("[bf16 trim ] gen ...", flush=True)
    trim = _gen(pipe, seed, fr, st, w, h)
    del pipe
    gc.collect(); torch.cuda.empty_cache()

    # ---- fp32 reference (stock masked attention, upcast) ----
    print("loading fp32 reference ...", flush=True)
    pipe32 = diffusers.DiffusionPipeline.from_pretrained(args.repo, torch_dtype=torch.float32).to("cuda")
    print(f"[fp32 gold ] gen {fr}f/{st}steps ...", flush=True)
    gold = _gen(pipe32, seed, fr, st, w, h)

    d_stock = _lpips_mean(loss_fn, gold, stock)
    d_trim = _lpips_mean(loss_fn, gold, trim)
    d_st = _lpips_mean(loss_fn, stock, trim)
    print("\n===== ACCURACY vs fp32 reference =====", flush=True)
    print(f"  LPIPS(fp32, bf16-stock) = {d_stock:.5f}", flush=True)
    print(f"  LPIPS(fp32, bf16-trim ) = {d_trim:.5f}", flush=True)
    print(f"  LPIPS(bf16-stock, trim) = {d_st:.5f}", flush=True)
    if d_stock is not None and d_trim is not None:
        verdict = "NOT less accurate (trim ~= stock vs fp32)" if d_trim <= d_stock * 1.25 + 0.01 \
            else "LESS accurate (trim farther from fp32 than stock)"
        print(f"\n  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
