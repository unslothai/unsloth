#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Validate the HunyuanVideo-1.5 padded-text attention trim: accuracy (stock vs trimmed forward
output on the SAME real inputs), per-forward speed, and torch.compile compatibility -- all at the
real production shape (default 121 frames / 480p).

Run: CUDA_VISIBLE_DEVICES=3 python scripts/hunyuan_trim_validate.py
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))


def _import_diffusers():
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


CAP: dict = {}


class _Stop(Exception):
    pass


def _capture_hook(module, args, kwargs):
    CAP["kwargs"] = {k: v for k, v in kwargs.items()}
    CAP["args"] = args
    raise _Stop


def _forward(transformer, no_grad = True):
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    with ctx:
        out = transformer(*CAP["args"], **CAP["kwargs"])
    return out[0] if isinstance(out, tuple) else out.sample


def _median_ms(
    fn,
    iters = 8,
    warmup = 2,
):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _compare(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    cos = torch.nn.functional.cosine_similarity(a, b, dim = 0).item()
    max_abs = (a - b).abs().max().item()
    denom = a.abs().max().item() or 1.0
    return cos, max_abs, max_abs / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v")
    ap.add_argument("--frames", type = int, default = 121)
    ap.add_argument("--width", type = int, default = 832)
    ap.add_argument("--height", type = int, default = 480)
    ap.add_argument("--compile", action = "store_true", help = "also test regional compile of blocks")
    args = ap.parse_args()

    diffusers = _import_diffusers()
    from core.inference.diffusion_attention import install_hunyuan_attention_trim
    from core.inference.video_families import detect_video_family

    dev = "cuda:0"
    print(f"loading {args.repo} ...", flush = True)
    pipe = diffusers.DiffusionPipeline.from_pretrained(args.repo, torch_dtype = torch.bfloat16).to(
        dev
    )
    fam = detect_video_family(args.repo) or detect_video_family("hunyuanvideo-1.5")
    print(
        f"family: {getattr(fam, 'name', None)} / transformer_class={getattr(fam, 'transformer_class', None)}",
        flush = True,
    )

    h = pipe.transformer.register_forward_pre_hook(_capture_hook, with_kwargs = True)
    print("capturing one real forward input ...", flush = True)
    try:
        pipe(
            prompt = "a cat playing piano",
            num_frames = args.frames,
            width = args.width,
            height = args.height,
            num_inference_steps = 1,
        )
    except _Stop:
        pass
    h.remove()

    k = CAP["kwargs"]
    ehs = k.get("encoder_hidden_states")
    m = k.get("encoder_attention_mask")
    ie = k.get("image_embeds")
    print(
        f"\ncaptured: encoder_hidden_states={tuple(ehs.shape)} mask_valid={m.bool().sum(1).tolist()}"
        f" image_embeds={tuple(ie.shape) if ie is not None else None}"
        f" image_all_zero={bool(torch.all(ie==0).item()) if ie is not None else None}",
        flush = True,
    )

    # ---- STOCK forward (reference) + timing ----
    transformer = pipe.transformer
    out_stock = _forward(transformer).detach().clone()
    t_stock = _median_ms(lambda: _forward(transformer))
    print(f"\nSTOCK   forward: {t_stock:8.2f} ms   out={tuple(out_stock.shape)}", flush = True)

    # ---- install trim, re-run same inputs ----
    engaged = install_hunyuan_attention_trim(pipe, fam, logger = None)
    print(f"install_hunyuan_attention_trim engaged = {engaged}", flush = True)
    out_trim = _forward(transformer).detach().clone()
    t_trim = _median_ms(lambda: _forward(transformer))

    cos, max_abs, rel = _compare(out_stock, out_trim)
    print(f"TRIM    forward: {t_trim:8.2f} ms   ({t_stock/t_trim:.2f}x faster)", flush = True)
    print(
        f"\nACCURACY stock-vs-trim:  cosine={cos:.8f}  max_abs={max_abs:.4e}  rel_max={rel:.4e}",
        flush = True,
    )
    finite = bool(torch.isfinite(out_trim).all().item())
    print(f"trim output finite: {finite}", flush = True)

    if args.compile:
        print(
            "\ncompiling blocks (compile_repeated_blocks, mode=default, dynamic=True) ...",
            flush = True,
        )
        try:
            for _a in ("recompile_limit", "cache_size_limit"):
                if hasattr(torch._dynamo.config, _a):
                    setattr(torch._dynamo.config, _a, 64)
            transformer.compile_repeated_blocks(fullgraph = False, dynamic = True)
            out_c = _forward(transformer).detach().clone()  # triggers compile
            t_c = _median_ms(lambda: _forward(transformer), iters = 5, warmup = 1)
            cos_c, ma_c, rel_c = _compare(out_stock, out_c)
            cnt = torch._dynamo.utils.counters
            print(f"TRIM+COMPILE forward: {t_c:8.2f} ms  ({t_stock/t_c:.2f}x vs stock)", flush = True)
            print(f"  accuracy vs stock: cosine={cos_c:.8f} max_abs={ma_c:.4e}", flush = True)
            print(
                f"  dynamo recompiles={sum(cnt['recompiles'].values()) if 'recompiles' in cnt else '?'}"
                f" graph_breaks={sum(cnt['graph_break'].values()) if 'graph_break' in cnt else 0}",
                flush = True,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"COMPILE FAILED: {type(exc).__name__}: {exc}", flush = True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
