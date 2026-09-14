#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eager vs compiled VAE decode on the SAME final latent.

Loads the shipped image pipeline with the VAE-decode compile OFF, renders once at a fixed seed to
capture the final latent and the eager decode, then replays that exact call through
``torch.compile(decode, fullgraph=False, dynamic=True)`` (and, optionally,
``mode="max-autotune-no-cudagraphs"``). Reports the first-call compile wall, the steady decode time,
max-abs / mean-abs on the decoded tensor and LPIPS between the two PNGs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def main(argv = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required = True)
    ap.add_argument("--model", required = True)
    ap.add_argument("--arm", default = "nvfp4")
    ap.add_argument("--nvfp4-backend", default = "flashinfer")
    ap.add_argument("--prequant-path", default = None)
    ap.add_argument("--gguf-filename", default = None)
    ap.add_argument("--base-repo", default = None)
    ap.add_argument("--resolution", type = int, default = 1024)
    ap.add_argument("--steps", type = int, required = True)
    ap.add_argument("--guidance", type = float, default = None)
    ap.add_argument("--seed", type = int, default = 1234)
    ap.add_argument(
        "--prompt", default = "a red sailboat on a calm blue lake at sunrise, crisp detail"
    )
    ap.add_argument("--out", required = True)
    ap.add_argument("--png-dir", required = True)
    ap.add_argument(
        "--backend-root",
        default = None,
        help = "Studio backend root; this checkout's studio/backend otherwise",
    )
    args = ap.parse_args(argv)

    root = Path(args.backend_root) if args.backend_root else REPO_ROOT / "studio" / "backend"
    sys.path.insert(0, str(root))
    os.environ["UNSLOTH_DIFFUSION_COMPILE_VAE"] = "0"
    if args.arm == "nvfp4":
        os.environ["UNSLOTH_NVFP4_BACKEND"] = args.nvfp4_backend

    import torch
    from nvfp4_budget_profile import (
        LpipsScorer,
        _resolve_guidance,
        neutralise_bitsandbytes,
        patch_hub_compat,
    )

    rec: dict = {
        "family": args.family,
        "arm": args.arm,
        "resolution": args.resolution,
        "steps": args.steps,
        "seed": args.seed,
        "backend_root": str(root),
    }
    rec["bitsandbytes_neutralised"] = neutralise_bitsandbytes()
    rec["hub_compat"] = patch_hub_compat()

    from core.inference.diffusion import get_diffusion_backend

    backend = get_diffusion_backend()
    load_kwargs: dict = {
        "hf_token": os.environ.get("HF_TOKEN"),
        "speed_mode": "default",
        "transformer_quant": None if args.arm == "bf16" else args.arm,
    }
    for key, value in (("gguf_filename", args.gguf_filename), ("base_repo", args.base_repo)):
        if value:
            load_kwargs[key] = value
    if args.arm == "nvfp4":
        load_kwargs["transformer_prequant_path"] = args.prequant_path
    backend.begin_load(args.model, **load_kwargs)
    while True:
        progress = backend.load_progress()
        if progress.get("phase") == "ready":
            break
        if progress.get("phase") == "error":
            raise RuntimeError(progress.get("error"))
        time.sleep(2)
    rec["speed_optims"] = backend.status().get("speed_optims")
    pipe = backend._state.pipe
    vae = pipe.vae
    rec["vae_class"] = type(vae).__name__
    rec["decode_compiled_on_load"] = "torch.compile" in type(vae.decode).__name__ or hasattr(
        vae.decode, "_torchdynamo_orig_callable"
    )

    eager_decode = vae.decode
    captured: dict = {}

    def capture(*a, **k):
        out = eager_decode(*a, **k)
        if "latent" not in captured:
            captured["args"] = a
            captured["kwargs"] = k
            captured["latent"] = a[0].detach().clone() if a else None
        return out

    vae.decode = capture
    gen_kwargs = dict(
        prompt = args.prompt,
        width = args.resolution,
        height = args.resolution,
        steps = args.steps,
        guidance = _resolve_guidance(args, args.model),
        seed = args.seed,
        batch_size = 1,
    )
    backend.generate(**gen_kwargs)
    vae.decode = eager_decode
    if captured.get("latent") is None:
        raise RuntimeError("no VAE decode call captured")
    latent = captured["latent"]
    rec["latent_shape"] = list(latent.shape)
    rec["latent_dtype"] = str(latent.dtype)

    def call(fn):
        args_ = (latent,) + tuple(captured["args"][1:])
        return fn(*args_, **captured["kwargs"])

    def sample_of(out):
        if hasattr(out, "sample"):
            return out.sample
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = call(fn)
        torch.cuda.synchronize()
        return time.perf_counter() - t0, sample_of(out).detach().float()

    _, ref = timed(eager_decode)
    eager_walls = [timed(eager_decode)[0] for _ in range(3)]
    rec["eager_decode_s"] = round(min(eager_walls), 4)

    results = {}
    for label, kwargs in (
        ("dynamic", {"fullgraph": False, "dynamic": True}),
        (
            "max_autotune",
            {"fullgraph": False, "dynamic": True, "mode": "max-autotune-no-cudagraphs"},
        ),
    ):
        torch.cuda.reset_peak_memory_stats()
        compiled = torch.compile(eager_decode, **kwargs)
        wall, got = timed(compiled)
        steady = [timed(compiled)[0] for _ in range(3)]
        diff = (got - ref).abs()
        results[label] = {
            "compile_wall_s": round(wall, 3),
            "steady_decode_s": round(min(steady), 4),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "nan_compiled": bool(torch.isnan(got).any()),
            "nan_eager": bool(torch.isnan(ref).any()),
        }
        results[label]["_tensor"] = got
        print(
            f"[{args.family}] {label}: {results[label]['compile_wall_s']}s compile, "
            f"steady {results[label]['steady_decode_s']}s, max_abs {results[label]['max_abs']:.4g}",
            flush = True,
        )

    # PNGs and LPIPS. Both tensors go through the same normalisation, so the metric is the decode
    # difference and nothing else.
    import numpy as np
    from PIL import Image

    png_dir = Path(args.png_dir)
    png_dir.mkdir(parents = True, exist_ok = True)

    def to_png(t, name):
        x = t
        while x.dim() > 4:
            x = x[:, :, 0]
        x = (x[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        arr = (x * 255).round().astype(np.uint8)
        Image.fromarray(arr).save(png_dir / name)
        return arr

    ref_np = to_png(ref, f"{args.family}_{args.arm}_eager.png")
    scorer = LpipsScorer("cuda")
    rec["lpips_backend"] = "nvfp4_budget_profile.LpipsScorer (lpips vgg)"
    for label, res in results.items():
        cand_np = to_png(res.pop("_tensor"), f"{args.family}_{args.arm}_{label}.png")
        res["lpips"] = scorer.score(ref_np, cand_np) if scorer is not None else None
        res["png_max_abs_uint8"] = int(np.abs(ref_np.astype(int) - cand_np.astype(int)).max())
    rec["results"] = results
    Path(args.out).parent.mkdir(parents = True, exist_ok = True)
    Path(args.out).write_text(json.dumps(rec, indent = 2, default = str) + "\n")
    print(json.dumps(rec["results"], indent = 2), flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
