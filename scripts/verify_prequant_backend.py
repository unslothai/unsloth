# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU verification of the Phase 9 pre-quantized load path through the real backend code.

Exercises the actual product functions (``load_prequantized_transformer`` and the runtime
``quantize_transformer``), not a reimplementation:

  prequant  -- load the checkpoint built by build_prequant_checkpoint.py via the real
               ``load_prequantized_transformer`` (meta-init + assign), measure GPU load peak,
               generate.
  runtime   -- the existing path: from_pretrained dense bf16 -> ``quantize_transformer`` on
               device, measure GPU load peak, generate (the LPIPS reference).

Asserts the prequant load peak is far below the dense one and the images match (LPIPS ~0).
Run each mode in its own process for a clean peak. One CUDA GPU."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
_RESEARCH = _REPO / "outputs" / "quant_research"
BACKEND = _REPO / "studio" / "backend"
BASE = "Tongyi-MAI/Z-Image-Turbo"
CKPT = os.environ.get("PREQUANT_CKPT", str(_RESEARCH / "prequant_fp8" / "transformer_fp8.pt"))
PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
OUT = Path(os.environ.get("PREQUANT_OUT_DIR", str(_RESEARCH / "prequant_verify_images")))

logging.basicConfig(level = logging.INFO, format = "%(message)s")
LOGGER = logging.getLogger("verify_prequant")

# Prequant and runtime produce the SAME quantized weights, so their images must be
# near-identical; anything above this LPIPS means the prequant path diverged and the run
# fails. The prequant load peak must also sit clearly below the dense runtime peak (the
# whole point of the path); require at least this fractional headroom.
LPIPS_MAX = 0.02
PREQUANT_PEAK_MAX_FRACTION = 0.75
_RUNTIME_PEAK_FILE = OUT / "runtime_peak.txt"


def _target(dtype):
    import types
    return types.SimpleNamespace(device = "cuda", dtype = dtype)


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


def _lpips(ref, arr):
    try:
        import lpips, torch

        fn = lpips.LPIPS(net = "alex", verbose = False).cuda().eval()

        def t(x):
            return (torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0).cuda()

        with torch.no_grad():
            return float(fn(t(ref), t(arr)).item())
    except Exception as exc:  # noqa: BLE001
        print(f"  (lpips: {type(exc).__name__})", flush = True)
        return None


def run(mode, steps, seed, res):
    sys.path.insert(0, str(BACKEND))
    import torch
    import diffusers
    from core.inference.diffusion_prequant import (
        ALLOW_LOCAL_PREQUANT_PATH_ENV,
        PrequantSource,
        load_prequantized_transformer,
    )
    from core.inference.diffusion_transformer_quant import quantize_transformer

    OUT.mkdir(parents = True, exist_ok = True)
    transformer_cls = diffusers.ZImageTransformer2DModel
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    if mode == "prequant":
        # A local checkpoint is refused unless its directory is allowlisted (unpickling an
        # arbitrary file is unsafe). This verifier's CKPT is operator-supplied and trusted,
        # so allowlist its directory here or the load returns None and measures nothing.
        ckpt_dir = os.path.dirname(os.path.realpath(CKPT))
        existing = os.environ.get(ALLOW_LOCAL_PREQUANT_PATH_ENV, "")
        os.environ[ALLOW_LOCAL_PREQUANT_PATH_ENV] = (
            ckpt_dir if not existing else existing + os.pathsep + ckpt_dir
        )
        source = PrequantSource(kind = "path", location = CKPT, filename = None)
        transformer = load_prequantized_transformer(
            transformer_cls,
            BASE,
            source,
            device = "cuda",
            dtype = torch.bfloat16,
            hf_token = None,
            scheme = "fp8",
            logger = LOGGER,
        )
        if transformer is None:
            print("prequant load FAILED (returned None)", flush = True)
            return 1
        pipe = diffusers.ZImagePipeline.from_pretrained(
            BASE, torch_dtype = torch.bfloat16, transformer = transformer
        )
        pipe.to("cuda")
        load_peak = torch.cuda.max_memory_allocated() / 1e9
        marker = getattr(transformer, "_unsloth_runtime_quant", None)
        print(f"[prequant] load_gpu_peak={load_peak:.1f} GB  marker={marker}", flush = True)
    else:  # runtime
        transformer = transformer_cls.from_pretrained(
            BASE, subfolder = "transformer", torch_dtype = torch.bfloat16
        ).to("cuda")
        pipe = diffusers.ZImagePipeline.from_pretrained(
            BASE, torch_dtype = torch.bfloat16, transformer = transformer
        )
        pipe.to("cuda")
        scheme = quantize_transformer(pipe, _target(torch.bfloat16), mode = "fp8", logger = LOGGER)
        load_peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"[runtime] engaged={scheme}  load_gpu_peak={load_peak:.1f} GB", flush = True)
        # Persist the dense reference peak so a later prequant run can enforce its VRAM win.
        _RUNTIME_PEAK_FILE.write_text(f"{load_peak:.6f}")

    img, dt = _gen(pipe, steps, seed, res)  # warmup
    img, dt = _gen(pipe, steps, seed, res)
    img.save(OUT / f"{mode}.png")
    print(f"[{mode}] gen={dt:.3f}s saved {mode}.png", flush = True)

    if mode != "prequant":
        return 0

    # Enforce the two invariants this verifier exists to check, so a broken prequant
    # checkpoint fails loudly instead of passing just because generation completed.
    ref_path = OUT / "runtime.png"
    if not ref_path.exists():
        print("FAIL: runtime reference image missing; run --mode runtime first", flush = True)
        return 1
    from PIL import Image

    lp = _lpips(np.array(Image.open(ref_path).convert("RGB")), np.array(img))
    print(f"[prequant] LPIPS_vs_runtime={lp}", flush = True)
    if lp is None:
        print("FAIL: LPIPS could not be computed (install lpips)", flush = True)
        return 1
    if lp > LPIPS_MAX:
        print(f"FAIL: LPIPS {lp:.4f} > {LPIPS_MAX} (prequant diverged from runtime)", flush = True)
        return 1
    if _RUNTIME_PEAK_FILE.exists():
        try:
            runtime_peak = float(_RUNTIME_PEAK_FILE.read_text().strip())
        except ValueError:
            runtime_peak = 0.0
        if runtime_peak > 0.0 and load_peak > runtime_peak * PREQUANT_PEAK_MAX_FRACTION:
            print(
                f"FAIL: prequant load peak {load_peak:.1f} GB not below "
                f"{PREQUANT_PEAK_MAX_FRACTION:.0%} of dense {runtime_peak:.1f} GB",
                flush = True,
            )
            return 1
    return 0


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices = ["prequant", "runtime"], required = True)
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--res", type = int, default = 1024)
    p.add_argument("--seed", type = int, default = 42)
    args = p.parse_args(argv)
    rc = run(args.mode, args.steps, args.seed, args.res)
    print("VERIFY-PREQUANT-DONE", flush = True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
