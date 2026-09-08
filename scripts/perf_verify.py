# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GPU verification for the diffusion performance pass (Phase 7).

Drives the real ``DiffusionBackend`` through several loads in one process and
checks, at a fixed seed:

  1. speed: ``default`` (compile + cudnn.benchmark + channels_last) vs ``off``
     -- expect a large denoise speedup at high PSNR (near-lossless).
  2. the TF32-leak fix: load ``max`` (flips global TF32 / cudnn.benchmark), unload,
     then load ``off`` -- the ``off`` image must be byte-identical (PSNR inf) to a
     fresh ``off`` baseline, proving the globals were restored on unload.
  3. ``balanced`` is now bit-identical: with VAE tiling restricted to the low tiers,
     streamed (group) offload should match the resident image (PSNR inf).

Run on one CUDA GPU with the GGUF + base repo cached.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def _psnr(a: "np.ndarray", b: "np.ndarray") -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0.0 else float(10.0 * np.log10((255.0**2) / mse))


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default = "unsloth/Z-Image-Turbo-GGUF")
    p.add_argument("--gguf", default = "z-image-turbo-Q4_K_M.gguf")
    p.add_argument(
        "--prompt",
        default = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed",
    )
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--width", type = int, default = 1024)
    p.add_argument("--height", type = int, default = 1024)
    p.add_argument("--out-dir", default = "outputs/perf_verify")
    args = p.parse_args(argv)

    import os

    import torch
    from core.inference.diffusion import DiffusionBackend

    out = Path(args.out_dir)
    out.mkdir(parents = True, exist_ok = True)
    backend = DiffusionBackend()
    token = os.environ.get("HF_TOKEN")

    def load(mode_speed = None, mode_mem = None):
        backend.begin_load(
            args.model,
            gguf_filename = args.gguf,
            hf_token = token,
            speed_mode = mode_speed,
            memory_mode = mode_mem,
        )
        deadline = time.time() + 2400
        while time.time() < deadline:
            ph = backend.load_progress().get("phase")
            if ph == "ready":
                return backend.status()
            if ph == "error":
                raise RuntimeError(f"load error: {backend.load_progress()}")
            time.sleep(0.5)
        raise RuntimeError("load timed out")

    def gen():
        torch.cuda.synchronize()
        t0 = time.time()
        img = backend.generate(
            prompt = args.prompt,
            width = args.width,
            height = args.height,
            steps = args.steps,
            guidance = 0.0,
            seed = args.seed,
            batch_size = 1,
        )["images"][0]
        torch.cuda.synchronize()
        return img, time.time() - t0

    def timed(
        mode_speed,
        *,
        warmup,
        iters,
        mem = None,
        tag = "",
    ):
        st = load(mode_speed, mem)
        for _ in range(warmup):
            gen()
        lats = []
        img = None
        for _ in range(iters):
            img, dt = gen()
            lats.append(dt)
        img.save(out / f"{tag}.png")
        backend.unload()
        med = sorted(lats)[len(lats) // 2]
        print(
            f"  [{tag}] speed={mode_speed} mem={mem} optims={st.get('speed_optims')} "
            f"tiling={st.get('vae_tiling')} median={med:.3f}s",
            flush = True,
        )
        return np.array(img), med

    print("== 1. speed: off vs default ==", flush = True)
    off_img, off_t = timed("off", warmup = 1, iters = 3, tag = "off")
    def_img, def_t = timed("default", warmup = 1, iters = 3, tag = "default")
    print(f"  PSNR(default vs off) = {_psnr(off_img, def_img):.1f} dB", flush = True)
    print(
        f"  speedup: off {off_t:.3f}s -> default {def_t:.3f}s "
        f"({(off_t-def_t)/off_t*100:+.1f}%)",
        flush = True,
    )

    print("== 2. TF32-leak fix: max then off must be byte-identical ==", flush = True)
    timed("max", warmup = 0, iters = 1, tag = "max")  # flips + should restore globals
    off2_img, _ = timed("off", warmup = 0, iters = 1, tag = "off2")
    leak_psnr = _psnr(off_img, off2_img)
    print(
        f"  PSNR(off-after-max vs off) = {leak_psnr:.1f} dB "
        f"({'OK byte-identical' if leak_psnr == float('inf') else 'LEAK! globals not restored'})",
        flush = True,
    )

    print("== 3. balanced is bit-identical (tiling off) ==", flush = True)
    bal_img, bal_t = timed("off", warmup = 0, iters = 1, mem = "balanced", tag = "balanced")
    bal_psnr = _psnr(off_img, bal_img)
    print(
        f"  PSNR(balanced vs off) = {bal_psnr:.1f} dB "
        f"({'OK bit-identical' if bal_psnr == float('inf') else 'differs'})",
        flush = True,
    )

    ok = (
        (leak_psnr == float("inf"))
        and (bal_psnr == float("inf"))  # check 3: balanced must be bit-identical to off
        and (def_t < off_t)
        and (_psnr(off_img, def_img) >= 30)
    )
    print(f"\nPERF-VERIFY {'OK' if ok else 'CHECK'}", flush = True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
