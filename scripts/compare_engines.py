# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Head-to-head: PyTorch (diffusers GGUF) vs native stable-diffusion.cpp.

Same Z-Image GGUF transformer, same VAE + text encoder, same resolution / steps /
seed, both resident (no CPU offload) on the same GPU. Reports per-engine compute
latency (model already loaded) so the denoise + VAE + TE work is compared fairly;
for sd.cpp it also reports the one-shot wall time (compute + the per-call model
reload, which a persistent sd-server would remove).

PyTorch runs first (load / warmup / median), is unloaded, then sd.cpp runs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

PROMPT = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed"
_DONE_RE = re.compile(r"generate_image completed in ([0-9.]+)s")


def _median(xs):
    return sorted(xs)[len(xs) // 2]


def bench_pytorch(repo, gguf, resolutions, steps, seed, iters):
    import torch
    from core.inference.diffusion import DiffusionBackend

    rows = []
    backend = DiffusionBackend()
    for speed in ("off", "default"):
        backend.begin_load(repo, gguf_filename = gguf, speed_mode = speed)
        deadline = time.time() + 1800  # 30 min: a stuck download/load must not hang forever
        while backend.load_progress().get("phase") != "ready":
            prog = backend.load_progress()
            if prog.get("phase") == "error":
                raise RuntimeError(prog)
            if time.time() > deadline:
                raise TimeoutError(f"load timed out (last progress: {prog})")
            time.sleep(0.5)
        for res in resolutions:

            def gen():
                torch.cuda.synchronize()
                t0 = time.time()
                backend.generate(
                    prompt = PROMPT,
                    width = res,
                    height = res,
                    steps = steps,
                    guidance = 0.0,
                    seed = seed,
                    batch_size = 1,
                )
                torch.cuda.synchronize()
                return time.time() - t0

            gen()  # warmup (compiles for `default`)
            med = _median([gen() for _ in range(iters)])
            rows.append(("pytorch", speed, res, med, None))
            print(f"  pytorch  speed={speed:7s} {res}px  compute={med:.3f}s", flush = True)
        backend.unload()
    return rows


def bench_sdcpp(binary, gguf, vae, llm, resolutions, steps, seed, iters):
    from core.inference.sd_cpp_args import SdCppGenParams, SdCppModelFiles
    from core.inference.sd_cpp_engine import SdCppEngine

    engine = SdCppEngine(binary = binary)
    if not engine.is_available():
        print("  sd.cpp binary not available; skipping", flush = True)
        return []
    files = SdCppModelFiles(diffusion_model = gguf, vae = vae, llm = llm)
    rows = []
    out_dir = Path("outputs/compare_engines")
    out_dir.mkdir(parents = True, exist_ok = True)
    for native in (None, "default"):  # resident-no-fa vs resident+--diffusion-fa
        for res in resolutions:
            params = SdCppGenParams(
                prompt = PROMPT, width = res, height = res, steps = steps, cfg_scale = 1.0, seed = seed
            )
            computes, walls = [], []
            for _ in range(iters):
                captured = {"c": None}

                def _log(ln):
                    m = _DONE_RE.search(ln)
                    if m:
                        captured["c"] = float(m.group(1))

                t0 = time.time()
                engine.generate(
                    files,
                    params,
                    output_path = str(out_dir / f"sd_{native}_{res}.png"),
                    offload = [],
                    native_speed = native,
                    on_log = _log,
                )
                walls.append(time.time() - t0)
                if captured["c"] is not None:
                    computes.append(captured["c"])
            med_c = _median(computes) if computes else None
            med_w = _median(walls)
            tag = "default(+fa)" if native == "default" else "off"
            rows.append(("sdcpp", tag, res, med_c, med_w))
            print(
                f"  sdcpp    speed={tag:12s} {res}px  compute={med_c}s  wall={med_w:.3f}s",
                flush = True,
            )
    return rows


def main(argv = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default = "unsloth/Z-Image-Turbo-GGUF")
    p.add_argument("--gguf-name", default = "z-image-turbo-Q4_K_M.gguf")
    p.add_argument("--sd-binary", default = None)
    p.add_argument(
        "--sd-gguf", default = None, help = "local gguf for sd.cpp (default: same as pytorch via cache)"
    )
    p.add_argument(
        "--vae",
        default = None,
        help = "VAE safetensors for sd.cpp (required when benchmarking the sd.cpp engine)",
    )
    p.add_argument(
        "--llm",
        default = None,
        help = "text-encoder GGUF for sd.cpp (required when benchmarking the sd.cpp engine)",
    )
    p.add_argument("--resolutions", default = "512,1024")
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--iters", type = int, default = 3)
    args = p.parse_args(argv)

    from huggingface_hub import hf_hub_download
    from core.inference.sd_cpp_engine import find_sd_cpp_binary

    resolutions = [int(x) for x in args.resolutions.split(",")]
    sd_gguf = args.sd_gguf or hf_hub_download(args.repo, args.gguf_name)
    binary = args.sd_binary or find_sd_cpp_binary()

    print("== PyTorch (diffusers GGUF) ==", flush = True)
    pt = bench_pytorch(args.repo, args.gguf_name, resolutions, args.steps, args.seed, args.iters)
    print("== stable-diffusion.cpp (native) ==", flush = True)
    sd = bench_sdcpp(
        binary, sd_gguf, args.vae, args.llm, resolutions, args.steps, args.seed, args.iters
    )

    print("\n==== COMPARISON (Z-Image-Turbo Q4, fixed seed, resident) ====", flush = True)
    print(f"{'engine':9s} {'config':13s} {'res':>5s} {'compute_s':>10s} {'wall_s':>8s}", flush = True)
    for eng, cfg, res, c, w in pt + sd:
        cs = f"{c:.3f}" if c is not None else "n/a"
        ws = f"{w:.3f}" if w is not None else "-"
        print(f"{eng:9s} {cfg:13s} {res:5d} {cs:>10s} {ws:>8s}", flush = True)
    print("COMPARE-DONE", flush = True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
