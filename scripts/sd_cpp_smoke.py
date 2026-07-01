# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end smoke for the native stable-diffusion.cpp engine.

Drives the real ``SdCppEngine`` over a built ``sd-cli`` and a set of split GGUF
assets (the same the diffusers path consumes), running one txt2img generation
and reporting wall time. This is the GPU/native analogue of
``scripts/diffusion_bench.py``: it proves the engine wiring (finder -> command
builder -> subprocess -> output PNG) works against real weights.

Example (Z-Image-Turbo on one GPU):

    SD_CLI_PATH=.../sd-cli CUDA_VISIBLE_DEVICES=6 python scripts/sd_cpp_smoke.py \\
        --family z-image \\
        --diffusion-model .../z-image-turbo-Q4_K_M.gguf \\
        --vae .../ae.safetensors \\
        --llm .../Qwen3-4B-Instruct-2507-Q4_K_M.gguf \\
        --memory-mode balanced --steps 8 --cfg-scale 1.0 --width 512 --height 512
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parent.parent / "studio" / "backend"
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.diffusion_memory import (  # noqa: E402
    MEMORY_MODE_BALANCED,
    MEMORY_MODE_FAST,
    MEMORY_MODE_LOW_VRAM,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
)
from core.inference.sd_cpp_args import (  # noqa: E402
    SdCppGenParams,
    SdCppModelFiles,
    offload_flags,
)
from core.inference.sd_cpp_engine import SdCppEngine, find_sd_cpp_binary  # noqa: E402

# memory-mode (user knob) -> sd.cpp offload policy, matching the diffusers planner.
_MODE_TO_POLICY = {
    MEMORY_MODE_FAST: OFFLOAD_NONE,
    MEMORY_MODE_BALANCED: OFFLOAD_GROUP,
    MEMORY_MODE_LOW_VRAM: OFFLOAD_MODEL,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description = "Native sd-cli engine smoke test.")
    p.add_argument("--binary", default = None, help = "sd-cli path (else env / finder)")
    p.add_argument("--family", default = "z-image")
    p.add_argument("--diffusion-model", required = True)
    p.add_argument("--vae", default = None)
    p.add_argument("--clip_l", default = None)
    p.add_argument("--t5xxl", default = None)
    p.add_argument("--llm", default = None)
    p.add_argument("--qwen2vl", default = None)
    p.add_argument(
        "--prompt",
        default = "A cinematic photograph of a red fox in a snowy forest at dawn, highly detailed",
    )
    p.add_argument("--negative-prompt", default = None)
    p.add_argument("--width", type = int, default = 512)
    p.add_argument("--height", type = int, default = 512)
    p.add_argument("--steps", type = int, default = 8)
    p.add_argument("--cfg-scale", type = float, default = 1.0)
    p.add_argument("--seed", type = int, default = 42)
    p.add_argument("--memory-mode", default = "balanced", choices = list(_MODE_TO_POLICY))
    p.add_argument("--out-image", default = "outputs/sdcpp_verify/sdcpp_smoke.png")
    p.add_argument("--timeout", type = float, default = 1800.0)
    args = p.parse_args(argv)

    binary = args.binary or find_sd_cpp_binary()
    engine = SdCppEngine(binary = binary)
    print(f"binary:    {engine.binary}", flush = True)
    print(f"available: {engine.is_available()}", flush = True)
    print(f"version:   {engine.version()}", flush = True)
    if not engine.is_available():
        print(
            "ERROR: sd-cli not found (set --binary / SD_CLI_PATH / UNSLOTH_SD_CPP_PATH).",
            flush = True,
        )
        return 2

    files = SdCppModelFiles(
        diffusion_model = args.diffusion_model,
        vae = args.vae,
        clip_l = args.clip_l,
        t5xxl = args.t5xxl,
        llm = args.llm,
        qwen2vl = args.qwen2vl,
    )
    params = SdCppGenParams(
        prompt = args.prompt,
        negative_prompt = args.negative_prompt,
        width = args.width,
        height = args.height,
        steps = args.steps,
        cfg_scale = args.cfg_scale,
        seed = args.seed,
    )
    policy = _MODE_TO_POLICY[args.memory_mode]
    off = offload_flags(policy)
    print(f"memory:    {args.memory_mode} -> policy={policy} -> flags={off}", flush = True)

    out = Path(args.out_image)
    t0 = time.time()
    result = engine.generate(
        files,
        params,
        output_path = str(out),
        offload = off,
        verbose = True,
        timeout = args.timeout,
        on_log = lambda ln: print(f"  [sd] {ln}", flush = True),
    )
    dt = time.time() - t0
    size_kb = result.stat().st_size / 1024 if result.is_file() else 0
    print(f"\nOK: generated {result} ({size_kb:.0f} KB) in {dt:.1f}s", flush = True)
    print("SD-CPP-SMOKE-OK", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
