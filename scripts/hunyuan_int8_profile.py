# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Why is HunyuanVideo-1.5 int8 DiT slower than dense+compile? Per-DiT-forward timing + dynamo
recompile/graph-break counting for dense vs int8, both under the loader's regional block compile.

Signal:
- steady-state per-forward time uniformly higher on int8 -> inherent int8 quant/dequant overhead
  (memory lever, not speed) -> memory-gate the auto-quant.
- erratic slow forwards / high recompile count / eager fallback -> a fixable compile inefficiency
  (int8 tensor subclass breaks dynamic-shape compile) -> fix the compile path.

Run:  CUDA_VISIBLE_DEVICES=3 python scripts/hunyuan_int8_profile.py --modes dense,int8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_REPO_ROOT / "studio" / "backend"), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from video_speedmem_bench import _build_pipe, _apply_levers, _target, PROMPT  # noqa: E402

_REPO = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v"


def _dynamo_counts():
    """(recompiles, graph_breaks, unique_compiles) from dynamo counters, best-effort."""
    try:
        from torch._dynamo.utils import counters

        rc = (
            sum(v for k, v in counters.get("recompiles", {}).items())
            if "recompiles" in counters
            else 0
        )
        gb = sum(counters.get("graph_break", {}).values())
        # total frames compiled
        stats = counters.get("stats", {})
        uc = stats.get("unique_graphs", 0)
        return rc, gb, uc
    except Exception:
        return -1, -1, -1


def _profile_mode(
    mode: str, *, steps: int, width: int, height: int, num_frames: int, guidance: float, seed: int
):
    import torch
    from core.inference.video_families import detect_video_family
    from core.inference.diffusion_cache import maybe_toggle_step_cache  # noqa: F401

    # fresh dynamo state per mode
    try:
        torch._dynamo.reset()
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass

    fam_obj = detect_video_family(_REPO)
    gvg = bool(getattr(fam_obj, "guidance_via_guider", False))
    default_steps = getattr(fam_obj, "default_steps", 50)

    pipe = _build_pipe(_REPO, False)
    cfg = dict(
        te = "none",
        vae = "none",
        dit = ("int8" if mode == "int8" else "none"),
        speed = "default",
        attn = "native",
        cache = "off",
    )
    engaged = _apply_levers(
        pipe,
        cfg,
        fam_name = "hunyuanvideo-1.5",
        fam_obj = fam_obj,
        force_fp32_vae = False,
        default_steps = default_steps,
    )
    print(
        f"[{mode}] dit_scheme={engaged['dit'] or 'dense'} speed={engaged.get('_effective_speed')}",
        flush = True,
    )

    # per-forward GPU timing via cuda events on the transformer.
    fwd_ms: list[float] = []
    starts: list = []

    def _pre(mod, args, kwargs):
        ev = torch.cuda.Event(enable_timing = True)
        ev.record()
        starts.append(ev)
        return None

    def _post(mod, args, output):
        end = torch.cuda.Event(enable_timing = True)
        end.record()
        torch.cuda.synchronize()
        if starts:
            fwd_ms.append(starts[-1].elapsed_time(end))

    h1 = pipe.transformer.register_forward_pre_hook(_pre, with_kwargs = True)
    h2 = pipe.transformer.register_forward_hook(_post)

    def _gen(tag):
        fwd_ms.clear()
        starts.clear()
        g = torch.Generator(device = "cuda").manual_seed(seed)
        kwargs = dict(
            prompt = PROMPT,
            width = width,
            height = height,
            num_frames = num_frames,
            num_inference_steps = steps,
            generator = g,
        )
        if gvg:
            guider = getattr(pipe, "guider", None)
            if guider is not None and hasattr(guider, "guidance_scale"):
                guider.guidance_scale = guidance
        else:
            kwargs["guidance_scale"] = guidance
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pipe(**kwargs)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        rc, gb, uc = _dynamo_counts()
        n = len(fwd_ms)
        srt = sorted(fwd_ms)
        med = srt[n // 2] if n else 0.0
        p10 = srt[max(0, n // 10)] if n else 0.0
        p90 = srt[min(n - 1, (9 * n) // 10)] if n else 0.0
        slow = sum(1 for x in fwd_ms if x > 2.0 * med) if med else 0
        print(
            f"[{mode}:{tag}] total={total:.2f}s n_fwd={n} med={med:.1f}ms "
            f"p10={p10:.1f} p90={p90:.1f} max={max(fwd_ms) if fwd_ms else 0:.1f} "
            f"slow(>2x med)={slow} | recompiles={rc} graph_breaks={gb} unique_graphs={uc}",
            flush = True,
        )
        return total

    _gen("warmup")  # pays compile
    _gen("timed")
    _gen("timed2")

    h1.remove()
    h2.remove()
    del pipe
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def main(argv = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default = "dense,int8")
    ap.add_argument("--steps", type = int, default = 30)
    ap.add_argument("--num-frames", type = int, default = 25)
    ap.add_argument("--width", type = int, default = 512)
    ap.add_argument("--height", type = int, default = 320)
    ap.add_argument("--guidance", type = float, default = 6.0)
    ap.add_argument("--seed", type = int, default = 42)
    args = ap.parse_args(argv)
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        _profile_mode(
            mode,
            steps = args.steps,
            width = args.width,
            height = args.height,
            num_frames = args.num_frames,
            guidance = args.guidance,
            seed = args.seed,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
