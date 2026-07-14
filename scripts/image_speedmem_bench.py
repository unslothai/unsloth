# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed + accuracy lever benchmark for the IMAGE diffusion backend (per-lever LPIPS).

Drives the SAME production lever functions the image loader calls -- ``apply_step_cache``,
``apply_attention_backend``, ``apply_speed_optims``, ``quantize_text_encoders``, the
compile-safe eager patches -- with the loader's own default arguments and order, so each
measured configuration reflects a real load. For each config it loads the pipeline fresh
(quant/compile mutate irreversibly), warms up (to pay the one-time compile), renders a
fixed prompt set at a fixed seed, and reports total latency, median per-step ms, peak
resident GB, and mean LPIPS(AlexNet) vs the bit-exact reference config (speed off,
native attention, uncached, dense) rendered at the same seed/settings.

Every generation starts from a clean step cache, exactly like the production backend:
diffusers keys FBCache residuals on the long-lived transformer and never resets them, so
without the per-generation reset the measured prompts would compare their first-block
residual against the PREVIOUS prompt's final one -- a state production never runs.
FBCache rows produced before this reset existed may overstate both the speedup and the
quality cost.

Lever isolation knobs (for before/after measurement of shipped fixes):
  --no-epc         force torch._inductor.config.emulate_precision_casts back off after
                   the speed layer enables it (the pre-fix compile numerics).
  --unarm-cache    restore the cache hooks' eager inner forwards after the speed layer
                   arms them (the pre-fix cache x compile composition).

Example:
    CUDA_VISIBLE_DEVICES=3 python scripts/image_speedmem_bench.py --family flux.1-dev \\
        --config compile --out outputs/image_speedmem
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _REPO_ROOT / "studio" / "backend"
for _p in (str(_BACKEND_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Fixed prompt set (the diffusion_quality.py defaults + one photographic subject) so the
# LPIPS mean is not hostage to a single composition.
PROMPTS = [
    "A cozy reading nook by a rain-streaked window, warm lamplight, a cat asleep on a stack of books",
    "A lone lighthouse on a rocky cliff at sunset, dramatic clouds, crashing waves, highly detailed",
    "A bustling night market street in the rain, neon signs reflected in puddles, cinematic",
    "A photograph of an astronaut riding a horse on the surface of the moon, detailed, 8k",
]

# Production defaults per family (diffusion_families.default_generation_params).
_FAMILIES: dict[str, dict[str, Any]] = {
    "qwen-image": {"repo": "Qwen/Qwen-Image", "family": "qwen-image"},
    "flux.1-dev": {"repo": "black-forest-labs/FLUX.1-dev", "family": "flux.1"},
    "flux.2-klein-4b": {"repo": "black-forest-labs/FLUX.2-klein-4B", "family": "flux.2-klein"},
    "sdxl": {"repo": "stabilityai/stable-diffusion-xl-base-1.0", "family": "sdxl"},
}

#                       te        speed       attn       cache
_CONFIGS: dict[str, dict[str, Any]] = {
    # bit-exact reference: everything off / native / dense.
    "reference": dict(te = "none", speed = "off", attn = "native", cache = "off"),
    # the non-compile floor: eager patches + attention auto-upgrade, no compile.
    "eager": dict(te = "none", speed = "eager", attn = "auto", cache = "off"),
    # the default dense tier (regional compile), uncached.
    "compile": dict(te = "none", speed = "default", attn = "auto", cache = "off"),
    # max tier (max-autotune regional compile + TF32 + fused QKV), uncached.
    "speedmax": dict(te = "none", speed = "max", attn = "auto", cache = "off"),
    # the default tier + FBCache (the auto path for 20+ step schedules).
    "fbcache": dict(te = "none", speed = "default", attn = "auto", cache = "fbcache"),
    # FBCache without compile (isolates the cache's own drift from the compile floor).
    "fbcache_eager": dict(te = "none", speed = "eager", attn = "auto", cache = "fbcache"),
    # TE quant isolation on the bit-exact stack: the conditioning perturbation ALONE.
    "te_fp8dyn": dict(te = "fp8_dynamic", speed = "off", attn = "native", cache = "off"),
    "te_fp8": dict(te = "fp8", speed = "off", attn = "native", cache = "off"),
}


def _sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _alloc_gb() -> float:
    import torch
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _peak_gb() -> float:
    import torch
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def _empty() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_LP: dict = {}


def _lpips_alex(ref_arr, arr) -> Optional[float]:
    """LPIPS(AlexNet) between two HxWx3 uint8 images (net on CPU). None if lpips missing."""
    try:
        import lpips
        import torch

        fn = _LP.get("fn")
        if fn is None:
            fn = lpips.LPIPS(net = "alex", verbose = False).eval()
            _LP["fn"] = fn

        def _t(a):
            import torch as _torch
            return _torch.from_numpy(a).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

        with torch.no_grad():
            return float(fn(_t(ref_arr), _t(arr)).item())
    except Exception:
        return None


def _import_diffusers():
    import torch  # noqa: F401
    import torchao  # noqa: F401
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


def _target():
    """Stand-in for DiffusionDeviceTarget: what the real lever functions read."""
    import torch
    return types.SimpleNamespace(
        device = "cuda",
        dtype = torch.bfloat16,
        supports_default_torch_compile = True,
    )


def _find_family(name: str):
    from core.inference.diffusion_families import _FAMILIES as ALL
    for fam in ALL:
        if fam.name == name:
            return fam
    raise SystemExit(f"unknown family '{name}'")


def _apply_levers(
    pipe,
    cfg: dict,
    *,
    fam_obj,
    no_epc: bool = False,
    unarm_cache: bool = False,
    logger = None,
) -> dict:
    """Apply the configured levers with the loader's own argument values, in the loader's
    order (diffusion.py): TE quant -> attention -> step cache -> eager patches -> speed."""
    from core.inference.diffusion_precision import quantize_text_encoders
    from core.inference.diffusion_attention import (
        apply_attention_backend,
        select_attention_backend,
    )
    from core.inference.diffusion_cache import apply_step_cache, _restore_hooked_block_inners
    from core.inference.diffusion_eager_patches import (
        install_compile_safe_patches,
        uninstall_patches,
    )
    from core.inference.diffusion_arch_patches import (
        install_arch_patches,
        uninstall_arch_patches,
    )
    from core.inference.diffusion_speed import apply_speed_optims

    tgt = _target()
    engaged: dict[str, Any] = {"te": None, "attn": None, "cache": None, "speed_optims": {}}

    if cfg["te"] != "none":
        engaged["te"] = quantize_text_encoders(
            pipe, tgt, mode = cfg["te"], family = fam_obj.name, logger = logger
        )

    speed_mode = cfg["speed"]
    engaged["attn"] = apply_attention_backend(
        pipe,
        select_attention_backend(
            tgt, None if cfg["attn"] == "auto" else cfg["attn"], speed_active = speed_mode != "off"
        ),
        logger = logger,
    )

    if cfg["cache"] != "off":
        engaged["cache"] = apply_step_cache(
            pipe, mode = cfg["cache"], quant_active = False, logger = logger
        )

    if speed_mode != "off":
        install_compile_safe_patches()
        install_arch_patches()
    else:
        uninstall_patches()
        uninstall_arch_patches()

    engaged["speed_optims"] = apply_speed_optims(
        pipe,
        tgt,
        is_gguf = False,
        family = fam_obj,
        speed_mode = speed_mode,
        cache_active = engaged["cache"] is not None,
        offload_active = False,
    )

    if no_epc:
        import torch
        cfg_ind = getattr(getattr(torch, "_inductor", None), "config", None)
        if cfg_ind is not None and hasattr(cfg_ind, "emulate_precision_casts"):
            cfg_ind.emulate_precision_casts = False
            engaged["epc_forced_off"] = True
    if unarm_cache:
        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            _restore_hooked_block_inners(transformer)
            engaged["cache_unarmed"] = True
    return engaged


def _reset_step_cache(pipe) -> None:
    """Clear stale FBCache residuals before a generation, mirroring the production
    backend's ``_reset_step_cache`` (diffusion.py): diffusers keys the residuals on the
    long-lived denoiser and never resets them itself, and the transformer-level entry
    point in diffusers 0.39 is ``_reset_stateful_cache`` (``reset_stateful_hooks`` lives
    only on the HookRegistry, so the getattr fallback is a silent no-op). Best-effort:
    an uncached denoiser (or SDXL's unet, which has no FBCache path) is a no-op."""
    denoiser = getattr(pipe, "transformer", None) or getattr(pipe, "unet", None)
    reset = getattr(denoiser, "_reset_stateful_cache", None) or getattr(
        denoiser, "reset_stateful_hooks", None
    )
    if callable(reset):
        try:
            reset()
        except Exception:
            pass


def _generate(
    pipe,
    fam_obj,
    *,
    steps: int,
    guidance: float,
    size: int,
    seed: int,
    limit: Optional[int] = None,
) -> tuple:
    """Render every prompt at a fixed per-prompt seed; returns (arrays, total_s, step_ms)."""
    import numpy as np
    import torch

    call_params = {}
    try:
        import inspect
        call_params = inspect.signature(pipe.__call__).parameters
    except (TypeError, ValueError):
        pass

    step_times: list[float] = []
    last: dict[str, float] = {}

    def _cb(p, i, t, kw):
        now = time.perf_counter()
        if "t" in last:
            step_times.append(now - last["t"])
        last["t"] = now
        return kw

    arrs = []
    total = 0.0
    for idx, prompt in enumerate(PROMPTS[: limit or len(PROMPTS)]):
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": size,
            "height": size,
            "generator": torch.Generator("cuda").manual_seed(seed + idx),
        }
        if fam_obj.cfg_kwarg in call_params:
            kwargs[fam_obj.cfg_kwarg] = guidance
        if "callback_on_step_end" in call_params:
            kwargs["callback_on_step_end"] = _cb
        last.clear()
        # Production resets the step cache before every generation; without this the
        # first step compares against the PREVIOUS prompt's final residual.
        _reset_step_cache(pipe)
        _sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            image = pipe(**kwargs).images[0]
        _sync()
        total += time.perf_counter() - t0
        arrs.append(np.array(image.convert("RGB")))
    med_step = sorted(step_times)[len(step_times) // 2] * 1000.0 if step_times else None
    return arrs, total, med_step, step_times


def main() -> None:
    ap = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    ap.add_argument("--family", required = True, choices = sorted(_FAMILIES))
    ap.add_argument("--config", required = True, choices = sorted(_CONFIGS))
    ap.add_argument("--steps", type = int, default = None, help = "override the family default")
    ap.add_argument("--size", type = int, default = 1024)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--out", default = "outputs/image_speedmem")
    ap.add_argument("--no-epc", action = "store_true")
    ap.add_argument("--unarm-cache", action = "store_true")
    ap.add_argument("--tag", default = None, help = "output row name (default: config name)")
    args = ap.parse_args()

    import logging

    logging.basicConfig(level = logging.INFO, format = "%(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("image_speedmem")

    fam_spec = _FAMILIES[args.family]
    cfg = _CONFIGS[args.config]
    tag = args.tag or args.config

    import numpy as np
    import torch

    diffusers = _import_diffusers()
    from core.inference.diffusion_families import default_generation_params

    fam_obj = _find_family(fam_spec["family"])
    steps, guidance = default_generation_params(fam_spec["repo"])
    if args.steps is not None:
        steps = args.steps

    out_dir = Path(args.out) / args.family
    out_dir.mkdir(parents = True, exist_ok = True)
    ref_npz = out_dir / f"ref_seed{args.seed}_st{steps}_{args.size}.npz"

    logger.info(
        "family=%s config=%s steps=%d guidance=%s size=%d seed=%d",
        args.family,
        args.config,
        steps,
        guidance,
        args.size,
        args.seed,
    )

    _reset_peak()
    t0 = time.perf_counter()
    pipe = diffusers.DiffusionPipeline.from_pretrained(fam_spec["repo"], torch_dtype = torch.bfloat16)
    load_s = time.perf_counter() - t0

    engaged = _apply_levers(
        pipe,
        cfg,
        fam_obj = fam_obj,
        no_epc = args.no_epc,
        unarm_cache = args.unarm_cache,
        logger = logger,
    )
    pipe.to("cuda")
    weights_gb = _alloc_gb()

    # Warmup: pays the one-time compile (and the cuDNN autotune) outside the timed runs.
    wt0 = time.perf_counter()
    _generate(
        pipe,
        fam_obj,
        steps = steps,
        guidance = guidance,
        size = args.size,
        seed = args.seed + 1000,
        limit = 1,
    )
    warmup_s = time.perf_counter() - wt0

    _reset_peak()
    arrs, total_s, med_step_ms, step_times = _generate(
        pipe, fam_obj, steps = steps, guidance = guidance, size = args.size, seed = args.seed
    )
    gen_peak = _peak_gb()

    # Persist / score against the reference.
    lpips_vals: list[float] = []
    if args.config == "reference" and not (args.no_epc or args.unarm_cache):
        np.savez_compressed(ref_npz, *arrs)
    if ref_npz.exists():
        ref = np.load(ref_npz)
        refs = [ref[k] for k in ref.files]
        for r, a in zip(refs, arrs):
            v = _lpips_alex(r, a)
            if v is not None:
                lpips_vals.append(v)

    from PIL import Image

    for i, a in enumerate(arrs):
        Image.fromarray(a).save(out_dir / f"{tag}_p{i}.png")

    row = {
        "family": args.family,
        "config": args.config,
        "tag": tag,
        "steps": steps,
        "guidance": guidance,
        "size": args.size,
        "seed": args.seed,
        "engaged": {k: v for k, v in engaged.items()},
        "load_s": round(load_s, 2),
        "warmup_s": round(warmup_s, 2),
        "total_gen_s": round(total_s, 2),
        "per_image_s": round(total_s / len(PROMPTS), 3),
        "median_step_ms": round(med_step_ms, 1) if med_step_ms else None,
        "step_times_s": [round(t, 4) for t in step_times],
        "weights_gb": round(weights_gb, 2),
        "gen_peak_gb": round(gen_peak, 2),
        "lpips_vs_ref_mean": round(sum(lpips_vals) / len(lpips_vals), 4) if lpips_vals else None,
        "lpips_vs_ref_per_prompt": [round(v, 4) for v in lpips_vals] or None,
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(row, indent = 2, default = str))
    print(json.dumps(row, indent = 2, default = str))

    del pipe
    _empty()


if __name__ == "__main__":
    main()
