# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed + memory lever benchmark for the VIDEO diffusion backend (B200).

The video default path already stacks several optimisations (verified in video.py): TE
auto-quant, DiT auto-quant when it fits resident, VAE auto (skipped for the fp32-VAE Wan
families), regional torch.compile (speed_mode "default"), cuDNN fused attention, and
First-Block-Cache auto-engaged at >= 20 steps. This benchmark drives the SAME real lever
functions the loader calls -- ``quantize_text_encoders`` / ``quantize_vae`` /
``quantize_transformer`` / ``apply_speed_optims`` / ``apply_attention_backend`` /
``apply_step_cache`` + ``maybe_toggle_step_cache`` -- with the loader's own default
arguments, so each measured configuration reflects a real load, not a synthetic one.

For each configuration it loads the full pipeline fresh (quant/compile mutate irreversibly),
warms up (to pay the one-time compile), then measures a short clip generation: total latency,
median per-step ms, peak resident GB, steady weight GB, and per-frame LPIPS(AlexNet) averaged
against the bit-exact reference config (everything off/dense/native). This isolates each
lever's contribution and answers: how much does the shipped default win, and do ``max``
compile / flash4 attention leave speed on the table for video.

Memory/timing idiom lifted from scripts/quant_speedmem_bench.py (reset_peak -> load ->
memory_allocated / max_memory_allocated with synchronize + perf_counter).

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/video_speedmem_bench.py --family wan2.2-ti2v-5b \\
        --configs reference,compile,cudnn,fbcache,ditquant,shipped,speedmax,flash4 \\
        --steps 30 --num-frames 25 --width 512 --height 320 --iters 3
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
# The video base repos live in the workspace BACKUP cache; honor an override but default to it.
os.environ.setdefault("HF_HOME", "/mnt/disks/unslothai/ubuntu/workspace_81/BACKUP_05/temp/hf_cache")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BACKEND_ROOT = _REPO_ROOT / "studio" / "backend"
for _p in (str(_BACKEND_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PROMPT = (
    "A cinematic drone shot flying over a misty mountain valley at sunrise, "
    "golden light, volumetric fog, highly detailed, smooth camera motion"
)

_FAMILIES: dict[str, dict[str, Any]] = {
    "wan2.2-ti2v-5b": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "vae_force_fp32": True,
        "guidance": 5.0,
    },
    "ltx-2": {"repo": "Lightricks/LTX-2", "vae_force_fp32": False, "guidance": 4.0},
    "hunyuanvideo-1.5": {
        "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "vae_force_fp32": False,
        "guidance": 6.0,
    },
    "hunyuanvideo-1.5-720p": {
        "repo": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "vae_force_fp32": False,
        "guidance": 6.0,
    },
    # Wan2.2-A14B is a dual-expert MoE (transformer + transformer_2); _apply_levers quantizes both.
    "wan2.2-t2v-a14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "vae_force_fp32": True,
        "guidance": 5.0,
    },
}


# ── cuda memory / timing helpers ───────────────────────────────────────────────
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


def _median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2] if xs else 0.0


_LP: dict = {}


def _lpips_alex(ref_arr, arr):
    """LPIPS(AlexNet) between two HxWx3 uint8 frames (net on CPU). None if lpips missing."""
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


def _frames_to_arrays(output) -> list:
    """Normalize a video pipeline output to a list of HxWx3 uint8 numpy frames."""
    import numpy as np

    frames = getattr(output, "frames", None)
    if frames is None:
        return []
    batch0 = frames[0]
    arrs = []
    for fr in batch0:
        if hasattr(fr, "convert"):  # PIL image
            arrs.append(np.array(fr.convert("RGB")))
        else:
            a = np.asarray(fr)
            if a.dtype != np.uint8:
                a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            arrs.append(a)
    return arrs


def _mean_luma(arrs: list) -> Optional[float]:
    """Mean Rec.601 luma over all frames (0-255). ~0 == black frames (the fp8 failure signal)."""
    import numpy as np

    if not arrs:
        return None
    vals = []
    for a in arrs:
        a = np.asarray(a).astype(np.float32)
        if a.ndim == 3 and a.shape[-1] >= 3:
            luma = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        else:
            luma = a
        vals.append(float(luma.mean()))
    return round(sum(vals) / len(vals), 3) if vals else None


def _mean_lpips(ref_arrs: list, arrs: list) -> Optional[float]:
    """Mean per-frame LPIPS over the min common frame count."""
    if not ref_arrs or not arrs:
        return None
    n = min(len(ref_arrs), len(arrs))
    vals = []
    for i in range(n):
        v = _lpips_alex(ref_arrs[i], arrs[i])
        if v is not None:
            vals.append(v)
    return round(sum(vals) / len(vals), 4) if vals else None


def _import_diffusers():
    import torch  # noqa: F401
    import torchao  # noqa: F401
    import diffusers.utils.import_utils as iu

    iu._bitsandbytes_available = False
    import diffusers

    return diffusers


def _target():
    """The object the real casters/optimisers read: a stand-in for DiffusionDeviceTarget.
    supports_default_torch_compile must be True or compile_eligible() bails."""
    import torch
    return types.SimpleNamespace(
        device = "cuda",
        dtype = torch.bfloat16,
        supports_default_torch_compile = True,
    )


# ── config matrix ──────────────────────────────────────────────────────────────
# Each config names the lever settings applied to a fresh pipe. Built UP from the
# bit-exact reference so each successive config isolates one lever's contribution;
# `shipped` is the current default; `speedmax`/`flash4` probe untapped headroom.
_CONFIGS: dict[str, dict[str, Any]] = {
    #                te      vae     dit     speed      attn      cache
    "reference": dict(te = "none", vae = "none", dit = "none", speed = "off", attn = "native", cache = "off"),
    "compile": dict(te = "none", vae = "none", dit = "none", speed = "default", attn = "native", cache = "off"),
    "cudnn": dict(te = "none", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "off"),
    "fbcache": dict(te = "none", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "auto"),
    "ditquant": dict(te = "none", vae = "none", dit = "auto", speed = "default", attn = "auto", cache = "auto"),
    "shipped": dict(te = "auto", vae = "auto", dit = "auto", speed = "default", attn = "auto", cache = "auto"),
    "speedmax": dict(te = "auto", vae = "auto", dit = "auto", speed = "max", attn = "auto", cache = "auto"),
    "flash4": dict(te = "auto", vae = "auto", dit = "auto", speed = "default", attn = "flash4", cache = "auto"),
    # diagnostics: isolate whether the DiT-quant + compile crash needs FBCache.
    "diag_ditq_default_nocache": dict(
        te = "none", vae = "none", dit = "auto", speed = "default", attn = "native", cache = "off"
    ),
    "diag_ditq_max_nocache": dict(
        te = "none", vae = "none", dit = "auto", speed = "max", attn = "native", cache = "off"
    ),
    "diag_ditq_nocompile": dict(
        te = "none", vae = "none", dit = "auto", speed = "eager", attn = "native", cache = "off"
    ),
    # which quant scheme survives torch.compile? (fp8/mslk fails "fake tensors"; test int8/mxfp8)
    "diag_ditint8_compile": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "native", cache = "off"
    ),
    "diag_ditmxfp8_compile": dict(
        te = "none", vae = "none", dit = "mxfp8", speed = "default", attn = "native", cache = "off"
    ),
    "diag_ditint8_fbcache": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "native", cache = "auto"
    ),
    # isolate the quant x FBCache over-caching interaction at production size:
    "ditfp8_nocache": dict(
        te = "none", vae = "none", dit = "auto", speed = "default", attn = "auto", cache = "off"
    ),
    "te_fbcache": dict(
        te = "auto", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "auto"
    ),
    "ditfp8_fbcache": dict(
        te = "none", vae = "none", dit = "auto", speed = "default", attn = "auto", cache = "auto"
    ),
    "ditint8_fbcache_prod": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "auto", cache = "auto"
    ),
    # mixed-fp8 vs int8 head-to-head (Phase 3): the DiT-quant accuracy comparison on Wan/Hunyuan.
    # fp8 here goes through the production quantize_transformer family exclude (input embedders
    # kept bf16), so it is only non-black if the mixed-fp8 wiring is live. cache on AND off.
    "ditfp8mixed_nocache": dict(
        te = "none", vae = "none", dit = "fp8", speed = "default", attn = "native", cache = "off"
    ),
    "ditfp8mixed_fbcache": dict(
        te = "none", vae = "none", dit = "fp8", speed = "default", attn = "auto", cache = "auto"
    ),
    "ditint8_nocache": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "native", cache = "off"
    ),
}


def _ref_cache_path(out, *, family, seed, steps, num_frames, width, height):
    """Reference-frame cache keyed by every parameter that changes the reference clip.

    The persisted reference (for the "run reference once, score other configs in parallel
    processes" workflow) shares the default --out dir across runs, so a single unkeyed
    ref_frames.npz would let a later reference-less run of a different family / seed / steps /
    frames / resolution score LPIPS against the wrong baseline. Key by all of them so a
    reference-less run only reuses a reference computed for the same parameters."""
    from pathlib import Path
    return Path(out) / (
        f"ref_frames_{family}_seed{seed}_st{steps}_f{num_frames}_{width}x{height}.npz"
    )


def _build_pipe(repo: str, force_fp32_vae: bool):
    import torch

    diffusers = _import_diffusers()
    # Wan-style VAEs decode in fp32 for numerical stability (the loader pins this via
    # vae_force_fp32). A scalar bf16 torch_dtype truncates the fp32-stored VAE weights at
    # load, and a later .to(float32) only widens the already-lossy values (banding), so the
    # bench would measure a decode path production never runs. Pin the VAE fp32 per-component
    # exactly like the production loader (video.py: {"vae": fp32, "default": bf16}).
    torch_dtype = torch.bfloat16
    if force_fp32_vae:
        torch_dtype = {"vae": torch.float32, "default": torch.bfloat16}
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, torch_dtype = torch_dtype)
    pipe = pipe.to("cuda")
    if force_fp32_vae and getattr(pipe, "vae", None) is not None:
        pipe.vae.to(torch.float32)  # belt-and-suspenders; a no-op on the primary path above
    return pipe


class _SecondExpertView:
    """Present ``pipe.transformer_2`` as ``.transformer`` so the single-DiT lever functions run on
    the second expert of a dual-expert MoE (Wan2.2-A14B) unforked -- mirrors the loader's
    _SecondDiTView (video.py). Attribute reads delegate to the real pipe except ``transformer``,
    and a ``transformer`` write (e.g. torch.compile reassigning it) is routed to ``transformer_2``."""

    def __init__(self, pipe):
        object.__setattr__(self, "_pipe", pipe)

    @property
    def transformer(self):
        return self._pipe.transformer_2

    def __getattr__(self, name):
        return getattr(self._pipe, name)

    def __setattr__(self, name, value):
        setattr(self._pipe, "transformer_2" if name == "transformer" else name, value)


def _apply_levers(
    pipe,
    cfg: dict,
    *,
    fam_name: str,
    fam_obj,
    force_fp32_vae: bool,
    default_steps: int,
    logger = None,
) -> dict:
    """Apply the configured levers with the loader's own argument values, in the loader's order:
    quant (dit -> te -> vae) THEN optimisation layers (cache -> attention -> speed). For a
    dual-expert MoE (pipe.transformer_2 present) every DiT-touching lever is applied to BOTH experts
    via _SecondExpertView, exactly like the loader, so A14B latency + accuracy are real."""
    from core.inference.diffusion_precision import quantize_text_encoders
    from core.inference.diffusion_vae_quant import quantize_vae
    from core.inference.diffusion_transformer_quant import quantize_transformer
    from core.inference.diffusion_speed import apply_speed_optims, snapshot_backend_flags
    from core.inference.diffusion_attention import select_attention_backend, apply_attention_backend
    from core.inference.diffusion_cache import (
        apply_step_cache,
        TC_FBCACHE,
        FBCACHE_MIN_STEPS,
    )

    tgt = _target()
    engaged = {
        "dit": None,
        "te": None,
        "vae": None,
        "attn": None,
        "cache": None,
        "speed_optims": {},
    }

    # DiT-touching levers run per expert: [pipe] for a single-DiT family, plus a second-expert view
    # for a dual-expert MoE. Each view exposes the expert as ``.transformer``.
    views = [pipe]
    if getattr(pipe, "transformer_2", None) is not None:
        views.append(_SecondExpertView(pipe))

    # DiT quant (pipeline kind, resident): mutates each expert's transformer in place.
    if cfg["dit"] not in ("none", "off"):
        schemes = [
            quantize_transformer(v, tgt, mode = cfg["dit"], family = fam_name, logger = logger)
            for v in views
        ]
        engaged["dit"] = schemes[0]
        engaged["dit_experts"] = schemes
        _empty()
    dit_quant_active = engaged["dit"] is not None

    # TE quant (once; text encoders are shared, not per-expert).
    if cfg["te"] not in ("none", "off"):
        engaged["te"] = quantize_text_encoders(
            pipe, tgt, mode = cfg["te"], family = fam_name, offload_active = False, logger = logger
        )
        _empty()

    # VAE quant (once; Wan force_fp32 pins dense inside quantize_vae regardless).
    if cfg["vae"] not in ("none", "off"):
        engaged["vae"] = quantize_vae(
            pipe,
            tgt,
            mode = cfg["vae"],
            family = fam_name,
            offload_active = False,
            force_fp32 = force_fp32_vae,
            logger = logger,
        )
        _empty()

    # ── optimisation layers ──
    snapshot_backend_flags()  # process-wide flags; benchmark process is short-lived so no restore.
    speed = cfg["speed"]
    speed_active = speed != "off"

    # A quantized DiT must be compiled (eager dynamic quant ~30x slower), matching the loader.
    if dit_quant_active and speed == "off":
        speed = "default"
        speed_active = True

    # Step cache FIRST (compile keys fullgraph off an active cache); per expert.
    cache_active = False
    if cfg["cache"] == "auto":
        cache_request = TC_FBCACHE if default_steps >= FBCACHE_MIN_STEPS else None
        if cache_request is not None:
            for v in views:
                engaged["cache"] = apply_step_cache(
                    v,
                    mode = cache_request,
                    threshold = None,
                    quant_active = dit_quant_active,
                    logger = logger,
                )
            cache_active = engaged["cache"] not in (None, "off")

    # Attention (per expert).
    backend = select_attention_backend(tgt, cfg["attn"], speed_active = speed_active)
    for v in views:
        engaged["attn"] = apply_attention_backend(v, backend, logger = logger)

    # Speed profile (per expert; compiles each denoiser).
    if speed != "off":
        for v in views:
            engaged["speed_optims"] = apply_speed_optims(
                v,
                tgt,
                is_gguf = False,
                family = fam_obj,
                speed_mode = speed,
                cache_active = cache_active,
                offload_active = False,
                logger = logger,
            )
    engaged["_effective_speed"] = speed
    return engaged


def _timed_video(
    pipe,
    *,
    steps,
    width,
    height,
    num_frames,
    guidance,
    seed,
    cache_mode,
    dit_quant_active,
    default_steps,
    guidance_via_guider = False,
    logger = None,
):
    """One clip generation. Re-checks FBCache per generation (maybe_toggle_step_cache) exactly
    like the loader, then times total + per-step. Returns (output, total_s, [per_step_ms])."""
    import torch

    from core.inference.diffusion_cache import maybe_toggle_step_cache, FBCACHE_MIN_STEPS

    if cache_mode == "auto":
        try:
            maybe_toggle_step_cache(
                pipe, steps = steps, quant_active = dit_quant_active, threshold = None, logger = logger
            )
        except Exception:
            pass

    g = torch.Generator(device = "cuda").manual_seed(seed)
    step_ts: list[float] = []
    last = [0.0]

    def _cb(pp, i, t, kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        if last[0]:
            step_ts.append((now - last[0]) * 1000.0)
        last[0] = now
        return kw

    kwargs = dict(
        prompt = PROMPT,
        width = width,
        height = height,
        num_frames = num_frames,
        num_inference_steps = steps,
        generator = g,
    )
    if guidance_via_guider:
        # HunyuanVideo-1.5: CFG lives on a guider component and __call__ takes no
        # guidance_scale / callback_on_step_end (the loader writes the scale onto pipe.guider).
        guider = getattr(pipe, "guider", None)
        if guider is not None and hasattr(guider, "guidance_scale"):
            try:
                guider.guidance_scale = guidance
            except Exception:
                pass
    else:
        kwargs["guidance_scale"] = guidance
        kwargs["callback_on_step_end"] = _cb

    _sync()
    t0 = time.perf_counter()
    out = pipe(**kwargs)
    _sync()
    return out, (time.perf_counter() - t0), step_ts


def _run_config(
    name: str,
    cfg: dict,
    *,
    family: str,
    steps: int,
    width: int,
    height: int,
    num_frames: int,
    seed: int,
    iters: int,
    out: Path,
    logger = None,
):
    import numpy as np

    from core.inference.video_families import detect_video_family

    spec = _FAMILIES[family]
    repo = spec["repo"]
    force_fp32 = spec.get("vae_force_fp32", False)
    guidance = spec.get("guidance", 5.0)
    fam_obj = detect_video_family(repo)
    default_steps = getattr(fam_obj, "default_steps", 50)
    gvg = bool(getattr(fam_obj, "guidance_via_guider", False))

    _empty()
    _reset_peak()
    # Fresh dynamo state per config so a prior config's compiled graphs cannot leak into this one
    # (each config builds a fresh pipe; without this, later configs in a multi-config run can be
    # measured against a dirty compile cache).
    try:
        import torch
        torch._dynamo.reset()
    except Exception:
        pass
    pipe = _build_pipe(repo, force_fp32)
    load_peak = _peak_gb()
    engaged = _apply_levers(
        pipe,
        cfg,
        fam_name = family,
        fam_obj = fam_obj,
        force_fp32_vae = force_fp32,
        default_steps = default_steps,
        logger = logger,
    )
    _empty()
    weights_gb = _alloc_gb()
    dit_active = engaged["dit"] is not None
    cache_mode = cfg["cache"]

    # warmup (pays the one-time compile / autotune)
    _timed_video(
        pipe,
        steps = steps,
        width = width,
        height = height,
        num_frames = num_frames,
        guidance = guidance,
        seed = seed,
        cache_mode = cache_mode,
        dit_quant_active = dit_active,
        default_steps = default_steps,
        guidance_via_guider = gvg,
        logger = logger,
    )
    _reset_peak()
    dts, steps_ms = [], []
    last_out = None
    for _ in range(iters):
        last_out, dt, st = _timed_video(
            pipe,
            steps = steps,
            width = width,
            height = height,
            num_frames = num_frames,
            guidance = guidance,
            seed = seed,
            cache_mode = cache_mode,
            dit_quant_active = dit_active,
            default_steps = default_steps,
            guidance_via_guider = gvg,
            logger = logger,
        )
        dts.append(dt)
        steps_ms.append(_median(st) if st else 0.0)
    gen_peak = _peak_gb()
    arrs = _frames_to_arrays(last_out)
    # save a mid frame for eyeballing
    try:
        if arrs:
            from PIL import Image
            Image.fromarray(arrs[len(arrs) // 2]).save(out / f"vid_{family}_{name}.png")
    except Exception:
        pass

    # Persist reference frames so parallel per-config processes can score LPIPS against them.
    if name == "reference" and arrs:
        try:
            import numpy as _np
            _np.savez_compressed(
                _ref_cache_path(
                    out,
                    family = family,
                    seed = seed,
                    steps = steps,
                    num_frames = num_frames,
                    width = width,
                    height = height,
                ),
                *arrs,
            )
        except Exception:
            pass

    row = {
        "config": name,
        "family": family,
        "levers": cfg,
        "dit_scheme": engaged["dit"] or "dense",
        "dit_experts": engaged.get("dit_experts"),
        "te_scheme": engaged["te"] or "dense",
        "vae_scheme": engaged["vae"] or "dense",
        "attn": engaged["attn"] or "native",
        "cache": engaged["cache"] or "off",
        "effective_speed": engaged.get("_effective_speed"),
        "speed_optims": engaged["speed_optims"],
        "load_peak_gb": round(load_peak, 2),
        "weights_gb": round(weights_gb, 2),
        "gen_peak_gb": round(gen_peak, 2),
        "gen_latency_s": round(_median(dts), 3),
        "per_step_ms": round(_median(steps_ms), 1),
        "n_frames": len(arrs),
        "mean_luma": _mean_luma(arrs),
    }
    del pipe
    _empty()
    return row, arrs


def main(argv = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--family", default = "wan2.2-ti2v-5b", choices = sorted(_FAMILIES))
    ap.add_argument(
        "--configs", default = ",".join(_CONFIGS), help = "comma list from: " + ",".join(_CONFIGS)
    )
    ap.add_argument("--steps", type = int, default = 30)
    ap.add_argument("--num-frames", type = int, default = 25)
    ap.add_argument("--width", type = int, default = 512)
    ap.add_argument("--height", type = int, default = 320)
    ap.add_argument("--seed", type = int, default = 42)
    ap.add_argument("--iters", type = int, default = 3)
    ap.add_argument("--out", default = "outputs/video_speedmem")
    args = ap.parse_args(argv)

    import logging

    logging.basicConfig(level = logging.INFO, format = "%(message)s")
    logger = logging.getLogger("videobench")

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    for n in names:
        if n not in _CONFIGS:
            raise SystemExit(f"unknown config '{n}'; choose from {list(_CONFIGS)}")

    print(
        f"== video speed+mem bench: family={args.family} configs={names} "
        f"steps={args.steps} frames={args.num_frames} {args.width}x{args.height} ==",
        flush = True,
    )

    rows = []
    ref_arrs = None
    # If not (re)computing the reference in this run, load persisted reference frames for LPIPS.
    if "reference" not in names:
        ref_npz = _ref_cache_path(
            out,
            family = args.family,
            seed = args.seed,
            steps = args.steps,
            num_frames = args.num_frames,
            width = args.width,
            height = args.height,
        )
        if ref_npz.exists():
            try:
                import numpy as _np
                with _np.load(ref_npz) as z:
                    ref_arrs = [z[k] for k in z.files]
            except Exception:
                ref_arrs = None
    for n in names:
        row, arrs = _run_config(
            n,
            _CONFIGS[n],
            family = args.family,
            steps = args.steps,
            width = args.width,
            height = args.height,
            num_frames = args.num_frames,
            seed = args.seed,
            iters = args.iters,
            out = out,
            logger = logger,
        )
        if n == "reference":
            ref_arrs = arrs
        row["lpips_vs_reference"] = _mean_lpips(ref_arrs, arrs) if ref_arrs is not None else None
        rows.append(row)
        print(
            f"  [{n}] {json.dumps({k: row[k] for k in ('dit_scheme','te_scheme','attn','cache','effective_speed','weights_gb','gen_peak_gb','gen_latency_s','per_step_ms','lpips_vs_reference')})}",
            flush = True,
        )

    # speedups relative to reference (if present)
    ref_lat = next((r["gen_latency_s"] for r in rows if r["config"] == "reference"), None)
    for r in rows:
        r["speedup_vs_reference"] = (
            round(ref_lat / r["gen_latency_s"], 3) if ref_lat and r["gen_latency_s"] else None
        )

    dest = out / f"video_{args.family}_{'-'.join(names)}.json"
    with open(dest, "w", encoding = "utf-8") as fh:
        json.dump(rows, fh, indent = 2)
    print(f"wrote {dest}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
