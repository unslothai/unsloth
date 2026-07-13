# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed + memory lever benchmark for the VIDEO diffusion backend (B200).

Drives the SAME lever functions the loader calls (``quantize_text_encoders`` / ``quantize_vae``
/ ``quantize_transformer`` / ``apply_speed_optims`` / ``apply_attention_backend`` /
``apply_step_cache`` + ``maybe_toggle_step_cache``) with the loader's own default arguments, so
each configuration reflects a real load.

Per config it loads the pipeline fresh (quant/compile mutate irreversibly), warms up, then
measures a short clip: total latency, median per-step ms, peak resident GB, steady weight GB, and
per-frame LPIPS(AlexNet) vs the bit-exact reference (everything off/dense/native). Isolates each
lever's contribution and whether ``max`` compile / flash4 leave speed on the table.

Memory/timing idiom lifted from scripts/quant_speedmem_bench.py.

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
# Each config names the lever settings for a fresh pipe, built UP from the bit-exact
# reference so each isolates one lever. `shipped` is the default; `speedmax`/`flash4`
# probe untapped headroom.
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
    # Companion-quant accuracy isolation vs the reference (uncached): TE-only and VAE-only on
    # the trim+cudnn+compile stack. With compile rounding fixed, the companions are the
    # next-largest divergence source.
    "diag_te_nocache": dict(
        te = "auto", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "off"
    ),
    "diag_vae_nocache": dict(
        te = "none", vae = "auto", dit = "none", speed = "default", attn = "auto", cache = "off"
    ),
    "ditfp8_fbcache": dict(
        te = "none", vae = "none", dit = "auto", speed = "default", attn = "auto", cache = "auto"
    ),
    "ditint8_fbcache_prod": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "auto", cache = "auto"
    ),
    # mixed-fp8 vs int8 head-to-head on Wan/Hunyuan. fp8 goes through the production family
    # exclude (input embedders kept bf16), so it is only non-black if that wiring is live.
    "ditfp8mixed_nocache": dict(
        te = "none", vae = "none", dit = "fp8", speed = "default", attn = "native", cache = "off"
    ),
    "ditfp8mixed_fbcache": dict(
        te = "none", vae = "none", dit = "fp8", speed = "default", attn = "auto", cache = "auto"
    ),
    "ditint8_nocache": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "native", cache = "off"
    ),
    # Hunyuan trim isolation: "cudnn" above is the same stack WITH the trim (auto-engaged
    # under any speed tier), so trim_off isolates its win. Only this row forces trim off.
    "trim_off": dict(
        te = "none",
        vae = "none",
        dit = "none",
        speed = "default",
        attn = "auto",
        cache = "off",
        trim = False,
    ),
    # Compile isolation at matched attention/trim: eager tier (no compile) vs "cudnn"
    # (default tier = regional compile).
    "eager_trim": dict(te = "none", vae = "none", dit = "none", speed = "eager", attn = "auto", cache = "off"),
    # int8 DiT baseline at the production attention stack (cudnn + trim), cache off,
    # comparable to the "cudnn" dense row.
    "int8_cudnn": dict(
        te = "none", vae = "none", dit = "int8", speed = "default", attn = "auto", cache = "off"
    ),
    # The full companion-quant stack WITHOUT step caching (te/vae auto + dit auto + compile
    # + cudnn + trim): the stacked-best default when the family's auto cache stays off.
    "shipped_nocache": dict(
        te = "auto", vae = "auto", dit = "auto", speed = "default", attn = "auto", cache = "off"
    ),
    # Compile-parity isolation: the "cudnn" compiled stack but with inductor's
    # emulate_precision_casts turned back OFF, to measure that parity flag's effect.
    "epc_off": dict(
        te = "none",
        vae = "none",
        dit = "none",
        speed = "default",
        attn = "auto",
        cache = "off",
        epc = False,
    ),
    # Explicit cache modes at the dense compiled stack (bypass AUTO), for FBCache-vs-MagCache
    # head-to-head rows at identical settings.
    "fbcache_explicit": dict(
        te = "none", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "fbcache"
    ),
    "magcache_explicit": dict(
        te = "none", vae = "none", dit = "none", speed = "default", attn = "auto", cache = "magcache"
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
    # Wan-style VAEs decode in fp32 (the loader pins this via vae_force_fp32). A scalar bf16
    # dtype truncates the fp32 VAE weights at load and a later .to(float32) only widens the
    # lossy values (banding), so pin the VAE fp32 per-component like the loader.
    torch_dtype = torch.bfloat16
    if force_fp32_vae:
        torch_dtype = {"vae": torch.float32, "default": torch.bfloat16}
    pipe = diffusers.DiffusionPipeline.from_pretrained(repo, torch_dtype = torch_dtype)
    # Stays on CPU: the caller applies levers FIRST, then places on CUDA (like the loader).
    # Placing the dense pipeline first would OOM configs whose quantized form fits, and
    # record a dense load peak for a quantized row.
    if force_fp32_vae and getattr(pipe, "vae", None) is not None:
        pipe.vae.to(torch.float32)  # belt-and-suspenders; a no-op on the primary path above
    return pipe


class _SecondExpertView:
    """Present ``pipe.transformer_2`` as ``.transformer`` so single-DiT lever functions run on an
    MoE second expert (Wan2.2-A14B), like the loader's _SecondDiTView. Reads delegate to the pipe
    except ``transformer``; a ``transformer`` write is routed to ``transformer_2``."""

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
    cache_threshold: Optional[float] = None,
    cache_quality: Optional[str] = None,
    logger = None,
) -> dict:
    """Apply the configured levers with the loader's own argument values, in the loader's order:
    quant (dit -> te -> vae) THEN optimisation layers (cache -> attention -> speed). For a
    dual-expert MoE (pipe.transformer_2 present) every DiT-touching lever is applied to BOTH experts
    via _SecondExpertView, exactly like the loader, so A14B latency + accuracy are real."""
    from core.inference.diffusion_precision import quantize_text_encoders
    from core.inference.diffusion_vae_quant import quantize_vae
    from core.inference.diffusion_transformer_quant import (
        quantize_transformer,
        is_int8_memory_fallback,
    )
    from core.inference.diffusion_speed import apply_speed_optims, snapshot_backend_flags
    from core.inference.diffusion_attention import (
        select_attention_backend,
        apply_attention_backend,
        install_hunyuan_attention_trim,
    )
    from core.inference.diffusion_cache import (
        apply_step_cache,
        auto_cache_mode,
        auto_cache_quality,
        normalize_cache_quality,
        FBCACHE_MIN_STEPS,
    )
    from core.inference.video import _step_cache_all_or_none

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

    # DiT quant (resident): mutates each expert's transformer in place. Mirror the loader's
    # dense-fit skip: for an AUTO request on an int8-fallback family (HunyuanVideo-1.5), run dense
    # when it fits resident (the bench always loads resident). Explicit int8/fp8 configs are
    # honored. Without this the auto rows would measure int8 where the loader runs dense.
    dense_fit_skip = cfg["dit"] == "auto" and is_int8_memory_fallback(tgt, fam_name)
    if cfg["dit"] not in ("none", "off") and not dense_fit_skip:
        schemes = [
            quantize_transformer(v, tgt, mode = cfg["dit"], family = fam_name, logger = logger)
            for v in views
        ]
        # All-or-none across experts, like the loader: a second-expert miss can't fall back to
        # dense, so reject here rather than publish timings for a pipeline users can't load.
        n_engaged = sum(1 for s in schemes if s is not None)
        if 0 < n_engaged < len(views):
            raise RuntimeError(
                f"dit quant '{cfg['dit']}' engaged on only {n_engaged}/{len(views)} experts; "
                "production rejects this partial state, so the row would be unloadable"
            )
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
    # Process-wide flags (cudnn.benchmark, TF32/fp16 accumulation, emulate_precision_casts):
    # snapshotted here and restored by _run_config after the row, like the unload path -- else
    # a speed-enabled row poisons a later reference row in the same process.
    engaged["_flags_snapshot"] = snapshot_backend_flags()
    speed = cfg["speed"]
    speed_active = speed != "off"

    # A quantized DiT must be compiled (eager dynamic quant ~30x slower), matching the loader.
    if dit_quant_active and speed == "off":
        speed = "default"
        speed_active = True

    # Step cache FIRST (compile keys fullgraph off an active cache); per expert.
    cache_active = False
    if cfg["cache"] in ("auto", "fbcache", "magcache"):
        # Per-family auto mode like the loader: MagCache for HunyuanVideo-1.5, FBCache
        # elsewhere. An explicit "fbcache"/"magcache" config bypasses the auto policy.
        if cfg["cache"] == "auto":
            cache_request = (
                auto_cache_mode(fam_name) if default_steps >= FBCACHE_MIN_STEPS else None
            )
        else:
            cache_request = cfg["cache"]
        if cache_request is not None:
            # Quality preset like the loader: an unset request takes the family's auto default.
            quality = normalize_cache_quality(cache_quality) or auto_cache_quality(fam_name)

            # All-or-none across MoE experts, exactly like the loader: overwriting engaged["cache"]
            # per expert would leave one expert cached and one dense on a partial engage while the
            # row reports the cache off -- a config that never runs in production. The shared helper
            # rolls the engaged expert(s) back so the row measures a real configuration.
            def _engage_cache(v: Any, expert: str) -> Optional[str]:
                return apply_step_cache(
                    v,
                    mode = cache_request,
                    threshold = cache_threshold,
                    quant_active = dit_quant_active,
                    family = fam_name,
                    steps = default_steps,
                    quality = quality,
                    expert = expert,
                    logger = logger,
                )

            engaged["cache"], cache_partial_reason = _step_cache_all_or_none(
                pipe, fam_obj, _engage_cache, logger = logger
            )
            if cache_partial_reason and logger is not None:
                logger.warning("benchmark cache disabled: %s", cache_partial_reason)
            cache_active = engaged["cache"] not in (None, "off")

    # HunyuanVideo-1.5 joint-attention trim (per expert), BEFORE the backend set like the loader.
    # Drops the ~99% zero-padded text tokens so the fused SDPA runs (~18x/DiT-forward, cosine ~1.0).
    # Gated on an active tier; no-op for non-Hunyuan families.
    trim_engaged = False
    if speed_active and cfg.get("trim", True):
        for v in views:
            trim_engaged = install_hunyuan_attention_trim(v, fam_obj, logger = logger) or trim_engaged
    engaged["attn_trim"] = trim_engaged

    # Attention (per expert).
    backend = select_attention_backend(tgt, cfg["attn"], speed_active = speed_active)
    for v in views:
        engaged["attn"] = apply_attention_backend(v, backend, logger = logger)

    # Speed profile: apply_speed_optims fans out over EVERY denoiser DiT internally
    # (transformer + transformer_2), so ONE pipe-level call covers a dual-expert MoE --
    # exactly like the production loader. Looping over the second-expert view would compile
    # and fuse transformer_2 a second time, so the warmup/compiled state these rows measure
    # would no longer match the loader this script mirrors.
    if speed != "off":
        engaged["speed_optims"] = apply_speed_optims(
            pipe,
            tgt,
            is_gguf = False,
            family = fam_obj,
            speed_mode = speed,
            cache_active = cache_active,
            offload_active = False,
            logger = logger,
        )
    engaged["_effective_speed"] = speed
    # emulate_precision_casts A/B: the speed layer sets it True; compile is lazy, so flipping
    # it back off here gives the pre-round-2 inductor numerics for the run ("epc_off" config).
    if not cfg.get("epc", True):
        try:
            import torch
            torch._inductor.config.emulate_precision_casts = False
            engaged["epc"] = False
        except Exception:
            pass
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
    cache_threshold = None,
    cache_quality = None,
    family = None,
    fam_obj = None,
    logger = None,
):
    """One clip generation. Re-checks the step cache per generation (maybe_toggle_step_cache)
    exactly like the loader, then times total + per-step. Returns (output, total_s, [per_step_ms])."""
    import torch

    from core.inference.diffusion_cache import (
        _disengage_step_cache,
        apply_step_cache,
        auto_cache_mode,
        auto_cache_quality,
        maybe_toggle_step_cache,
        normalize_cache_quality,
    )
    from core.inference.video import _step_cache_all_or_none

    # Re-check the step cache per generation exactly like the loader, and route it through the
    # SAME transactional helper (_step_cache_all_or_none) the load path uses. Toggling each expert
    # view independently (swallowing per-expert results/exceptions) can engage one expert while
    # transformer_2 fails, so the row would time a mixed cached/uncached MoE state -- a config
    # production never runs, and one whose stale engagement would poison later rows in the matrix.
    # The helper rolls back a partial engage so every timed row measures a real configuration.
    if cache_mode == "auto":
        def _toggle_cache(view: Any, expert: str) -> Optional[str]:
            return maybe_toggle_step_cache(
                view,
                steps = steps,
                quant_active = dit_quant_active,
                threshold = cache_threshold,
                mode = auto_cache_mode(family),
                family = family,
                quality = normalize_cache_quality(cache_quality) or auto_cache_quality(family),
                expert = expert,
                logger = logger,
            )

        _step_cache_all_or_none(pipe, fam_obj, _toggle_cache, logger = logger)
    elif cache_mode == "magcache":
        # An explicit magcache never toggles off, but production re-engages it when the step
        # count differs (marker carries "#s{steps}"; endswith, not substring). _apply_levers
        # installed at default_steps, so a --steps override would else time a stale curve. Wrapped
        # in the all-or-none helper so a mixed resize rolls back instead of timing a split pair.
        def _resize_magcache(view: Any, expert: str) -> Optional[str]:
            transformer = getattr(view, "transformer", None)
            marker = getattr(transformer, "_unsloth_step_cache", None)
            if not marker or str(marker).endswith(f"#s{int(steps)}"):
                return "magcache"  # already sized for these steps
            _disengage_step_cache(
                transformer,
                reason = f"explicit magcache re-interpolating for {steps} steps",
                logger = logger,
            )
            return apply_step_cache(
                view,
                mode = "magcache",
                threshold = cache_threshold,
                quant_active = dit_quant_active,
                family = family,
                steps = steps,
                quality = normalize_cache_quality(cache_quality) or auto_cache_quality(family),
                expert = expert,
                logger = logger,
            )

        _step_cache_all_or_none(pipe, fam_obj, _resize_magcache, logger = logger)

    # Clear step-cache residuals before EVERY generation, like production: diffusers keys
    # them on the long-lived transformer, so measured iterations would otherwise start
    # against the previous clip's cache state. Best-effort, uncached is a no-op.
    for name in ("transformer", "transformer_2"):
        module = getattr(pipe, name, None)
        reset = getattr(module, "_reset_stateful_cache", None) or getattr(
            module, "reset_stateful_hooks", None
        )
        if callable(reset):
            try:
                reset()
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
    restore_step = None
    if guidance_via_guider:
        # HunyuanVideo-1.5: CFG lives on a guider component; __call__ takes no guidance_scale
        # (the loader writes the scale onto pipe.guider).
        guider = getattr(pipe, "guider", None)
        if guider is not None and hasattr(guider, "guidance_scale"):
            try:
                guider.guidance_scale = guidance
            except Exception:
                pass
        # __call__ ignores callback_on_step_end here, so the _cb step timer never fires and
        # per_step_ms would publish 0.0 for every Hunyuan row. Time the denoise via a
        # scheduler.step wrapper instead, restored after generation.
        sched = getattr(pipe, "scheduler", None)
        orig_step = getattr(sched, "step", None)
        if callable(orig_step):

            def _timed_step(*a, **k):
                torch.cuda.synchronize()
                now = time.perf_counter()
                if last[0]:
                    step_ts.append((now - last[0]) * 1000.0)
                last[0] = now
                return orig_step(*a, **k)

            sched.step = _timed_step
            restore_step = lambda: setattr(sched, "step", orig_step)  # noqa: E731
    else:
        kwargs["guidance_scale"] = guidance
        kwargs["callback_on_step_end"] = _cb

    _sync()
    t0 = time.perf_counter()
    try:
        out = pipe(**kwargs)
    finally:
        if restore_step is not None:
            restore_step()
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
    cache_threshold: Optional[float] = None,
    cache_quality: Optional[str] = None,
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
    # Fresh dynamo state per config so a prior config's compiled graphs can't leak in.
    try:
        import torch
        torch._dynamo.reset()
    except Exception:
        pass
    # Loader order: build on CPU, apply levers, THEN place on CUDA. Placing dense first would
    # OOM configs whose quantized form fits, and misrecord load_peak_gb.
    pipe = _build_pipe(repo, force_fp32)
    engaged = _apply_levers(
        pipe,
        cfg,
        fam_name = family,
        fam_obj = fam_obj,
        force_fp32_vae = force_fp32,
        default_steps = default_steps,
        cache_threshold = cache_threshold,
        cache_quality = cache_quality,
        logger = logger,
    )
    pipe = pipe.to("cuda")
    _sync()
    load_peak = _peak_gb()
    _empty()
    weights_gb = _alloc_gb()
    dit_active = engaged["dit"] is not None
    cache_mode = cfg["cache"]

    # warmup (pays the one-time compile / autotune)
    warmup_t0 = time.perf_counter()
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
        cache_threshold = cache_threshold,
        cache_quality = cache_quality,
        family = family,
        fam_obj = fam_obj,
        logger = logger,
    )
    warmup_s = time.perf_counter() - warmup_t0
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
            cache_threshold = cache_threshold,
            cache_quality = cache_quality,
            family = family,
            fam_obj = fam_obj,
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
    # Persist EVERY config's frames too, so a row generated before the reference exists can
    # be LPIPS-rescored offline instead of publishing null.
    if arrs:
        try:
            import numpy as _np
            _np.savez_compressed(out / f"frames_{family}_{name}.npz", *arrs)
        except Exception:
            pass

    # Report the GENERATION-time cache state, not the load-time one: the per-generation recheck can
    # toggle an auto cache off or re-size a magcache. The marker's mode prefix IS the live state.
    cache_marker = getattr(getattr(pipe, "transformer", None), "_unsloth_step_cache", None)
    row = {
        "config": name,
        "family": family,
        "levers": cfg,
        "dit_scheme": engaged["dit"] or "dense",
        "dit_experts": engaged.get("dit_experts"),
        "te_scheme": engaged["te"] or "dense",
        "vae_scheme": engaged["vae"] or "dense",
        "attn": engaged["attn"] or "native",
        "cache": str(cache_marker).split("@")[0] if cache_marker else "off",
        "cache_at_load": engaged["cache"] or "off",
        "effective_speed": engaged.get("_effective_speed"),
        "speed_optims": engaged["speed_optims"],
        "attn_trim": engaged.get("attn_trim", False),
        "cache_threshold": cache_threshold,
        "cache_quality": cache_quality,
        "cache_marker": cache_marker,
        "load_peak_gb": round(load_peak, 2),
        "weights_gb": round(weights_gb, 2),
        "gen_peak_gb": round(gen_peak, 2),
        "warmup_s": round(warmup_s, 3),
        "gen_latency_s": round(_median(dts), 3),
        "per_step_ms": round(_median(steps_ms), 1),
        "n_frames": len(arrs),
        "mean_luma": _mean_luma(arrs),
    }
    del pipe
    _empty()
    # Restore the process-wide backend flags this config's speed layer mutated (snapshot from
    # _apply_levers), like the unload path, so a later config starts from clean flags.
    try:
        from core.inference.diffusion_speed import restore_backend_flags
        restore_backend_flags(engaged.get("_flags_snapshot"))
    except Exception:
        pass
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
    ap.add_argument(
        "--cache-threshold",
        type = float,
        default = None,
        help = "FBCache residual-diff threshold override (None -> the production default)",
    )
    ap.add_argument(
        "--cache-quality",
        default = None,
        choices = ("quality", "balanced", "fast"),
        help = "Step-cache quality preset (None -> the family's production auto default)",
    )
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
    row_arrs: list = []
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
            cache_threshold = args.cache_threshold,
            cache_quality = args.cache_quality,
            logger = logger,
        )
        if n == "reference":
            ref_arrs = arrs
        row["lpips_vs_reference"] = _mean_lpips(ref_arrs, arrs) if ref_arrs is not None else None
        rows.append(row)
        row_arrs.append(arrs)
        print(
            f"  [{n}] {json.dumps({k: row[k] for k in ('dit_scheme','te_scheme','attn','cache','effective_speed','weights_gb','gen_peak_gb','gen_latency_s','per_step_ms','lpips_vs_reference')})}",
            flush = True,
        )

    # Rows finalized BEFORE the reference clip was generated scored lpips_vs_reference as None;
    # rescore them now that the reference frames exist.
    if ref_arrs is not None:
        for r, arrs in zip(rows, row_arrs):
            if r["lpips_vs_reference"] is None:
                r["lpips_vs_reference"] = _mean_lpips(ref_arrs, arrs)

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
