# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in step caching for the diffusion transformer (First-Block-Cache).

Across denoising steps a DiT's output changes little once the trajectory settles, so most of
the transformer can be reused. First-Block-Cache (FBCache) computes the first block, and if
its residual barely changed from the previous step (within ``threshold``) it skips the
remaining blocks and reuses their cached output. diffusers ships it natively
(``transformer.enable_cache(FirstBlockCacheConfig(...))`` for CacheMixin models, or the
standalone ``apply_first_block_cache`` hook).

Measured on Flux.1-dev (28 steps, 1024px, B200): ~1.4x on top of torch.compile (2.83 ->
2.03 s) at LPIPS ~0.08 vs the no-cache output -- deep inside the speed-for-quality bar.

OFF by default and a deliberate per-load opt-in, because the win scales with step count: a
few-step distilled model (e.g. Z-Image-Turbo at ~8 steps) has almost no headroom and a
single skipped step is a large fraction of the trajectory, so caching is for many-step
models (Flux / Qwen-Image). It composes with torch.compile only with ``fullgraph=False``
(the cache's compiler-disabled decision is a graph break), which the speed layer switches to
automatically when a cache is engaged. Best-effort: an incompatible model (e.g. a transformer
whose block signature the hook does not recognise) is caught and the load proceeds uncached.
torch / diffusers imported lazily.
"""

from __future__ import annotations

from typing import Any, Optional

TC_OFF = "off"
TC_AUTO = "auto"
TC_FBCACHE = "fbcache"
TC_MAGCACHE = "magcache"
TC_MODES = (TC_FBCACHE, TC_MAGCACHE)

# FBCache residual thresholds: higher skips more steps (faster, lower quality). The dense
# bf16 default; a quantised transformer shifts the residual distribution, so it needs a
# higher threshold for the cache to trigger at all (per ParaAttention's fp8 guidance).
DEFAULT_FBCACHE_THRESHOLD = 0.08
QUANT_FBCACHE_THRESHOLD = 0.12

# MagCache (diffusers >= 0.39): skips whole steps from a PRE-CALIBRATED residual-magnitude
# curve with an accumulated-error budget, a consecutive-skip cap, and a no-skip retention
# window over the early steps -- so unlike FBCache the divergence from the uncached
# trajectory is bounded. Measured on HunyuanVideo-1.5-720p (B200, 50 steps, 720p clip):
# threshold 0.12 = 1.5x end-to-end at LPIPS 0.147 vs the same uncached stack with the SAME
# composition (FBCache at its 0.08 default reached 2.4x but LPIPS 0.54: a brighter,
# visibly different clip -- why the fbcache auto policy excludes this family).
DEFAULT_MAGCACHE_THRESHOLD = 0.12
MAGCACHE_MAX_SKIP_STEPS = 3
MAGCACHE_RETENTION_RATIO = 0.2

# ── cache quality presets ──────────────────────────────────────────────────────────
# A user-facing speed/accuracy knob over the step cache's internals (threshold, skip cap,
# retention window). "balanced" is exactly the pre-knob shipped behaviour; "quality"
# trades most of the cache speedup for a near-lossless clip; "fast" skips more
# aggressively. An explicit transformer_cache_threshold always overrides the preset's
# threshold (the preset still supplies the magcache skip cap / retention window).
CQ_QUALITY = "quality"
CQ_BALANCED = "balanced"
CQ_FAST = "fast"
CACHE_QUALITY_LEVELS = (CQ_QUALITY, CQ_BALANCED, CQ_FAST)

# MagCache preset -> (threshold, max_skip_steps, retention_ratio). Calibrated on
# HunyuanVideo-1.5-720p (B200, 1280x720, 33 frames, 50 steps, pairwise LPIPS vs the same
# uncached trim+cudnn+compile stack, WITH the compiled hook inners below): quality
# (0.06, 2, 0.3) = 1.64x at LPIPS 0.050 (30 steps: 1.63x at 0.093) vs balanced
# (0.12, 3, 0.2) = 2.17x at LPIPS 0.129 (30 steps: 2.02x at 0.201). Skip counts bind on
# the cap + retention window below threshold ~0.12, which is why quality tightens all
# three rather than just the threshold.
_MAGCACHE_QUALITY_PRESETS: dict[str, tuple[float, int, float]] = {
    CQ_QUALITY: (0.06, 2, 0.3),
    CQ_BALANCED: (DEFAULT_MAGCACHE_THRESHOLD, MAGCACHE_MAX_SKIP_STEPS, MAGCACHE_RETENTION_RATIO),
    CQ_FAST: (0.24, MAGCACHE_MAX_SKIP_STEPS, MAGCACHE_RETENTION_RATIO),
}

# FBCache preset -> threshold (dense, quant-active). "balanced" keeps the measured
# defaults (0.08 dense / 0.12 quantised); "quality" halves the trigger so the cache only
# reuses when the first-block residual is nearly static; "fast" uses the quantised
# threshold everywhere.
_FBCACHE_QUALITY_THRESHOLDS: dict[str, tuple[float, float]] = {
    CQ_QUALITY: (0.04, 0.06),
    CQ_BALANCED: (DEFAULT_FBCACHE_THRESHOLD, QUANT_FBCACHE_THRESHOLD),
    CQ_FAST: (QUANT_FBCACHE_THRESHOLD, 0.15),
}


def normalize_cache_quality(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested cache quality; None / "" / "auto" -> None (the loader
    resolves it per family via ``auto_cache_quality``). Raises ValueError for an
    unsupported value."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized == "auto":
        return None
    if normalized not in CACHE_QUALITY_LEVELS:
        raise ValueError(
            f"Unsupported transformer_cache_quality '{value}'. Use one of: auto, "
            f"{', '.join(CACHE_QUALITY_LEVELS)}."
        )
    return normalized


# Families whose UNSET cache quality resolves to the near-lossless "quality" preset
# instead of "balanced". HunyuanVideo-1.5 (both repacks) measured with the compiled
# cache inners (see _compile_hooked_block_inners): quality = 1.63-1.64x at pairwise
# LPIPS 0.05 (50 steps) / 0.09 (30 steps) vs balanced's 2.02-2.17x at 0.13-0.20 --
# the accuracy-first default keeps most of the speedup at under half the drift, and
# balanced / fast stay one explicit request away. Families without a measured quality
# point keep balanced (their pre-knob behaviour).
_FAMILY_AUTO_CACHE_QUALITY: dict[str, str] = {
    "hunyuanvideo-1.5": CQ_QUALITY,
    "hunyuanvideo-1.5-720p": CQ_QUALITY,
}


def auto_cache_quality(family: Optional[str]) -> str:
    """The cache quality preset an UNSET request resolves to for ``family``."""
    return _FAMILY_AUTO_CACHE_QUALITY.get(str(family or "").strip().lower(), CQ_BALANCED)


# The auto policy's step-count bar: FBCache's win scales with step count (each skipped
# step is a larger quality hit on a short trajectory), so auto engages it only at 20+
# steps -- full "dev"-style schedules (28+) qualify, distilled turbo models (4-9) never do.
FBCACHE_MIN_STEPS = 20


# Per-family MagCache magnitude-ratio curves (MagCacheConfig.mag_ratios), calibrated with
# diffusers' calibrate mode on the family base checkpoints at the default 50-step schedule
# (720p clip, B200). The curve is checkpoint-dependent but highly stable where it matters:
# the CFG cond/uncond branches differ by <= 0.014 and a 30-step calibration matches the
# 50-step curve within 0.027 after nearest-interpolation, so ONE curve per family is
# enough -- diffusers interpolates it to the actual step count. Conditional-branch curve
# per the MagCache calibration guidance.
_MAGCACHE_720P_RATIOS = (
    1.0,
    1.0226,
    1.0093,
    1.001,
    1.0008,
    1.0001,
    0.9995,
    1.0003,
    0.9998,
    0.9993,
    0.9994,
    0.9993,
    0.9997,
    1.0002,
    0.9994,
    0.9985,
    0.9987,
    0.9997,
    0.9979,
    0.9987,
    0.9985,
    0.9982,
    0.9977,
    0.998,
    0.9979,
    0.9971,
    0.9968,
    0.9967,
    0.9964,
    0.9965,
    0.9959,
    0.9954,
    0.995,
    0.9938,
    0.9942,
    0.9924,
    0.9924,
    0.9907,
    0.9905,
    0.9878,
    0.9867,
    0.9845,
    0.9808,
    0.9773,
    0.9715,
    0.9652,
    0.9529,
    0.9347,
    0.9011,
    0.83,
)
_MAGCACHE_480P_RATIOS = (
    1.0,
    1.0077,
    1.0138,
    1.0043,
    1.0029,
    0.9986,
    0.9966,
    1.0,
    1.0006,
    0.9996,
    0.9993,
    0.9986,
    1.0,
    0.9993,
    0.9966,
    0.9986,
    0.9988,
    0.9991,
    0.998,
    0.9977,
    0.9976,
    0.9971,
    0.9973,
    0.9969,
    0.996,
    0.9961,
    0.9949,
    0.9958,
    0.9933,
    0.9942,
    0.9941,
    0.9926,
    0.9929,
    0.9916,
    0.9923,
    0.9887,
    0.99,
    0.9882,
    0.9865,
    0.9833,
    0.9827,
    0.9791,
    0.9763,
    0.9718,
    0.9657,
    0.9563,
    0.9454,
    0.9264,
    0.8967,
    0.8382,
)
# Wan2.2-TI2V-5B, calibrated at 1280x704 / 33 frames / 50 steps on the family base
# checkpoint (B200, trim-less compiled stack). Cond/uncond branches agree within
# 0.0008, so one (conditional) curve serves both CFG contexts.
_MAGCACHE_WAN5B_RATIOS = (
    1.0,
    0.9906,
    0.9996,
    0.9936,
    0.9968,
    0.9958,
    0.9956,
    0.9953,
    0.9957,
    0.9954,
    0.9941,
    0.9958,
    0.9933,
    0.9938,
    0.9948,
    0.9936,
    0.9948,
    0.9925,
    0.994,
    0.9927,
    0.9913,
    0.9919,
    0.9918,
    0.9907,
    0.989,
    0.9901,
    0.9892,
    0.9903,
    0.9884,
    0.9868,
    0.9851,
    0.9848,
    0.9849,
    0.9831,
    0.9818,
    0.9804,
    0.9781,
    0.9756,
    0.9733,
    0.9717,
    0.9688,
    0.9646,
    0.9611,
    0.9559,
    0.9503,
    0.9443,
    0.938,
    0.9315,
    0.9227,
    0.9208,
)

# All curves are calibrated at the family's default 50-step schedule. A single-DiT
# curve therefore has 50 entries and MagCacheConfig interpolates it to the actual step
# count. A dual-expert MoE (Wan2.2-A14B) runs each expert on a SLICE of the schedule
# (the boundary_ratio split) and the MagCache hook counts each expert's OWN forwards
# from 0, so each expert carries its own curve, keyed "family::transformer_2" for the
# second expert, whose length is the number of steps that expert ran during the 50-step
# calibration; engage-time scales it proportionally to the requested step count (the
# boundary split is a fixed fraction of the schedule for a given checkpoint).
_MAGCACHE_CALIBRATION_STEPS = 50

_MAGCACHE_FAMILY_RATIOS: dict[str, tuple[float, ...]] = {
    "hunyuanvideo-1.5": _MAGCACHE_480P_RATIOS,
    "hunyuanvideo-1.5-720p": _MAGCACHE_720P_RATIOS,
    "wan2.2-ti2v-5b": _MAGCACHE_WAN5B_RATIOS,
}


def _magcache_ratio_key(family: Optional[str], expert: Optional[str]) -> str:
    """The `_MAGCACHE_FAMILY_RATIOS` key for a (family, expert) pair: the bare family
    name for the primary ``transformer``, ``family::expert`` for a second expert."""
    fam = str(family or "").strip().lower()
    exp = str(expert or "").strip().lower()
    if exp in ("", "transformer"):
        return fam
    return f"{fam}::{exp}"

# Families whose AUTO step-cache decision engages MagCache instead of FBCache. On
# HunyuanVideo-1.5 FBCache free-runs (no skip cap, no error budget) and derails the
# trajectory (LPIPS 0.54 + a luma shift at its default threshold), while MagCache holds
# the same composition at 1.5x -- see the constants above. On Wan2.2-TI2V-5B both modes
# stay composition-true, but MagCache dominates the accuracy/speed frontier (B200,
# 1280x704/33f/50 steps, pairwise LPIPS vs the same uncached compiled stack): balanced
# MagCache 1.65x at 0.034 vs FBCache 0.08 at 1.49x/0.031, and at the fast points 1.73x
# at 0.044 vs 1.71x at 0.083 -- FBCache's error grows unboundedly past its threshold
# while MagCache's budget caps it. On Wan2.2-A14B (dual-expert MoE) the OPPOSITE holds
# (B200, 1280x720/33f/50 steps, per-expert calibrated curves, same pairwise protocol):
# FBCache 0.12 at 2.88x/0.128 dominates balanced MagCache (1.80x/0.145) and FBCache
# 0.08 sits at 1.28x/0.098 vs MagCache quality's 1.14x/0.074 -- the 16-step high-noise
# expert leaves MagCache too few forwards to skip within its error budget -- so the
# family keeps the FBCache default and no calibrated curve ships (an explicit magcache
# request runs uncached with a warning rather than engaging a measured-worse mode).
# Every other family keeps the measured FBCache default. An EXPLICIT
# "fbcache"/"magcache" request always wins.
_FAMILY_AUTO_CACHE_MODE: dict[str, str] = {
    "hunyuanvideo-1.5": TC_MAGCACHE,
    "hunyuanvideo-1.5-720p": TC_MAGCACHE,
    "wan2.2-ti2v-5b": TC_MAGCACHE,
}


def auto_cache_mode(family: Optional[str]) -> str:
    """The cache mode the AUTO policy engages for ``family`` (mode only; the step-count
    bar and the engage call are the caller's job). MagCache additionally needs a
    calibrated ratio curve: a family routed here without one runs uncached (the
    apply_step_cache magcache branch checks), never silently falls back to FBCache."""
    return _FAMILY_AUTO_CACHE_MODE.get(str(family or "").strip().lower(), TC_FBCACHE)


def normalize_transformer_cache(value: Optional[str]) -> Optional[str]:
    """Lower/strip a requested cache mode; None / "" / "none" / "off" -> None (disabled),
    "auto" -> TC_AUTO (the loader decides from the step count).

    Raises ValueError for an unsupported value so a bad request is rejected cheaply."""
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_")
    if not normalized or normalized in ("none", "off"):
        return None
    if normalized == TC_AUTO:
        return TC_AUTO
    if normalized not in TC_MODES:
        raise ValueError(
            f"Unsupported transformer_cache '{value}'. Use one of: off, auto, "
            f"{', '.join(TC_MODES)}."
        )
    return normalized


# Transformer block classes whose FBCache metadata is missing from the installed
# diffusers. The First-Block-Cache hook reads each block's (hidden_states,
# encoder_hidden_states) return layout from TransformerBlockRegistry; diffusers 0.39
# registers the HunyuanVideo 1.0 blocks but not the 1.5 ones, so enable_cache raises
# "Model class HunyuanVideo15TransformerBlock not registered" on a DiT that is
# otherwise fully cache-compatible: CacheMixin, one homogeneous ``transformer_blocks``
# list of residual-additive dual-stream blocks returning (hidden_states,
# encoder_hidden_states) -- the exact layout of the registered 1.0 block. Keyed by the
# TRANSFORMER class name so only a family that needs the patch pays for it, and probed
# via TransformerBlockRegistry.get first so a diffusers release that ships the
# registration natively makes this a no-op.
#   transformer class -> ((block module, block class, hs index, ehs index), ...)
#
# LTX-2 is DELIBERATELY absent: its LTX2VideoTransformerBlock is also unregistered in
# diffusers 0.39, but it returns (hidden_states, audio_hidden_states) -- a JOINT
# video+audio stream -- while both cache hook families cache/skip only the single
# ``hidden_states`` stream and, on a skipped step, fetch the parameter literally named
# ``encoder_hidden_states`` (the TEXT embeddings) for the second return slot. A naive
# registration would therefore feed text embeddings into the next block's audio input
# on every skipped step. Step caching for LTX-2 needs a dual-stream cache
# implementation, not a metadata entry; until then the family runs uncached (verified:
# enable_cache raises "not registered" and the load proceeds uncached, and the
# distilled LTX-2.3 checkpoints run 8-step schedules below FBCACHE_MIN_STEPS anyway).
_EXTRA_BLOCK_METADATA: dict[str, tuple[tuple[str, str, int, Optional[int]], ...]] = {
    "HunyuanVideo15Transformer3DModel": (
        (
            "diffusers.models.transformers.transformer_hunyuan_video15",
            "HunyuanVideo15TransformerBlock",
            0,
            1,
        ),
    ),
}


def _ensure_block_metadata_registered(transformer: Any, logger: Any = None) -> None:
    """Register the missing FBCache block metadata for ``transformer``'s family (see
    ``_EXTRA_BLOCK_METADATA``). Best-effort: a failure just leaves enable_cache to raise
    its own error and the load runs uncached, exactly as before this patch."""
    specs = _EXTRA_BLOCK_METADATA.get(type(transformer).__name__)
    if not specs:
        return
    try:
        import importlib

        from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
        for module_name, cls_name, hs_index, ehs_index in specs:
            block_cls = getattr(importlib.import_module(module_name), cls_name)
            try:
                TransformerBlockRegistry.get(block_cls)
                continue  # a newer diffusers registers it natively
            except ValueError:
                pass
            TransformerBlockRegistry.register(
                block_cls,
                TransformerBlockMetadata(
                    return_hidden_states_index = hs_index,
                    return_encoder_hidden_states_index = ehs_index,
                ),
            )
            if logger is not None:
                logger.info("diffusion.cache: registered %s block metadata for fbcache", cls_name)
    except Exception as exc:  # noqa: BLE001 -- best-effort; enable_cache surfaces the real error
        _warn(logger, "block metadata registration", exc)


def _invalidate_child_registry_cache(transformer: Any) -> None:
    """Drop the HookRegistry's cached child-registry list after (un)installing hooks.

    ``cache_context`` propagates the state context through ``_get_child_registries``,
    which diffusers 0.39 caches on first use. An UNCACHED generation already calls
    ``cache_context`` (the pipeline wraps every denoise call), creating the
    transformer-level registry with an EMPTY cached child list -- so a later
    ``enable_cache`` (the auto step-count toggle engaging FBCache mid-session) installs
    block hooks that ``_set_context`` never reaches, and the first cached forward dies
    with "No context is set". Invalidate the stale cache so the next ``cache_context``
    rebuilds it over the freshly hooked blocks. Best-effort and cheap (one attribute)."""
    registry = getattr(transformer, "_diffusers_hook", None)
    if registry is not None and getattr(registry, "_child_registries_cache", None) is not None:
        try:
            registry._child_registries_cache = None
        except Exception:  # noqa: BLE001 -- diffusers internals moved; leave as-is
            pass


# diffusers' cache hook registry names whose compute branch we re-point at a compiled
# inner forward (leader = the measuring first block, block = the remaining ones); both
# hook families share the fn_ref layout.
_CACHE_HOOK_NAMES = (
    "mag_cache_leader_block_hook",
    "mag_cache_block_hook",
    "fbc_leader_block_hook",
    "fbc_block_hook",
)


def _compile_hooked_block_inners(transformer: Any, logger: Any = None) -> int:
    """Restore the regional compile on cache-hooked blocks' COMPUTED steps.

    ``enable_cache`` replaces each block's ``forward`` with the hook's ``new_forward``
    (stashing the pre-hook bound method in ``fn_ref.original_forward``), whose skip
    decision is data-dependent Python: MagCache ``@torch.compiler.disable``s the whole
    ``new_forward`` (recursive -- the compute branch runs EAGER), and even FBCache's
    traceable ``new_forward`` graph-breaks around its disabled threshold decision,
    which on some archs (measured: Qwen-Image) drops the compute branch's call into
    ``original_forward`` out of the compiled region -- the block's regional compile
    artifact (``_compiled_call_impl``) is never reached and the cache forfeits the
    compile win on every non-skipped step. An explicitly ``torch.compile``d callable
    re-enables
    dynamo for its own extent even inside a disabled frame, so re-pointing
    ``fn_ref.original_forward`` at a compiled wrapper of the same bound method restores
    compiled compute steps while the skip decision stays eager exactly as designed.
    Measured (B200, scripts/image_speedmem_bench.py): Qwen-Image FBCache computed steps
    91.8 -> 71.2 ms (= the uncached compiled rate), 1.21x end to end; FLUX.1-dev is
    neutral (its FBCache ``new_forward`` happens to trace, so computed steps were
    already compiled -- same-process armed vs unarmed latents bit-identical); on the
    video DiT balanced MagCache went 39.4 -> 26.9 s at 50 steps.

    Only blocks the speed layer actually compiled are armed (``_compiled_call_impl``
    guard -- eager tiers stay untouched), and only when ``original_forward`` is a plain
    bound method (a stacked hook chain, e.g. offload, captures a partial and is
    skipped). Idempotent via the ``_unsloth_orig_inner`` marker; best-effort. Returns
    the number of hooks armed."""
    try:
        import torch
    except Exception:  # noqa: BLE001 -- no torch, nothing to arm
        return 0
    armed = 0
    try:
        for module in transformer.modules():
            registry = getattr(module, "_diffusers_hook", None)
            if registry is None or getattr(module, "_compiled_call_impl", None) is None:
                continue
            hooks = getattr(registry, "hooks", None) or {}
            for name in _CACHE_HOOK_NAMES:
                hook = hooks.get(name)
                fn_ref = getattr(hook, "fn_ref", None) if hook is not None else None
                orig = getattr(fn_ref, "original_forward", None)
                if orig is None or getattr(hook, "_unsloth_orig_inner", None) is not None:
                    continue
                if getattr(orig, "__self__", None) is None:
                    continue  # not the plain bound method; arming would miss the block
                # fullgraph=False / dynamic=True: a cache is active by definition (its
                # decision points graph-break) and this matches the default tier the
                # regional compile used. Dynamo caches per code object, so re-arming
                # after a toggle is effectively free (~0.03 s).
                fn_ref.original_forward = torch.compile(orig, fullgraph = False, dynamic = True)
                hook._unsloth_orig_inner = orig
                armed += 1
    except Exception as exc:  # noqa: BLE001 -- best-effort: the cache still works eager
        _warn(logger, "cache-hook inner compile", exc)
        return armed
    if armed and logger is not None:
        logger.info(
            "diffusion.cache: %d cache-hooked block(s) armed with compiled inner forwards",
            armed,
        )
    return armed


def _restore_hooked_block_inners(transformer: Any) -> None:
    """Undo ``_compile_hooked_block_inners``: put the plain bound methods back and clear
    the markers. MUST run before ``disable_cache`` -- ``remove_hook`` splices
    ``fn_ref.original_forward`` back into ``module.forward``, and leaving the compiled
    wrapper there would pin a stale compiled callable onto the uncached path."""
    try:
        modules = list(transformer.modules())
    except Exception:  # noqa: BLE001 -- not a torch module (tests/fakes): nothing armed
        return
    for module in modules:
        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            continue
        hooks = getattr(registry, "hooks", None) or {}
        for name in _CACHE_HOOK_NAMES:
            hook = hooks.get(name)
            orig = getattr(hook, "_unsloth_orig_inner", None) if hook is not None else None
            if orig is None:
                continue
            try:
                hook.fn_ref.original_forward = orig
                hook._unsloth_orig_inner = None
            except Exception:  # noqa: BLE001 -- per-hook best-effort
                pass


def _pipeline_opens_cache_context(pipe: Any) -> bool:
    """Whether the pipeline enters ``transformer.cache_context(...)`` in its denoise loop.
    The First-Block-Cache hook requires it at run time, and a CacheMixin transformer alone
    does NOT guarantee it: Flux Kontext / img2img / inpaint / controlnet reuse the CacheMixin
    FluxTransformer2DModel but never open a cache_context. Read from the pipeline ``__call__``
    source, resolved off the instance so a per-expert proxy view (``_SecondDiTView``)
    delegates to the real pipe; if it cannot be read, report False so the cache stays off."""
    import inspect

    call = getattr(pipe, "__call__", None)
    if call is None:
        return False
    try:
        src = inspect.getsource(call)
    except (OSError, TypeError):
        return False
    # Match the actual call `cache_context(` -- a bare mention in a comment/docstring lacks
    # the paren, so this does not false-positive on prose.
    return "cache_context(" in src


def apply_step_cache(
    pipe: Any,
    *,
    mode: Optional[str],
    threshold: Optional[float] = None,
    quant_active: bool = False,
    family: Optional[str] = None,
    steps: Optional[int] = None,
    quality: Optional[str] = None,
    expert: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Engage step caching on ``pipe.transformer``. Returns the mode actually engaged, or
    None when disabled / unsupported (the load then runs uncached). ``threshold`` overrides
    the default; ``quant_active`` raises the FBCache default so the cache still triggers on
    a quantised transformer. ``quality`` picks the preset parameter set (threshold + the
    magcache skip cap / retention window); an explicit ``threshold`` still wins over the
    preset's threshold. The magcache mode additionally needs ``family`` (to look up the
    calibrated ratio curve) and ``steps`` (MagCache interpolates that curve over the
    configured step count and sizes its no-skip retention window from it); a dual-expert
    MoE caller passes ``expert`` (the pipe attribute the view exposes, e.g.
    "transformer_2") so each expert gets ITS OWN calibrated curve -- the experts split
    the schedule at the boundary timestep, and the hook counts each expert's own
    forwards from 0, so one shared full-schedule curve would be misaligned for both.
    Best-effort: never raises for an incompatible model."""
    mode = normalize_transformer_cache(mode)
    if mode is None or mode == TC_AUTO:
        # AUTO must be resolved by the loader (step-count policy) before reaching the
        # engage call; treat a stray auto as off rather than crashing the load.
        return None
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    quality = normalize_cache_quality(quality) or CQ_BALANCED
    if mode == TC_MAGCACHE:
        preset_thr, mag_skip, mag_retention = _MAGCACHE_QUALITY_PRESETS[quality]
        thr = threshold if threshold is not None else preset_thr
    else:
        dense_thr, quant_thr = _FBCACHE_QUALITY_THRESHOLDS[quality]
        thr = threshold if threshold is not None else (quant_thr if quant_active else dense_thr)
    # Engage only via the transformer's native enable_cache (the diffusers CacheMixin path):
    # the lower-level apply_first_block_cache hook would install on a non-CacheMixin
    # transformer too (e.g. Z-Image), whose pipeline opens no cache_context and would crash
    # the first generation -- so a model without enable_cache runs uncached per the
    # best-effort contract instead of being reported as cached and then failing.
    enable_cache = getattr(transformer, "enable_cache", None)
    if not callable(enable_cache):
        _warn(logger, mode, RuntimeError("transformer has no cache_context (not a CacheMixin)"))
        return None
    # A CacheMixin transformer is necessary but NOT sufficient: the First-Block-Cache hook
    # raises "No context is set" on the first forward unless the PIPELINE wraps its denoise
    # loop in transformer.cache_context(...). Flux Kontext / img2img / inpaint / controlnet
    # reuse the CacheMixin FluxTransformer2DModel yet their __call__ opens no cache_context,
    # so engaging FBCache there would crash every default generation -- run uncached instead.
    if not _pipeline_opens_cache_context(pipe):
        _warn(
            logger, mode, RuntimeError("pipeline __call__ opens no cache_context; running uncached")
        )
        return None
    # Some cache-compatible block classes are missing from the installed diffusers'
    # FBCache metadata registry (HunyuanVideo-1.5); register them before enable_cache.
    # Both hook families (FBCache / MagCache) read the same block metadata.
    _ensure_block_metadata_registered(transformer, logger)
    try:
        if mode == TC_MAGCACHE:
            ratio_key = _magcache_ratio_key(family, expert)
            ratios = _MAGCACHE_FAMILY_RATIOS.get(ratio_key)
            if ratios is None:
                # No silent FBCache fallback: the family was routed to magcache exactly
                # because FBCache derails it, so an uncalibrated checkpoint runs uncached.
                _warn(
                    logger,
                    mode,
                    RuntimeError(f"no calibrated mag_ratios for '{ratio_key}'"),
                )
                return None
            if not steps or int(steps) <= 0:
                _warn(logger, mode, RuntimeError("magcache needs the step count to engage"))
                return None
            from diffusers.hooks import MagCacheConfig

            # A full-schedule curve (one entry per calibration step) interpolates to the
            # requested step count directly. An expert SUB-curve (dual-expert MoE) covers
            # only that expert's slice of the calibration schedule, and the hook indexes
            # it by the expert's own forward count, so scale its configured step count by
            # the same steps/calibration ratio: the boundary split is a fixed fraction of
            # the schedule, so the expert runs ~len(ratios) * steps / 50 forwards.
            num_steps = int(steps)
            if len(ratios) != _MAGCACHE_CALIBRATION_STEPS:
                num_steps = max(
                    1, round(len(ratios) * int(steps) / _MAGCACHE_CALIBRATION_STEPS)
                )
            config: Any = MagCacheConfig(
                threshold = thr,
                max_skip_steps = mag_skip,
                retention_ratio = mag_retention,
                num_inference_steps = num_steps,
                mag_ratios = list(ratios),
            )
            # The curve is interpolated over the CONFIGURED step count, so the marker
            # carries it: the auto toggle re-engages on a step-count change.
            marker = f"{mode}@{thr}#s{int(steps)}"
        else:
            try:
                from diffusers import FirstBlockCacheConfig
            except ImportError:  # older diffusers exports it only from diffusers.hooks
                from diffusers.hooks import FirstBlockCacheConfig

            config = FirstBlockCacheConfig(threshold = thr)
            marker = f"{mode}@{thr}"
        enable_cache(config)
        # A prior uncached generation may have frozen an empty child-registry list on
        # the transformer's HookRegistry; the block hooks just installed would then
        # never receive the cache context. Must follow every enable_cache.
        _invalidate_child_registry_cache(transformer)
        # If the blocks are already regionally compiled (the generation-time toggle
        # path: compile ran at load), re-point the fresh hooks' compute branch at
        # compiled inners; the load path (cache before compile) is armed by
        # _compile_repeated_blocks instead. No-op when nothing is compiled.
        _compile_hooked_block_inners(transformer, logger)
        try:
            transformer._unsloth_step_cache = marker
        except Exception:  # noqa: BLE001 — marker is best-effort
            pass
        if logger is not None:
            logger.info("diffusion.cache: %s engaged (threshold=%s)", mode, thr)
        return mode
    except Exception as exc:  # noqa: BLE001 — incompatible model -> run uncached
        # enable_cache can fail after hooking some blocks; drop any partial hooks so
        # the reported-uncached model doesn't actually run half-cached. Any armed
        # compiled inners must be restored FIRST (remove_hook splices original_forward
        # back into module.forward).
        _restore_hooked_block_inners(transformer)
        try:
            transformer.disable_cache()
        except Exception:  # noqa: BLE001
            pass
        _warn(logger, mode, exc)
        return None


def effective_denoise_steps(steps: int, strength: Optional[float]) -> int:
    """The number of steps diffusers ACTUALLY denoises for a request.

    An image-conditioned workflow with ``strength`` < 1 (img2img / upscale / inpaint) runs
    only a fraction of ``num_inference_steps``: diffusers' ``get_timesteps`` computes
    ``init_timestep = min(int(num_inference_steps * strength), num_inference_steps)`` and
    denoises exactly ``init_timestep`` steps -- the product is FLOORED, not rounded. The auto
    step-cache policy must key on THIS count -- e.g. a 28-step upscale at strength 0.35 runs
    ``int(9.8) = 9`` real steps, exactly the short trajectory FBCache should stay off (each
    skipped step is a large quality hit). ``strength`` None (txt2img / reference) or >= 1 -> the
    full count.
    """
    s = int(steps)
    if strength is None or float(strength) >= 1.0:
        return s
    return max(1, min(int(s * float(strength)), s))


def effective_request_strength(
    request_strength: Optional[float],
    has_init_image: bool,
    pipe_accepts_strength: bool,
    pipe_default_strength: Any,
) -> Optional[float]:
    """The strength the pipe will ACTUALLY apply, for keying the auto step-cache policy.

    Only image-conditioned pipelines that take ``strength`` apply it (txt2img / a pipe without
    the kwarg run the full trajectory -> None). When the request omits ``strength`` the loader
    does NOT pass the kwarg, so the pipe runs its OWN signature default (< 1 for every img2img /
    inpaint pipeline here, e.g. 0.6); the policy must key on that default, not the full step
    count, or FBCache engages on a fraction of the advertised steps. A non-numeric default
    (``inspect.Parameter.empty``) falls back to the full count (None).
    """
    if not (has_init_image and pipe_accepts_strength):
        return None
    if request_strength is not None:
        return request_strength
    return pipe_default_strength if isinstance(pipe_default_strength, (int, float)) else None


def _disengage_step_cache(
    transformer: Any,
    *,
    reason: str,
    logger: Any = None,
) -> bool:
    """disable_cache + clear the marker; True when the transformer is now uncached."""
    disable_cache = getattr(transformer, "disable_cache", None)
    if not callable(disable_cache):
        return False
    try:
        # Before remove_hook splices fn_ref.original_forward back into module.forward:
        # the compiled inner wrappers must not leak onto the uncached path.
        _restore_hooked_block_inners(transformer)
        disable_cache()
        transformer._unsloth_step_cache = None
        if logger is not None:
            logger.info("diffusion.cache: step cache disengaged (%s)", reason)
        return True
    except Exception as exc:  # noqa: BLE001 -- keep the cache rather than crash
        _warn(logger, "step cache disable", exc)
        return False


def maybe_toggle_step_cache(
    pipe: Any,
    *,
    steps: int,
    quant_active: bool = False,
    threshold: Optional[float] = None,
    mode: str = TC_FBCACHE,
    family: Optional[str] = None,
    quality: Optional[str] = None,
    expert: Optional[str] = None,
    logger: Any = None,
) -> Optional[str]:
    """Generation-time enable/disable for an AUTO cache decision, keyed on the actual
    step count: engage ``mode`` (the family's auto cache mode) at ``FBCACHE_MIN_STEPS``
    or more, run uncached below it. Idempotent (the ``_unsloth_step_cache`` marker tracks
    the engaged state), so calling it on every generation is cheap -- except a magcache
    step-count change, which re-engages so the ratio curve is re-interpolated over the
    actual schedule. Only the loader's auto path calls this; an explicit user choice is
    never toggled. Returns the mode now active (or None when uncached)."""
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    engaged = getattr(transformer, "_unsloth_step_cache", None)
    want = int(steps) >= FBCACHE_MIN_STEPS
    if (
        want
        and engaged
        and mode == TC_MAGCACHE
        # endswith, not substring: "#s5" would match inside "#s50".
        and not str(engaged).endswith(f"#s{int(steps)}")
        and _disengage_step_cache(
            transformer, reason = f"magcache re-interpolating for {steps} steps", logger = logger
        )
    ):
        engaged = None
    if want and not engaged:
        return apply_step_cache(
            pipe,
            mode = mode,
            threshold = threshold,
            quant_active = quant_active,
            family = family,
            steps = steps,
            quality = quality,
            expert = expert,
            logger = logger,
        )
    if not want and engaged:
        if _disengage_step_cache(
            transformer,
            reason = f"auto: {steps} steps < {FBCACHE_MIN_STEPS}",
            logger = logger,
        ):
            return None
        return mode
    return mode if engaged else None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.cache: %s unavailable (%s); running uncached", what, exc)
