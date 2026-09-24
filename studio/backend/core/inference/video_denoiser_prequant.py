# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed a conventional video pipeline with hosted pre-quantized denoisers, all-or-nothing.

The MoE second expert has no filename fallback (it would load expert 1). Torch-free at import."""

from __future__ import annotations

from typing import Any, Optional

MOE_SECOND_DENOISER = "transformer_2"


def denoiser_components(fam: Any) -> tuple[str, ...]:
    """Pipeline attribute name(s) of this family's denoiser(s), from the registry."""
    if getattr(fam, "is_moe", False):
        return ("transformer", MOE_SECOND_DENOISER)
    return ("transformer",)


def denoiser_prequant_sources(
    fam: Any, scheme: Optional[str], base_repo: Optional[str]
) -> Optional[dict[str, Any]]:
    """``{component: PrequantSource}`` for EVERY denoiser, or None (never partial); never raises."""
    wanted = (scheme or "").strip().lower()
    if wanted in ("", "auto", "off", "none"):
        return None
    try:
        from .diffusion_prequant import resolve_prequant_source
        from .video_families import video_family_prequant_task_specific

        sources: dict[str, Any] = {}
        for component in denoiser_components(fam):
            task = None if component == "transformer" else component
            if task is not None and not video_family_prequant_task_specific(fam, wanted, task):
                # Without its own row, ``task`` resolves expert 1's file: loads fine, wrong weights.
                return None
            sources[component] = resolve_prequant_source(
                fam,
                wanted,
                base_repo = base_repo,
                task = task,
            )
            if sources[component] is None:
                return None
        return sources or None
    except Exception:  # noqa: BLE001 -- an unanswerable registry keeps the dense denoiser
        return None


def denoiser_prequant_pipe_kwargs(
    fam: Any,
    base_repo: str,
    *,
    scheme: Optional[str],
    dtype: Any,
    device: str,
    hf_token: Optional[str] = None,
    target: Any = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Overrides for assembly, or ``{}`` (dense bf16 + runtime quant) unless all denoisers seed."""
    try:
        import gc

        sources = denoiser_prequant_sources(fam, scheme, base_repo)
        if not sources:
            return {}
        if target is not None:
            from .diffusion_transformer_quant import dense_transformer_supported
            if not dense_transformer_supported(target):
                return {}
        import diffusers

        from .diffusion_prequant import load_prequantized_transformer
        from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES

        transformer_cls = getattr(diffusers, str(getattr(fam, "transformer_class", "")), None)
        if transformer_cls is None:
            _warn(
                logger,
                scheme or "",
                ValueError(
                    f"transformer class {getattr(fam, 'transformer_class', None)!r} "
                    "not found in diffusers"
                ),
            )
            return {}
        seeded: dict[str, Any] = {}
        for component, source in sources.items():
            module = load_prequantized_transformer(
                transformer_cls,
                base_repo,
                source,
                device = device,
                dtype = dtype,
                hf_token = hf_token,
                scheme = scheme,
                min_features = DEFAULT_MIN_LINEAR_FEATURES,
                # Stamp which expert this is: once loaded the two are indistinguishable.
                config_subfolder = component,
                component = component,
                cache_dir = cache_dir,
                local_files_only = local_files_only,
                logger = logger,
            )
            if module is None:
                # ALL or none; drop what loaded so host memory is freed before the dense build.
                seeded.clear()
                gc.collect()
                if logger is not None:
                    logger.info(
                        "video.denoiser_prequant: no usable %s checkpoint for %s %s, so the "
                        "dense denoiser(s) are loaded and quantised at runtime instead",
                        scheme,
                        getattr(fam, "name", None),
                        component,
                    )
                return {}
            seeded[component] = module
            # Never let two experts coexist in host memory.
            gc.collect()
        if logger is not None:
            logger.info(
                "video.denoiser_prequant: seeded %s pre-quantized denoiser(s) for %s (%s)",
                len(seeded),
                getattr(fam, "name", None),
                scheme,
            )
        return seeded
    except Exception as exc:  # noqa: BLE001 -- seeding is an optimisation, never a blocker
        _warn(logger, "pipe_kwargs", exc)
        return {}


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("video.denoiser_prequant: %s failed: %s", what, exc)
