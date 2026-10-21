# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed a conventional video pipeline with hosted pre-quantized denoisers, all-or-nothing.

The MoE's second expert uses the ``task`` slot with NO filename fallback: it would load expert 1 as expert 2.
"""

from __future__ import annotations

from typing import Any, Optional

MOE_SECOND_DENOISER = "transformer_2"


def hosted_denoiser_refused(
    source: Any,
    scheme: Optional[str],
    base_repo: Optional[str],
    *,
    component: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> bool:
    """Whether the loader would refuse hosted ``source`` from its header, asked BEFORE anything is staged:
    the same base / min_features / component the loader checks, so a stale artifact is neither fetched nor
    allowed to drop the dense shards. Offline (``HF_HUB_OFFLINE`` or a ``local_files_only`` load scope)
    only a cached file is read. Unknown is never a refusal. Never raises."""
    if getattr(source, "kind", None) != "repo":
        return False
    try:
        from .diffusion import hub_cache_dir
        from .diffusion_prequant import hosted_prequant_refusal
        from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES

        return (
            hosted_prequant_refusal(
                source,
                str(scheme or "").strip().lower(),
                base = base_repo,
                min_features = DEFAULT_MIN_LINEAR_FEATURES,
                component = component,
                hf_token = hf_token,
                cache_dir = hub_cache_dir(),
            )
            is not None
        )
    except Exception:  # noqa: BLE001 -- an unanswerable probe keeps today's plan
        return False


def denoiser_components(fam: Any) -> tuple[str, ...]:
    if getattr(fam, "is_moe", False):
        return ("transformer", MOE_SECOND_DENOISER)
    return ("transformer",)


def denoiser_prequant_sources(
    fam: Any,
    scheme: Optional[str],
    base_repo: Optional[str],
    *,
    hf_token: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """``{component: PrequantSource}`` for EVERY denoiser, or None (never partial). A hosted source whose
    header the loader would refuse (``hosted_denoiser_refused``) counts as none, so planning keeps the
    dense shards and never fetches it. Never raises."""
    wanted = (scheme or "").strip().lower()
    if wanted in ("", "auto", "off", "none"):
        return None
    try:
        from .diffusion_nvfp4_flag import nvfp4_blocked

        if nvfp4_blocked(wanted):
            return None
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
        for component, source in sources.items():
            if hosted_denoiser_refused(
                source, wanted, base_repo, component = component, hf_token = hf_token
            ):
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
    """Component overrides for pipeline assembly, or ``{}`` (not seeded whole: re-plan memory at bf16)."""
    try:
        import gc

        sources = denoiser_prequant_sources(fam, scheme, base_repo, hf_token = hf_token)
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
                config_subfolder = component,
                component = component,
                cache_dir = cache_dir,
                local_files_only = local_files_only,
                logger = logger,
            )
            if module is None:
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
            from .diffusion_prequant_cache import note_hit, path_in_cache

            if getattr(source, "kind", None) == "path" and path_in_cache(source.location):
                note_hit(source.location, logger)
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
