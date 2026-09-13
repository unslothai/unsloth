# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed a CONVENTIONAL video pipeline with hosted pre-quantized denoiser(s).

The dual-expert MoE's second denoiser is addressed through the ``task`` slot of
``prequant_filenames``, which deliberately gets no filename fallback: falling back would load
expert 1 as expert 2 and pass every check on the way. Seeding is ALL-OR-NOTHING across experts.
Torch-free at import so planning code can ask what would be seeded without paying for diffusers.
"""

from __future__ import annotations

from typing import Any, Optional

# Filename key, not a workflow: the pipeline attribute and the task string are the same word so the family table reads as "this row is transformer_2's artifact".
MOE_SECOND_DENOISER = "transformer_2"


def denoiser_components(fam: Any) -> tuple[str, ...]:
    """The pipeline attribute name(s) this family's denoiser(s) live under.

    Registry-only, unlike ``video._transformer_names`` which reads a BUILT pipe."""
    if getattr(fam, "is_moe", False):
        return ("transformer", MOE_SECOND_DENOISER)
    return ("transformer",)


def denoiser_prequant_sources(
    fam: Any, scheme: Optional[str], base_repo: Optional[str]
) -> Optional[dict[str, Any]]:
    """``{component: PrequantSource}`` for EVERY denoiser this family has, or None.

    None rather than a partial dict when any component is unresolved: a dict missing one expert
    reads as "some coverage" at every caller. Pure and never raises: it runs on the
    download-planning path.
    """
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
                # Without a row of its own, ``task`` resolves the first expert's file, which would load cleanly and denoise the second half of the schedule with the wrong weights.
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
    """Component overrides for pipeline assembly, or ``{}`` when the model cannot be seeded whole.

    ``{}`` is not a failure: assembly builds the dense bf16 DiT and the caller's runtime
    ``quantize_transformer`` rewrites it in place, so a memory plan made against the artifact's
    size has to be rebuilt at bf16 when this returns empty."""
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
                # Which expert this is, on both sides: the two experts are indistinguishable once loaded, so the stamp is what turns a mis-addressed artifact into a refusal instead of a wrong picture.
                config_subfolder = component,
                component = component,
                cache_dir = cache_dir,
                local_files_only = local_files_only,
                logger = logger,
            )
            if module is None:
                # ALL or none: the seeded half would be met by a dense-quantised other half nobody measured. Drop what was loaded so the pickle-sized host memory goes back before the dense build starts.
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
            # The pickle is ~7 GB per A14B expert and the next iteration allocates the next one; collect now so the two never coexist.
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
