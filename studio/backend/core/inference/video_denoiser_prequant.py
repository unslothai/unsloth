# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed a CONVENTIONAL video pipeline with hosted pre-quantized denoiser(s).

The video loader has always quantised its DiT the runtime way: ``from_pretrained`` materialises
the dense bf16 denoiser and ``quantize_transformer`` rewrites it in place. That downloads the full
bf16 shards on every fresh machine and peaks at the dense size even though the steady state is
half of it. The modular (MiniMax-H3) workflow already avoids both by seeding a hosted checkpoint
into the pipeline before the components are built; this module is the same move for the
conventional families, expressed as ``pipe_kwargs`` overrides in the shape
``diffusion_te_prequant.te_prequant_pipe_kwargs`` uses for the text encoders.

Two things make the denoiser harder than the encoder. A dual-expert MoE (Wan2.2-A14B) has TWO
denoisers, ``transformer`` and ``transformer_2``, which share a config and a key set, so the
second artifact is addressed through the ``task`` slot of ``prequant_filenames`` -- a task-specific
name gets no filename fallback, and here that guard is the whole point: falling back would load
expert 1 as expert 2 and pass every check on the way. And seeding is ALL-OR-NOTHING, the same rule
the runtime block applies ("engaged on only N/M experts"): a half-seeded MoE is a model nobody
measured, so one component that will not load means no component is seeded and the load runs
dense exactly as it does today.

Pure orchestration and torch-free at import: every heavy import happens inside the functions, so
planning code can ask what would be seeded without paying for diffusers.
"""

from __future__ import annotations

from typing import Any, Optional

# The MoE second expert's checkpoint is named through this task slot. It is a filename key, not a
# workflow: the pipeline attribute and the task string are deliberately the same word so the
# family table reads as "this row is transformer_2's artifact".
MOE_SECOND_DENOISER = "transformer_2"


def denoiser_components(fam: Any) -> tuple[str, ...]:
    """The pipeline attribute name(s) this family's denoiser(s) live under.

    ``("transformer",)`` for a single-DiT family, plus ``"transformer_2"`` for a dual-expert MoE.
    Registry-only, unlike ``video._transformer_names`` which reads a BUILT pipe: this runs before
    anything is assembled, which is the whole point of seeding."""
    if getattr(fam, "is_moe", False):
        return ("transformer", MOE_SECOND_DENOISER)
    return ("transformer",)


def denoiser_prequant_sources(
    fam: Any, scheme: Optional[str], base_repo: Optional[str]
) -> Optional[dict[str, Any]]:
    """``{component: PrequantSource}`` for EVERY denoiser this family has, or None.

    None rather than a partial dict when any component is unresolved: the caller may only drop the
    dense shards, budget the artifact's size or seed at all when the whole model is covered, and a
    dict missing one expert reads as "some coverage" at every one of those sites.

    The non-default components resolve through the ``task`` slot, which is what gives the second
    expert its own filename and, deliberately, no fallback to the first one's.

    Pure (registry only, no IO, no torch) and never raises: it runs on the download-planning path.
    """
    wanted = (scheme or "").strip().lower()
    if wanted in ("", "auto", "off", "none"):
        return None
    try:
        from .diffusion_prequant import resolve_prequant_source
        from .video_families import video_family_prequant_task_specific

        sources: dict[str, Any] = {}
        for component in denoiser_components(fam):
            # The first denoiser is the task-agnostic row every family already writes; only the extra experts are named
            # per component, so a single-DiT family resolves exactly what it resolved before.
            task = None if component == "transformer" else component
            if task is not None and not video_family_prequant_task_specific(fam, wanted, task):
                # Without a row of its own, ``task`` resolves the task-AGNOSTIC name, i.e. the first expert's file.
                # The two experts share a base, a class, a config and a key set, so it would load cleanly and denoise
                # the second half of the schedule with the wrong weights. Uncovered is the only honest answer.
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
    """Component overrides for pipeline assembly: ``{<component>: <pre-quantized denoiser>}`` for
    every denoiser of ``fam``, or ``{}`` when the model cannot be seeded whole.

    ``{}`` is not a failure, it is today's behaviour: assembly builds the dense bf16 DiT and the
    caller's runtime ``quantize_transformer`` rewrites it in place. The caller must treat it that
    way -- a memory plan made against the artifact's size has to be rebuilt at bf16 when this
    returns empty, exactly as the pre-cast text-encoder injection is handled next door.

    Each component is loaded and then released from this function's own bookkeeping before the
    next is opened: an A14B expert deserializes from a ~7 GB pickle, and holding two at once on
    the host is the difference between a load and an OOM on a 16 GB machine."""
    try:
        import gc

        sources = denoiser_prequant_sources(fam, scheme, base_repo)
        if not sources:
            return {}
        # Gated like the runtime quant it replaces: a host that cannot run a torchao denoiser has nothing to do with a
        # pre-quantized one either, and finding that out after a multi-GB fetch is the expensive order.
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
                # Reject a checkpoint baked under a different Linear filter, exactly as the image and modular paths do,
                # so a seeded denoiser and a runtime-quantised one stay the same model.
                min_features = DEFAULT_MIN_LINEAR_FEATURES,
                # Which expert this is, on both sides: the config subfolder it is built from and the metadata stamp the
                # loader checks it against. The two experts are indistinguishable once loaded, so the stamp is what
                # turns a mis-addressed artifact into a refusal instead of a wrong picture.
                config_subfolder = component,
                component = component,
                cache_dir = cache_dir,
                # A load nobody asked for may not fetch a multi-GB checkpoint either; a cache miss returns None and the
                # dense path below takes over.
                local_files_only = local_files_only,
                logger = logger,
            )
            if module is None:
                # ALL or none, the same rule the runtime block applies across experts: the seeded half would be met by
                # a dense-quantised other half nobody measured. Drop what was loaded so the pickle-sized host memory
                # goes back before the dense build starts.
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
            # The checkpoint's pickle is dead the moment load_prequantized_transformer returns, but it is ~7 GB per
            # A14B expert and the next iteration allocates the next one; collect now so the two never coexist.
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
