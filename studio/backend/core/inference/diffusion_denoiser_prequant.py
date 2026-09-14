# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed an official image PIPELINE pick with its hosted pre-quantized denoiser.

The image twin of ``video_denoiser_prequant``. A ``kind == "pipeline"`` pick assembles every
component from one repo, so the only way to run a quantised denoiser without first downloading and
materialising the released bf16 shards is to hand ``from_pretrained`` a ``transformer=`` that is
already quantised. That is what this builds.

Torch-free at import, so the download planner can ask what WOULD be seeded without paying for
diffusers: the decision has to be settled before a byte moves, or the plan stages the dense shards
the load never opens and fetches the replacement inline, outside the load's progress, cancel and
disk preflight.
"""

from __future__ import annotations

from typing import Any, Optional

from .diffusion_families import IDEOGRAM4_FAMILY_NAME
from .diffusion_krea2 import KREA2_FAMILY_NAME

# The planner DECIDED against the seed, which is a different thing from never having asked: a load whose pull kept the
# dense shards on this decline must not re-take the decision against post-eviction free memory and then fetch the
# artifact inline. Same sentinel contract as the video loader's ``DENOISER_SEED_DECLINED``.
PIPELINE_SEED_DECLINED = "__declined__"

# The one component a seeded image pipeline covers. Single-valued rather than a tuple, unlike video's dual-expert MoE:
# every seedable image family here has exactly one denoiser, and ``pipeline_seed_supported`` refuses the families that
# do not.
DENOISER_COMPONENT = "transformer"

# Families whose pipeline is assembled PER COMPONENT rather than through ``pipeline_cls.from_pretrained(**pipe_kwargs)``
# (krea ships transformers-5.x configs the 4.x line cannot parse, ideogram the same Qwen stack) plus, for ideogram, a
# second denoiser this seed does not cover. Their assemblers never see ``pipe_kwargs``, so a seed offered to them would
# be silently dropped AFTER the plan had already left their dense shards out of the pull.
_UNSEEDABLE_PIPELINE_FAMILIES = (KREA2_FAMILY_NAME, IDEOGRAM4_FAMILY_NAME)


def pipeline_seed_supported(fam: Any) -> bool:
    """Whether a seeded denoiser reaches this family's pipeline assembly at all."""
    name = str(getattr(fam, "name", "") or "")
    return bool(name) and name not in _UNSEEDABLE_PIPELINE_FAMILIES


def denoiser_prequant_source(
    fam: Any,
    scheme: Optional[str],
    *,
    base_repo: Optional[str],
    path_override: Optional[str] = None,
) -> Optional[Any]:
    """The ``PrequantSource`` a pipeline pick would seed its denoiser from, or None.

    ``usable_prequant_source`` rather than ``resolve_prequant_source``: a local override the loader
    would refuse (outside the allowlist, absent, or baked for another scheme) must read as no source
    here, or the plan drops the dense shards for a checkpoint that is then rejected after eviction.
    Pure and never raises: it runs on the download-planning path.
    """
    wanted = (scheme or "").strip().lower()
    if wanted in ("", "auto", "off", "none") or wanted == PIPELINE_SEED_DECLINED:
        return None
    try:
        from .diffusion_prequant import usable_prequant_source
        return usable_prequant_source(fam, wanted, path_override = path_override, base_repo = base_repo)
    except Exception:  # noqa: BLE001 -- an unanswerable registry keeps the released bf16 denoiser
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
    path_override: Optional[str] = None,
    fast_accum: Optional[bool] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    logger: Any = None,
) -> dict[str, Any]:
    """``{"transformer": module}`` for pipeline assembly, or ``{}`` when it cannot be seeded.

    ``{}`` is not a failure: assembly then builds the released bf16 denoiser and the caller's
    in-memory ``quantize_transformer`` rewrites it in place. It IS a re-plan, though -- a memory plan
    made against the artifact's size no longer describes the build -- and it is a download, since the
    plan that chose this left the dense shards out of the pull.
    """
    try:
        if not pipeline_seed_supported(fam):
            return {}
        source = denoiser_prequant_source(
            fam, scheme, base_repo = base_repo, path_override = path_override
        )
        if source is None:
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
                str(scheme or ""),
                ValueError(
                    f"transformer class {getattr(fam, 'transformer_class', None)!r} "
                    "not found in diffusers"
                ),
            )
            return {}
        module = load_prequantized_transformer(
            transformer_cls,
            base_repo,
            source,
            device = device,
            dtype = dtype,
            hf_token = hf_token,
            scheme = scheme,
            # Reject a checkpoint built with a different Linear filter, so a seeded denoiser and a runtime-quantised
            # one cover the same layers.
            min_features = DEFAULT_MIN_LINEAR_FEATURES,
            # Only enforced when the caller pinned fp8 fast-accum; a checkpoint that baked the other choice falls back
            # to the dense build rather than running a kernel nobody asked for.
            fast_accum = fast_accum,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
        )
        if module is None:
            if logger is not None:
                logger.info(
                    "diffusion.denoiser_prequant: no usable %s checkpoint for %s, so the released "
                    "denoiser is loaded and quantised at runtime instead",
                    scheme,
                    getattr(fam, "name", None),
                )
            return {}
        if logger is not None:
            logger.info(
                "diffusion.denoiser_prequant: seeded the %s pre-quantized denoiser for %s (%s)",
                scheme,
                getattr(fam, "name", None),
                source.location,
            )
        return {DENOISER_COMPONENT: module}
    except Exception as exc:  # noqa: BLE001 -- seeding is an optimisation, never a blocker
        _warn(logger, "pipe_kwargs", exc)
        return {}


def prequant_artifact_label(source: Any) -> Optional[str]:
    """``prequant:<repo>/<file>`` for a hosted artifact, ``prequant:<path>`` for a local override.

    What the resolved record reports so a user can tell WHICH checkpoint produced the pixels: the
    scheme alone does not distinguish the hosted artifact from a runtime quantise of the released
    weights, and the two do not render the same image.
    """
    if source is None:
        return None
    kind = getattr(source, "kind", None)
    location = getattr(source, "location", None)
    if not location:
        return None
    if kind != "repo":
        return f"prequant:{location}"
    filename = getattr(source, "filename", None)
    return f"prequant:{location}/{filename}" if filename else f"prequant:{location}"


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.denoiser_prequant: %s failed: %s", what, exc)
