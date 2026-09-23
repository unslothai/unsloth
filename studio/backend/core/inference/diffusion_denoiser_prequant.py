# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seed an official image PIPELINE pick with its hosted pre-quantized denoiser.

A ``kind == "pipeline"`` pick assembles every component from one repo, so the only way to run a
quantised denoiser without materialising the bf16 shards is handing ``from_pretrained`` an
already-quantised ``transformer=``. Torch-free at import."""

from __future__ import annotations

from typing import Any, Optional

from .diffusion_families import IDEOGRAM4_FAMILY_NAME
from .diffusion_krea2 import KREA2_FAMILY_NAME

# The planner DECIDED against the seed, unlike never having asked: the pull kept the dense shards, so
# the loader must not re-take the decision and fetch the artifact inline.
PIPELINE_SEED_DECLINED = "__declined__"

DENOISER_COMPONENT = "transformer"

# Families assembled PER COMPONENT, plus ideogram's second denoiser: their assemblers never see
# ``pipe_kwargs``, so a seed would be dropped after the plan had dropped their dense shards.
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
    ``usable_prequant_source``, not ``resolve_prequant_source``: an override the loader would refuse
    must read as no source here. Never raises."""
    wanted = (scheme or "").strip().lower()
    if wanted in ("", "auto", "off", "none") or wanted == PIPELINE_SEED_DECLINED:
        return None
    try:
        from .diffusion_prequant import usable_prequant_source

        return usable_prequant_source(fam, wanted, path_override=path_override, base_repo=base_repo)
    except Exception:  # noqa: BLE001 -- an unanswerable registry keeps the released bf16 denoiser
        return None


def denoiser_prequant_cached(
    fam: Any,
    scheme: Optional[str],
    *,
    base_repo: Optional[str],
    path_override: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> bool:
    """Whether the artifact this seed would open is ALREADY on disk, answered without a Hub call.
    The offline twin of the ``model_info`` probe the online plan makes, so a load that may not
    download can still seed from the cache an earlier load built. Never raises."""
    source = denoiser_prequant_source(fam, scheme, base_repo=base_repo, path_override=path_override)
    if source is None:
        return False
    if getattr(source, "kind", None) != "repo":
        # ``usable_prequant_source`` already proved a local override present and baked for this scheme.
        return True
    try:
        from .diffusion_prequant import prequant_checkpoint_cached

        return prequant_checkpoint_cached(source, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 -- an unreadable cache is not proof the artifact is there
        return False


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
    """``{"transformer": module}`` for pipeline assembly, or ``{}`` when it cannot be seeded, which
    obliges the caller to re-plan at bf16 and restore the dropped shards."""
    try:
        if not pipeline_seed_supported(fam):
            return {}
        source = denoiser_prequant_source(
            fam, scheme, base_repo=base_repo, path_override=path_override
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
            device=device,
            dtype=dtype,
            hf_token=hf_token,
            scheme=scheme,
            min_features=DEFAULT_MIN_LINEAR_FEATURES,
            fast_accum=fast_accum,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            logger=logger,
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


def prequant_artifact_label(source: Any, module: Any = None) -> Optional[str]:
    """``prequant:<repo>/<file>`` for a hosted artifact, ``prequant:<path>`` for a local override;
    the scheme alone cannot tell either from a runtime quantise. ``module`` is the seeded denoiser,
    which carries the file that really loaded: a repo holding only ``fallback_filename`` serves
    that one, and ``source.filename`` would then name a file nobody fetched."""
    if source is None:
        return None
    kind = getattr(source, "kind", None)
    location = getattr(source, "location", None)
    if not location:
        return None
    if kind != "repo":
        return f"prequant:{location}"
    filename = _loaded_filename(module) or getattr(source, "filename", None)
    return f"prequant:{location}/{filename}" if filename else f"prequant:{location}"


def _loaded_filename(module: Any) -> Optional[str]:
    path = getattr(module, "_unsloth_prequant_path", None)
    if not path:
        return None
    import os

    return os.path.basename(str(path)) or None


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.denoiser_prequant: %s failed: %s", what, exc)
