# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit int8 image pick on a ROCm RDNA card reads the family's hosted int8 ConvRot DiT as native W8A8 Linears.

CUDA seeds that file as torchao tensors. ROCm has no torchao path (``native_quant_host``), so the planner kept the dense
bf16 shards and the loader quantised them at load into the native W8A8 ConvRot build. The hosted file holds exactly
that build (same layers, same g256 rotation, per-row int8 codes and float32 scales), so the same Linears are read from
it instead (``diffusion_native_hosted.load_native_from_hosted``, which refuses any file that is not that build).

Only a load whose runtime build would be W8A8 qualifies, since the hosted file is W8A8 only: an explicit int8 request on
the compiled RDNA tier (``rdna_explicit_int8_reason``) for a family whose explicit or AUTO table measured W8A8
(``rdna_explicit_w8a8`` / ``rdna_auto_w8a8``), or with ``UNSLOTH_NATIVE_INT8_ACT=1``. AUTO is never changed and
weight-only families keep the dense shards. Kill switch ``UNSLOTH_ROCM_IMAGE_HOSTED_INT8=0``;
``UNSLOTH_NATIVE_INT8_ACT=0`` also keeps the old path. Torch-free at import."""

from __future__ import annotations

import os
from typing import Any, Optional

ROCM_IMAGE_HOSTED_INT8_ENV = "UNSLOTH_ROCM_IMAGE_HOSTED_INT8"


def rocm_image_hosted_int8_enabled() -> bool:
    raw = (os.environ.get(ROCM_IMAGE_HOSTED_INT8_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _hosted_convrot_file(fam: Any) -> Optional[str]:
    profile = getattr(fam, "profile", None)
    spec = getattr(profile, "int8_convrot", None) if profile is not None else None
    if (
        spec is None
        or not getattr(spec, "filename", None)
        or int(getattr(spec, "group", 0) or 0) <= 0
    ):
        return None
    return str(spec.filename)


def rocm_image_hosted_int8_reason(fam: Any, requested: Optional[str], target: Any) -> Optional[str]:
    """Why an EXPLICIT int8 image request on this ROCm RDNA card reads the hosted int8 ConvRot DiT as native W8A8,
    else None (the dense shards are quantised at load, as before). Host-level only: the caller still requires a
    resident ``pipeline`` load on a compiling speed tier. Never raises."""
    try:
        if fam is None or not rocm_image_hosted_int8_enabled():
            return None
        from .diffusion_transformer_quant import (
            TQ_INT8,
            int8_convrot_enabled,
            native_int8_act,
            normalize_transformer_quant,
            rdna_auto_w8a8_reason,
            rdna_explicit_int8_reason,
            rdna_explicit_w8a8_reason,
        )

        if normalize_transformer_quant(requested) != TQ_INT8:
            return None
        name = getattr(fam, "name", None)
        if _hosted_convrot_file(fam) is None or not int8_convrot_enabled(name):
            return None
        # ROCm RDNA explicit int8 on the compiled tier only (checks ROCm, bf16, UNSLOTH_NATIVE_INT8_ACT=0, the fused
        # W8A8 kernel and the compile tier); every other host keeps its existing route.
        if rdna_explicit_int8_reason(target, name) is None:
            return None
        from .diffusion_native_quant import int8_rotation_group

        if int(int8_rotation_group() or 0) <= 0:
            return None
        # The same W8A8 rule the load applies to the runtime build (weight-only families keep the dense shards).
        return (
            rdna_explicit_w8a8_reason(target, name)
            or rdna_auto_w8a8_reason(target, name)
            or ("UNSLOTH_NATIVE_INT8_ACT=1 runs int8 W8A8" if native_int8_act(target) else None)
        )
    except Exception:  # noqa: BLE001 - unanswerable keeps the dense shards
        return None


def load_rocm_hosted_transformer(
    fam: Any,
    transformer_cls: Any,
    base_repo: str,
    source: Any,
    *,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Any:
    """The native W8A8 denoiser read from ``source``'s hosted int8 ConvRot file, or None (the caller restores the
    dense shards). Never raises."""
    try:
        from .diffusion_families import pinned_revision_kwargs
        from .diffusion_native_cache import native_spec
        from .diffusion_native_hosted import load_native_from_hosted
        from .diffusion_prequant import _resolve_checkpoint_path
        from .diffusion_transformer_quant import TQ_INT8

        path = _resolve_checkpoint_path(
            source,
            hf_token,
            cache_dir,
            local_files_only = local_files_only,
            scheme = TQ_INT8,
            logger = logger,
        )
        if path is None:
            return None
        name = getattr(fam, "name", None)
        module = load_native_from_hosted(
            path,
            transformer_cls,
            base_repo,
            spec = native_spec(TQ_INT8, name, act_int8 = True),
            family = name,
            component = "transformer",
            config_kwargs = {
                "token": hf_token,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                **pinned_revision_kwargs(base_repo),
            },
            base = base_repo,
            logger = logger,
            tag = "diffusion.rocm_hosted_int8",
        )
        if module is None:
            return None
        try:
            module._unsloth_prequant_path = str(path)
        except Exception:  # noqa: BLE001 - the artifact label only
            pass
        return module
    except Exception as exc:  # noqa: BLE001 - a hosted problem keeps the dense build
        if logger is not None:
            logger.warning("diffusion.rocm_hosted_int8: hosted read skipped (%s)", exc)
        return None
