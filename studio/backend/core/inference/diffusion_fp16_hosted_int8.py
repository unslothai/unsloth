# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An int8 image load on an fp16-only NVIDIA card (Turing) reads the family's hosted int8 ConvRot DiT as native W8A8.

torchao refuses an fp16 card, so the planner keeps the released bf16 shards and the loader quantises them at load
into the torchao-free W8A8 ConvRot build (``fp16_card_int8_scheme``): Qwen-Image-2.1 on a T4 reads its dense DiT and
spends about 43 s in that build on every load, and nothing caches it. The hosted int8 ConvRot file holds the same
build (same layers, g256 rotation, per-row int8 codes and float32 scales, made from the bf16 weights), so the native
Linears are read from it instead (``load_native_from_hosted`` refuses any file that is not that build), computing and
holding every dense tensor in the card's dtype as the at-load build does.

The hosted codes are not the at-load codes: about 7.4% of the Qwen-Image-2.1 codes differ by one step and the per-row
scales by -0.4% to +0.6% (same weights: every unquantised tensor is bit-equal to the released shards; the float16
rounding of the at-load build is not the cause), and the hosted reconstruction error is 1.1% to 1.7% higher per layer.
Images are close, not bit-identical, so only families scored against a dense reference take the route.

Used only when the hosted file is in the local Hugging Face cache, or the load is online (it is then fetched once).
Kill switch ``UNSLOTH_FP16_CARD_HOSTED_INT8=0``. Torch-free at import."""

from __future__ import annotations

import os
from typing import Any, Optional

FP16_CARD_HOSTED_INT8_ENV = "UNSLOTH_FP16_CARD_HOSTED_INT8"
# Families whose hosted read was scored on a T4 against a dense float16 reference (16 prompts). Others keep the at-load
# build until measured: the hosted codes are not the at-load codes (see the module docstring).
FP16_CARD_HOSTED_INT8_FAMILIES = frozenset({"qwen-image-2.1"})
_TAG = "diffusion.fp16_hosted_int8"


def fp16_card_hosted_int8_enabled() -> bool:
    raw = (os.environ.get(FP16_CARD_HOSTED_INT8_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def fp16_card_hosted_int8_reason(
    fam: Any, native_scheme: Optional[str], target: Any
) -> Optional[str]:
    """Why this load reads the hosted int8 ConvRot DiT instead of quantising the dense one, else None: the load's
    native build is the fp16-card int8 W8A8 ConvRot one and the family hosts that file. Never raises."""
    try:
        if fam is None or not fp16_card_hosted_int8_enabled():
            return None
        from .diffusion_native_quant import int8_rotation_group
        from .diffusion_rocm_hosted_int8 import _hosted_convrot_file
        from .diffusion_transformer_quant import (
            TQ_INT8,
            fp16_card_int8_act,
            fp16_card_int8_host,
            int8_convrot_enabled,
        )

        if (
            native_scheme != TQ_INT8
            or not fp16_card_int8_host(target)
            or not fp16_card_int8_act(target)
        ):
            return None
        name = getattr(fam, "name", None)
        if name not in FP16_CARD_HOSTED_INT8_FAMILIES:
            return None
        if _hosted_convrot_file(fam) is None or not int8_convrot_enabled(name):
            return None
        if int(int8_rotation_group() or 0) <= 0:
            return None
        return "the fp16-only card's int8 build is W8A8 ConvRot, which the hosted int8 ConvRot file holds"
    except Exception:  # noqa: BLE001 - unanswerable keeps the at-load build
        return None


def fp16_card_hosted_int8_transformer(
    fam: Any,
    target: Any,
    transformer_cls: Any,
    *,
    base: Optional[str],
    config_source: Optional[str],
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Any:
    """The native W8A8 ``transformer`` read from the hosted int8 ConvRot file in the card's dtype, or None (the dense
    shards load and are quantised at load, as before). Never raises."""
    try:
        if not base or transformer_cls is None:
            return None
        from .diffusion_denoiser_prequant import denoiser_prequant_source
        from .diffusion_rocm_hosted_int8 import load_rocm_hosted_transformer
        from .diffusion_transformer_quant import TQ_INT8

        source = denoiser_prequant_source(
            fam,
            TQ_INT8,
            base_repo = base,
            local_files_only = local_files_only,
            hf_token = hf_token,
            cache_dir = cache_dir,
        )
        if source is None or getattr(source, "kind", None) != "repo":
            return None
        module = load_rocm_hosted_transformer(
            fam,
            transformer_cls,
            base,
            source,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
            compute_dtype = getattr(target, "dtype", None),
            config_source = config_source,
            tag = _TAG,
        )
        if module is not None and logger is not None:
            logger.info(
                "%s: %s int8 denoiser read from the hosted ConvRot checkpoint as native W8A8 Linears in %s "
                "(no dense shards read, no load-time quantise; %s=0 keeps the at-load build)",
                _TAG,
                getattr(fam, "name", None),
                str(getattr(target, "dtype", None)).replace("torch.", ""),
                FP16_CARD_HOSTED_INT8_ENV,
            )
        return module
    except Exception as exc:  # noqa: BLE001 - a hosted problem keeps the at-load build
        if logger is not None:
            logger.warning("%s: hosted read skipped (%s)", _TAG, exc)
        return None


def fp16_card_hosted_int8_cached(
    fam: Any,
    native_scheme: Optional[str],
    target: Any,
    *,
    base: Optional[str],
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """The cached hosted int8 ConvRot file this load will read, or None: the load prefetch then skips the dense DiT
    shards it no longer reads. Cache only, never the Hub. Never raises."""
    try:
        if not base or fp16_card_hosted_int8_reason(fam, native_scheme, target) is None:
            return None
        from .diffusion_denoiser_prequant import denoiser_prequant_source
        from .diffusion_prequant import cached_checkpoint_path
        from .diffusion_rocm_hosted_int8 import rocm_convrot_source
        from .diffusion_transformer_quant import TQ_INT8

        source = denoiser_prequant_source(
            fam, TQ_INT8, base_repo = base, local_files_only = True, cache_dir = cache_dir
        )
        convrot = rocm_convrot_source(source) if source is not None else None
        if convrot is None:
            return None
        path = cached_checkpoint_path(convrot, cache_dir = cache_dir)
        return str(path) if path else None
    except Exception:  # noqa: BLE001 - a prefetch hint only
        return None
