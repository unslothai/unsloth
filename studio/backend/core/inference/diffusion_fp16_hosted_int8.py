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

The file is read from the local Hugging Face cache only. An online load plans it (``fp16_card_hosted_int8_plan_source``)
in place of the dense DiT shards, so the download plan, its byte total and the staged assets name what is read; a
load that did not stage it (offline, unplanned) keeps the dense build. Kill switch ``UNSLOTH_FP16_CARD_HOSTED_INT8=0``. Torch-free at import."""

from __future__ import annotations

import os
from typing import Any, Optional

FP16_CARD_HOSTED_INT8_ENV = "UNSLOTH_FP16_CARD_HOSTED_INT8"
# Families whose hosted read was scored on a T4 against a dense float16 reference (Qwen-Image-2.1 16 prompts, FLUX.3
# klein 8; MOSS-Image against float32, see below). Others keep the at-load build until measured: the hosted codes are
# not the at-load codes (see the module docstring).
FP16_CARD_HOSTED_INT8_FAMILIES = frozenset({"qwen-image-2.1", "flux.3-klein", "moss-image"})
# AUTO takes the hosted build on any host: MOSS-Image SFT on a T4, hosted 14.51 s/step LPIPS 0.023 vs per-row int8
# 37.35 s/step LPIPS 0.029 and dense float32 36.01 s/step.
FP16_CARD_HOSTED_INT8_AUTO_FAMILIES = frozenset({"moss-image"})
# Measured host RAM peak (VmHWM, MiB, T4 51 GB host); a host below it plus the small-host reserve keeps the per-row build.
FP16_CARD_HOSTED_INT8_AUTO_HOST_PEAK_MIB: dict[str, int] = {"moss-image": 30000}
_TAG = "diffusion.fp16_hosted_int8"


def fp16_card_hosted_int8_enabled() -> bool:
    raw = (os.environ.get(FP16_CARD_HOSTED_INT8_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _official_hub_base(fam: Any, base: Optional[str]) -> bool:
    """Whether ``base`` is the family's released Hub repo (or a mirror of it). The hosted file stands in for that
    DiT only: a local pipeline (an edited or fine-tuned copy, whatever its folder name), a Hub fine-tune and a sibling
    distill under the same family (Qwen-Image-2.1-Turbo, not scored on a T4) keep their own weights, quantised at load.
    """
    try:
        official = str(getattr(fam, "base_repo", "") or "")
        if not base or not official:
            return False
        from .diffusion_families import _is_local_path, canonical_base
        from .diffusion_prequant import _HUB_REPO_ID

        text = str(base).strip()
        if not _HUB_REPO_ID.fullmatch(text) or _is_local_path(text):
            return False
        return canonical_base(text).lower() == canonical_base(official).lower()
    except Exception:  # noqa: BLE001 - unanswerable keeps the at-load build
        return False


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
    logger: Any = None,
) -> Any:
    """The native W8A8 ``transformer`` read from the cached hosted int8 ConvRot file in the card's dtype, or None (the
    dense shards load and are quantised at load, as before). Cache only: the download plan stages the file, so the
    load never pulls bytes it did not account for. The module carries ``_unsloth_prequant_artifact`` (the file that
    ran, for the resolved status). Raises only an allocation failure."""
    try:
        if not _official_hub_base(fam, base) or transformer_cls is None:
            return None
        from .diffusion_denoiser_prequant import denoiser_prequant_source, prequant_artifact_label
        from .diffusion_rocm_hosted_int8 import load_rocm_hosted_transformer, rocm_convrot_source
        from .diffusion_transformer_quant import TQ_INT8

        source = denoiser_prequant_source(
            fam,
            TQ_INT8,
            base_repo = base,
            local_files_only = True,
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
            local_files_only = True,
            logger = logger,
            compute_dtype = getattr(target, "dtype", None),
            config_source = config_source,
            tag = _TAG,
        )
        if module is None:
            return None
        try:
            module._unsloth_prequant_artifact = prequant_artifact_label(
                rocm_convrot_source(source) or source, module
            )
        except Exception:  # noqa: BLE001 - the status label only
            pass
        if logger is not None:
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
        from .diffusion_native_hosted import raise_if_allocation_failure

        raise_if_allocation_failure(exc)
        if logger is not None:
            logger.warning("%s: hosted read skipped (%s)", _TAG, exc)
        return None


def _dense_dit_fits_host(
    fam: Any, base: Optional[str], cache_dir: Optional[str]
) -> tuple[Optional[bool], str]:
    """(fits, why) for the dense float16 DiT in host RAM via ``dense_load_fits``; fits is None when unpriced."""
    import torch

    from .diffusion_small_host import (
        StoredComponent,
        dense_load_fits,
        host_commit_mib,
        host_ram_mib,
        resolve_snapshot_dir,
        snapshot_weights_complete,
        stored_components,
    )

    comp = None
    source = "family table"
    snapshot = resolve_snapshot_dir(base, cache_dir) if base else None
    if snapshot is not None:
        stored = stored_components(snapshot)
        if "transformer" in stored and snapshot_weights_complete(
            snapshot, skip = tuple(n for n in stored if n != "transformer")
        ):
            comp, source = stored["transformer"], "snapshot"
    if comp is None:
        from .diffusion_auto_policy import family_bf16_components_gb

        table = family_bf16_components_gb(fam, base)
        if not table or not table[0]:
            return None, "DiT size unknown"
        comp = StoredComponent(
            "transformer", int(table[0] * 1000.0**3 / (1024.0 * 1024.0)), "bfloat16"
        )
    total, available = host_ram_mib()
    fits, dense = dense_load_fits(
        {"transformer": comp},
        torch.float16,
        host_total_mib = total,
        host_available_mib = available,
        host_commit_mib = host_commit_mib(),
    )
    return fits, f"dense float16 DiT ~{dense} MiB ({source}), {available} MiB available host RAM"


def hosted_auto_host_ram_shortfall(fam: Any) -> Optional[str]:
    """Why this host is too small for AUTO's measured hosted build of ``fam``, else None. Never raises."""
    try:
        name = getattr(fam, "name", None)
        peak = FP16_CARD_HOSTED_INT8_AUTO_HOST_PEAK_MIB.get(name)
        if name not in FP16_CARD_HOSTED_INT8_AUTO_FAMILIES or peak is None:
            return None
        from .diffusion_small_host import host_ram_mib, host_reserve_mib

        total, available = host_ram_mib()
        if available is None:
            return "host RAM unreadable"
        need = int(peak) + host_reserve_mib(total)
        if available >= need:
            return None
        return (
            f"the hosted int8 ConvRot build peaks at ~{peak} MiB of host RAM (+{need - int(peak)} MiB reserve), "
            f"{available} MiB available: AUTO keeps the at-load per-row int8 build"
        )
    except Exception:  # noqa: BLE001 - unanswerable keeps the at-load route
        return "host RAM unreadable"


def small_host_hosted_int8_reason(
    fam: Any,
    target: Any,
    *,
    base: Optional[str],
    cache_dir: Optional[str] = None,
    require_cached: bool = False,
    has_lora: bool = False,
) -> Optional[str]:
    """Why AUTO takes int8 from the hosted ConvRot file on this fp16-only card, else None (AUTO is unchanged).
    Replaces the small-host per-row quantise at load (~200 s on a 12.7 GB T4) when the dense DiT does not fit, and on
    any host for ``FP16_CARD_HOSTED_INT8_AUTO_FAMILIES``. Never raises."""
    try:
        if fam is None or has_lora or not _official_hub_base(fam, base):
            return None
        from .diffusion_small_host import small_host_disabled
        from .diffusion_transformer_quant import TQ_INT8, fp16_card_native_scheme

        if small_host_disabled():
            return None
        name = getattr(fam, "name", None)
        scheme = fp16_card_native_scheme(target, TQ_INT8, family = name)
        if scheme != TQ_INT8 or fp16_card_hosted_int8_reason(fam, scheme, target) is None:
            return None
        measured = name in FP16_CARD_HOSTED_INT8_AUTO_FAMILIES
        if measured and hosted_auto_host_ram_shortfall(fam) is not None:
            return None
        if not measured:
            fits, why = _dense_dit_fits_host(fam, base, cache_dir)
            if fits is not False:
                return None
        if (
            require_cached
            and fp16_card_hosted_int8_cached(fam, scheme, target, base = base, cache_dir = cache_dir)
            is None
        ):
            return None
        if measured:
            return (
                "measured on a T4: the hosted int8 ConvRot W8A8 build is faster and closer to the float32 reference "
                f"than the at-load int8 build, and this host's RAM covers its ~"
                f"{FP16_CARD_HOSTED_INT8_AUTO_HOST_PEAK_MIB.get(name)} MiB peak "
                f"({FP16_CARD_HOSTED_INT8_ENV}=0 keeps the at-load route)"
            )
        return (
            f"{why}: the low-RAM route would quantise the dense DiT per row at load; the hosted int8 ConvRot file "
            f"holds this card's W8A8 build ({FP16_CARD_HOSTED_INT8_ENV}=0 keeps the at-load route)"
        )
    except Exception:  # noqa: BLE001 - unanswerable keeps the old route
        return None


def fp16_card_hosted_int8_planned_scheme(
    fam: Any,
    target: Any,
    transformer_quant: Optional[str],
    *,
    kind: str,
    speed_mode: Optional[str] = None,
    forces_offload: bool = False,
    has_lora: bool = False,
    base: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[str]:
    """The native scheme the load will build on this fp16-only card, settled before any download the way the load
    settles it (an explicit pick, AUTO's measured Turing int8, or AUTO on a host the dense DiT does not fit:
    ``small_host_hosted_int8_reason``), else None. Never raises."""
    try:
        if kind != "pipeline" or fam is None or has_lora:
            return None
        from .diffusion_transformer_quant import (
            TQ_INT8,
            fp16_card_native_scheme,
            native_quant_scheme,
            normalize_transformer_quant,
            turing_auto_int8_reason,
        )

        name = getattr(fam, "name", None)
        raw = transformer_quant
        if raw is None or str(raw).strip().lower() in ("", "auto"):
            # Speed off rewrites AUTO to off (``diffusion_speed.SPEED_OFF``).
            if speed_mode is not None and str(speed_mode).strip().lower() == "off":
                return None
            if forces_offload or (
                turing_auto_int8_reason(target, name) is None
                and small_host_hosted_int8_reason(fam, target, base = base, cache_dir = cache_dir)
                is None
            ):
                return None
            pinned = TQ_INT8
        else:
            pinned = normalize_transformer_quant(raw)
        if native_quant_scheme(target, pinned, family = name) is not None:
            return None
        return fp16_card_native_scheme(target, pinned, family = name)
    except Exception:  # noqa: BLE001 - unanswerable keeps the dense shards in the plan
        return None


def fp16_card_hosted_int8_plan_source(
    fam: Any,
    native_scheme: Optional[str],
    target: Any,
    *,
    base: Optional[str],
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
) -> Any:
    """The hosted int8 ConvRot source the download plan stages in place of the dense DiT shards, or None (the plan
    keeps them). The same gate and header check the load applies, so the plan never drops the dense shards for a file
    the load refuses. Never raises."""
    try:
        if (
            not _official_hub_base(fam, base)
            or fp16_card_hosted_int8_reason(fam, native_scheme, target) is None
        ):
            return None
        from .diffusion_denoiser_prequant import denoiser_prequant_source
        from .diffusion_prequant import _planning_min_features, hosted_prequant_refusal
        from .diffusion_rocm_hosted_int8 import rocm_convrot_source
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
        convrot = rocm_convrot_source(source)
        if convrot is None:
            return None
        if (
            hosted_prequant_refusal(
                convrot,
                TQ_INT8,
                base = base,
                min_features = _planning_min_features(None),
                cache_dir = cache_dir,
                local_files_only = local_files_only,
            )
            is not None
        ):
            return None
        return convrot
    except Exception:  # noqa: BLE001 - unanswerable keeps the dense shards in the plan
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
        if (
            not _official_hub_base(fam, base)
            or fp16_card_hosted_int8_reason(fam, native_scheme, target) is None
        ):
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
