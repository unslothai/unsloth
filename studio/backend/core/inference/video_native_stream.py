# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native int8 video denoisers built per shard into the local prequant cache when the dense load would not fit host
RAM (Wan2.2-T2V-A14B: two fp32 14B experts, 57 GB at bf16, on a 53 GB host); the dense experts are never held in RAM.
``UNSLOTH_VIDEO_NATIVE_STREAM_BUILD=0`` keeps the dense load, ``=1`` streams even where it fits.
"""

from __future__ import annotations

import gc
import json
import os
import time
import types
from typing import Any, Callable, Optional

VIDEO_NATIVE_STREAM_ENV = "UNSLOTH_VIDEO_NATIVE_STREAM_BUILD"
_MIB = 1024 * 1024
_WEIGHT_NAMES = ("model", "diffusion_pytorch_model")


def _env() -> str:
    return (os.environ.get(VIDEO_NATIVE_STREAM_ENV) or "").strip().lower()


def stream_disabled() -> bool:
    return _env() in ("0", "off", "false", "no")


def stream_forced() -> bool:
    return _env() in ("1", "on", "true", "yes", "force")


def decide(
    dense_mib: Optional[int],
    *,
    available_mib: Optional[int],
    reserve_mib: int,
    forced: bool = False,
    disabled: bool = False,
) -> tuple[bool, str]:
    """(engage, reason). Engages only when the dense pipeline load provably exceeds available host RAM (or forced);
    an unreadable size or host keeps the dense load, as before."""
    if disabled:
        return False, f"{VIDEO_NATIVE_STREAM_ENV}=0"
    if forced:
        return True, f"{VIDEO_NATIVE_STREAM_ENV}=1"
    if dense_mib is None:
        return False, "dense denoiser size unreadable"
    if available_mib is None:
        return False, "host RAM unreadable"
    need = int(dense_mib) + int(reserve_mib)
    if need <= int(available_mib):
        return False, (
            f"dense load (~{int(dense_mib)} MiB + {int(reserve_mib)} MiB reserve) fits "
            f"{int(available_mib)} MiB available host RAM"
        )
    return True, (
        f"dense load needs ~{int(dense_mib)} MiB + {int(reserve_mib)} MiB reserve of host RAM, "
        f"{int(available_mib)} MiB available"
    )


def _component_mib(
    source: str,
    component: str,
    cache_dir: Optional[str],
    *,
    weights_names: tuple,
    keep_in_fp32: Any = None,
) -> Optional[int]:
    """MiB ``component`` occupies once loaded at half precision (fp32 tensors halve, kept-fp32 names stay), 0 with
    no weight file, None when a header is unreadable."""
    from .diffusion_native_cache import _dense_shards
    from .video_small_host import loaded_mib

    for name in weights_names:
        shards = _dense_shards(source, component, cache_dir, weights_name = name)
        if shards:
            return loaded_mib(
                types.SimpleNamespace(shards = shards, parts = {"keep_in_fp32": keep_in_fp32})
            )
    return 0


def dense_load_mib(
    source: str,
    denoisers: tuple,
    *,
    cache_dir: Optional[str],
    transformer_cls: Any,
    skip: tuple = (),
) -> tuple[Optional[int], dict]:
    """(total MiB, ``{component: MiB}``) the dense pipeline load holds in host RAM: every denoiser plus the weighted
    companions ``model_index.json`` lists that are not already injected (``skip``). Denoiser size None = unknown."""
    from . import diffusion_prequant_cache as _cache

    keep = list(getattr(transformer_cls, "_keep_in_fp32_modules", None) or ())
    sizes: dict = {}
    for component in denoisers:
        mib = _component_mib(
            source,
            component,
            cache_dir,
            weights_names = ("diffusion_pytorch_model",),
            keep_in_fp32 = keep,
        )
        if not mib:
            return None, sizes
        sizes[component] = int(mib)
    try:
        index = _cache._local_file(str(source), "model_index.json", cache_dir)
        names = []
        if index is not None:
            with open(index, "r", encoding = "utf-8") as fh:
                raw = json.load(fh) or {}
            names = [
                k
                for k, v in raw.items()
                if not k.startswith("_")
                and isinstance(v, (list, tuple))
                and len(v) == 2
                and all(v)
                and k not in denoisers
                and k not in skip
            ]
        for component in names:
            mib = _component_mib(source, component, cache_dir, weights_names = _WEIGHT_NAMES)
            if mib:
                sizes[component] = int(mib)
    except Exception:  # noqa: BLE001 -- companions only sharpen the estimate; the denoisers decide alone
        pass
    return sum(sizes.values()), sizes


def build_into_cache(
    fam: Any,
    spec: Optional[dict],
    transformer_cls: Any,
    *,
    target: Any,
    dtype: Any,
    base: Optional[str],
    source: Optional[str],
    config_kwargs: Optional[dict] = None,
    skip: tuple = (),
    host_quant_device: Any = None,
    logger: Any = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> dict:
    """``{component: native module}`` for EVERY denoiser, built per shard and read back from the prequant cache, or
    ``{}`` for the dense load. A cancel returns ``{}`` (the caller re-checks it); an out-of-memory raises."""
    if spec is None or transformer_cls is None or not source:
        return {}
    if getattr(target, "device", None) != "cuda":
        return {}
    from . import diffusion_native_cache as NC
    from .diffusion_native_quant import quantise_host_linears_on
    from .diffusion_small_host import host_ram_mib, host_reserve_mib
    from .video_denoiser_prequant import denoiser_components

    if not NC.stream_build_enabled():
        return {}
    components = tuple(denoiser_components(fam))
    source = str(source)
    cache_dir = (config_kwargs or {}).get("cache_dir")
    try:
        dense_mib, sizes = dense_load_mib(
            source, components, cache_dir = cache_dir, transformer_cls = transformer_cls, skip = skip
        )
        total, available = host_ram_mib()
        engaged, why = decide(
            dense_mib,
            available_mib = available,
            reserve_mib = host_reserve_mib(total),
            forced = stream_forced(),
            disabled = stream_disabled(),
        )
    except Exception as exc:  # noqa: BLE001 -- unanswerable: the dense load, as before
        engaged, why, sizes = False, f"sizing failed ({type(exc).__name__}: {exc})", {}
    if logger is not None:
        logger.info(
            "video.native_stream: engaged=%s (%s); %s",
            engaged,
            why,
            ", ".join(f"{k}={v} MiB" for k, v in sizes.items()) or "no sizes",
        )
    if not engaged:
        return {}
    t0 = time.perf_counter()
    family = getattr(fam, "name", None)
    if logger is not None:
        logger.info(
            "video.native_stream: building the native %s %s of %s per shard into the local prequant cache "
            "(the dense denoisers are not loaded; %s=0 loads them dense)",
            spec.get("scheme"),
            "/".join(components),
            family,
            VIDEO_NATIVE_STREAM_ENV,
        )

    def build(component: str) -> Any:
        """One expert per shard, or None (logged; a cancel raises)."""
        with quantise_host_linears_on(host_quant_device):
            module = NC.build_from_shards(
                transformer_cls,
                source,
                spec,
                family = family,
                dtype = dtype,
                component = component,
                config_kwargs = config_kwargs,
                logger = logger,
                cancel_check = cancel_check,
                cast_floats = True,
                reraise_oom = True,
            )
        if module is None:
            if cancel_check is not None:
                cancel_check()
            why = "could not be built per shard"
        elif NC.build_spec(module, family) != spec:
            module, why = None, "built a recipe other than the cache key's"
        if module is None and logger is not None:
            logger.warning(
                "video.native_stream: %s %s; loading the dense denoisers", component, why
            )
        return module

    kept: dict = {}
    paths: dict = {}
    for component in components:
        if cancel_check is not None:
            cancel_check()
        hit = NC.lookup(
            fam, spec, base_repo = base, source_base = source, component = component, logger = logger
        )
        if hit is not None:
            paths[component] = hit
            continue
        module = build(component)
        if module is None:
            return {}
        path = NC.save(
            module,
            fam,
            spec,
            base_repo = base,
            source_base = source,
            component = component,
            logger = logger,
            cancel_check = cancel_check,
        )
        if path is None:
            # Unwritable cache: keep this expert in memory, still never dense.
            kept[component] = module
        else:
            paths[component] = path
        module = None
        gc.collect()
    out: dict = dict(kept)
    for component in list(paths):
        if cancel_check is not None:
            cancel_check()
        module = NC.load(
            paths[component],
            transformer_cls,
            source,
            component = component,
            spec = spec,
            config_kwargs = config_kwargs,
            logger = logger,
            cancel_check = cancel_check,
        )
        if module is None:
            if cancel_check is not None:
                cancel_check()
            # Evicted since written (small cache budget): rebuild, kept in memory.
            if logger is not None:
                logger.warning(
                    "video.native_stream: the cached %s did not read back; rebuilding it per shard",
                    component,
                )
            module = build(component)
            if module is None:
                return {}
            paths.pop(component)
            kept[component] = module
        out[component] = module
    if logger is not None:
        logger.info(
            "video.native_stream: %d native %s denoiser(s) ready in %.1f s (%d from the local prequant cache, "
            "%d kept in memory)",
            len(out),
            spec.get("scheme"),
            time.perf_counter() - t0,
            len(paths),
            len(kept),
        )
    return {c: out[c] for c in components}
