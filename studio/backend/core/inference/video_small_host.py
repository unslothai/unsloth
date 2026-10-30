# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stages cast-cache misses too big for host RAM onto the card, same bits;
UNSLOTH_VIDEO_SMALL_HOST=0 turns off, =1 stages every miss that fits the card"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

SMALL_HOST_ENV = "UNSLOTH_VIDEO_SMALL_HOST"
_MIB = 1024 * 1024
# device bytes the staged load needs above its own weights (CUDA context grows, allocator rounding)
_DEVICE_MARGIN_MIB = 768
# a candidate this small converts in RAM even on a small host
_STAGE_MIN_MIB = 512


def _env() -> str:
    return (os.environ.get(SMALL_HOST_ENV) or "").strip().lower()


def stage_disabled() -> bool:
    return _env() in ("0", "off", "false", "no")


def stage_forced() -> bool:
    return _env() in ("1", "on", "true", "yes", "force")


def _keep_regex(patterns: Any) -> Optional[Any]:
    pats = [str(p) for p in (patterns or ()) if p]
    if not pats:
        return None
    # transformers' own rule (modeling_utils: ``(^|\.)module($|\.)``)
    return re.compile("|".join(rf"((^|\.){re.escape(p)}($|\.))" for p in pats))


def loaded_mib(cand: Any) -> Optional[int]:
    """MiB ``cand`` occupies once loaded at its half dtype: fp32 tensors halve except ``_keep_in_fp32_modules``."""
    from .diffusion_cast_cache import _shard_header

    keep = _keep_regex((getattr(cand, "parts", None) or {}).get("keep_in_fp32"))
    total = 0
    for shard in getattr(cand, "shards", None) or ():
        header = _shard_header(shard)
        if header is None:
            return None
        for name, entry in header.items():
            if not isinstance(entry, dict):
                continue
            try:
                begin, end = (int(v) for v in entry.get("data_offsets") or (0, 0))
            except (TypeError, ValueError):
                return None
            size = end - begin
            tag = entry.get("dtype")
            if tag == "F64":
                size = size // 4 if keep is None or not keep.search(name) else size // 2
            elif tag == "F32" and (keep is None or not keep.search(name)):
                size //= 2
            total += size
    return -(-total // _MIB)


@dataclass(frozen = True)
class StagePlan:
    engaged: bool
    reason: str
    stage: tuple = ()
    dense_mib: int = 0
    available_mib: Optional[int] = None
    reserve_mib: int = 0
    skipped: dict = field(default_factory = dict)


def plan_stage(
    misses: dict,
    *,
    available_mib: Optional[int],
    reserve_mib: int,
    device_free_mib: Optional[int],
    forced: bool = False,
    disabled: bool = False,
) -> StagePlan:
    """``misses``: component -> loaded MiB. Stage largest first (each must fit the card) until the rest fits host RAM."""
    if disabled:
        return StagePlan(False, f"{SMALL_HOST_ENV}=0")
    sizes = {k: int(v) for k, v in misses.items() if v is not None}
    dense = sum(sizes.values())
    if not sizes:
        return StagePlan(False, "no cast-cache miss to convert", dense_mib = 0)
    if available_mib is None and not forced:
        return StagePlan(False, "host RAM unreadable", dense_mib = dense)
    fits = available_mib is not None and dense + reserve_mib <= int(available_mib)
    if fits and not forced:
        return StagePlan(
            False,
            f"dense conversion ({dense} MiB) fits {available_mib} MiB available host RAM",
            dense_mib = dense,
            available_mib = available_mib,
            reserve_mib = reserve_mib,
        )
    stage = []
    skipped = {}
    left = dense
    for name, mib in sorted(sizes.items(), key = lambda kv: (-kv[1], kv[0])):
        if not forced and available_mib is not None and left + reserve_mib <= int(available_mib):
            break
        if mib < _STAGE_MIN_MIB:
            skipped[name] = "small"
            continue
        if device_free_mib is None or mib + _DEVICE_MARGIN_MIB > int(device_free_mib):
            skipped[name] = (
                f"needs {mib + _DEVICE_MARGIN_MIB} MiB of the card, {device_free_mib} MiB free"
            )
            continue
        stage.append(name)
        left -= mib
    why = (
        f"{SMALL_HOST_ENV}=1"
        if forced
        else f"dense conversion needs ~{dense} MiB + {reserve_mib} MiB reserve of host RAM, "
        f"{available_mib} MiB available"
    )
    return StagePlan(bool(stage), why, tuple(stage), dense, available_mib, reserve_mib, skipped)


def _component_dir(cand: Any) -> str:
    return str(Path(cand.config_path).parent)


def _device_load_kwargs(cand: Any, device: Any) -> dict:
    from .diffusion_cast_cache import _load_kwargs

    kwargs = _load_kwargs(cand)
    kwargs.pop("output_loading_info", None)
    kwargs["device_map"] = {"": str(device)}
    return kwargs


def stage_on_device(
    cand: Any,
    device: Any,
    *,
    logger: Any = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> bool:
    """Load ``cand`` alone onto ``device``, write its cast-cache entry from the device tensors, free it."""
    import torch

    from .diffusion_cast_cache import _state_names, save

    t0 = time.perf_counter()
    module = None
    try:
        module = cand.cls.from_pretrained(_component_dir(cand), **_device_load_kwargs(cand, device))
        module.eval()
        if cancel_check is not None:
            cancel_check()
        names = _state_names(module)
        if not names:
            return False
        cand.names = names
        path = save(cand, module, logger = logger, check = cancel_check)
        if logger is not None:
            logger.info(
                "video.small_host: %s staged on %s in %.1f s (%s)",
                cand.component,
                device,
                time.perf_counter() - t0,
                "cast cache written" if path else "not written",
            )
        return bool(path)
    finally:
        module = None
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def _device_free_mib(device: Any) -> Optional[int]:
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info(dev)
        return int(free // _MIB)
    except Exception:  # noqa: BLE001
        return None


def stage_misses(
    cands: list,
    device: Any,
    *,
    logger: Any = None,
    cancel_check: Optional[Callable[[], None]] = None,
) -> list:
    """Stages cands a dense load would not fit, returning them; never raises except on cancel."""
    if not cands or device is None:
        return []
    try:
        import torch

        if torch.device(device).type != "cuda":
            return []
        from .diffusion_cast_cache import cached_components
        from .diffusion_small_host import host_ram_mib, host_reserve_mib

        cached = set(cached_components(cands))
        misses = {c.component: loaded_mib(c) for c in cands if c.component not in cached}
        total, available = host_ram_mib()
        plan = plan_stage(
            misses,
            available_mib = available,
            reserve_mib = host_reserve_mib(total),
            device_free_mib = _device_free_mib(device),
            forced = stage_forced(),
            disabled = stage_disabled(),
        )
        if logger is not None and (plan.engaged or plan.skipped or misses):
            logger.info(
                "video.small_host: engaged=%s (%s); stage=%s%s",
                plan.engaged,
                plan.reason,
                list(plan.stage),
                f" skipped={json.dumps(plan.skipped)}" if plan.skipped else "",
            )
        done = []
        by_name = {c.component: c for c in cands}
        for name in plan.stage:
            if cancel_check is not None:
                cancel_check()
            if stage_on_device(by_name[name], device, logger = logger, cancel_check = cancel_check):
                done.append(name)
        return done
    except Exception as exc:  # noqa: BLE001 - the dense load still runs
        if cancel_check is not None:
            cancel_check()
        if logger is not None:
            logger.warning(
                "video.small_host: staging skipped (%s: %s)", type(exc).__name__, str(exc)[:200]
            )
        return []
