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
from typing import Any, Callable, Iterable, Optional

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
    # component -> MiB of it placed on the card when it is staged split across the card and host RAM
    split: dict = field(default_factory = dict)
    # MiB the same load converts in host RAM outside the cast cache (a bf16 encoder cast to fp16)
    others_mib: int = 0


def plan_stage(
    misses: dict,
    *,
    available_mib: Optional[int],
    reserve_mib: int,
    device_free_mib: Optional[int],
    forced: bool = False,
    disabled: bool = False,
    others_mib: int = 0,
    source_castable: Optional[Iterable] = None,
) -> StagePlan:
    """``misses``: component -> loaded MiB. Stage largest first until the rest fits host RAM; ``others_mib`` is what
    the same load converts in host RAM besides them (it cannot be staged, so it is priced, not moved). A miss larger
    than the card is staged split: the card holds what fits, host RAM the rest (when that rest fits). Only a
    component in ``source_castable`` (None = every one) can be split, since its cast is written from the source
    shards; any other one too big for the card is skipped and the planner moves on."""
    if disabled:
        return StagePlan(False, f"{SMALL_HOST_ENV}=0")
    sizes = {k: int(v) for k, v in misses.items() if v is not None}
    dense = sum(sizes.values())
    others = max(0, int(others_mib or 0))
    if not sizes:
        return StagePlan(False, "no cast-cache miss to convert", dense_mib = 0, others_mib = others)
    if available_mib is None and not forced:
        return StagePlan(False, "host RAM unreadable", dense_mib = dense, others_mib = others)
    beside = f" + {others} MiB of other converted components" if others else ""
    fits = available_mib is not None and dense + others + reserve_mib <= int(available_mib)
    if fits and not forced:
        return StagePlan(
            False,
            f"dense conversion ({dense} MiB{beside}) fits {available_mib} MiB available host RAM",
            dense_mib = dense,
            available_mib = available_mib,
            reserve_mib = reserve_mib,
            others_mib = others,
        )
    stage = []
    skipped = {}
    split = {}
    left = dense
    for name, mib in sorted(sizes.items(), key = lambda kv: (-kv[1], kv[0])):
        if (
            not forced
            and available_mib is not None
            and left + others + reserve_mib <= int(available_mib)
        ):
            break
        if mib < _STAGE_MIN_MIB:
            skipped[name] = "small"
            continue
        card = None if device_free_mib is None else int(device_free_mib) - _DEVICE_MARGIN_MIB
        if card is not None and mib <= card:
            stage.append(name)
            left -= mib
            continue
        # Split: the card's share plus a host share that fits beside the rest of the load's conversions.
        host_room = (
            None
            if available_mib is None
            else int(available_mib) - reserve_mib - others - (left - mib)
        )
        if (
            card is not None
            and (source_castable is None or name in source_castable)
            and card >= _STAGE_MIN_MIB
            and host_room is not None
            and mib - card <= host_room
        ):
            stage.append(name)
            split[name] = card
            left -= mib
            continue
        skipped[name] = (
            f"needs {mib + _DEVICE_MARGIN_MIB} MiB of the card, {device_free_mib} MiB free, and its host share "
            f"does not fit {host_room} MiB"
        )
    why = (
        f"{SMALL_HOST_ENV}=1"
        if forced
        else f"dense conversion needs ~{dense} MiB{beside} + {reserve_mib} MiB reserve of host RAM, "
        f"{available_mib} MiB available"
    )
    return StagePlan(
        bool(stage), why, tuple(stage), dense, available_mib, reserve_mib, skipped, split, others
    )


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


def converted_others_mib(
    source: Any,
    cands: list,
    *,
    torch_dtype: Any,
    cache_dir: Optional[str] = None,
    revision: Optional[str] = None,
    skip: Any = (),
) -> int:
    """MiB the pipeline load converts in host RAM outside the cast cache: half-precision shards of a component that
    is not a cast candidate but loads at the other half dtype (bf16 -> fp16 on a float16 card), which become anonymous
    memory beside the cast-cache misses. 0 when unreadable."""
    try:
        import torch

        from . import diffusion_cast_cache as cc

        index = cc._read_json(
            cc._cache._local_file(str(source), "model_index.json", cache_dir, revision)
        )
        if not isinstance(index, dict):
            return 0
        names = {c.component for c in cands}
        skip = set(skip or ())
        total = 0
        for component, spec in index.items():
            if component.startswith("_") or component in skip or component in names:
                continue
            if (
                not isinstance(spec, (list, tuple))
                or len(spec) != 2
                or spec[0] not in cc._LIBRARIES
            ):
                continue
            dtype = cc._component_dtype(torch_dtype, component)
            if not cc._is_half(dtype):
                continue
            want = "F16" if dtype == torch.float16 else "BF16"
            for shard in (
                cc._shards_of(str(source), component, cc._STEMS[spec[0]], cache_dir, revision) or ()
            ):
                header = cc._shard_header(shard) or {}
                for name, entry in header.items():
                    if name == "__metadata__" or not isinstance(entry, dict):
                        continue
                    if entry.get("dtype") in ("F16", "BF16") and entry.get("dtype") != want:
                        begin, end = (int(v) for v in entry.get("data_offsets") or (0, 0))
                        total += end - begin
        return -(-total // _MIB)
    except Exception:  # noqa: BLE001 - unpriced, as before
        return 0


def stage_misses(
    cands: list,
    device: Any,
    *,
    logger: Any = None,
    cancel_check: Optional[Callable[[], None]] = None,
    others_mib: int = 0,
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
            others_mib = others_mib,
            source_castable = {
                c.component for c in cands if getattr(c, "library", None) == "diffusers"
            },
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
        from .diffusion_cast_cache import save_from_source

        for name in plan.stage:
            if cancel_check is not None:
                cancel_check()
            if plan.split.get(name):
                # A card / host split through accelerate's cpu offload brings its host share back as meta tensors
                # (nothing to write); the source shards cast one tensor at a time instead, card untouched. If that
                # fails the component is left to the normal load.
                t0 = time.perf_counter()
                if save_from_source(by_name[name], logger = logger, check = cancel_check) is not None:
                    done.append(name)
                    if logger is not None:
                        logger.info(
                            "video.small_host: %s cast written from its source shards in %.1f s (host RAM: one "
                            "tensor at a time); the load maps it",
                            name,
                            time.perf_counter() - t0,
                        )
                continue
            if stage_on_device(by_name[name], device, logger = logger, cancel_check = cancel_check):
                done.append(name)
        if plan.skipped and not stage_disabled():
            # Neither the card nor a card / host split takes it: write its cast from the source shards one tensor at a
            # time, so the load maps the cached file instead of converting the whole component in host RAM. This runs
            # even when nothing else was staged (a lone DiT too big for the card is the case it is for).
            for name, why in plan.skipped.items():
                if name not in by_name or why == "small":
                    continue
                if cancel_check is not None:
                    cancel_check()
                t0 = time.perf_counter()
                if save_from_source(by_name[name], logger = logger, check = cancel_check) is not None:
                    done.append(name)
                    if logger is not None:
                        logger.info(
                            "video.small_host: %s cast written from its source shards in %.1f s (host RAM: one "
                            "tensor at a time); the load maps it",
                            name,
                            time.perf_counter() - t0,
                        )
        return done
    except Exception as exc:  # noqa: BLE001 - the dense load still runs
        if cancel_check is not None:
            cancel_check()
        if logger is not None:
            logger.warning(
                "video.small_host: staging skipped (%s: %s)", type(exc).__name__, str(exc)[:200]
            )
        return []
