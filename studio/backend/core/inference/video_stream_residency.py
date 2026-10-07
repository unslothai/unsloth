# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the top-level group and a block prefix of a group-offloaded Wan denoiser resident, re-fitted per request.

Room = free + unused cache + already resident - need. Need is the streamed floor + base overhead + the runtime estimate,
or, for a request no larger (pixels x frames) than a completed one, that request's measured peak x 1.15 + 1 GiB; the
estimate stays the lower bound. Resident groups keep their host copy, so streaming them again is a pointer swap.
Kill switches: UNSLOTH_VIDEO_DIT_RESIDENT=0, UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS=0 (top-level group only),
UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED=0 (estimate only).
"""

from __future__ import annotations

import os
from typing import Any, Optional

VIDEO_DIT_RESIDENT_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT"
VIDEO_DIT_RESIDENT_BLOCKS_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS"
VIDEO_DIT_RESIDENT_MEASURED_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED"
# A measured peak is scaled by this and a fixed slack is kept on top (allocator fragmentation, cuDNN workspaces).
MEASURED_PEAK_MARGIN = 1.15
MEASURED_SLACK_MIB = 1024

_COUNTS: dict[str, int] = {
    "fits": 0,
    "measured_fits": 0,
    "recorded": 0,
    "promoted_mib": 0,
    "released_mib": 0,
    "resident_mib": 0,
    "groups": 0,
}


def counts() -> dict[str, int]:
    return dict(_COUNTS)


def _off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def residency_enabled() -> bool:
    return not _off(VIDEO_DIT_RESIDENT_ENV)


def blocks_enabled() -> bool:
    return not _off(VIDEO_DIT_RESIDENT_BLOCKS_ENV)


def measured_enabled() -> bool:
    return not _off(VIDEO_DIT_RESIDENT_MEASURED_ENV)


def applies(
    family_name: Optional[str],
    *,
    is_moe: bool,
    offload_policy: Optional[str],
    device: Any,
    modular: bool = False,
) -> bool:
    return (
        residency_enabled()
        and not modular
        and not is_moe
        and str(family_name or "").startswith("wan")
        and offload_policy == "group"
        and str(device or "").startswith("cuda")
    )


def resident_bytes(module: Any) -> int:
    from .diffusion_memory import _offload_groups

    total = 0
    for group in _offload_groups(module) or ():
        if getattr(group, "_unsloth_resident", False):
            total += int(getattr(group, "_unsloth_resident_bytes", 0) or 0)
    return total


def resident_mib(module: Any) -> int:
    return resident_bytes(module) >> 20


def _top_group_mib(module: Any) -> int:
    from .diffusion_memory import _offload_groups, _storage_nbytes
    for group in _offload_groups(module) or ():
        if getattr(group, "offload_leader", None) is module:
            seen: set = set()
            total = 0
            for t in (
                [p for m in group.modules for p in m.parameters()]
                + [b for m in group.modules for b in m.buffers()]
                + list(group.parameters or [])
                + list(group.buffers or [])
            ):
                if id(t) not in seen:
                    seen.add(id(t))
                    total += sum(_storage_nbytes(t))
            return -(-total // (1 << 20))
    return 0


def room_mib(
    *,
    free_mib: int,
    unused_cache_mib: int,
    resident_mib_now: int,
    floor_mib: int,
    width: int,
    height: int,
    frames: int,
) -> int:
    from .diffusion_memory import DEFAULT_BASE_OVERHEAD_MIB, estimate_video_runtime_mib

    available = int(free_mib) + max(0, int(unused_cache_mib)) + max(0, int(resident_mib_now))
    need = (
        int(floor_mib)
        + DEFAULT_BASE_OVERHEAD_MIB
        + estimate_video_runtime_mib(width = width, height = height, num_frames = frames)
    )
    return max(0, available - need)


def measured_room_mib(
    *, free_mib: int, unused_cache_mib: int, resident_mib_now: int, peak_extra_mib: int
) -> int:
    available = int(free_mib) + max(0, int(unused_cache_mib)) + max(0, int(resident_mib_now))
    need = int(max(0, int(peak_extra_mib)) * MEASURED_PEAK_MARGIN) + MEASURED_SLACK_MIB
    return max(0, available - need)


def _measured_extra_mib(module: Any, work: int) -> Optional[int]:
    """The largest recorded peak among completed requests at least as large as ``work``; None when none covers it."""
    peaks = getattr(module, "_unsloth_video_peaks", None) or {}
    covering = [extra for w, extra in peaks.items() if w >= work]
    return max(covering) if covering else None


def record_request_peak(pipe: Any, *, logger: Any = None) -> Optional[int]:
    """Record the request's peak allocation above its start; call only after decode and export."""
    module = getattr(pipe, "transformer", None)
    pending = getattr(module, "_unsloth_video_pending", None)
    if module is None or pending is None:
        return None
    try:
        import torch

        module._unsloth_video_pending = None
        work, start, dev = pending
        extra = (int(torch.cuda.max_memory_allocated(dev)) - int(start)) >> 20
        if extra <= 0:
            return None
        peaks = getattr(module, "_unsloth_video_peaks", None)
        if not isinstance(peaks, dict):
            peaks = {}
            module._unsloth_video_peaks = peaks
        peaks[work] = max(extra, peaks.get(work, 0))
        _COUNTS["recorded"] += 1
        if logger is not None:
            logger.info("video.dit_resident: request peak %d MiB above its start recorded", extra)
        return extra
    except Exception:  # noqa: BLE001 -- unrecorded keeps the estimate
        return None


def fit_for_request(
    pipe: Any,
    *,
    device: Any,
    floor_mib: Optional[int],
    width: int,
    height: int,
    frames: int,
    logger: Any = None,
) -> Optional[int]:
    """Re-fit ``pipe.transformer``'s resident set for one request; returns resident MiB or None. Never raises: a
    failure streams every group again (plain group offload)."""
    module = getattr(pipe, "transformer", None)
    if module is None or floor_mib is None:
        return None
    try:
        import torch

        from .diffusion_memory import _keep_groups_resident, release_resident_groups

        dev = torch.device(device)
        if dev.type != "cuda":
            return None
        free, _total = torch.cuda.mem_get_info(dev)
        unused = int(torch.cuda.memory_reserved(dev)) - int(torch.cuda.memory_allocated(dev))
        current = resident_mib(module)
        module._unsloth_video_pending = None
        work = int(width) * int(height) * int(frames)
        room = room_mib(
            free_mib = int(free) >> 20,
            unused_cache_mib = unused >> 20,
            resident_mib_now = current,
            floor_mib = int(floor_mib),
            width = width,
            height = height,
            frames = frames,
        )
        measured = _measured_extra_mib(module, work) if measured_enabled() else None
        if measured is not None:
            wider = measured_room_mib(
                free_mib = int(free) >> 20,
                unused_cache_mib = unused >> 20,
                resident_mib_now = current,
                peak_extra_mib = measured,
            )
            if wider > room:
                room = wider
                _COUNTS["measured_fits"] += 1
        if not blocks_enabled():
            top = _top_group_mib(module)
            room = min(room, top) if top else 0
        released = promoted = 0
        over = resident_bytes(module) - (room << 20)
        if over > 0:
            restore = release_resident_groups(
                pipe, -(-over // (1 << 20)), logger, denoisers_only = True, reason = "a larger request"
            )
            if restore is not None:
                released = max(0, current - resident_mib(module))
        after_release = resident_mib(module)
        if room > after_release:
            torch.cuda.empty_cache()
            _keep_groups_resident(module, room, dev, logger)
            promoted = max(0, resident_mib(module) - after_release)
        kept = resident_mib(module)
        torch.cuda.reset_peak_memory_stats(dev)
        module._unsloth_video_pending = (work, int(torch.cuda.memory_allocated(dev)), dev)
        try:
            module._unsloth_resident_room = room if room > 0 else None
        except AttributeError:
            pass
        _COUNTS["fits"] += 1
        _COUNTS["promoted_mib"] += promoted
        _COUNTS["released_mib"] += released
        _COUNTS["resident_mib"] = kept
        from .diffusion_memory import _offload_groups

        _COUNTS["groups"] = sum(
            1 for g in _offload_groups(module) or () if getattr(g, "_unsloth_resident", False)
        )
        if logger is not None and (promoted or released):
            logger.info(
                "video.dit_resident: %d MiB of the streamed denoiser resident (%d groups) for %dx%d at %d frames; "
                "room %d MiB (%s), +%d / -%d MiB; free %d, unused cache %d, floor %d MiB",
                kept,
                _COUNTS["groups"],
                width,
                height,
                frames,
                room,
                f"measured peak {measured} MiB" if measured is not None else "estimate",
                promoted,
                released,
                int(free) >> 20,
                unused >> 20,
                int(floor_mib),
            )
        return kept
    except Exception as exc:  # noqa: BLE001 -- residency is a speed-up; stream everything again
        if logger is not None:
            logger.warning("video.dit_resident: streaming every group again (%s)", exc)
        try:
            from .diffusion_memory import release_resident_groups
            release_resident_groups(
                pipe, 1 << 30, logger, denoisers_only = True, reason = "a residency failure"
            )
        except Exception:  # noqa: BLE001
            pass
        return None
