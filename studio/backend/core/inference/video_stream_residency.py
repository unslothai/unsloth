# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep part of a block-streamed Wan denoiser on the GPU, re-fitted per request.

A Wan load that does not fit the card (Wan2.2-TI2V-5B on a 15 GB T4) streams its denoiser with diffusers group
offloading. Two costs come with that, both on every denoiser call (two per step under CFG):

- the top-level group (patch embedding, condition embedder, output head) has no copy stream, so its ~190 MiB is
  uploaded from pageable host memory on the compute stream before each call;
- every block is uploaded again, although the denoise peak leaves several GB of the card unused.

Here the top-level group and then a prefix of the blocks stay resident within the room the request leaves: free memory
plus the allocator's unused cache plus what is already resident, minus the streamed floor, the base overhead and the
request's runtime estimate (``estimate_video_runtime_mib``, which covers the decode). The room is re-read for every
request, so a longer clip streams some blocks again first. Resident groups keep their host copy, so streaming them
again is a pointer swap.

After a request completes (decode and export included), its peak CUDA allocation above what was allocated when it
started is recorded per load. A later request no larger than a recorded one (pixels x frames) is sized from that
measured peak instead: room = free + unused cache + resident - measured peak x 1.15 - 1 GiB. The estimate stays the
floor of the room, so the measured path only ever widens it, and a cancelled or failed request records nothing.

Kill switches: ``UNSLOTH_VIDEO_DIT_RESIDENT=0`` streams everything as before; ``UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS=0``
keeps only the top-level group resident; ``UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED=0`` sizes every request from the
estimate.
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
    """Single-DiT Wan families streamed by group offloading on a CUDA device."""
    return (
        residency_enabled()
        and not modular
        and not is_moe
        and str(family_name or "").startswith("wan")
        and offload_policy == "group"
        and str(device or "").startswith("cuda")
    )


def resident_bytes(module: Any) -> int:
    """Bytes of ``module``'s offload groups currently held resident by this module's placement."""
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
    """MiB the resident groups may occupy for this request (never negative)."""
    from .diffusion_memory import DEFAULT_BASE_OVERHEAD_MIB, estimate_video_runtime_mib

    available = int(free_mib) + max(0, int(unused_cache_mib)) + max(0, int(resident_mib_now))
    need = (
        int(floor_mib)
        + DEFAULT_BASE_OVERHEAD_MIB
        + estimate_video_runtime_mib(width = width, height = height, num_frames = frames)
    )
    return max(0, available - need)


def measured_room_mib(
    *,
    free_mib: int,
    unused_cache_mib: int,
    resident_mib_now: int,
    peak_extra_mib: int,
) -> int:
    """Room from a measured request peak: what the request allocated above its start, plus margin and slack."""
    available = int(free_mib) + max(0, int(unused_cache_mib)) + max(0, int(resident_mib_now))
    need = int(max(0, int(peak_extra_mib)) * MEASURED_PEAK_MARGIN) + MEASURED_SLACK_MIB
    return max(0, available - need)


def _measured_extra_mib(module: Any, work: int) -> Optional[int]:
    """The recorded peak of the smallest completed request at least as large as ``work``; None when none covers it."""
    peaks = getattr(module, "_unsloth_video_peaks", None) or {}
    covering = [extra for w, extra in peaks.items() if w >= work]
    return max(covering) if covering else None


def record_request_peak(pipe: Any, *, logger: Any = None) -> Optional[int]:
    """Record the completed request's peak CUDA allocation above its start (on the device its fit read). Call only
    after decode and export."""
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
    """Re-fit the resident set of ``pipe.transformer`` for one request. Returns the resident MiB, or None when skipped.
    Never raises: any failure streams every group again, which is the plain group-offload placement."""
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
        # an earlier request that never reached record_request_peak (cancelled, failed) leaves nothing measured
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
            # contiguous room for the promoted weights; the request's own allocations come after
            torch.cuda.empty_cache()
            _keep_groups_resident(module, room, dev, logger)
            promoted = max(0, resident_mib(module) - after_release)
        kept = resident_mib(module)
        torch.cuda.reset_peak_memory_stats(dev)
        module._unsloth_video_pending = (work, int(torch.cuda.memory_allocated(dev)), dev)
        try:
            # release_resident_groups' restore reads it back; keep it at this request's room
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
