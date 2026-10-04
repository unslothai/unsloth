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

Kill switches: ``UNSLOTH_VIDEO_DIT_RESIDENT=0`` streams everything as before; ``UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS=0``
keeps only the top-level group resident.
"""

from __future__ import annotations

import os
from typing import Any, Optional

VIDEO_DIT_RESIDENT_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT"
VIDEO_DIT_RESIDENT_BLOCKS_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS"

_COUNTS: dict[str, int] = {"fits": 0, "promoted_mib": 0, "released_mib": 0, "resident_mib": 0, "groups": 0}


def counts() -> dict[str, int]:
    return dict(_COUNTS)


def _off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def residency_enabled() -> bool:
    return not _off(VIDEO_DIT_RESIDENT_ENV)


def blocks_enabled() -> bool:
    return not _off(VIDEO_DIT_RESIDENT_BLOCKS_ENV)


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
        room = room_mib(
            free_mib = int(free) >> 20,
            unused_cache_mib = unused >> 20,
            resident_mib_now = current,
            floor_mib = int(floor_mib),
            width = width,
            height = height,
            frames = frames,
        )
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
            _keep_groups_resident(module, room, dev, logger)
            promoted = max(0, resident_mib(module) - after_release)
        kept = resident_mib(module)
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
                "room %d MiB, +%d / -%d MiB",
                kept,
                _COUNTS["groups"],
                width,
                height,
                frames,
                room,
                promoted,
                released,
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
