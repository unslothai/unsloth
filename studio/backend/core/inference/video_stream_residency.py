# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the top-level group and a block prefix of a block-streamed Wan or HunyuanVideo-1.5 denoiser resident, re-fitted
per request.

Room = free + unused cache + already resident - need. Need is the streamed floor + base overhead + the runtime estimate,
or, for a request no larger (pixels x frames) than a completed one, that request's measured peak x 1.15 + 1 GiB; the
estimate stays the lower bound. Resident groups keep their host copy, so streaming them again is a pointer swap.
Kill switches: UNSLOTH_VIDEO_DIT_RESIDENT=0, UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS=0 (top-level group only),
UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED=0 (estimate only).

Calibration is not Wan-only: ``arm_request`` arms the same peak measurement for any video family, and every recorded
peak also lands in a process-wide table keyed by the loaded model (``calibration_key``), so a reload of the same model
and the offload refusal (``measured_extra_mib``) reuse it. ``UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED=0`` turns reuse off.
"""

from __future__ import annotations

import os
from typing import Any, Optional

VIDEO_DIT_RESIDENT_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT"
VIDEO_DIT_RESIDENT_BLOCKS_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT_BLOCKS"
VIDEO_DIT_RESIDENT_MEASURED_ENV = "UNSLOTH_VIDEO_DIT_RESIDENT_MEASURED"
RESIDENT_POLICIES = ("group", "streaming")
# Single-DiT families measured under the fit (MiniMax-H3 has its own controller, video_minimax_h3_residency).
RESIDENT_FAMILY_PREFIXES = ("wan", "hunyuanvideo-1.5")
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
        and str(family_name or "").startswith(RESIDENT_FAMILY_PREFIXES)
        and offload_policy in RESIDENT_POLICIES
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


# calibration key -> {work (pixels x frames): peak MiB above the request's start}; survives a reload of the same model
_PEAKS: dict = {}


def calibration_key(
    family: Any,
    repo: Any = None,
    offload_policy: Any = None,
    device: Any = None,
    **extra: Any,
) -> tuple:
    return (
        str(family or ""),
        str(repo or ""),
        str(offload_policy or ""),
        str(
            device or ""
        ),  # the card too: a different GPU model can pick different kernels and workspaces
        tuple(sorted((k, str(v)) for k, v in extra.items())),
    )


def with_extra(key: Any, name: str, value: Any) -> Any:
    """``key`` with one keyword entry replaced (e.g. the step-cache mode a request actually ran with)."""
    if not isinstance(key, tuple) or len(key) != 5:
        return key
    extras = dict(key[4])
    extras[name] = str(value)
    return (*key[:4], tuple(sorted(extras.items())))


def retag(pipe: Any, name: str, value: Any) -> Any:
    """Rewrite one entry of the denoiser's calibration key in place (the peak recorded later lands there)."""
    module = getattr(pipe, "transformer", None)
    key = getattr(module, "_unsloth_video_calibration_key", None)
    if key is None:
        return None
    new = with_extra(key, name, value)
    try:
        module._unsloth_video_calibration_key = new
    except Exception:  # noqa: BLE001
        return key
    return new


def _covering(peaks: dict, work: int) -> Optional[int]:
    covering = [extra for w, extra in (peaks or {}).items() if w >= work]
    return max(covering) if covering else None


def _store_key(key: Any) -> Any:
    """``key`` as persisted: the device in it is a process-local ordinal (``cuda:0``), so the GPU's name rides along
    and workers with different ``CUDA_VISIBLE_DEVICES`` never share one card's peaks with another."""
    if not isinstance(key, tuple) or len(key) != 5:
        return key
    if dict(key[4]).get("revision", "?") in ("", "None", "unknown"):
        return None  # the exact weights are unknown (an unreadable local build): nothing persists for them
    try:
        from .diffusion_measured_peaks import device_name
        name = device_name(key[3]) if key[3] else None
    except Exception:  # noqa: BLE001
        name = None
    return (key, name)


def _key_peaks(key: Any) -> dict:
    """This process's peaks for ``key``, merged with the persisted store's (other processes' measurements) at every
    read; the store's own mtime cache keeps that a stat when nothing changed."""
    peaks = _PEAKS.setdefault(key, {})
    try:
        from .diffusion_measured_peaks import work_peaks
        for work, extra in work_peaks(_store_key(key)).items():
            if extra > peaks.get(work, 0):
                peaks[work] = extra
    except Exception:  # noqa: BLE001 - no store: this process's peaks only
        pass
    return peaks


def measured_extra_mib(key: Any, work: int) -> Optional[int]:
    if key is None or not measured_enabled():
        return None
    return _covering(_key_peaks(key), int(work))


def _measured_extra_mib(module: Any, work: int) -> Optional[int]:
    """The largest recorded peak among completed requests at least as large as ``work``; None when none covers it.
    A tagged module reads only the peaks of its calibration key (same load AND request mode), which also carries the
    peaks it recorded itself; the module's own mode-blind table is the fallback for an untagged one."""
    key = getattr(module, "_unsloth_video_calibration_key", None)
    if key is not None:
        return _covering(_key_peaks(key), work)
    return _covering(getattr(module, "_unsloth_video_peaks", None) or {}, work)


def arm_request(
    pipe: Any,
    *,
    key: Any,
    width: int,
    height: int,
    frames: int,
    device: Any,
    rearm: bool = True,
) -> bool:
    """Tag the denoiser with ``key`` and arm the peak measurement ``record_request_peak`` closes. ``rearm=False``
    after ``fit_for_request`` only tags. CUDA only; never raises."""
    module = getattr(pipe, "transformer", None)
    if module is None:
        return False
    try:
        import torch

        dev = torch.device(device)
        if dev.type != "cuda":
            return False
        module._unsloth_video_calibration_key = key
        if not rearm:
            return True
        dev = _open_window(module, dev)
        torch.cuda.reset_peak_memory_stats(dev)
        work = int(width) * int(height) * int(frames)
        module._unsloth_video_pending = (work, int(torch.cuda.memory_allocated(dev)), dev)
        module._unsloth_video_failed_captures = _failed_captures()
        return True
    except Exception:  # noqa: BLE001 -- an unarmed request keeps the estimate
        return False


def _open_window(module: Any, dev: Any) -> Any:
    """Register this request's peak measurement on its card (``diffusion_measured_peaks.open_window``), replacing a
    stale one; returns the indexed device."""
    import torch

    from .diffusion_measured_peaks import close_window, open_window

    if dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    stale = getattr(module, "_unsloth_video_window", None)
    if stale is not None:
        close_window(stale[0], stale[1])
    module._unsloth_video_window = (dev, open_window(dev))
    before = _capture_generation(dev)
    graphs = _live_graphs(dev)
    generation = _capture_generation(dev)
    accounted = getattr(module, "_unsloth_video_graphs", None)
    # a capture counts its graph before its generation, so an unchanged generation around the count makes 0 consistent
    if graphs == 0 and generation == before:
        module._unsloth_video_graphs = accounted = 0
        module._unsloth_video_graph_gen = generation
    elif accounted is None or graphs < accounted:
        # the load's own graphs (every request starts above them), or fewer after a reset freed some
        first = accounted is None
        module._unsloth_video_graphs = accounted = graphs
        if first or graphs <= 0:
            # nothing alive on the card; a drop to a non-zero count may be another pipeline's reset, so a capture
            # this one made unrecorded keeps the generation dirty
            module._unsloth_video_graph_gen = generation
    if getattr(module, "_unsloth_video_graph_gen", None) is None:
        module._unsloth_video_graph_gen = generation
    # graphs a discarded request recorded stay allocated, also one that replaced an evicted graph (same count): a
    # peak above that start would omit them
    module._unsloth_video_graphs_clean = (
        graphs == accounted and generation <= module._unsloth_video_graph_gen
    )
    return dev


def _capture_generation(dev: Any) -> int:
    try:
        from .diffusion_cuda_graph import capture_generation
        return capture_generation(getattr(dev, "index", None))
    except Exception:  # noqa: BLE001
        return -1


def _live_graphs(dev: Any) -> int:
    try:
        from .diffusion_cuda_graph import live_graph_count
        return live_graph_count(getattr(dev, "index", None))
    except Exception:  # noqa: BLE001
        return -1


def discard_request(pipe: Any) -> None:
    """Drop a request's armed measurement without recording it (cancel, failure, refusal): its window would otherwise
    mark every later measurement on the card as overlapped. A no-op after ``record_request_peak``."""
    module = getattr(pipe, "transformer", None)
    if module is None:
        return
    try:
        module._unsloth_video_pending = None
        window = getattr(module, "_unsloth_video_window", None)
        module._unsloth_video_window = None
        if window is not None:
            from .diffusion_measured_peaks import close_window
            close_window(window[0], window[1])
    except Exception:  # noqa: BLE001
        pass


def _failed_captures() -> int:
    try:
        from .diffusion_cuda_graph import failed_capture_count
        return failed_capture_count()
    except Exception:  # noqa: BLE001
        return 0


def _graphs_off_for_load(handles: tuple) -> bool:
    """True when every whole-step graph is off for good (declined or poisoned, no per-block fallback); per-call refusals keep the key."""
    if not handles:
        return False
    for h in handles:
        if (
            getattr(h, "fallback", None) is not None
            or getattr(h, "fallback_handle", None) is not None
        ):
            return False
        # load-wide only: a per-key drop (probe OOM, unpaid replay) leaves other shapes free to capture
        placement = getattr(h, "placement", None)
        declined = placement is not None and getattr(placement, "_declined", None) is not None
        dropped = declined and not getattr(h, "cache", None)
        if not (dropped or getattr(h, "poisoned", False) is True):
            return False
    return True


def live_speed_optims(pipe: Any, speed_optims: Any) -> tuple:
    """``speed_optims`` minus ``cuda_graph`` once this load's graphs are off for good, so the key matches what runs."""
    optims = tuple(speed_optims or ())
    if "cuda_graph" not in optims:
        return optims
    try:
        if _graphs_off_for_load(tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())):
            return tuple(o for o in optims if o != "cuda_graph")
        return optims
    except Exception:  # noqa: BLE001 - unreadable: keep the load's record
        return optims


def _keyed_graphs_live(pipe: Any, key: Any) -> bool:
    """False when ``key`` names CUDA graphs that ``live_status`` reports dropped or never engaged for this load."""
    try:
        optims = dict(key[4]).get("speed_optims", "")
        if "cuda_graph" not in str(optims):
            return True
        from .diffusion_cuda_graph import live_status

        _, live = live_status(
            None, ("cuda_graph",), tuple(getattr(pipe, "_unsloth_cuda_graphs", ()) or ())
        )
        return "cuda_graph" in live
    except Exception:  # noqa: BLE001 - unreadable: no key names this render
        return False


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
        window = getattr(module, "_unsloth_video_window", None)
        module._unsloth_video_window = None
        if window is not None:
            from .diffusion_measured_peaks import close_window

            # closed after the read: a render arming in between would otherwise reset the peak unseen
            if not close_window(window[0], window[1]):
                return None  # another render on the card reset the peak meanwhile
        if extra <= 0 or not getattr(module, "_unsloth_video_graphs_clean", True):
            return None
        peaks = getattr(module, "_unsloth_video_peaks", None)
        if not isinstance(peaks, dict):
            peaks = {}
            module._unsloth_video_peaks = peaks
        peaks[work] = max(extra, peaks.get(work, 0))
        key = getattr(module, "_unsloth_video_calibration_key", None)
        # a failed capture absorbed an OOM: this peak is that attempt's, keep it out of the keyed and persisted peaks
        # a load keyed with CUDA graphs whose graphs no longer run (a poisoned wrapper after a failed capture runs
        # eager without counting a new failure) records an eager peak a graph-capturing load would undershoot
        if (
            key is not None
            and _failed_captures() == getattr(module, "_unsloth_video_failed_captures", 0)
            and _keyed_graphs_live(pipe, key)
        ):
            graphs, generation = _live_graphs(dev), _capture_generation(dev)
            stored_key = _store_key(key)
            try:
                from .diffusion_memory import stream_window_pad_mib

                # on the streamed baseline, as image peaks: a later request of the key may stream groups this request
                # kept resident; an unknown placement enters neither the keyed table nor the store
                pad = stream_window_pad_mib(pipe)
            except Exception:  # noqa: BLE001
                pad = None
            held: list = []
            if pad is not None:
                shared = _key_peaks(key)
                shared[work] = max(extra + pad, shared.get(work, 0))
                # an unpersistable key keeps this process's table only, where max wins: nothing lower can be stored
                # later
                if stored_key is None:
                    held.append(True)
                else:
                    try:
                        from .diffusion_measured_peaks import record_work
                        record_work(stored_key, work, extra + pad, held = held)
                    except Exception:  # noqa: BLE001 - persistence is best-effort
                        pass
            if held:
                # the store holds this sample: it covers the graphs the request recorded; unpersisted, they stay dirty
                module._unsloth_video_graphs = graphs
                module._unsloth_video_graph_gen = generation
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
            torch.cuda.empty_cache()
            _keep_groups_resident(module, room, dev, logger)
            promoted = max(0, resident_mib(module) - after_release)
        kept = resident_mib(module)
        dev = _open_window(module, dev)
        torch.cuda.reset_peak_memory_stats(dev)
        module._unsloth_video_pending = (work, int(torch.cuda.memory_allocated(dev)), dev)
        module._unsloth_video_failed_captures = _failed_captures()
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
