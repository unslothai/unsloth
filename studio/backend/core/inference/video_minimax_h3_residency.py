# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keeps as much of a MiniMax-H3 denoiser on device as fits: top-level group first, then a prefix of
its block groups; re-fitted per request; demotion is free; UNSLOTH_H3_DIT_RESIDENT=0 keeps all
streamed."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

H3_DIT_RESIDENT_ENV = "UNSLOTH_H3_DIT_RESIDENT"

# Decimal GB each H3 render phase needs on top of the resident set (B200 peaks).
H3_STREAM_WINDOW_GB = 1.5
#   per million pixel-frames
H3_ACTIVATION_GB_PER_MPIXEL_FRAME = 0.08
#   fraction of the activations: 1-2 GB MLP buffers fragment the caching allocator (OOM at 1344x768x124 without it)
H3_FRAGMENTATION_FRACTION = 0.5
H3_VAE_DECODE_GB = 6.0
H3_PHASE_OVERHEAD_GB = 1.8
#   top-level group: 0.81 GB int8
H3_TOP_LEVEL_GB = 1.0


def h3_dit_resident_enabled() -> bool:
    return str(os.environ.get(H3_DIT_RESIDENT_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def h3_phase_need_gb(
    width: int,
    height: int,
    num_frames: int,
    *,
    te_streamed_gb: float,
    fragmentation: bool = True,
    top_gb: float = 0.0,
) -> float:
    """Device GB a request needs beyond the resident set: its largest phase. ``fragmentation`` is slack only the
    residency plan keeps; the refusal floor leaves it out so residency never refuses what the streamed tier admitted."""
    activations = H3_ACTIVATION_GB_PER_MPIXEL_FRAME * width * height * num_frames / 1_000_000
    slack = H3_FRAGMENTATION_FRACTION if fragmentation else 0.0
    denoise = H3_STREAM_WINDOW_GB + top_gb + activations * (1.0 + slack)
    from .video_minimax_h3_sol_vae import sol_decode_need_gb

    # 0.0 unless UNSLOTH_H3_SOL_VAE=1: Sol's weights + one whole-clip decode peak
    decode = max(H3_VAE_DECODE_GB, sol_decode_need_gb(width, height, num_frames))
    return max(te_streamed_gb, denoise, decode) + H3_PHASE_OVERHEAD_GB


def _tensor_bytes(tensor: Any) -> int:
    try:
        from .diffusion_prequant import tensor_payload_bytes
        return int(tensor_payload_bytes(tensor))
    except Exception:  # noqa: BLE001
        return int(tensor.numel()) * int(tensor.element_size())


def _group_tensors(group: Any) -> list:
    out: list = []
    seen: set = set()
    for module in getattr(group, "modules", None) or ():
        for tensor in list(module.parameters()) + list(module.buffers()):
            if id(tensor) not in seen:
                seen.add(id(tensor))
                out.append(tensor)
    for tensor in list(getattr(group, "parameters", None) or ()) + list(
        getattr(group, "buffers", None) or ()
    ):
        if id(tensor) not in seen:
            seen.add(id(tensor))
            out.append(tensor)
    return out


def group_payload_bytes(group: Any) -> int:
    return sum(_tensor_bytes(t) for t in _group_tensors(group))


def _group_hook(module: Any) -> Any:
    try:
        from diffusers.hooks import group_offloading as go

        registry = getattr(module, "_diffusers_hook", None)
        if registry is None:
            return None
        return registry.get_hook(getattr(go, "_GROUP_OFFLOADING", "group_offloading"))
    except Exception:  # noqa: BLE001
        return None


def h3_offload_groups(transformer: Any) -> tuple[Optional[Any], list]:
    """(top-level group, block groups in execution order) of a block-streamed H3 denoiser; (None, []) if not streamed."""
    top_hook = _group_hook(transformer)
    top = getattr(top_hook, "group", None)
    blocks: list = []
    seen: set = set()
    for block in getattr(transformer, "transformer_blocks", None) or ():
        group = getattr(_group_hook(block), "group", None)
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            blocks.append(group)
    return top, blocks


def _noop() -> None:
    return None


def _resident_onload(group: Any) -> Any:
    """A resident group's onload kicks the prefetch, else the first streamed block behind it is never prefetched."""
    prefetcher = getattr(group, "_unsloth_prefetcher", None)
    kick = getattr(prefetcher, "kick", None)
    if not callable(kick):
        return _noop

    def onload_(*_a: Any, **_k: Any) -> None:
        kick()

    return _outside_inference_mode(onload_)


def is_resident(group: Any) -> bool:
    return bool(getattr(group, "_unsloth_resident", False))


# A resident group's host copy is only read again on demote, so it is released while the group is resident and
# rebuilt from the device copy (bit-exact) on demote: the H3 L4 load gives those ~11 GB to the conditioner instead.
H3_RESIDENT_UNPIN_ENV = "UNSLOTH_H3_RESIDENT_UNPIN"
_RELEASED_ATTR = "_unsloth_host_released"
# Floor (MiB) of the available host RAM a rebuild leaves; the load's own reserve (``encoder_resident_keep_mib``, what
# the conditioner share was sized to leave) applies when larger. Below it the host-yield callback gives RAM back first.
H3_REBUILD_KEEP_MIB = 2048


def h3_resident_unpin_enabled() -> bool:
    return str(os.environ.get(H3_RESIDENT_UNPIN_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _host_leaf_bytes(tensor: Any) -> int:
    return sum(int(leaf.numel()) * int(leaf.element_size()) for leaf in _host_leaves(tensor))


def released_host_bytes(group: Any) -> int:
    return int(getattr(group, _RELEASED_ATTR, 0) or 0)


def release_host_copy(group: Any, logger: Any = None) -> int:
    """Drop the host copy of a resident ``group`` (its weights live on the device); returns bytes released, 0 if kept.
    Kept when off, when a background pin still owns the group, or when any weight still reads its host copy."""
    if not h3_resident_unpin_enabled() or not is_resident(group) or released_host_bytes(group):
        return 0
    cpu = getattr(group, "cpu_param_dict", None)
    if not isinstance(cpu, dict) or not cpu:
        return 0
    pinner = getattr(group, "_unsloth_background_pin", None)
    wait = getattr(pinner, "wait", None)
    if callable(wait):
        try:
            wait(group)
        except Exception:  # noqa: BLE001 -- unknown pin state: keep the copy
            return 0
    tensors = _group_tensors(group)
    if any(t not in cpu for t in tensors):
        return 0
    for t in tensors:
        host = cpu[t]
        if getattr(host, "device", None) is None or host.device.type != "cpu" or _reads(t, host):
            return 0  # a weight still points at its host copy: not on the device
    nbytes = sum(_host_leaf_bytes(cpu[t]) for t in tensors)
    group.cpu_param_dict = {}
    setattr(group, _RELEASED_ATTR, max(1, nbytes))
    return nbytes


def _host_copy_of(tensor: Any) -> Any:
    """Host copy of a device ``tensor`` in the layout ``_pin_host_tensor`` gives (pinned where CUDA is up)."""
    import torch
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if is_traceable_wrapper_subclass(tensor):
        names, ctx = tensor.__tensor_flatten__()
        inner = {name: _host_copy_of(getattr(tensor, name)) for name in names}
        return type(tensor).__tensor_unflatten__(inner, ctx, tensor.size(), tensor.stride())
    data = tensor.detach()
    if type(data) is torch.nn.Parameter:
        data = data.data
    host = data.to("cpu", copy = True)
    if not torch.cuda.is_available():
        return host
    return _pin_host_tensor(host)


def rebuild_host_copy(
    group: Any,
    *,
    host_yield: Any = None,
    logger: Any = None,
) -> int:
    """Re-create a released group's host copy from its device weights (an exact copy). ``host_yield(nbytes)`` is asked
    for RAM first when the copy would leave under ``H3_REBUILD_KEEP_MIB`` available. Returns bytes rebuilt."""
    need = released_host_bytes(group)
    if not need:
        return 0
    import torch

    if host_yield is not None:
        try:
            from .diffusion_small_host import encoder_resident_keep_mib, host_ram_mib
            total, available = host_ram_mib()
            if available is not None:
                keep = max(H3_REBUILD_KEEP_MIB, int(encoder_resident_keep_mib(total)))
                deficit = (need >> 20) + keep - int(available)
                if deficit > 0:
                    host_yield(deficit << 20)
        except Exception as exc:  # noqa: BLE001 -- rebuild anyway
            if logger is not None:
                logger.warning(
                    "video.h3_residency: host RAM check before a rebuild failed: %s", exc
                )
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    host = {}
    with torch.no_grad():
        for t in _group_tensors(group):
            source = t if _is_wrapper(t) else t.data
            host[t] = _host_copy_of(source)
    group.cpu_param_dict = host
    setattr(group, _RELEASED_ATTR, 0)
    return need


def _is_wrapper(tensor: Any) -> bool:
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass
    return is_traceable_wrapper_subclass(tensor)


# A slab still alive only for a few small weights of other groups (first-fit packs them into earlier slabs' tails)
# is compacted: those weights move to their own pinned copies so the slab is freed. Fraction of the slab still used.
H3_COMPACT_MAX_LIVE_FRACTION = 0.25
_H3_COMPACT_MIN_SLAB = 64 << 20
# The compaction's new pinned copies exist before any old slab is freed and pinned pages cannot be reclaimed: an L4 at
# a 16 GiB budget was OOM-killed when the first request's re-fit re-pinned 13.2 GB of whole weights next to a
# conditioner share sized to leave only the load's reserve. Priced (default): only the inner tensors that live in the
# compacted slab move, the move is checked against the load's reserve (the conditioner share yields first) and slabs
# that do not fit wait for a later re-fit. ``UNSLOTH_H3_COMPACT_PRICED=0`` restores the unpriced whole-weight move.
H3_COMPACT_PRICED_ENV = "UNSLOTH_H3_COMPACT_PRICED"


def h3_compact_priced_enabled() -> bool:
    return str(os.environ.get(H3_COMPACT_PRICED_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _storage_key(leaf: Any) -> Optional[int]:
    try:
        if leaf.device.type != "cpu" or leaf.numel() == 0:
            return None
        return int(leaf.untyped_storage().data_ptr())
    except Exception:  # noqa: BLE001 -- no storage (meta / sparse): never compacted
        return None


def _pin_cost(leaf: Any) -> int:
    """Pinned bytes ``_pin_host_tensor`` takes for plain ``leaf``: page-exact when registered, else the host
    allocator's power-of-two block."""
    import torch

    from .diffusion_pinned_arena import _packable, _pow2_ceil

    nbytes = int(leaf.numel()) * int(leaf.element_size())
    if type(leaf) is torch.Tensor and _packable(leaf):
        return -(-nbytes // 4096) * 4096
    return _pow2_ceil(nbytes)


def _repin_leaves_in(
    src: Any,
    keys: set,
    depth: int = 0,
) -> tuple[Any, int]:
    """``src`` with only its inner tensors stored in one of ``keys`` re-pinned (exact copies); the others are shared.
    Returns ``(src, 0)`` when none is."""
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if depth < 4 and is_traceable_wrapper_subclass(src):
        names, ctx = src.__tensor_flatten__()
        inner = {}
        moved = 0
        for name in names:
            inner[name], n = _repin_leaves_in(getattr(src, name), keys, depth + 1)
            moved += n
        if not moved:
            return src, 0
        return type(src).__tensor_unflatten__(inner, ctx, src.size(), src.stride()), moved
    if _storage_key(src) not in keys:
        return src, 0
    return _pin_host_tensor(src), int(src.numel()) * int(src.element_size())


def _compact_headroom_bytes(cost: int, host_yield: Any, logger: Any) -> Optional[int]:
    """Bytes of new pinned copies the host can take above the load's reserve, after asking ``host_yield`` (the
    conditioner's resident share) for the deficit. None when host RAM is unreadable (no bound)."""
    from .diffusion_small_host import encoder_resident_keep_mib, host_ram_mib

    total, available = host_ram_mib()
    if available is None:
        return None
    keep = max(H3_REBUILD_KEEP_MIB, int(encoder_resident_keep_mib(total)))
    deficit = -(-cost >> 20) + keep - int(available)
    if deficit > 0 and host_yield is not None:
        try:
            host_yield(deficit << 20)
        except Exception as exc:  # noqa: BLE001 -- price against what is available now
            if logger is not None:
                logger.warning(
                    "video.h3_residency: host RAM yield before a compaction failed: %s", exc
                )
        _total, now = host_ram_mib()
        if now is not None:
            available = now
    return max(0, int(available) - keep) << 20


def compact_host_slabs(
    groups: list,
    logger: Any = None,
    *,
    host_yield: Any = None,
    report: Optional[dict] = None,
) -> int:
    """Move the host weights that keep a mostly-released pinned slab alive into fresh pinned copies (exact copies,
    re-pointed like the arena's own pin), so the slab returns to the OS. Returns bytes of slabs freed (estimated).
    ``report["deferred"]``: eligible slabs left for a later call because their copies did not fit the host reserve."""
    from .diffusion_pinned_arena import _repoint

    priced = h3_compact_priced_enabled()
    live: dict = {}
    sizes: dict = {}
    for group in groups:
        cpu = getattr(group, "cpu_param_dict", None) or {}
        for tensor, host in list(cpu.items()):
            for leaf in _host_leaves(host):
                key = _storage_key(leaf)
                if key is None:
                    continue
                sizes[key] = int(leaf.untyped_storage().nbytes())
                live.setdefault(key, []).append(
                    (group, tensor, host, int(leaf.numel()) * int(leaf.element_size()), leaf)
                )
    eligible = [
        key
        for key, entries in live.items()
        if sizes[key] >= _H3_COMPACT_MIN_SLAB
        and sum(e[3] for e in entries) <= H3_COMPACT_MAX_LIVE_FRACTION * sizes[key]
    ]
    chosen = eligible
    deferred = 0
    if priced and eligible:
        costs = {key: sum(_pin_cost(e[4]) for e in live[key]) for key in eligible}
        room = _compact_headroom_bytes(sum(costs.values()), host_yield, logger)
        if room is not None:
            chosen = []
            for key in eligible:
                if costs[key] <= room:
                    chosen.append(key)
                    room -= costs[key]
            deferred = len(eligible) - len(chosen)
    keys = set(chosen)
    freed = moved = 0
    for key in chosen:
        for group, tensor, host, _n, _leaf in live[key]:
            cpu = group.cpu_param_dict
            if cpu.get(tensor) is not host:
                continue  # already moved through another leaf
            if priced:
                new, n = _repin_leaves_in(host, keys)
            else:
                new, n = _pin_host_tensor(host), _host_leaf_bytes(host)
            cpu[tensor] = new
            if _reads(tensor, host):
                _repoint(tensor, new)
            moved += n
        freed += sizes[key]
    del live
    if report is not None:
        report["deferred"] = deferred
    if deferred and logger is not None:
        logger.info(
            "video.h3_residency: %d pinned slab(s) left for a later re-fit (their copies do not fit the host reserve)",
            deferred,
        )
    if freed:
        try:
            from .diffusion_offload_prefetch import bump_placement_epoch
            bump_placement_epoch()
        except Exception:  # noqa: BLE001
            pass
        if logger is not None:
            logger.info(
                "video.h3_residency: compacted %.2f GB of pinned slabs (moved %.1f MB of other blocks' weights)",
                freed / 1e9,
                moved / 1e6,
            )
    return freed


def _host_empty_cache() -> None:
    try:
        from .diffusion_memory import _host_empty_cache as empty
        empty()
    except Exception:  # noqa: BLE001
        pass


# A fully resident initial fit skips the up-front block pin (18 s, 20 GB pinned transient on 48 GB); leftovers pin after.
H3_DEFER_RESIDENT_PIN_ENV = "UNSLOTH_H3_DEFER_RESIDENT_PIN"
_PIN_DEFERRED_ATTR = "_unsloth_pin_deferred"


def h3_defer_resident_pin_enabled() -> bool:
    return str(os.environ.get(H3_DEFER_RESIDENT_PIN_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _onload_pageable(group: Any) -> bool:
    """diffusers' stream onload of a pin-deferred group, read from the pageable host copy without pinning."""
    from contextlib import nullcontext

    cpu = getattr(group, "cpu_param_dict", None)
    process = getattr(group, "_process_tensors_from_modules", None)
    if not isinstance(cpu, dict) or not cpu or not callable(process):
        return False
    import torch

    stream = getattr(group, "stream", None)
    if stream is not None:
        stream.synchronize()
    current = torch.cuda.current_stream() if stream is not None else None
    with nullcontext() if stream is None else torch.cuda.stream(stream):
        process(cpu, default_stream = current)
    return True


def _host_alias(tensor: Any) -> Any:
    """A host-copy object distinct from ``tensor`` over the same storage. diffusers' lazy copy of a torchao weight is
    the weight itself, so the onload's swap_tensors would carry that "host copy" to the device as well."""
    if _is_wrapper(tensor):
        names, ctx = tensor.__tensor_flatten__()
        inner = {name: getattr(tensor, name) for name in names}
        return type(tensor).__tensor_unflatten__(inner, ctx, tensor.size(), tensor.stride())
    return tensor.detach()


def defer_resident_block_pins(
    residency: "H3Residency",
    budget_bytes: int,
    logger: Any = None,
) -> bool:
    """Engage the deferral if ``budget_bytes`` holds top + every block; call before the prefetch and initial fit."""
    if not h3_defer_resident_pin_enabled() or not h3_resident_unpin_enabled():
        return False
    if not residency.usable:
        return False
    top, n = residency.plan(budget_bytes)
    if (residency.top is not None and not top) or n != len(residency.blocks):
        return False
    for group in residency.blocks:
        cpu = getattr(group, "cpu_param_dict", None)
        if (
            not getattr(group, "low_cpu_mem_usage", False)
            or getattr(group, "stream", None) is None
            or getattr(group, "offload_to_disk_path", None)
            or not isinstance(cpu, dict)
            or not cpu
            or is_resident(group)
        ):
            return False
    try:
        aliases = [
            {t: _host_alias(src) for t, src in group.cpu_param_dict.items() if src is t}
            for group in residency.blocks
        ]
    except Exception:  # noqa: BLE001 -- unknown wrapper: pin as the lazy path does
        return False
    for group, alias in zip(residency.blocks, aliases):
        group.cpu_param_dict.update(alias)
        group.low_cpu_mem_usage = False
        setattr(group, _PIN_DEFERRED_ATTR, True)
    if logger is not None:
        logger.info(
            "video.h3_block_pin: %d block groups will be resident, pin deferred (%.2f GB)",
            len(residency.blocks),
            sum(residency.block_bytes) / 1e9,
        )
    return True


def pin_deferred_block_groups(transformer: Any, logger: Any = None) -> tuple[int, int]:
    """Pin the deferred blocks the initial fit left streamed; returns (groups pinned, block groups)."""
    _top, blocks = h3_offload_groups(transformer)
    left = [g for g in blocks if getattr(g, _PIN_DEFERRED_ATTR, False) and not is_resident(g)]
    if not left:
        return 0, len(blocks)
    for group in left:
        group.low_cpu_mem_usage = True
        setattr(group, _PIN_DEFERRED_ATTR, False)
    if logger is not None:
        logger.info(
            "video.h3_block_pin: %d of %d pin-deferred block groups stay streamed, pinning them now",
            len(left),
            len(blocks),
        )
    return pin_streamed_block_groups(transformer, logger = logger)


def make_resident(group: Any) -> None:
    """Onload ``group`` once and make its hooks' onload / offload no-ops, so it stays on the device."""
    import torch

    if is_resident(group):
        return
    promoted = False
    if getattr(group, _PIN_DEFERRED_ATTR, False):
        try:
            promoted = _outside_inference_mode(_onload_pageable)(group)
        except Exception:  # noqa: BLE001 -- diffusers' onload re-copies every tensor from the host copy
            promoted = False
    if not promoted:
        # Instance attribute: an instance override is an onload path too, except the prefetcher's own.
        prefetcher = getattr(group, "_unsloth_prefetcher", None)
        owns = getattr(prefetcher, "owns", None)
        if callable(owns) and owns(group):
            _outside_inference_mode(type(group).onload_)(group)
        else:
            group.onload_()
    stream = getattr(group, "stream", None)
    if stream is not None:
        stream.synchronize()
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    group._unsloth_saved_io = (group.__dict__.get("onload_"), group.__dict__.get("offload_"))
    group.onload_ = _resident_onload(group)
    group.offload_ = _noop
    group._unsloth_resident = True
    if getattr(group, _PIN_DEFERRED_ATTR, False):
        setattr(group, _PIN_DEFERRED_ATTR, False)


def demote(
    group: Any,
    *,
    host_yield: Any = None,
    logger: Any = None,
    rebuild: bool = True,
) -> None:
    """Undo ``make_resident``: restore the streaming onload / offload and move the weights back to the host copy.
    A released host copy is rebuilt from the device first; ``rebuild=False`` (unload) drops the weights instead."""
    import torch

    if not is_resident(group):
        return
    if released_host_bytes(group):
        if rebuild:
            nbytes = rebuild_host_copy(group, host_yield = host_yield, logger = logger)
            if logger is not None:
                logger.info(
                    "video.h3_residency: rebuilt a demoted block's host copy from the device (%.2f GB)",
                    nbytes / 1e9,
                )
        else:
            group._unsloth_resident = False
            return
    saved_on, saved_off = getattr(group, "_unsloth_saved_io", (None, None))
    for name, saved in (("onload_", saved_on), ("offload_", saved_off)):
        if saved is not None:
            setattr(group, name, saved)
        else:
            group.__dict__.pop(name, None)
    group._unsloth_resident = False
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    group.offload_()


class H3Residency:
    """The resident set of one streamed H3 denoiser: the top-level group and a prefix of its block groups."""

    def __init__(
        self,
        transformer: Any,
        device: Any,
        *,
        logger: Any = None,
    ) -> None:
        self.top, self.blocks = h3_offload_groups(transformer)
        self.device = device
        self.logger = logger
        self.top_bytes = group_payload_bytes(self.top) if self.top is not None else 0
        self.block_bytes = [group_payload_bytes(g) for g in self.blocks]
        self.max_blocks = 0
        # host_yield(nbytes): frees host RAM (the conditioner's resident share) before a demoted block is rebuilt
        self.host_yield: Any = None
        # host_top_up(): hands RAM a re-fit released to the conditioner's resident share (after the load only)
        self.host_top_up: Any = None
        self.released_bytes = 0
        self.topped_bytes = 0
        # bytes of slabs the last ``apply`` compacted; a compaction that could not fit every slab retries next re-fit
        self.compacted_bytes = 0
        self.compact_pending = False
        # video_minimax_h3_te.H3TeDeviceShare
        self.te_share: Any = None

    @property
    def usable(self) -> bool:
        return bool(self.blocks)

    def resident_blocks(self) -> int:
        n = 0
        for group in self.blocks:
            if not is_resident(group):
                break
            n += 1
        return n

    def resident_bytes(self) -> int:
        total = self.top_bytes if (self.top is not None and is_resident(self.top)) else 0
        return total + sum(b for g, b in zip(self.blocks, self.block_bytes) if is_resident(g))

    def device_bytes(self) -> int:
        te = getattr(self, "te_share", None)
        return self.resident_bytes() + (te.resident_bytes() if te is not None else 0)

    def te_budget(self, budget_bytes: int, top: bool, n: int) -> int:
        """Zero unless the whole denoiser is resident: it is read every step, the conditioner once."""
        if n < len(self.blocks) or (self.top is not None and not top):
            return 0
        planned = (self.top_bytes if top else 0) + sum(self.block_bytes[:n])
        return max(0, int(budget_bytes) - planned)

    def plan(
        self,
        budget_bytes: int,
        *,
        cap: Optional[int] = None,
    ) -> tuple[bool, int]:
        """(top resident, number of resident blocks) that fit ``budget_bytes``; top first, then the block prefix."""
        if budget_bytes <= 0 or self.top is None and not self.blocks:
            return False, 0
        top = self.top is not None and self.top_bytes <= budget_bytes
        left = budget_bytes - (self.top_bytes if top else 0)
        n = 0
        limit = len(self.blocks) if cap is None else min(cap, len(self.blocks))
        while n < limit and self.block_bytes[n] <= left:
            left -= self.block_bytes[n]
            n += 1
        return top, n

    def apply(self, top: bool, n: int) -> None:
        """Demote first (frees the device), then promote."""
        import torch

        for i in range(len(self.blocks) - 1, n - 1, -1):
            demote(self.blocks[i], host_yield = getattr(self, "host_yield", None), logger = self.logger)
        if self.top is not None and not top:
            demote(self.top, host_yield = getattr(self, "host_yield", None), logger = self.logger)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.top is not None and top:
            make_resident(self.top)
        for i in range(n):
            make_resident(self.blocks[i])
        # Block groups only: the top-level group keeps its own pinned copy (it is re-pinned by its own path).
        freed = sum(release_host_copy(self.blocks[i], logger = self.logger) for i in range(n))
        self.compacted_bytes = 0
        if freed or getattr(self, "compact_pending", False):
            groups = list(self.blocks) + ([self.top] if self.top is not None else [])
            report: dict = {}
            try:
                self.compacted_bytes = compact_host_slabs(
                    groups,
                    logger = self.logger,
                    host_yield = getattr(self, "host_yield", None),
                    report = report,
                )
            except Exception as exc:  # noqa: BLE001 -- the released copies are still dropped
                if self.logger is not None:
                    self.logger.warning("video.h3_residency: slab compaction skipped: %s", exc)
            self.compact_pending = bool(report.get("deferred"))
            _host_empty_cache()
        self.released_bytes = sum(released_host_bytes(g) for g in self.blocks)

    def fit(
        self,
        budget_bytes: int,
        *,
        initial: bool = False,
    ) -> tuple[bool, int]:
        """Re-fit the resident set to ``budget_bytes`` (the device bytes the resident set may occupy)."""
        top, n = self.plan(budget_bytes)
        te = getattr(self, "te_share", None)
        te_left = self.te_budget(budget_bytes, top, n) if te is not None else 0
        te_n = te.plan(te_left) if te is not None else 0
        if te is not None:
            # the conditioner gives its share back before any denoiser block is promoted
            te.shrink(te_n)
        self.apply(top, n)
        if te is not None:
            try:
                te.grow(te_n)
            except Exception as exc:  # noqa: BLE001 -- the conditioner streams as before
                te.shrink(0)
                if self.logger is not None:
                    self.logger.warning("video.h3_te_resident_device: staying streamed: %s", exc)
            te.report(te_left, force = initial)
        if initial:
            self.max_blocks = n
        released = getattr(self, "released_bytes", 0)
        top_up = getattr(self, "host_top_up", None)
        # A compaction may have taken the conditioner share back to fit its copies: it is re-sized after the pins.
        compacted = getattr(self, "compacted_bytes", 0) and h3_compact_priced_enabled()
        if (released > getattr(self, "topped_bytes", 0) or compacted) and callable(top_up):
            try:
                top_up()
            except Exception as exc:  # noqa: BLE001 -- the conditioner stays mapped
                if self.logger is not None:
                    self.logger.warning("video.h3_residency: conditioner top-up failed: %s", exc)
            self.topped_bytes = released
        elif released < getattr(self, "topped_bytes", 0):
            self.topped_bytes = released
        if self.logger is not None:
            self.logger.info(
                "video.h3_residency: %d of %d denoiser blocks resident (+ top-level %s), %.2f GB on the device, "
                "budget %.2f GB, %.2f GB of resident host copies released",
                n,
                len(self.blocks),
                "resident" if top else "streamed",
                self.resident_bytes() / 1e9,
                budget_bytes / 1e9,
                self.released_bytes / 1e9,
            )
        return top, n


def release_all(residency: "H3Residency", *, logger: Any = None) -> None:
    """Return every group of ``residency`` to streaming, including one whose onload failed partway. Never raises."""
    import torch

    te = getattr(residency, "te_share", None)
    if te is not None:
        try:
            te.shrink(0)
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("video.h3_te_resident_device: could not release the share: %s", exc)
    groups = list(reversed(residency.blocks))
    if residency.top is not None:
        groups.append(residency.top)
    for group in groups:
        try:
            if is_resident(group):
                demote(group, host_yield = getattr(residency, "host_yield", None), logger = logger)
            else:
                group.offload_()
        except Exception as exc:  # noqa: BLE001
            if logger is not None:
                logger.warning("video.h3_residency: could not release a group: %s", exc)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def h3_held_host_bytes(*modules: Any) -> dict[str, int]:
    """Host bytes the given components hold (tensors and group-offload host copies), pinned vs pageable, de-duplicated
    by storage; pinned bytes rounded up to a power of two as torch's host allocator reserves them."""
    seen: set = set()
    pinned = 0
    pageable = 0

    def _add(tensor: Any) -> None:
        nonlocal pinned, pageable
        flatten = getattr(tensor, "__tensor_flatten__", None)
        if callable(flatten) and type(tensor).__name__ not in ("Tensor", "Parameter"):
            try:
                for name in flatten()[0]:
                    _add(getattr(tensor, name))
                return
            except Exception:  # noqa: BLE001
                pass
        try:
            if tensor.device.type != "cpu":
                return
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key in seen or storage.nbytes() == 0:
                return
            seen.add(key)
            nbytes = int(storage.nbytes())
            if tensor.is_pinned():
                pinned += 1 << (nbytes - 1).bit_length()
            else:
                pageable += nbytes
        except Exception:  # noqa: BLE001
            return

    for module in modules:
        if module is None:
            continue
        try:
            for tensor in list(module.parameters()) + list(module.buffers()):
                _add(tensor)
            for sub in module.modules():
                group = getattr(_group_hook(sub), "group", None)
                for tensor in (getattr(group, "cpu_param_dict", None) or {}).values():
                    _add(tensor)
        except Exception:  # noqa: BLE001
            continue
    return {"pinned": pinned, "pageable": pageable}


H3_COMPILE_BELOW_HOOKS_ENV = "UNSLOTH_H3_COMPILE_BELOW_HOOKS"


def _is_original_forward(fn: Any, module: Any) -> bool:
    return getattr(fn, "__self__", None) is module and getattr(fn, "__func__", None) is getattr(
        type(module), "forward", None
    )


def compile_blocks_below_offload_hooks(transformer: Any, logger: Any = None) -> int:
    """Compile each streamed block's forward, not _call_impl, so hook state stays out of guards; UNSLOTH_H3_COMPILE_BELOW_HOOKS=0 keeps old placement."""
    if str(os.environ.get(H3_COMPILE_BELOW_HOOKS_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return 0
    kwargs = getattr(transformer, "_unsloth_regional_compile_kwargs", None)
    if not isinstance(kwargs, dict):
        return 0
    import torch

    guard = getattr(transformer, "_unsloth_compile_guard", None)
    moved = 0
    for module in list(transformer.modules()):
        if getattr(module, "_compiled_call_impl", None) is None or _group_hook(module) is None:
            continue
        registry = getattr(module, "_diffusers_hook", None)
        fn_refs = list(getattr(registry, "_fn_refs", None) or ())
        target = next(
            (ref for ref in fn_refs if _is_original_forward(getattr(ref, "forward", None), module)),
            None,
        )
        if target is None:
            continue
        original = target.forward
        compiled = torch.compile(original, **kwargs)
        if guard is not None and callable(getattr(guard, "wrap", None)):
            target.forward = guard.wrap(compiled, original, transformer)
            guard.restores.append(lambda ref = target, fn = original: setattr(ref, "forward", fn))
        else:
            target.forward = compiled
        module._compiled_call_impl = None
        moved += 1
    if moved and logger is not None:
        logger.info(
            "video.h3_compile: %d streamed denoiser blocks compile below their offload hooks", moved
        )
    return moved


H3_TOP_GROUP_PIN_ENV = "UNSLOTH_H3_TOP_GROUP_PIN"


def pin_streamed_top_level_group(transformer: Any, logger: Any = None) -> bool:
    """diffusers builds top group without a stream, so each forward uploads it from pageable memory; UNSLOTH_H3_TOP_GROUP_PIN=0 keeps diffusers' group."""
    if str(os.environ.get(H3_TOP_GROUP_PIN_ENV, "")).strip().lower() in ("0", "off", "false", "no"):
        return False
    from .video_minimax_h3_te import h3_te_pin_allowed

    top, blocks = h3_offload_groups(transformer)
    if not h3_te_pin_allowed(
        group_payload_bytes(top) if top is not None else 0, getattr(top, "onload_device", None)
    ):
        if logger is not None:
            logger.info(
                "video.h3_top_group: left as diffusers built it (pinning not allowed on this host)"
            )
        return False
    stream = next(
        (getattr(g, "stream", None) for g in blocks if getattr(g, "stream", None) is not None), None
    )
    why = None
    if top is None:
        why = "no top-level group"
    elif stream is None:
        why = "the blocks have no copy stream"
    elif getattr(top, "stream", None) is not None:
        why = "the top-level group already streams"
    elif getattr(top, "offload_to_disk_path", None) or not callable(
        getattr(top, "_init_cpu_param_dict", None)
    ):
        why = "disk offload / unknown diffusers group"
    elif getattr(top, "_unsloth_pinned_top", False) or is_resident(top):
        why = "another path already pinned or holds it"
    if why is not None:
        if logger is not None:
            logger.info("video.h3_top_group: left as diffusers built it (%s)", why)
        return False
    saved = (
        top.stream,
        top.low_cpu_mem_usage,
        top.record_stream,
        top.non_blocking,
        top.cpu_param_dict,
    )
    try:
        from .diffusion_pinned_arena import pinned_arena_for_group_offload

        top.stream = stream
        top.low_cpu_mem_usage = False
        top.record_stream = True
        # Blocking: nothing prefetches the top-level group.
        top.non_blocking = False
        with pinned_arena_for_group_offload():
            top.cpu_param_dict = top._init_cpu_param_dict()
    except Exception as exc:  # noqa: BLE001 -- keep diffusers' group as it was
        (
            top.stream,
            top.low_cpu_mem_usage,
            top.record_stream,
            top.non_blocking,
            top.cpu_param_dict,
        ) = saved
        if logger is not None:
            logger.warning(
                "video.h3_top_group: pinned copy refused, keeping the pageable group: %s", exc
            )
        return False
    if logger is not None:
        logger.info(
            "video.h3_top_group: the streamed denoiser's top-level group (%.2f GB) onloads from a pinned copy",
            group_payload_bytes(top) / 1e9,
        )
    return True


H3_BLOCK_PIN_ENV = "UNSLOTH_H3_STREAM_BLOCK_PIN"


def pin_streamed_block_groups(transformer: Any, logger: Any = None) -> tuple[int, int]:
    """Pin a lazily streamed denoiser's block groups in place, group by group, while the pin budget holds.

    The up-front plan prices a pin as extra host RAM, so 12 / 16 GB hosts stream H3 lazily: every onload re-pins
    behind a host sync and the stream prefetch stays off. The arena re-points each weight at its slab, freeing the
    pageable copy; the budget is re-read per group, so a surviving pageable copy stops the pin.
    Returns (groups pinned, block groups). ``UNSLOTH_H3_STREAM_BLOCK_PIN=0`` off."""
    if str(os.environ.get(H3_BLOCK_PIN_ENV, "")).strip().lower() in ("0", "off", "false", "no"):
        return 0, 0
    _top, blocks = h3_offload_groups(transformer)
    lazy = [
        g
        for g in blocks
        if getattr(g, "low_cpu_mem_usage", False)
        and getattr(g, "stream", None) is not None
        and not getattr(g, "offload_to_disk_path", None)
        and callable(getattr(g, "_init_cpu_param_dict", None))
        and not is_resident(g)
    ]
    if not lazy:
        return 0, len(blocks)
    from .diffusion_memory import _pinned_memory_capped
    from .diffusion_pinned_arena import pin_arena_enabled, pinned_arena_for_group_offload
    from .video_minimax_h3_te import h3_te_pin_allowed

    # Per-tensor pin_memory() keeps the pageable copy until the first offload; capped hosts pin each onload anyway.
    if not pin_arena_enabled() or _pinned_memory_capped():
        return 0, len(blocks)
    pinned = 0
    with pinned_arena_for_group_offload() as arena:
        if arena is None:
            return 0, len(blocks)
        for group in lazy:
            tensors = _group_tensors(group)
            if any(getattr(t, "device", None) is None or t.device.type != "cpu" for t in tensors):
                break
            if not h3_te_pin_allowed(
                group_payload_bytes(group), getattr(group, "onload_device", None)
            ):
                break
            saved = (group.low_cpu_mem_usage, group.cpu_param_dict)
            try:
                group.low_cpu_mem_usage = False
                group.cpu_param_dict = group._init_cpu_param_dict()
            except Exception as exc:  # noqa: BLE001 -- this group and the rest stay lazily pinned
                group.low_cpu_mem_usage, group.cpu_param_dict = saved
                if logger is not None:
                    logger.warning("video.h3_block_pin: stopped at a group: %s", exc)
                break
            pinned += 1
        payload, reserved = arena.payload_bytes, arena.reserved_bytes
    if pinned == len(blocks) and len(lazy) == len(blocks):
        transformer._unsloth_pin_arena_bytes = (payload, reserved)
    if logger is not None:
        logger.info(
            "video.h3_block_pin: %d of %d streamed block groups pinned in place (%.2f GB in %.2f GB of slabs)",
            pinned,
            len(blocks),
            payload / 1e9,
            reserved / 1e9,
        )
    return pinned, len(blocks)


H3_BLOCK_BACKGROUND_PIN_ENV = "UNSLOTH_H3_BACKGROUND_BLOCK_PIN"
# Groups queued ahead of the swap: keeps 8 copy threads busy while bounding pageable sources held beside pins.
_H3_PIN_LOOKAHEAD = 2


def _host_leaves(tensor: Any) -> list:
    from .diffusion_pinned_arena import _plain_leaves
    return [leaf for leaf in _plain_leaves(tensor) if leaf.numel() > 0]


def _pin_host_tensor(src: Any) -> Any:
    """Pinned copy of ``src``; torchao wrapper subclasses are rebuilt from pinned inner tensors like ``PinnedArena.pin``."""
    import torch
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    from .diffusion_memory import registered_host_copy
    from .diffusion_pinned_arena import _packable

    if is_traceable_wrapper_subclass(src):
        names, ctx = src.__tensor_flatten__()
        inner = {name: _pin_host_tensor(getattr(src, name)) for name in names}
        return type(src).__tensor_unflatten__(inner, ctx, src.size(), src.stride())
    if type(src) is torch.Tensor and _packable(src):
        try:
            return registered_host_copy(src)
        except Exception:  # noqa: BLE001 -- registration refused: the allocator's pin for this tensor
            pass
    return src.pin_memory()


def _reads(tensor: Any, src: Any) -> bool:
    try:
        if getattr(tensor, "device", None) is None or tensor.device.type != "cpu":
            return False
        mine, theirs = _host_leaves(tensor), _host_leaves(src)
        return len(mine) == len(theirs) and all(
            a.data_ptr() == b.data_ptr() for a, b in zip(mine, theirs)
        )
    except Exception:  # noqa: BLE001 -- unreadable: leave it, its next offload re-points it
        return False


def _h3_block_pinner_cls() -> Any:
    from .diffusion_memory import _FAST_PIN_THREADS, _bump_placement_epoch, _GroupPinner
    from .diffusion_pinned_arena import _repoint

    class _H3BlockPinner(_GroupPinner):
        """Pins torchao weights too and re-reads the budget per group; refused groups are pinned per onload."""

        def __init__(
            self,
            module: Any,
            groups: list,
            device: Any,
            logger: Any = None,
            admit: Any = None,
        ):
            super().__init__(module, groups, device, logger, fast = True)
            self.admit = admit
            self.groups_pinned = 0
            self.stopped_at: Optional[int] = None
            # (seconds since start, pinned bytes) after each group
            self.timeline: list = []

        def _unpinned(self, group: Any) -> list:
            out = []
            for tensor, src in list(group.cpu_param_dict.items()):
                if getattr(src, "device", None) is None or src.device.type != "cpu":
                    continue
                leaves = _host_leaves(src)
                if leaves and not all(leaf.is_pinned() for leaf in leaves):
                    out.append((tensor, src))
            return out

        def _run(self) -> None:
            import time

            import torch

            start = time.perf_counter()
            failed = None
            pool = None
            try:
                if (
                    getattr(self.device, "type", None) == "cuda"
                    and getattr(self.device, "index", None) is not None
                ):
                    torch.cuda.set_device(self.device)
                from concurrent.futures import ThreadPoolExecutor

                pool = ThreadPoolExecutor(
                    max_workers = _FAST_PIN_THREADS,
                    thread_name_prefix = "unsloth-fast-pin",
                    initializer = torch.cuda.set_device,
                    initargs = (torch.cuda.current_device(),),
                )
                order = list(self.groups)
                queued: dict = {}
                nxt = 0
                for i, group in enumerate(order):
                    if self._stop.is_set():
                        break
                    while (
                        nxt < min(i + 1 + _H3_PIN_LOOKAHEAD, len(order)) and self.stopped_at is None
                    ):
                        ahead = order[nxt]
                        if self.admit is not None and not self.admit(ahead):
                            self.stopped_at = nxt
                            break
                        queued[id(ahead)] = [
                            (tensor, src, pool.submit(_pin_host_tensor, src))
                            for tensor, src in self._unpinned(ahead)
                        ]
                        nxt += 1
                    if id(group) not in queued:
                        break
                    placed = [
                        (tensor, src, self._fast_result(future, src))
                        for tensor, src, future in queued.pop(id(group))
                    ]
                    cpu = group.cpu_param_dict
                    for tensor, src, pinned in placed:
                        if cpu.get(tensor) is src:
                            cpu[tensor] = pinned
                            if _reads(tensor, src):
                                _repoint(tensor, pinned)
                            self.pinned += sum(leaf.nbytes for leaf in _host_leaves(src))
                    del placed
                    self.groups_pinned += 1
                    self.timeline.append((time.perf_counter() - start, self.pinned))
                    # Done before the epoch: a capture check that sees the new epoch must also see this group ready.
                    self._done[id(group)].set()
                    _bump_placement_epoch()
            except Exception as exc:  # noqa: BLE001 -- the rest is pinned per onload
                failed = exc
            finally:
                if pool is not None:
                    pool.shutdown(wait = True, cancel_futures = True)
                for event in self._done.values():
                    event.set()
                if failed is not None:
                    _bump_placement_epoch()
            if self.logger is not None:
                try:
                    if failed is not None:
                        self.logger.warning(
                            "video.h3_block_pin: background pin stopped (%s); the rest pins on each onload",
                            failed,
                        )
                    self.logger.info(
                        "video.h3_block_pin: %d of %d streamed block groups pinned in place in the background "
                        "(%.2f GB in %.1f s, onloads waited %.1f s%s)",
                        self.groups_pinned,
                        len(self.groups),
                        self.pinned / 1e9,
                        time.perf_counter() - start,
                        self.waited_s,
                        ", stopped by the pin budget" if self.stopped_at is not None else "",
                    )
                except Exception:  # noqa: BLE001
                    pass

    return _H3BlockPinner


def defer_streamed_block_pins(
    pipe: Any,
    transformer: Any,
    device: Any,
    logger: Any = None,
) -> Any:
    """Background pin of streamed groups, off with UNSLOTH_H3_BACKGROUND_BLOCK_PIN=0; onloads wait on it."""
    off = ("0", "off", "false", "no")
    if str(os.environ.get(H3_BLOCK_PIN_ENV, "")).strip().lower() in off:
        return None
    if str(os.environ.get(H3_BLOCK_BACKGROUND_PIN_ENV, "")).strip().lower() in off:
        return None
    _top, blocks = h3_offload_groups(transformer)
    lazy = [
        g
        for g in blocks
        if getattr(g, "low_cpu_mem_usage", False)
        and getattr(g, "stream", None) is not None
        and not getattr(g, "offload_to_disk_path", None)
        and isinstance(getattr(g, "cpu_param_dict", None), dict)
        and not is_resident(g)
    ]
    if not blocks or len(lazy) != len(blocks):
        return None
    from .diffusion_memory import (
        _PENDING_PINS_ATTR,
        _background_pin_enabled,
        _fast_pin_supported,
        _pinned_memory_capped,
        install_group_pin_wait,
    )
    from .diffusion_pinned_arena import pin_arena_enabled
    from .video_minimax_h3_te import h3_te_pin_allowed

    if not pin_arena_enabled() or _pinned_memory_capped():
        return None
    if not _background_pin_enabled(device) or not _fast_pin_supported(device):
        return None
    for group in lazy:
        cpu = group.cpu_param_dict
        for tensor in _group_tensors(group):
            if (
                getattr(tensor, "device", None) is None
                or tensor.device.type != "cpu"
                or tensor not in cpu
            ):
                return None
    if not h3_te_pin_allowed(group_payload_bytes(lazy[0]), device):
        return None
    pending = getattr(pipe, _PENDING_PINS_ATTR, None)
    if pending is None:
        pending = []
        try:
            setattr(pipe, _PENDING_PINS_ATTR, pending)
        except Exception:  # noqa: BLE001 -- nowhere to park it: pin up front
            return None
    install_group_pin_wait()
    pinner = _h3_block_pinner_cls()(
        transformer,
        lazy,
        device,
        logger,
        admit = lambda g: h3_te_pin_allowed(group_payload_bytes(g), device),
    )
    # cpu_param_dict already maps each tensor to its pageable host copy; onloads read it until the pin swaps it
    for group in lazy:
        group.low_cpu_mem_usage = False
    pending.append(pinner)
    pinner.start()
    if logger is not None:
        logger.info(
            "video.h3_block_pin: pinning %d streamed block groups in place in the background (%.2f GB)",
            len(lazy),
            sum(group_payload_bytes(g) for g in lazy) / 1e9,
        )
    return pinner


H3_STREAM_PREFETCH_ENV = "UNSLOTH_H3_STREAM_PREFETCH"
H3_STREAM_CARRY_ENV = "UNSLOTH_H3_STREAM_CARRY"


def h3_stream_carry_enabled() -> bool:
    """Cross-forward carry for H3's scheduler.step-driven loop; ``UNSLOTH_H3_STREAM_CARRY=0`` off."""
    return str(os.environ.get(H3_STREAM_CARRY_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def h3_stream_prefetch_depth(block_bytes: list, default_depth: int) -> int:
    """Groups copied ahead: as many as ``H3_STREAM_WINDOW_GB`` (reserved by the residency plan) holds next to the
    running block, capped by ``default_depth``, at least 1."""
    largest = max(block_bytes) if block_bytes else 0
    if largest <= 0:
        return max(1, int(default_depth))
    fits = int(H3_STREAM_WINDOW_GB * 1e9 // largest) - 1
    return max(1, min(int(default_depth), fits))


def h3_stream_prefetch_window(top_bytes: int, block_bytes: list) -> int:
    """Bytes the prefetcher may hold on the device: the top-level group plus two blocks, diffusers' own footprint."""
    largest = max(block_bytes) if block_bytes else 0
    return int(max(0, top_bytes) + 2 * largest)


def install_h3_stream_prefetch(
    transformer: Any,
    device: Any,
    logger: Any = None,
) -> int:
    """Drive a block-streamed H3 denoiser's offload groups with the event-fenced prefetch (no host sync per onload).
    Run after the top-level pin, before the residency fit; returns groups covered. ``UNSLOTH_H3_STREAM_PREFETCH=0`` off."""
    if str(os.environ.get(H3_STREAM_PREFETCH_ENV, "")).strip().lower() in (
        "0",
        "off",
        "false",
        "no",
    ):
        return 0
    try:
        from .diffusion_offload_prefetch import install_group_prefetch, prefetch_depth
    except ImportError:
        return 0
    top, blocks = h3_offload_groups(transformer)
    if not blocks:
        return 0
    from .diffusion_memory import _pinned_memory_capped

    if not _pinned_memory_capped() and any(getattr(g, "low_cpu_mem_usage", False) for g in blocks):
        # Unpinned host copies: the prefetcher would pin a fresh copy per onload, two blocks ahead (more host RAM, no speed-up).
        if logger is not None:
            logger.info(
                "video.h3_prefetch: left to diffusers' onload (the streamed blocks have unpinned host copies)"
            )
        return 0
    depth = h3_stream_prefetch_depth([group_payload_bytes(g) for g in blocks], prefetch_depth())

    if _pinned_memory_capped():
        depth = 1  # each in-flight group is pinned on the fly; diffusers' own path holds two near a ~1 GiB cap
    # Prefetcher refuses replaced onload_: lift the outside-inference_mode wrappers (torchao int8 cannot be re-pointed inside), install, then rewrap.
    groups = _all_offload_groups(transformer)
    # Unstreamed top group keeps its wrapper, so generic top-group adoption cannot pin it.
    kept = top if top is not None and getattr(top, "stream", None) is None else None
    lifted = [g for g in groups if g is not kept and _lift_outside_inference_wrappers(g)]
    try:
        covered = install_group_prefetch(transformer, device, logger, depth = depth)
    except Exception:  # noqa: BLE001 -- install_group_prefetch never raises today; keep the wrappers either way
        covered = 0
    for group in lifted:
        for name in ("onload_", "offload_"):
            setattr(group, name, _outside_inference_mode(getattr(group, name)))
    if covered:
        from .diffusion_offload_prefetch import module_prefetcher

        prefetcher = module_prefetcher(transformer)
        for group in groups:
            if getattr(group, "_unsloth_prefetcher", None) is prefetcher:
                # owns() compares the instance onload_ with this
                group._unsloth_prefetch_onload = group.__dict__.get("onload_")
        if prefetcher is not None:
            # (depth + 1) x the 0.8 GB top-level group would leave no block ahead once that group streams (12 GB).
            prefetcher.window = h3_stream_prefetch_window(
                group_payload_bytes(top) if top is not None else 0,
                [group_payload_bytes(g) for g in blocks],
            )
            # a streamed top-level group runs for the whole forward, so only keeping it hides its upload
            prefetcher.keep_first = h3_stream_carry_enabled()
            # end() puts groups copied ahead but never run back on the host copy; kick() queues copies
            for name in ("end", "kick"):
                setattr(prefetcher, name, _outside_inference_mode(getattr(prefetcher, name)))
    return covered


def _all_offload_groups(transformer: Any) -> list:
    out: list = []
    seen: set = set()
    for sub in transformer.modules():
        group = getattr(_group_hook(sub), "group", None)
        if group is not None and id(group) not in seen:
            seen.add(id(group))
            out.append(group)
    return out


def _lift_outside_inference_wrappers(group: Any) -> bool:
    """Drop the instance onload_ / offload_ that only re-enter the class methods outside inference_mode."""
    attrs = getattr(group, "__dict__", {})
    if is_resident(group) or not any(name in attrs for name in ("onload_", "offload_")):
        return False
    for name in ("onload_", "offload_"):
        attrs.pop(name, None)
    return True


def _outside_inference_mode(fn: Any) -> Any:
    import torch

    def call(*args: Any, **kwargs: Any) -> Any:
        with torch.inference_mode(False), torch.no_grad():
            return fn(*args, **kwargs)

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    return disable(call) if callable(disable) else call


H3_VAE_PINNED_SWAP_ENV = "UNSLOTH_H3_VAE_PINNED_SWAP"
H3_VAE_PAGEABLE_SWAP_ENV = "UNSLOTH_H3_VAE_PAGEABLE_SWAP"


def install_pinned_swap(
    module: Any,
    *,
    logger: Any = None,
    label: str = "component",
) -> bool:
    """Stock rotation uploads from pageable memory and copies back every park; device moves use pinned host copy; UNSLOTH_H3_VAE_PINNED_SWAP=0 keeps stock moves.
    Where pinning is not allowed the swap keeps the module's own pageable host copy, so a park is still a re-point
    (UNSLOTH_H3_VAE_PAGEABLE_SWAP=0 keeps stock moves there)."""
    off = ("0", "off", "false", "no")
    if str(os.environ.get(H3_VAE_PINNED_SWAP_ENV, "")).strip().lower() in off:
        return False
    if module is None or getattr(module, "_unsloth_pinned_swap", None) is not None:
        return False
    import torch

    from .video_minimax_h3_te import _module_payload_bytes, h3_te_pin_allowed, pin_module_in_place

    if not torch.cuda.is_available():
        return False
    tensors = list(module.parameters()) + list(module.buffers())
    if not tensors or any(t.device.type != "cpu" for t in tensors):
        return False
    if any(type(t.data) is not torch.Tensor for t in tensors):
        return False
    pinned = h3_te_pin_allowed(_module_payload_bytes(module))
    if not pinned and str(os.environ.get(H3_VAE_PAGEABLE_SWAP_ENV, "")).strip().lower() in off:
        return False
    if pinned:
        try:
            pin_module_in_place(module, repoint = True)
        except Exception as exc:  # noqa: BLE001 -- stock moves
            if logger is not None:
                logger.warning("video.h3_pinned_swap: %s not pinned (%s)", label, exc)
            return False
    host: dict[int, Any] = {}
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type == "cpu" and (tensor.is_pinned() or not pinned):
            host[id(tensor)] = tensor.data
    stock_to = module.to

    def _swap_to(*args: Any, **kwargs: Any) -> Any:
        try:
            device, dtype, _non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)
        except Exception:  # noqa: BLE001
            return stock_to(*args, **kwargs)
        if device is None or dtype is not None or memory_format is not None:
            return stock_to(*args, **kwargs)
        device = torch.device(device)
        fallback = False
        for tensor in list(module.parameters()) + list(module.buffers()):
            pinned = host.get(id(tensor))
            if pinned is None or pinned.dtype != tensor.dtype or pinned.shape != tensor.shape:
                fallback = True
                continue
            if device.type == "cpu":
                tensor.data = pinned
            elif tensor.device != device:
                tensor.data = pinned.to(device, non_blocking = True)
        if fallback:
            return stock_to(*args, **kwargs)
        return module

    module.to = _swap_to
    module._unsloth_pinned_swap = host
    if logger is not None:
        logger.info(
            "video.h3_pinned_swap: %s moves by re-pointing at a %s host copy (%.2f GB)",
            label,
            "pinned" if pinned else "pageable",
            sum(t.numel() * t.element_size() for t in host.values()) / 1e9,
        )
    return True


H3_ROTATION_PARK_ENV = "UNSLOTH_H3_ROTATION_PARK"
_PARK_ATTR = "_unsloth_h3_rotation_park"
_COMPONENTS_MANAGER_LOGGER = "diffusers.modular_pipelines.components_manager"
_FALLBACK_WARNING = "no combination of models to offload to cpu is found"


def h3_rotation_park_enabled() -> bool:
    return str(os.environ.get(H3_ROTATION_PARK_ENV, "")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


class _DropFallbackWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return _FALLBACK_WARNING not in record.getMessage()
        except Exception:  # noqa: BLE001 -- an unformattable record is not ours to drop
            return True


class H3QuietOffloadStrategy:
    """The wrapped strategy, minus its "offloading all models" warning (evict-all is this tier's policy)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def __call__(self, hooks: Any, model_id: Any, model: Any, execution_device: Any) -> Any:
        log = logging.getLogger(_COMPONENTS_MANAGER_LOGGER)
        quiet = _DropFallbackWarning()
        log.addFilter(quiet)
        try:
            return self.inner(
                hooks = hooks, model_id = model_id, model = model, execution_device = execution_device
            )
        finally:
            log.removeFilter(quiet)


def install_h3_rotation_park(
    pipe: Any,
    *,
    park: bool,
    logger: Any = None,
) -> bool:
    """Streamed tiers only. ``park`` (streamed denoiser): evict the rotation when a render returns, not at the next
    forward, so the next request's VRAM floor stops counting the last decode's audio VAE."""
    if pipe is None or not h3_rotation_park_enabled():
        return False
    manager = getattr(pipe, "_components_manager", None)
    hooks = list(getattr(manager, "model_hooks", None) or ())
    if not hooks or getattr(manager, "_auto_offload_device", None) is None:
        return False
    wrapped: dict[int, Any] = {}

    def _quiet(strategy: Any) -> Any:
        if strategy is None or isinstance(strategy, H3QuietOffloadStrategy):
            return strategy
        if id(strategy) not in wrapped:
            wrapped[id(strategy)] = H3QuietOffloadStrategy(strategy)
        return wrapped[id(strategy)]

    for hook in hooks:
        inner = getattr(hook, "hook", None)
        if inner is not None and hasattr(inner, "offload_strategy"):
            inner.offload_strategy = _quiet(inner.offload_strategy)
    if hasattr(manager, "_offload_strategy"):
        manager._offload_strategy = _quiet(manager._offload_strategy)
    setattr(pipe, _PARK_ATTR, bool(park))
    if logger is not None:
        logger.info(
            "video.h3_rotation: evicting every resident component is this tier's policy, so the diffusers "
            "fallback warning is muted; %s %s (%s=0 keeps both as before)",
            ", ".join(str(getattr(hook, "model_id", "?")) for hook in hooks),
            "park on the host when a render returns, so the next request's VRAM floor and resident set see that "
            "room free"
            if park
            else "stay until they are evicted (the conditioner's pre-hook may be skipped by a prompt-cache hit)",
            H3_ROTATION_PARK_ENV,
        )
    return True


def park_h3_rotation(pipe: Any) -> bool:
    """Evict resident rotating components (the pre-hooks' own eviction); never raises."""
    if pipe is None or not getattr(pipe, _PARK_ATTR, False):
        return False
    try:
        manager = getattr(pipe, "_components_manager", None)
        device = getattr(manager, "_auto_offload_device", None)
        if manager is None or device is None:
            return False
        from .diffusion_prequant import _evict_rotation_hook

        _evict_rotation_hook(manager, device)(None, None)
        return True
    except Exception:  # noqa: BLE001 -- the next render's pre-hook evicts them as before
        return False
