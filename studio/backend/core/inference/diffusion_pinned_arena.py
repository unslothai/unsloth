# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One pinned host copy of a streamed module, packed into power-of-two slabs instead of one pin per tensor.

torch's host allocator rounds each pin up to a power of two (H3's 19.45 GB int8 denoiser held 33.8 GB pinned), and the
pageable source survived until each group's first onload. The slab copy replaces the parameter's storage on the spot.

Used by every up-front group-offload pin (image streaming / group / resident tiers, streamed encoders, prequant, H3):
a 23.8 GB bf16 Flux DiT held 37.2 GB pinned per tensor, 24.3 GB packed.

Kill switch: ``UNSLOTH_DIFFUSION_PIN_ARENA=0`` restores per-tensor ``pin_memory()``.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable, Iterator, Optional

PIN_ARENA_ENV = "UNSLOTH_DIFFUSION_PIN_ARENA"
DEFAULT_SLAB_BYTES = 1 << 30
_ALIGN = 512


def pin_arena_enabled() -> bool:
    return str(os.environ.get(PIN_ARENA_ENV, "1")).strip().lower() not in (
        "0",
        "off",
        "false",
        "no",
    )


def _alloc_slab(nbytes: int) -> Any:
    """One pinned slab: registered memory, else a power-of-two torch pinned request (no rounding)."""
    import torch

    from .diffusion_memory import registered_pinned_empty

    slab = registered_pinned_empty(nbytes)
    if slab is not None:
        return slab
    return torch.empty(nbytes, dtype = torch.uint8, pin_memory = True)


def _pow2_ceil(n: int) -> int:
    return 1 << max(0, (int(n) - 1).bit_length())


def _slab_size(nbytes: int, slab_bytes: int, remaining: Optional[int]) -> int:
    """A new slab for an ``nbytes`` tensor. With ``remaining`` (bytes still to place, this tensor included) known, the
    tail slabs halve with what is left (largest power of two not above it) instead of a whole slab per small module."""
    size = slab_bytes
    if remaining is not None and remaining < slab_bytes:
        size = 1 << (max(int(remaining), 1).bit_length() - 1)
    return max(size, _pow2_ceil(nbytes))


def _first_fit(
    entries: list[list[Any]], nbytes: int, capacity: Callable[[Any], int]
) -> Optional[tuple[int, int]]:
    """(slab index, offset) of ``nbytes`` in the first ``[slab, used]`` entry with room, marked taken; else None."""
    for index, entry in enumerate(entries):
        start = (entry[1] + _ALIGN - 1) // _ALIGN * _ALIGN
        if start + nbytes <= capacity(entry[0]):
            entry[1] = start + nbytes
            return index, start
    return None


def _packable(tensor: Any) -> bool:
    # Only CPU tensors: group offload's ``_to_cpu`` hands a device-resident one to the stock per-tensor pin.
    return (
        tensor.device.type == "cpu"
        and tensor.is_contiguous()
        and tensor.numel() > 0
        and not tensor.dtype.is_complex
    )


class PinnedArena:
    """First-fit bump allocator over pinned uint8 slabs. Views keep their slab alive, so dropping the
    last tensor that points into a slab returns it to torch's host cache like any pinned tensor.

    ``expected_bytes`` (the packable bytes about to be pinned) shrinks the tail slabs to what is left."""

    def __init__(
        self,
        slab_bytes: int = DEFAULT_SLAB_BYTES,
        expected_bytes: Optional[int] = None,
    ) -> None:
        self.slab_bytes = _pow2_ceil(slab_bytes)
        self.expected_bytes = expected_bytes
        self._slabs: list[list[Any]] = []  # [slab tensor, used bytes]
        self.payload_bytes = 0

    @property
    def reserved_bytes(self) -> int:
        return sum(int(s.numel()) for s, _ in self._slabs)

    def _take(self, nbytes: int) -> Any:
        hit = _first_fit(self._slabs, nbytes, lambda slab: int(slab.numel()))
        if hit is not None:
            slab, start = self._slabs[hit[0]][0], hit[1]
            return slab[start : start + nbytes]
        remaining = (
            None
            if self.expected_bytes is None
            else max(nbytes, self.expected_bytes - self.payload_bytes)
        )
        slab = _alloc_slab(_slab_size(nbytes, self.slab_bytes, remaining))
        self._slabs.append([slab, nbytes])
        return slab[:nbytes]

    def pin_plain(self, tensor: Any) -> Any:
        """A pinned copy of a plain CPU tensor, as a view into a slab. Falls back to ``pin_memory()``
        for anything the packing cannot represent exactly."""
        if tensor.device.type != "cpu" or not _packable(tensor):
            return tensor.pin_memory()
        nbytes = tensor.numel() * tensor.element_size()
        dst = self._take(nbytes).view(tensor.dtype).view(tensor.shape)
        dst.copy_(tensor)
        self.payload_bytes += nbytes
        return dst

    def pin(self, tensor: Any) -> Any:
        """Pinned copy of ``tensor``. A traceable wrapper subclass (torchao Int8Tensor and friends) is
        rebuilt from pinned inner tensors, which is what its own ``pin_memory()`` produces."""
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if is_traceable_wrapper_subclass(tensor):
            names, ctx = tensor.__tensor_flatten__()
            inner = {name: self.pin(getattr(tensor, name)) for name in names}
            return type(tensor).__tensor_unflatten__(inner, ctx, tensor.size(), tensor.stride())
        return self.pin_plain(tensor)


def _plain_leaves(tensor: Any, depth: int = 0) -> list:
    """The plain tensors ``PinnedArena.pin`` would copy for ``tensor`` (inner tensors of a wrapper subclass)."""
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if depth < 4 and is_traceable_wrapper_subclass(tensor):
        names, _ = tensor.__tensor_flatten__()
        return [leaf for name in names for leaf in _plain_leaves(getattr(tensor, name), depth + 1)]
    return [tensor]


def packable_bytes(tensors: Any) -> int:
    """Bytes of ``tensors`` the arena packs (the rest goes to ``pin_memory()``): its ``expected_bytes``."""
    return sum(
        int(leaf.numel()) * int(leaf.element_size())
        for tensor in tensors
        for leaf in _plain_leaves(tensor)
        if _packable(leaf)
    )


def arena_reserved_bytes(tensors: Any, slab_bytes: int = DEFAULT_SLAB_BYTES) -> int:
    """Pinned bytes an arena sized by ``packable_bytes(tensors)`` reserves for them: its packing replayed in order, and
    a tensor it hands back to ``pin_memory()`` priced at the host allocator's power-of-two rounding."""
    tensors = list(tensors)
    slab_bytes = _pow2_ceil(slab_bytes)
    expected = packable_bytes(tensors)
    slabs: list[list[int]] = []  # [capacity, used]
    loose = placed = 0
    for tensor in tensors:
        for leaf in _plain_leaves(tensor):
            nbytes = int(leaf.numel()) * int(leaf.element_size())
            if nbytes <= 0:
                continue
            if not _packable(leaf):
                loose += _pow2_ceil(nbytes)
                continue
            if _first_fit(slabs, nbytes, int) is None:
                slabs.append(
                    [_slab_size(nbytes, slab_bytes, max(nbytes, expected - placed)), nbytes]
                )
            placed += nbytes
    return loose + sum(cap for cap, _ in slabs)


def _repoint(param: Any, pinned: Any) -> None:
    """Make ``param`` read the pinned copy now, so its pageable storage is freed during setup.
    Group offloading's own offload does exactly this after every forward."""
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass
    if is_traceable_wrapper_subclass(param):
        names, _ = pinned.__tensor_flatten__()
        for name in names:
            setattr(param, name, getattr(pinned, name))
    else:
        param.data = pinned


@contextlib.contextmanager
def pinned_arena_for_group_offload(
    *,
    enabled: Optional[bool] = None,
    slab_bytes: int = DEFAULT_SLAB_BYTES,
    expected_bytes: Optional[int] = None,
) -> Iterator[Optional[PinnedArena]]:
    """Routes group-offload pinned copies through one arena while active; yields None when disabled."""
    if enabled is None:
        enabled = pin_arena_enabled()
    if not enabled:
        yield None
        return
    try:
        from diffusers.hooks import group_offloading as go
    except Exception:  # noqa: BLE001 -- no group offloading in this diffusers
        yield None
        return
    group_cls = getattr(go, "ModuleGroup", None)
    original = group_cls.__dict__.get("_to_cpu") if group_cls is not None else None
    if original is None:
        yield None
        return
    arena = PinnedArena(slab_bytes, expected_bytes)

    def _to_cpu(tensor: Any, low_cpu_mem_usage: bool) -> Any:
        if (
            low_cpu_mem_usage
            or getattr(tensor, "device", None) is None
            or tensor.device.type != "cpu"
        ):
            return original.__func__(tensor, low_cpu_mem_usage)
        try:
            pinned = arena.pin(tensor)
        except Exception:  # noqa: BLE001 -- any surprise keeps the stock per-tensor pin
            return original.__func__(tensor, low_cpu_mem_usage)
        try:
            _repoint(tensor, pinned)
        except Exception:  # noqa: BLE001 -- the stock path re-points at the first offload anyway
            pass
        return pinned

    group_cls._to_cpu = staticmethod(_to_cpu)
    try:
        yield arena
    finally:
        group_cls._to_cpu = original
