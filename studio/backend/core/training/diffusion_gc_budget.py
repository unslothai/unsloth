# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint only as many DiT blocks as the card needs.

A DiT LoRA run checkpoints every transformer block, so each block's forward runs twice per step (forward, then again
in backward). After streaming the text encoders, a 4-bit run peaks at 3 to 6 GiB, which leaves most of a 24 to 80 GB
card idle while the step pays for the recompute. Here step 1 runs with every block checkpointed, except the first
block of each block class, whose kept activations are measured. After step 1 the free memory left on the card decides
how many blocks keep their activations instead of recomputing them; the rest stay checkpointed. Skipping a block's
recompute runs the same kernels on the same inputs, so the loss and gradients are unchanged.

An out-of-memory error in a step that keeps activations turns the plan off and reruns that step with every block
checkpointed, from the same inputs, so the run carries on exactly as a fully checkpointed run would.

``UNSLOTH_DIFFUSION_GC_MODE``: unset or ``auto`` uses this plan on CUDA (not ROCm); ``plain`` checkpoints every block
and ``partial`` every Nth, as before.
"""

from __future__ import annotations

import functools
import os
from typing import Any, Optional

GC_MODE_ENV = "UNSLOTH_DIFFUSION_GC_MODE"
# Free memory kept back after the plan: allocator fragmentation, the CUDA workspace, a preview round's VAE decode.
_MARGIN_MIN_BYTES = 2 << 30
_MARGIN_FRACTION = 0.10
# Only this share of the remaining headroom is handed to kept activations.
_HEADROOM_SHARE = 0.85


def gc_budget_mode() -> str:
    raw = (os.environ.get(GC_MODE_ENV, "") or "auto").strip().lower()
    return raw if raw in ("plain", "partial") else "auto"


def _is_rocm() -> bool:
    try:
        import torch
        return bool(getattr(torch.version, "hip", None))
    except Exception:  # noqa: BLE001
        return True


def _owner(function: Any) -> Any:
    while isinstance(function, functools.partial):
        function = function.func
    return getattr(function, "__self__", function)


def _proxy_numel(args: tuple, kwargs: dict) -> int:
    """Elements of the block's sequence inputs (floating tensors of rank >= 3): activations scale with these."""
    import torch

    total = 0
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.ndim >= 3 and value.is_floating_point():
            total += value.numel()
    return total


def _output_bytes(out: Any) -> int:
    import torch

    if isinstance(out, torch.Tensor):
        return out.numel() * out.element_size()
    if isinstance(out, (tuple, list)):
        return sum(_output_bytes(o) for o in out)
    return 0


def _memory_allocated() -> int:
    import torch
    return torch.cuda.memory_allocated()


class _Checkpoint:
    """The non-reentrant checkpoint the plain path uses, block kwargs bound (non-reentrant would read debug / context_fn)."""

    def __call__(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        import torch.utils.checkpoint as ck

        fn = getattr(ck, "_unsloth_pristine_checkpoint", None) or ck.checkpoint
        if kwargs:
            function = functools.partial(function, **kwargs)
        return fn(function, *args, use_reentrant = False)


class BudgetedCheckpoint:
    """``gradient_checkpointing_func`` for a diffusers model: checkpoints every block except those in ``keep``."""

    def __init__(self, base: Optional[Any] = None):
        self.base = base or _Checkpoint()
        self.keep: frozenset = frozenset()
        self.probing = True
        self.planned = False
        self.disabled = False
        self.index: dict[int, int] = {}
        self.block_class: dict[int, str] = {}
        self.block_proxy: dict[int, int] = {}
        # Kept bytes per proxy element, measured on the first block of each class.
        self.class_cost: dict[str, float] = {}
        self.kept_bytes = 0
        # Largest canvas area seen while measuring (bucketed runs only).
        self.probe_area: Optional[int] = None

    def _block_index(self, owner: Any) -> int:
        key = id(owner)
        idx = self.index.get(key)
        if idx is None:
            if self.planned:
                # A caller first seen after the plan (a per-call closure, not a block): checkpointed, never recorded.
                return -1
            idx = len(self.index)
            self.index[key] = idx
        return idx

    def __call__(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        owner = _owner(function)
        idx = self._block_index(owner)
        if self.probing and not self.disabled:
            cls = type(owner).__name__
            self.block_class[idx] = cls
            self.block_proxy[idx] = max(self.block_proxy.get(idx, 0), _proxy_numel(args, kwargs))
            if cls not in self.class_cost:
                return self._measure(cls, idx, function, args, kwargs)
        if idx in self.keep and not self.disabled:
            return function(*args, **kwargs)
        return self.base(function, *args, **kwargs)

    def _measure(self, cls: str, idx: int, function: Any, args: tuple, kwargs: dict) -> Any:
        before = _memory_allocated()
        out = function(*args, **kwargs)
        kept = max(0, _memory_allocated() - before - _output_bytes(out))
        self.class_cost[cls] = kept / max(1, self.block_proxy.get(idx, 0))
        return out

    def note_canvas(self, area: int) -> None:
        self.probe_area = max(self.probe_area or 0, int(area))

    def estimate(self, idx: int) -> int:
        cost = self.class_cost.get(self.block_class.get(idx, ""))
        if cost is None:
            return -1
        return int(cost * self.block_proxy.get(idx, 0))

    def plan(
        self,
        projected_peak: int,
        free_bytes: int,
        reserved_bytes: int,
        total_bytes: int,
        max_canvas_area: Optional[int] = None,
    ) -> int:
        """Pick the blocks that keep their activations from step 1's measurements; returns how many. A bucketed run
        scales the step-1 sizes (and the peak above the weights) to its largest canvas."""
        self.probing = False
        self.planned = True
        if self.disabled:
            return 0
        scale = 1.0
        if max_canvas_area and self.probe_area:
            scale = max(1.0, max_canvas_area / self.probe_area)
        margin = max(_MARGIN_MIN_BYTES, int(total_bytes * _MARGIN_FRACTION))
        headroom = int((free_bytes + reserved_bytes - projected_peak - margin) * _HEADROOM_SHARE)
        keep, used = [], 0
        for idx in sorted(self.block_class):
            need = int(self.estimate(idx) * scale)
            if need < 0 or used + need > headroom:
                break
            keep.append(idx)
            used += need
        self.keep = frozenset(keep)
        self.kept_bytes = used
        return len(keep)

    def fall_back(self) -> bool:
        """Every block checkpointed from now on; True when that changed anything."""
        changed = not self.disabled and (bool(self.keep) or self.probing)
        self.disabled = True
        self.probing = False
        self.keep = frozenset()
        return changed

    @property
    def can_retry(self) -> bool:
        """A fully checkpointed rerun would use less memory than this step: measuring, or blocks kept."""
        return not self.disabled and (self.probing or bool(self.keep))

    @property
    def total_blocks(self) -> int:
        return len(self.block_class)


def _to_host(obj: Any, pin: bool) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        if obj.device.type == "cpu":
            return obj.detach().clone()
        host = torch.empty(obj.shape, dtype = obj.dtype, pin_memory = pin)
        # Pinned + non_blocking: queued on the current stream, so it is complete before any later copy back.
        return host.copy_(obj.detach(), non_blocking = pin)
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_host(o, pin) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_host(v, pin) for k, v in obj.items()}
    return obj


def _to_device(obj: Any, device: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking = obj.is_pinned())
    if isinstance(obj, (tuple, list)):
        return type(obj)(_to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


class StepReplay:
    """One optimizer step's micro-batch inputs, kept on the host, and the CPU / CUDA RNG state right before each
    micro-batch's forward, so an out-of-memory rerun replays the step exactly: same inputs, same dropout masks and
    in-forward draws, and the RNG left where one successful pass would have left it. The device copies are not kept,
    so gradient accumulation still holds only one micro-batch's inputs on the card."""

    def __init__(self, device: str):
        self.device = device
        self.entries: list = []

    def clear(self) -> None:
        self.entries.clear()

    def _rng_state(self) -> tuple:
        import torch
        cuda = torch.cuda.get_rng_state() if self.device == "cuda" else None
        return torch.get_rng_state(), cuda

    def _set_rng_state(self, state: tuple) -> None:
        import torch

        cpu, cuda = state
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)

    def record(self, inputs: tuple) -> None:
        """Call right before the micro-batch's forward."""
        self.entries.append((_to_host(inputs, pin = self.device == "cuda"), self._rng_state()))

    def replay(self, fn: Any) -> list:
        """Rerun every recorded micro-batch through ``fn``; returns its results in order."""
        out = []
        for host, state in self.entries:
            self._set_rng_state(state)
            out.append(fn(*_to_device(host, self.device)))
        return out


def is_out_of_memory(exc: BaseException) -> bool:
    """The caching allocator's OutOfMemoryError, or a raw CUDA allocation failure (cuBLAS workspace, bnb)."""
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:  # noqa: BLE001
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def enable_budgeted_checkpointing(model: Any, device: str) -> Optional[BudgetedCheckpoint]:
    """Install the budgeted checkpoint on a diffusers model; None when the mode, device or model does not allow it
    (the caller then checkpoints every block as before)."""
    if gc_budget_mode() != "auto" or device != "cuda" or _is_rocm():
        return None
    enable = getattr(model, "enable_gradient_checkpointing", None)
    if not callable(enable):
        return None
    try:
        import inspect
        if "gradient_checkpointing_func" not in inspect.signature(enable).parameters:
            return None
    except (TypeError, ValueError):
        return None
    budget = BudgetedCheckpoint()
    enable(gradient_checkpointing_func = budget)
    return budget
