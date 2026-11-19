# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checkpoint only as many DiT blocks as the card needs.

A DiT LoRA run checkpoints every transformer block, so each block's forward runs twice per step (forward, then again
in backward). After streaming the text encoders, a 4-bit run peaks at 3 to 6 GiB, which leaves most of a 24 to 80 GB
card idle. Step 1 measures the first block of each class; afterwards free memory decides how many blocks skip the
recompute (same kernels, same inputs, so loss and gradients are unchanged). An out-of-memory error halves the kept set
(every block checkpointed once nothing is left) and reruns the step from the same inputs.

``UNSLOTH_DIFFUSION_GC_MODE``: unset or ``auto`` uses this plan on CUDA (not ROCm); ``plain`` checkpoints every block
and ``partial`` every Nth, as before.
"""

from __future__ import annotations

import functools
import gc
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
        self.scale = 1.0
        self.step_downs = 0
        # The room the plan had for kept activations; what they leave of it is ``spare_bytes``.
        self.headroom = 0

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
        """Pick the blocks that keep activations; returns how many. Bucketed runs scale to the largest canvas."""
        self.probing = False
        self.planned = True
        if self.disabled:
            return 0
        scale = 1.0
        if max_canvas_area and self.probe_area:
            scale = max(1.0, max_canvas_area / self.probe_area)
        self.scale = scale
        margin = max(_MARGIN_MIN_BYTES, int(total_bytes * _MARGIN_FRACTION))
        headroom = int((free_bytes + reserved_bytes - projected_peak - margin) * _HEADROOM_SHARE)
        self.headroom = max(0, headroom)
        self._choose(sorted(self.block_class), headroom)
        return len(self.keep)

    @property
    def spare_bytes(self) -> int:
        """Planned room the kept activations leave unused (0 once disabled or stepped down)."""
        if self.disabled or self.step_downs:
            return 0
        return max(0, self.headroom - self.kept_bytes)

    def _choose(self, order: list, room: int) -> None:
        # A block that does not fit (or an unmeasured one) stays checkpointed; a cheaper one after it can still fit.
        keep, used = [], 0
        for idx in order:
            need = int(self.estimate(idx) * self.scale)
            if need < 0 or used + need > room:
                continue
            keep.append(idx)
            used += need
        self.keep = frozenset(keep)
        self.kept_bytes = used

    def step_down(self) -> bool:
        """After an out-of-memory error: keep at most half the kept bytes and at least one block fewer; with nothing
        left (or while still measuring) every block is checkpointed from now on. Returns whether blocks are still kept."""
        if self.disabled:
            return False
        if self.probing or not self.keep:
            self.fall_back()
            return False
        order, before, kept = sorted(self.keep), self.keep, self.kept_bytes
        self._choose(order, kept // 2)
        if self.keep == before:
            # Blocks measured at zero bytes: drop the last one so the retry still changes something.
            self._choose(order[:-1], kept)
        if not self.keep:
            self.fall_back()
            return False
        self.step_downs += 1
        return True

    def fall_back(self) -> bool:
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
        self.entries.append((_to_host(inputs, pin = self.device == "cuda"), self._rng_state()))

    def replay(self, fn: Any) -> list:
        out = []
        for host, state in self.entries:
            self._set_rng_state(state)
            out.append(fn(*_to_device(host, self.device)))
        return out


def plan_from_device(
    budget: BudgetedCheckpoint,
    fb_peak: int,
    alloc_before_step: int,
    max_canvas_area: Optional[int] = None,
    held_bytes: int = 0,
) -> None:
    """After the first optimizer step: keep as many blocks' activations as free memory less ``held_bytes`` allows.
    Never raises: on any error every block stays checkpointed."""
    import logging

    import torch

    log = logging.getLogger(__name__)
    try:
        # The optimizer's first step allocates its moments, which every later step carries.
        projected = int(fb_peak) + max(0, torch.cuda.memory_allocated() - int(alloc_before_step))
        projected += max(0, int(held_bytes))
        free, total = torch.cuda.mem_get_info()
        kept = budget.plan(
            projected, free, torch.cuda.memory_reserved(), total, max_canvas_area = max_canvas_area
        )
    except Exception as exc:  # noqa: BLE001 -- an optimisation only: every block stays checkpointed
        budget.fall_back()
        log.info("gradient checkpointing budget not applied: %s", exc)
        return
    log.info(
        "gradient checkpointing: keeping activations for %d of %d blocks (%.2f GiB)",
        kept,
        budget.total_blocks,
        budget.kept_bytes / (1 << 30),
    )


def step_down_message(budget: BudgetedCheckpoint) -> str:
    if budget.keep and not budget.disabled:
        return (
            "Ran out of GPU memory while keeping block activations; keeping "
            f"{len(budget.keep)} of {budget.total_blocks} blocks from this step on."
        )
    return (
        "Ran out of GPU memory while keeping block activations; every block is checkpointed "
        "from this step on."
    )


# Returned by BudgetedSteps.forward_backward when the micro-batch ran out of memory and the step can rerun.
OUT_OF_MEMORY = object()


class BudgetedSteps:
    """The training loop's side of a BudgetedCheckpoint (None: plain calls, nothing recorded).

    Per optimizer step: ``start_step``; per micro-batch ``forward_backward(fn, inputs)``. On ``OUT_OF_MEMORY`` the
    caller drops its references to the micro-batch, then ``rerun(fn)`` steps the plan down and replays every
    micro-batch of the step so far from host copies (same RNG), returning their results; the caller restarts its
    accumulators from those. Around the first optimizer update: ``before_update`` / ``after_update`` plan the budget."""

    def __init__(
        self,
        budget: Optional[BudgetedCheckpoint],
        device: str,
        optimizer: Any,
        warn: Optional[Any] = None,
        on_step_down: Optional[Any] = None,
    ):
        self.budget = budget
        self.device = device
        self.optimizer = optimizer
        self.warn = warn
        self.on_step_down = on_step_down
        self.replay = StepReplay(device) if budget is not None else None
        self.reruns = 0
        self._pending: Optional[tuple] = None

    def start_step(self) -> None:
        if self.replay is not None:
            self.replay.clear()

    def forward_backward(self, fn: Any, inputs: tuple) -> Any:
        budget = self.budget
        if budget is None:
            return fn(*inputs)
        if budget.can_retry:
            self.replay.record(inputs)
        try:
            return fn(*inputs)
        except Exception as exc:  # noqa: BLE001 -- re-raised unless a rerun can use less memory
            if not budget.can_retry or not is_out_of_memory(exc):
                raise
        # Outside the except block, so the failed graph is freed with its traceback.
        return OUT_OF_MEMORY

    def rerun(self, fn: Any) -> list:
        budget = self.budget
        while True:
            budget.step_down()
            self.reruns += 1
            # Partial gradients of the step are already accumulated: the whole step reruns.
            self.optimizer.zero_grad(set_to_none = True)
            gc.collect()
            _empty_cache(self.device)
            if self.on_step_down is not None:
                self.on_step_down()
            if self.warn is not None:
                self.warn(step_down_message(budget))
            try:
                results = self.replay.replay(fn)
            except Exception as exc:  # noqa: BLE001
                if not budget.can_retry or not is_out_of_memory(exc):
                    raise
                continue
            if not budget.can_retry:
                self.replay.clear()
            return results

    def before_update(self) -> None:
        import torch

        budget = self.budget
        self._pending = None
        if budget is not None and not budget.planned:
            self._pending = (torch.cuda.max_memory_allocated(), torch.cuda.memory_allocated())

    def after_update(self, max_canvas_area: Optional[int] = None) -> None:
        """Call only after an update that applied (an fp16 skipped step allocates no optimizer state)."""
        pending, self._pending = self._pending, None
        if pending is not None:
            plan_from_device(self.budget, pending[0], pending[1], max_canvas_area)


def _empty_cache(device: str) -> None:
    try:
        import torch
        module = getattr(torch, device, None)
        if module is not None and hasattr(module, "empty_cache"):
            module.empty_cache()
    except Exception:  # noqa: BLE001
        pass


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
