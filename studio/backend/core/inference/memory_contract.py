# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One canonical memory estimate, and the two legacy shapes projected from it.

Studio answers "how much memory would this load take" on two routes:

* ``POST /api/inference/estimate-memory`` -- the Load Model panel
* ``GET  /api/models/kv-cache-estimate``  -- the Hub memory bar

They already share the ``_gguf_memory_breakdown`` planner, so the arithmetic
cannot drift. What used to drift is everything around it, and the sharp edge is
that ``weights_bytes`` exists on both, is an ``int`` on both, and means
different things: every resident file on the inference route, the quant file
alone on the models route. Nothing in the type system separates those, so a
caller reading the wrong one is simply wrong, quietly.

This module is the fix. :func:`build_memory_estimate` turns a planner breakdown
into the canonical :class:`MemoryEstimate`, whose two unambiguous fields replace
that one ambiguous one. The two ``project_*`` functions then map the canonical
model back onto each route's existing wire shape, byte for byte, so no client
sees a change. The legacy meaning of ``weights_bytes`` is applied in exactly one
place per route, right here, where the two sit next to each other and the
difference is impossible to miss.

Everything here is pure: no I/O, no probing, no network. The planner does that
work; this only renames and reshapes what it returned.
"""

from __future__ import annotations

from typing import Any, Optional

from models.inference import MemoryEstimate

__all__ = [
    "EMPTY_BREAKDOWN",
    "build_memory_estimate",
    "project_estimate_memory_response",
    "project_kv_cache_estimate",
]


class _EmptyBreakdown:
    """Stands in for a planner run that did not happen.

    Every attribute is absent, so :func:`build_memory_estimate`'s ``getattr``
    defaults apply and the planner-derived fields come back at their "not
    computed" values. ``gpu_bytes`` in particular resolves to ``None`` rather
    than ``0``, which is the distinction that matters: never ran, as opposed to
    ran and found nothing on the card.

    A class rather than ``None`` so callers have one object to pass and the
    projection stays total, instead of every call site growing its own branch.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<EMPTY_BREAKDOWN>"


EMPTY_BREAKDOWN = _EmptyBreakdown()


class _Unset:
    """Distinguishes "not overridden" from an override whose value is None.

    ``gpu_bytes`` needs all three states -- a number, a real ``None`` meaning the
    planner never ran, and "take whatever the breakdown had" -- and ``None``
    cannot carry the third.
    """

    __slots__ = ()

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        return False


_UNSET = _Unset()


def build_memory_estimate(
    breakdown: Any,
    *,
    quant_file_bytes: int,
    native_context: Optional[int] = None,
    gpu_floor_bytes: Optional[int] = None,
    floor_can_offload: bool = False,
    context_is_pinned: bool = True,
    inherited_device_pin: bool = False,
    spec_unpriced: bool = False,
    context_fitted: Optional[int] = None,
    moe_offload_unmodelled: bool = False,
    gpu_bytes: Any = _UNSET,
    compute_bytes: Any = _UNSET,
    total_bytes: Any = _UNSET,
    n_ctx: Any = _UNSET,
) -> MemoryEstimate:
    """quant_file_bytes comes from the caller, as 0 when unknown rather than the resident total."""
    resident = int(getattr(breakdown, "weights_bytes", 0) or 0)
    quant = int(quant_file_bytes or 0)
    # Deliberately not clamped against resident: the figures come from different sources
    return MemoryEstimate(
        available = True,
        reason = None,
        quant_file_bytes = quant,
        resident_files_bytes = resident,
        kv_bytes = int(getattr(breakdown, "kv_bytes", 0) or 0),
        kv_checkpoint_bytes = int(getattr(breakdown, "kv_checkpoint_bytes", 0) or 0),
        compute_bytes = (
            int(getattr(breakdown, "compute_bytes", 0) or 0)
            if isinstance(compute_bytes, _Unset)
            else int(compute_bytes or 0)
        ),
        drafter_runtime_bytes = int(getattr(breakdown, "drafter_runtime_bytes", 0) or 0),
        drafter_runtime_gpu_bytes = int(getattr(breakdown, "drafter_runtime_gpu_bytes", 0) or 0),
        projector_runtime_bytes = int(getattr(breakdown, "projector_runtime_bytes", 0) or 0),
        drafter_kv_unsized = bool(getattr(breakdown, "drafter_kv_unsized", False)),
        adapters_unsized = bool(getattr(breakdown, "adapters_unsized", False)),
        total_bytes = (
            int(getattr(breakdown, "total_bytes", 0) or 0)
            if isinstance(total_bytes, _Unset)
            else int(total_bytes or 0)
        ),
        # Not `or 0`: zero means an all-CPU launch and must stay distinct from None
        gpu_bytes = (
            (None if getattr(breakdown, "gpu_bytes", None) is None else int(breakdown.gpu_bytes))
            if isinstance(gpu_bytes, _Unset)
            else (None if gpu_bytes is None else int(gpu_bytes))
        ),
        gpu_floor_bytes = None if gpu_floor_bytes is None else int(gpu_floor_bytes),
        floor_can_offload = bool(floor_can_offload),
        kv_estimable = bool(getattr(breakdown, "kv_estimable", True)),
        kv_on_gpu = bool(getattr(breakdown, "kv_on_gpu", True)),
        n_ctx = (
            int(getattr(breakdown, "n_ctx", 0) or 0)
            if isinstance(n_ctx, _Unset)
            else int(n_ctx or 0)
        ),
        native_context = native_context,
        context_fitted = context_fitted,
        cache_type_kv = getattr(breakdown, "cache_type_kv", None),
        n_parallel = int(getattr(breakdown, "n_parallel", 1) or 1),
        layer_count = getattr(breakdown, "layer_count", None),
        gpu_layers = getattr(breakdown, "gpu_layers", None),
        moe_offload_unmodelled = bool(moe_offload_unmodelled),
        context_is_pinned = bool(context_is_pinned),
        inherited_device_pin = bool(inherited_device_pin),
        spec_unpriced = bool(spec_unpriced),
    )


def project_estimate_memory_response(estimate: MemoryEstimate) -> dict:
    """Here weights_bytes is the resident-files total, which this route has always meant by it."""
    return {
        "available": estimate.available,
        "reason": estimate.reason,
        "weights_bytes": estimate.resident_files_bytes,
        "kv_bytes": estimate.kv_bytes,
        "kv_checkpoint_bytes": estimate.kv_checkpoint_bytes,
        "compute_bytes": estimate.compute_bytes,
        "drafter_runtime_bytes": estimate.drafter_runtime_bytes,
        "drafter_runtime_gpu_bytes": estimate.drafter_runtime_gpu_bytes,
        "projector_runtime_bytes": estimate.projector_runtime_bytes,
        "drafter_kv_unsized": estimate.drafter_kv_unsized,
        "adapters_unsized": estimate.adapters_unsized,
        "total_bytes": estimate.total_bytes,
        "gpu_bytes": estimate.gpu_bytes or 0,
        "kv_estimable": estimate.kv_estimable,
        "kv_on_gpu": estimate.kv_on_gpu,
        "n_ctx": estimate.n_ctx,
        "context_fitted": estimate.context_fitted,
        "context_is_pinned": estimate.context_is_pinned,
        "gpu_floor_bytes": estimate.gpu_floor_bytes,
        "floor_can_offload": estimate.floor_can_offload,
        "cache_type_kv": estimate.cache_type_kv,
        "n_parallel": estimate.n_parallel,
        "layer_count": estimate.layer_count,
        "gpu_layers": estimate.gpu_layers,
        "moe_offload_unmodelled": estimate.moe_offload_unmodelled,
    }


def project_kv_cache_estimate(
    estimate: MemoryEstimate,
    *,
    kv_bytes: Optional[int] = None,
    spec_bytes: Optional[int] = None,
    spec_fixed_bytes: Optional[int] = None,
    projector_bytes: Optional[int] = None,
    kv_checkpoint_bytes: Optional[int] = None,
) -> dict:
    """weights_bytes is the quant file alone, and None (not 0) means no such term for this route."""
    return {
        "weights_bytes": estimate.quant_file_bytes or None,
        "kv_bytes": kv_bytes or None,
        "native_context": estimate.native_context,
        "spec_bytes": spec_bytes,
        "n_ctx": estimate.n_ctx,
        "projector_bytes": projector_bytes,
        "kv_checkpoint_bytes": kv_checkpoint_bytes,
        "spec_fixed_bytes": spec_fixed_bytes,
        # Deliberately NOT `or None`. Zero means an all-CPU launch.
        "gpu_bytes": estimate.gpu_bytes,
        "compute_bytes": estimate.compute_bytes or None,
        "total_bytes": estimate.total_bytes or None,
        "gpu_floor_bytes": estimate.gpu_floor_bytes,
        "context_is_pinned": estimate.context_is_pinned,
        "inherited_device_pin": estimate.inherited_device_pin,
        "spec_unpriced": estimate.spec_unpriced,
    }
