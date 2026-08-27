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
    context_is_pinned: bool = True,
    inherited_device_pin: bool = False,
    spec_unpriced: bool = False,
    moe_offload_unmodelled: bool = False,
    gpu_bytes: Any = _UNSET,
    compute_bytes: Any = _UNSET,
    total_bytes: Any = _UNSET,
    n_ctx: Any = _UNSET,
) -> MemoryEstimate:
    """Normalize a planner breakdown into the canonical estimate.

    *breakdown* is a ``_GgufMemoryBreakdown``, taken structurally rather than by
    import so this module does not depend on a route module. Its own
    ``weights_bytes`` field carries the RESIDENT-FILES meaning, which is why it
    lands in ``resident_files_bytes`` and never in ``quant_file_bytes``.

    *quant_file_bytes* has to be supplied by the caller because the planner does
    not carry it: the planner is given a resolved config and reports what the
    launch would hold, while the size of the one file the user picked is known
    only to whatever resolved that file. Callers that genuinely do not know it
    should pass 0 rather than the resident total, since a wrong number here is
    exactly the confusion this module exists to end.

    The four trailing overrides exist because ``/kv-cache-estimate`` derives those
    figures from its own planner calls, with its own None handling, and they are
    not the breakdown's. They are passed IN rather than assigned onto the returned
    model afterwards: Pydantic does not validate assignment by default, so
    mutating the model post-construction puts whatever it is handed straight onto
    the wire. Measured, not assumed -- ``m.gpu_bytes = "not an int"`` succeeds on
    pydantic 2.13, and so does setting a declared ``int`` field to ``None``.
    """
    resident = int(getattr(breakdown, "weights_bytes", 0) or 0)
    quant = int(quant_file_bytes or 0)
    # Deliberately NOT clamped against `resident`, though the quant file is by
    # definition one of the resident files and so cannot really be larger.
    #
    # The two figures do not come from the same place. `resident` is what the
    # planner measured from the files it opened; `quant` is what resolved the
    # user's chosen file, which may be a listing size or a stat of a different
    # path. They agree in production and diverge whenever anything stubs one
    # side, and a clamp there does not catch a bug -- it silently replaces the
    # caller's real number with an unrelated one. The first draft of this
    # function clamped, and the contract-freeze suite caught it truncating a
    # 4.1 GB quant to 373 bytes.
    return MemoryEstimate(
        available = True,
        reason = None,
        quant_file_bytes = quant,
        resident_files_bytes = resident,
        kv_bytes = int(getattr(breakdown, "kv_bytes", 0) or 0),
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
        # Not `or 0`: zero is a real answer (an all-CPU launch) and must survive
        # distinct from None. See the field's own description.
        gpu_bytes = (
            (None if getattr(breakdown, "gpu_bytes", None) is None else int(breakdown.gpu_bytes))
            if isinstance(gpu_bytes, _Unset)
            else (None if gpu_bytes is None else int(gpu_bytes))
        ),
        gpu_floor_bytes = None if gpu_floor_bytes is None else int(gpu_floor_bytes),
        kv_estimable = bool(getattr(breakdown, "kv_estimable", True)),
        kv_on_gpu = bool(getattr(breakdown, "kv_on_gpu", True)),
        n_ctx = (
            int(getattr(breakdown, "n_ctx", 0) or 0)
            if isinstance(n_ctx, _Unset)
            else int(n_ctx or 0)
        ),
        native_context = native_context,
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
    """The ``EstimateMemoryResponse`` shape, for ``POST /estimate-memory``.

    ``weights_bytes`` here is the RESIDENT-FILES total, which is what this route
    has always meant by it and what the Load Model panel itemizes against.
    """
    return {
        "available": estimate.available,
        "reason": estimate.reason,
        # The aggregate meaning. See the module docstring.
        "weights_bytes": estimate.resident_files_bytes,
        "kv_bytes": estimate.kv_bytes,
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
    """The ``GET /kv-cache-estimate`` shape, for the Hub memory bar.

    ``weights_bytes`` here is the QUANT FILE ALONE, which is what this route has
    always meant by it: the bar draws its weights segment from this and prints it
    beside the download size on the same row, so folding the projector or a
    drafter in would make the two disagree on screen.

    The keyword terms are this route's OWN itemization, computed by the route
    rather than by the planner, so they are passed through instead of derived
    here. ``kv_bytes`` is among them deliberately: this route prices the target
    cache itself and its figure is not interchangeable with the planner's.
    Reaching into the estimate for it would silently swap one for the other.

    ``None`` is meaningful throughout and is preserved -- this route uses ``None``
    for "no such term", never ``0``, and the frontend's ``estimateIsUnsized()``
    distinguishes them.

    The one field where ``None`` and ``0`` differ in the OTHER direction is
    ``gpu_bytes``, which is passed straight through: see its field description.
    """
    return {
        # The quant-file meaning. See the module docstring.
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
