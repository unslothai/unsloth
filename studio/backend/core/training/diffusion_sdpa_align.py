# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One compiled graph for every caption length: inductor's SDPA layout constraint guards on whether an attention
mask's sizes / strides are multiples of 8 (``is_aligned_realized_tensor`` and the last-dim check in
``sdpa_constraint``). Under ``dynamic=True`` that guard lands on the caption length: Qwen-Image-2.1 builds a
``[L, L]`` causal mask over its text tokens, so the first caption whose token count is a multiple of 8 recompiled
every block mid-run (L4, 512 px: a 27 to 29 s stall at step 16 in every compiled arm, bf16 / fp8 / int8 alike).

The wrapper answers those alignment questions only when they are statically known and never adds a guard; an
undecided one takes the path inductor already takes for a misaligned mask (a copy into an aligned layout), which is
correct for any length. It only applies inside the SDPA constraint, so no other guard changes.
``UNSLOTH_DIFFUSION_SDPA_ALIGN_GUARD=1`` keeps torch's guard."""

from __future__ import annotations

import os
import threading
from typing import Any

SDPA_ALIGN_GUARD_ENV = "UNSLOTH_DIFFUSION_SDPA_ALIGN_GUARD"

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def _kept() -> bool:
    return (os.environ.get(SDPA_ALIGN_GUARD_ENV) or "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def _wrap(original: Any) -> Any:
    from torch._inductor.virtualized import V

    def sdpa_constraint_unguarded(fx_node, *args, **kwargs):
        sizevars = V.graph.sizevars
        shadowed = "guard_or_false" in vars(sizevars)
        if shadowed:  # nested or already unguarded: nothing to do
            return original(fx_node, *args, **kwargs)
        sizevars.guard_or_false = sizevars.statically_known_true
        try:
            return original(fx_node, *args, **kwargs)
        finally:
            del sizevars.guard_or_false

    sdpa_constraint_unguarded._unsloth_original = original
    return sdpa_constraint_unguarded


def install_sdpa_align_without_guard() -> int:
    """Wrap every registered SDPA layout constraint; returns how many ops it covers (0: not installed). Idempotent,
    never raises."""
    if _kept():
        return 0
    with _LOCK:
        if "wrapped" in _STATE:
            return _STATE["wrapped"]
        count = 0
        try:
            import torch._inductor.lowering as lowering

            original = getattr(lowering, "sdpa_constraint", None)
            table = getattr(lowering, "_maybe_layout_constraints", None)
            if callable(original) and isinstance(table, dict):
                wrapped = _wrap(original)
                for op, constraint in list(table.items()):
                    if constraint is original:
                        table[op] = wrapped
                        count += 1
        except Exception:  # noqa: BLE001 -- an inductor without these internals keeps its own behaviour
            count = 0
        _STATE["wrapped"] = count
        return count
