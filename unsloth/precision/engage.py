# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engagement: proof that a requested low-precision path really ran.

`record` stores how many modules a recipe converted on the model (`model._unsloth_precision_engaged`).
`require` turns "converted nothing" into an error under UNSLOTH_STRICT_ENGAGE=1 (certification runs), and is a
no-op otherwise, so callers keep their own warning. `count_lowp_gemms` counts the scaled GEMM ops that
actually dispatched inside a block of code."""

from __future__ import annotations

import contextlib
import os
from typing import Dict

__all__ = [
    "EngagementError",
    "strict_engage",
    "record",
    "engaged",
    "require",
    "LOWP_GEMM_OPS",
    "count_lowp_gemms",
]

_ENV = "UNSLOTH_STRICT_ENGAGE"


class EngagementError(RuntimeError):
    """A requested feature silently fell back (strict mode)."""


def strict_engage() -> bool:
    return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


def record(
    model,
    recipe: str,
    count: int,
    detail = None,
) -> None:
    try:
        table = getattr(model, "_unsloth_precision_engaged", None)
        if not isinstance(table, dict):
            table = {}
        table[recipe] = {
            "modules": int(count),
            **({"detail": detail} if detail is not None else {}),
        }
        model._unsloth_precision_engaged = table
    except Exception:
        pass


def engaged(model) -> Dict[str, dict]:
    return dict(getattr(model, "_unsloth_precision_engaged", None) or {})


def require(
    recipe: str,
    count: int,
    why: str = "",
) -> bool:
    """True when engaged. Count 0 under strict mode raises EngagementError; otherwise returns False."""
    if count > 0:
        return True
    if strict_engage():
        raise EngagementError(
            f"Unsloth: {recipe} was requested but engaged on 0 modules"
            + (f" ({why})" if why else "")
            + f". {_ENV}=1 refuses silent fallbacks; unset it to train in the original precision."
        )
    return False


# aten ops that run a low-precision GEMM (names resolved lazily: absent ops are skipped).
LOWP_GEMM_OPS = ("_scaled_mm", "_scaled_mm_v2", "_scaled_grouped_mm", "_scaled_grouped_mm_v2")


@contextlib.contextmanager
def count_lowp_gemms():
    """Count low-precision GEMM dispatches: `with count_lowp_gemms() as c: ...; c['_scaled_mm']`."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    targets = {}
    for name in LOWP_GEMM_OPS:
        op = getattr(torch.ops.aten, name, None)
        if op is not None:
            targets[name] = op
    counts = {name: 0 for name in targets}

    class _Counter(TorchDispatchMode):
        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            packet = getattr(func, "overloadpacket", None)
            for name, op in targets.items():
                if packet is op:
                    counts[name] += 1
            return func(*args, **(kwargs or {}))

    with _Counter():
        yield counts
