# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Engagement: proof that a requested low-precision path really ran (strict mode: UNSLOTH_STRICT_ENGAGE=1)."""

from __future__ import annotations

import contextlib
import os
import threading
from typing import Dict

__all__ = [
    "EngagementError",
    "strict_engage",
    "record",
    "engaged",
    "require",
    "LOWP_GEMM_OPS",
    "LOWP_GEMM_CUSTOM_OPS",
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

# Opaque custom ops that run a low-precision GEMM inside: dispatch sees only the outer op, so each call counts
# once at its boundary. Matched by qualified name, so a namespace that never loaded (no fbgemm) costs nothing.
# Dequantize-then-matmul ops (mxfp4_bounded_dense_matmul, fp8 weight dequant, fp8_block_gemv, which upcasts
# the weight before its dot) are not low-precision GEMMs.
LOWP_GEMM_CUSTOM_OPS = (
    "unsloth::nvfp4_qat_linear",
    "unsloth::fp8_block_matmul_triton",
    "unsloth::fp8_block_matmul_torchao",
    "unsloth::fp8_block_small_m_linear",
    "unsloth_zoo::mxfp4_ogs_moe",
    "fbgemm::f8f8bf16",
    "fbgemm::f8f8bf16_tensorwise",
    "fbgemm::f8f8bf16_rowwise",
    "fbgemm::f8f8bf16_blockwise",
    "fbgemm::f8f8bf16_cublas",
)


# Counters open right now; eager launchers that bypass dispatch (Triton fp8 kernels) report through note_lowp_gemm.
_ACTIVE = []
# ids of the counters already inside a counted GEMM on this thread: a GEMM that delegates to another counted
# launcher or op (torchao block matmul -> Triton) counts once, at its outer boundary.
_INSIDE = threading.local()


def _inside():
    ids = getattr(_INSIDE, "ids", None)
    if ids is None:
        ids = _INSIDE.ids = set()
    return ids


def note_lowp_gemm(name):
    """Count one GEMM in every open counter not already inside a counted GEMM; returns the ids to release
    with `release_lowp_gemm` once the call returns."""
    inside = _inside()
    entered = [id(counts) for counts in _ACTIVE if id(counts) not in inside]
    for counts in _ACTIVE:
        if id(counts) in entered:
            counts[name] = counts.get(name, 0) + 1
    inside.update(entered)
    return entered


def release_lowp_gemm(entered):
    _inside().difference_update(entered)


@contextlib.contextmanager
def count_lowp_gemms():
    """Count low-precision GEMM dispatches: `with count_lowp_gemms() as c: ...; c['_scaled_mm']`, and
    `c['unsloth::nvfp4_qat_linear']` for opaque custom ops (LOWP_GEMM_CUSTOM_OPS)."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    names = {}
    for name in LOWP_GEMM_OPS:
        if getattr(torch.ops.aten, name, None) is not None:
            names[f"aten::{name}"] = name
    for name in LOWP_GEMM_CUSTOM_OPS:
        names[name] = name
    counts = {key: 0 for key in names.values()}

    class _Counter(TorchDispatchMode):
        # Skip Dynamo tracing (a plain mode breaks fullgraph); older torch ignores this hook.
        @classmethod
        def ignore_compile_internals(cls):
            return True

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            key = names.get(getattr(getattr(func, "_schema", None), "name", None))
            inside = _inside()
            if key is None or id(counts) in inside:
                return func(*args, **(kwargs or {}))
            counts[key] += 1
            inside.add(id(counts))
            try:
                return func(*args, **(kwargs or {}))
            finally:
                inside.discard(id(counts))

    _ACTIVE.append(counts)
    try:
        with _Counter():
            yield counts
    finally:
        # By identity: nested counters start as equal dicts, and list.remove compares by value.
        for i, active in enumerate(_ACTIVE):
            if active is counts:
                del _ACTIVE[i]
                break
        _inside().discard(id(counts))
