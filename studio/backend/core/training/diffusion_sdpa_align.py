# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One compiled graph for every caption length: inductor's SDPA layout constraint guards on whether an attention
mask's sizes / strides are multiples of 8 (``is_aligned_realized_tensor`` and the last-dim check in
``sdpa_constraint``). Under ``dynamic=True`` that guard lands on the caption length: Qwen-Image-2.1 builds a
``[L, L]`` causal mask over its text tokens, so the first caption whose token count is a multiple of 8 recompiled
every block mid-run (L4, 512 px: a 27 to 29 s stall at step 16 in every compiled arm, bf16 / fp8 / int8 alike).

Inside the SDPA constraint only, an alignment question (every relation in it is ``Mod(x, n) == 0`` or already
statically decided) is answered when it is statically known and never guarded; an undecided one takes the path
inductor already takes for a misaligned mask (a copy into an aligned layout), which is correct for any length.
Every other question there keeps torch's own ``guard_or_false``. While installed, a ``cache_key_tag`` keeps warm
FX / AOT caches of the two lowerings apart. ``uninstall_sdpa_align_without_guard`` restores torch's constraint and
tag; ``UNSLOTH_DIFFUSION_SDPA_ALIGN_GUARD=1`` never installs it."""

from __future__ import annotations

import os
import threading
from typing import Any

SDPA_ALIGN_GUARD_ENV = "UNSLOTH_DIFFUSION_SDPA_ALIGN_GUARD"
CACHE_TAG = "|unsloth_sdpa_align=unguarded"

_LOCK = threading.Lock()
# "ops": {op: original constraint} wrapped by this module; "tag": the cache_key_tag before install.
_STATE: dict[str, Any] = {}


def _kept() -> bool:
    return (os.environ.get(SDPA_ALIGN_GUARD_ENV) or "").strip().lower() in (
        "1",
        "on",
        "true",
        "yes",
    )


def internals_supported() -> bool:
    """The exact torch internals the wrapper relies on: ``SizeVarAllocator.guard_or_false`` and
    ``statically_known_true``, an ``is_aligned_realized_tensor`` that asks ``guard_or_false``, and the SDPA
    constraint in ``_maybe_layout_constraints``. torch 2.4 to 2.7 lack them (their SDPA constraint never guards
    through ``guard_or_false``), so there the wrapper stays out. Never raises."""
    try:
        import inspect

        import torch._inductor.ir as ir
        import torch._inductor.lowering as lowering
        from torch._inductor.sizevars import SizeVarAllocator

        if not (
            callable(getattr(SizeVarAllocator, "guard_or_false", None))
            and callable(getattr(SizeVarAllocator, "statically_known_true", None))
        ):
            return False
        aligned = getattr(ir, "is_aligned_realized_tensor", None)
        if not callable(aligned) or "guard_or_false" not in inspect.getsource(aligned):
            return False
        return callable(getattr(lowering, "sdpa_constraint", None)) and isinstance(
            getattr(lowering, "_maybe_layout_constraints", None), dict
        )
    except Exception:  # noqa: BLE001 -- unknown internals: stay out
        return False


def _is_mod_zero(rel: Any) -> bool:
    import sympy

    if not isinstance(rel, sympy.Eq):
        return False
    for side, other in ((rel.lhs, rel.rhs), (rel.rhs, rel.lhs)):
        if other == 0 and (
            isinstance(side, sympy.Mod) or type(side).__name__ in ("Mod", "PythonMod")
        ):
            return True
    return False


def is_alignment_question(expr: Any, sizevars: Any) -> bool:
    """True when ``expr`` asks only whether sizes / strides are multiples of something: at least one
    ``Mod(x, n) == 0`` relation, and every other relation in it already statically decided."""
    import sympy

    try:
        rels = [expr] if isinstance(expr, sympy.core.relational.Relational) else []
        rels += [r for r in expr.atoms(sympy.core.relational.Relational) if r is not expr]
    except Exception:  # noqa: BLE001 -- not a sympy expression: not ours to answer
        return False
    has_mod = False
    for rel in rels:
        if _is_mod_zero(rel):
            has_mod = True
            continue
        try:
            decided = sizevars.statically_known_true(rel) or sizevars.statically_known_true(
                sympy.Not(rel)
            )
        except Exception:  # noqa: BLE001
            decided = False
        if not decided:
            return False
    return has_mod


def _wrap(original: Any) -> Any:
    from torch._inductor.virtualized import V

    def sdpa_constraint_unguarded_alignment(fx_node, *args, **kwargs):
        sizevars = V.graph.sizevars
        if "guard_or_false" in vars(sizevars):  # nested call: the outer one already answers
            return original(fx_node, *args, **kwargs)
        guarded = sizevars.guard_or_false

        def guard_or_false(expr):
            if is_alignment_question(expr, sizevars):
                return bool(sizevars.statically_known_true(expr))
            return guarded(expr)

        sizevars.guard_or_false = guard_or_false
        try:
            return original(fx_node, *args, **kwargs)
        finally:
            del sizevars.guard_or_false

    sdpa_constraint_unguarded_alignment._unsloth_original = original
    return sdpa_constraint_unguarded_alignment


def _cache_config():
    import torch
    cconfig = getattr(getattr(torch, "compiler", None), "config", None)
    return cconfig if cconfig is not None and hasattr(cconfig, "cache_key_tag") else None


def install_sdpa_align_without_guard() -> int:
    """Wrap every registered SDPA layout constraint; returns how many ops it covers (0: not installed). Idempotent,
    never raises. With the opt-out set, a wrapper an earlier run in this process installed is taken out."""
    if _kept():
        uninstall_sdpa_align_without_guard()
        return 0
    if not internals_supported():
        return 0
    with _LOCK:
        if _STATE.get("ops"):
            return len(_STATE["ops"])
        try:
            import torch._inductor.lowering as lowering

            original = getattr(lowering, "sdpa_constraint", None)
            table = getattr(lowering, "_maybe_layout_constraints", None)
            if not callable(original) or not isinstance(table, dict):
                return 0
            wrapped = _wrap(original)
            ops = {}
            for op, constraint in list(table.items()):
                if constraint is original:
                    table[op] = wrapped
                    ops[op] = original
            if not ops:
                return 0
            _STATE["ops"] = ops
            cconfig = _cache_config()
            if cconfig is not None:
                prev = cconfig.cache_key_tag or ""
                _STATE["tag"] = prev
                if CACHE_TAG not in prev:
                    cconfig.cache_key_tag = prev + CACHE_TAG
            return len(ops)
        except Exception:  # noqa: BLE001 -- an inductor without these internals keeps its own behaviour
            return 0


def uninstall_sdpa_align_without_guard() -> int:
    """Put torch's SDPA constraint and the previous ``cache_key_tag`` back; returns how many ops it restored."""
    with _LOCK:
        ops = _STATE.pop("ops", None) or {}
        tag = _STATE.pop("tag", None)
        restored = 0
        try:
            import torch._inductor.lowering as lowering
            table = getattr(lowering, "_maybe_layout_constraints", None)
            if isinstance(table, dict):
                for op, original in ops.items():
                    if getattr(table.get(op), "_unsloth_original", None) is original:
                        table[op] = original
                        restored += 1
        except Exception:  # noqa: BLE001
            pass
        if tag is not None:
            cconfig = _cache_config()
            if cconfig is not None and CACHE_TAG in (cconfig.cache_key_tag or ""):
                cconfig.cache_key_tag = tag
        return restored


def installed() -> bool:
    return bool(_STATE.get("ops"))
