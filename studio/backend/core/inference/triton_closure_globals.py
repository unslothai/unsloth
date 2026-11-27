# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton 3.2 / 3.3 (torch 2.6 / 2.7, the newest on CUDA 11.8) resolve a JIT kernel's names in its module globals
only: a kernel built inside a factory that reads ``tl``, a helper ``@triton.jit`` function or a constant from the
factory's closure fails to lower there (``NameError: 'tl' is not defined``). Newer Triton also reads the closure."""

from __future__ import annotations

import functools
from typing import Any, Callable


def _kernels_in(
    obj: Any,
    seen: set,
    depth: int = 0,
) -> list:
    if obj is None or depth > 6 or id(obj) in seen:
        return []
    seen.add(id(obj))
    fn = getattr(obj, "fn", None)
    if callable(fn) and hasattr(fn, "__code__") and hasattr(obj, "run"):
        # a JITFunction: itself, plus the JIT helpers it calls through its closure
        out = [obj]
        for cell in fn.__closure__ or ():
            out += _kernels_in(cell.cell_contents, seen, depth + 1)
        return out
    if isinstance(obj, (tuple, list, set, frozenset)):
        return [k for o in obj for k in _kernels_in(o, seen, depth + 1)]
    if isinstance(obj, dict):
        return [k for o in obj.values() for k in _kernels_in(o, seen, depth + 1)]
    if hasattr(obj, "__code__"):
        # a plain launcher: the kernels it captures
        return [
            k
            for c in getattr(obj, "__closure__", None) or ()
            for k in _kernels_in(c.cell_contents, seen, depth + 1)
        ]
    ns = getattr(obj, "__dict__", None)
    if isinstance(ns, dict) and type(obj).__module__ == "types":
        return [k for o in ns.values() for k in _kernels_in(o, seen, depth + 1)]
    return []


def publish(result: Any) -> Any:
    """Copy the closure names of every JIT kernel reachable from ``result`` into that kernel's module globals.
    A module global of the same name bound to a different object is left alone (the kernel then sees it, as before)."""
    for kernel in _kernels_in(result, set()):
        fn = kernel.fn
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
            try:
                fn.__globals__.setdefault(name, cell.cell_contents)
            except ValueError:  # an empty cell: the factory never bound it
                continue
    return result


def publishes_closure_globals(factory: Callable) -> Callable:
    """Decorator for a kernel factory: its kernels resolve their closure names on Triton 3.2 / 3.3 too."""

    @functools.wraps(factory)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return publish(factory(*args, **kwargs))

    return wrapper
