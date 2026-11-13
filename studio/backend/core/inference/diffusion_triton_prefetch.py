# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Triton sm_8x prefetch fix for 8-bit tl.dot loops (triton-lang/triton#12486).

From Triton 3.8 the sm_8x prefetch pass carries an 8-bit dot's next K slice across the loop one
byte per register, so int8 / fp8 tl.dot loops spill (255 registers, ~550 B of stack) and run up
to ~7x slower than on 3.7.1. The pass is skipped only for TTIR modules that hold an 8-bit tt.dot
on sm_8x; every other kernel, card and Triton version compiles exactly as before.

Triton's cache key does not see the skip. That is harmless: the skip only moves when the next
K slice is read from shared memory, the MMA sequence is unchanged and the results are the same,
so a cached cubin from a process without the fix (or the reverse) differs in speed, never in output.
"""

from __future__ import annotations

import functools
import inspect
import os
import re
import threading

_TRITON_PREFETCH_ENV = "UNSLOTH_TRITON_8BIT_PREFETCH_FIX"
_TRITON_PREFETCH_SENTINEL = "__unsloth_8bit_prefetch_fix__"
_TRITON_PREFETCH_VERSIONS = ((3, 8), (3, 9))  # checked to still carry the bug
_TRITON_PREFETCH_PIPELINE = re.compile(
    r"if\s+capability\s*//\s*10\s*==\s*8\s*:\s*\n\s*passes\.ttgpuir\.add_prefetch\(pm\)"
)
_TRITON_8BIT_DOT = re.compile(r"\btt\.dot\b[^\n]*?:\s*tensor<[0-9x]*x(?:i8|f8E[0-9A-Za-z]+)>")


def _triton_8bit_prefetch_fix_mode():
    return (os.environ.get(_TRITON_PREFETCH_ENV) or "1").strip().lower()


def _triton_ttir_has_8bit_dot(mod):
    try:
        text = mod.str_nodebug() if hasattr(mod, "str_nodebug") else str(mod)
    except Exception:
        return False
    return "tt.dot" in text and _TRITON_8BIT_DOT.search(text) is not None


def fix_triton_sm8x_8bit_prefetch():
    """Returns True when the fix is installed (now or before), False when it does not apply."""
    if _triton_8bit_prefetch_fix_mode() in ("0", "off", "false", "no"):
        return False
    try:
        import triton
        from triton._C.libtriton import passes
        from triton.backends.nvidia import compiler as nvidia_compiler
    except Exception:
        return False
    backend = getattr(nvidia_compiler, "CUDABackend", None)
    ttgpuir = getattr(passes, "ttgpuir", None)
    stage = None if backend is None else backend.__dict__.get("make_ttgir")
    make_ttgir = getattr(stage, "__func__", stage)
    original_prefetch = getattr(ttgpuir, "add_prefetch", None)
    if not callable(make_ttgir) or not callable(original_prefetch):
        return False
    if getattr(make_ttgir, _TRITON_PREFETCH_SENTINEL, False):
        return True
    match = re.match(r"(\d+)\.(\d+)", str(getattr(triton, "__version__", "")))
    version = (int(match.group(1)), int(match.group(2))) if match else None
    if version not in _TRITON_PREFETCH_VERSIONS and _triton_8bit_prefetch_fix_mode() != "force":
        return False
    try:
        source = inspect.getsource(make_ttgir)
    except Exception:
        return False
    if _TRITON_PREFETCH_PIPELINE.search(source) is None:
        return False

    # The skip flag lives on the installed pass wrapper, so whichever copy of this fix runs
    # second drives the same flag the first one installed.
    state = getattr(original_prefetch, _TRITON_PREFETCH_SENTINEL, None)
    if not isinstance(state, threading.local):
        state = threading.local()

        def add_prefetch(*args, **kwargs):
            if getattr(state, "active", False):
                return None
            return original_prefetch(*args, **kwargs)

        setattr(add_prefetch, _TRITON_PREFETCH_SENTINEL, state)
        add_prefetch.__wrapped__ = original_prefetch
        ttgpuir.add_prefetch = add_prefetch

    @functools.wraps(make_ttgir)
    def patched_make_ttgir(*args, **kwargs):
        mod = args[0] if args else kwargs.get("mod")
        capability = args[3] if len(args) > 3 else kwargs.get("capability")
        skip = (
            isinstance(capability, int)
            and capability // 10 == 8
            and _triton_8bit_prefetch_fix_mode() not in ("0", "off", "false", "no")
            and _triton_ttir_has_8bit_dot(mod)
        )
        if not skip:
            return make_ttgir(*args, **kwargs)
        previous = getattr(state, "active", False)
        state.active = True
        try:
            return make_ttgir(*args, **kwargs)
        finally:
            state.active = previous

    setattr(patched_make_ttgir, _TRITON_PREFETCH_SENTINEL, True)
    backend.make_ttgir = staticmethod(patched_make_ttgir)
    return True
