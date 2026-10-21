# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backport of torch 2.14's polynomial-gcd proof in ``SizeVarAllocator.statically_known_multiple_of``.

torch 2.12 / 2.13 cannot prove ``(k*a - k*b) % (a - b) == 0`` and raise inductor ``CantSplit`` under
``torch.compile(dynamic=True)`` (Qwen-Image-2.1 torchao fp8 / int8, FLUX.1 single-stream blocks). Probe-gated, not
version-gated; only turns False into True when the gcd equals the denominator up to sign.
Kill switch: ``UNSLOTH_DIFFUSION_INDUCTOR_BACKPORTS=0``."""

from __future__ import annotations

import os
import threading
from typing import Any

BACKPORTS_ENV = "UNSLOTH_DIFFUSION_INDUCTOR_BACKPORTS"

# Mirrors torch 2.14 ``_MAX_SYMBOLS_FOR_EXPENSIVE_SYMPY_OPS`` / ``_MAX_ADD_TERMS_FOR_POLY_GCD``.
_MAX_SYMBOLS = 20
_MAX_ADD_TERMS = 20

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def backports_disabled() -> bool:
    return (os.environ.get(BACKPORTS_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _gcd_proves_multiple(allocator: Any, numerator: Any, denominator: Any) -> bool:
    import sympy

    if numerator == 0:
        return True
    try:
        # sympy gcd is over the rationals: gcd((a + b) / 2, a + b) == a + b, so require integer polynomials.
        for expr in (numerator, denominator):
            if any(not number.is_integer for number in expr.atoms(sympy.Number)):
                return False
    except Exception:  # noqa: BLE001
        return False
    try:
        from torch.utils._sympy.functions import simple_floordiv_gcd
    except Exception:  # noqa: BLE001 - very old torch: only the polynomial gcd below
        simple_floordiv_gcd = None
    try:
        symbols = set(getattr(numerator, "free_symbols", ())) | set(
            getattr(denominator, "free_symbols", ())
        )
    except Exception:  # noqa: BLE001
        return False
    if len(symbols) > _MAX_SYMBOLS:
        return False

    def covers(gcd: Any) -> bool:
        if gcd == 1 or gcd == -1:
            return False
        if gcd == denominator or gcd == -denominator:
            return True
        simplify = getattr(allocator, "simplify", sympy.expand)
        return simplify(gcd - denominator) == 0 or simplify(gcd + denominator) == 0

    wide = any(
        isinstance(e, sympy.Add) and len(e.args) > _MAX_ADD_TERMS for e in (numerator, denominator)
    )
    try:
        if simple_floordiv_gcd is not None and covers(simple_floordiv_gcd(numerator, denominator)):
            return True
        if not wide and covers(sympy.gcd(numerator, denominator)):
            return True
    except (sympy.PolynomialError, TypeError, ValueError):
        return False
    return False


def _stock_proves(allocator_cls: Any) -> bool:
    """Whether this torch's own check already proves the canonical failing case (torch 2.14+)."""
    import sympy

    a, b = sympy.symbols("s0 s1", integer = True, positive = True)
    allocator = allocator_cls()
    return bool(allocator.statically_known_multiple_of(4096 * a - 4096 * b, a - b))


def install(logger: Any = None) -> bool:
    """Idempotently patch ``SizeVarAllocator.statically_known_multiple_of``. True when the backport is active."""
    if backports_disabled():
        return False
    with _LOCK:
        if "original" in _STATE:
            return True
        if _STATE.get("not_needed"):
            return False
        try:
            from torch._inductor.sizevars import SizeVarAllocator
        except Exception:  # noqa: BLE001 - no inductor, nothing to fix
            return False
        original = getattr(SizeVarAllocator, "statically_known_multiple_of", None)
        if original is None:
            return False
        try:
            if _stock_proves(SizeVarAllocator):
                _STATE["not_needed"] = True
                return False
        except Exception:  # noqa: BLE001 - a probe failure is not proof the stock check works; still patch
            pass

        def statically_known_multiple_of(self: Any, numerator: Any, denominator: Any) -> bool:
            if original(self, numerator, denominator):
                return True
            try:
                import sympy
                if isinstance(denominator, (int, sympy.Integer)):
                    return False
                return _gcd_proves_multiple(self, numerator, denominator)
            except Exception:  # noqa: BLE001 - a failed proof is the stock answer
                return False

        statically_known_multiple_of._unsloth_backport = True  # type: ignore[attr-defined]
        statically_known_multiple_of.__wrapped__ = original  # type: ignore[attr-defined]
        SizeVarAllocator.statically_known_multiple_of = statically_known_multiple_of
        _STATE["original"] = original
        _STATE["cls"] = SizeVarAllocator
    if logger is not None:
        logger.info(
            "diffusion.speed: backported torch 2.14 symbolic divisibility proof (inductor CantSplit fix)"
        )
    return True


def uninstall() -> None:
    with _LOCK:
        original = _STATE.pop("original", None)
        cls = _STATE.pop("cls", None)
        _STATE.pop("not_needed", None)
        if original is not None and cls is not None:
            cls.statically_known_multiple_of = original


def is_installed() -> bool:
    return "original" in _STATE


def proof_available() -> bool:
    """Whether inductor can prove the stream-merge divisibility: stock torch 2.14+ or this backport installed."""
    install()
    return is_installed() or bool(_STATE.get("not_needed"))
