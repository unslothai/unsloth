# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared guard against `unsloth` / `unsloth_zoo` namespace-package shadows.

Dependency-free (stdlib only) so every subprocess entry point can call it
before any heavy ML import. See `ensure_real_packages` for the full rationale.
"""

from __future__ import annotations

import os
import sys


def ensure_real_packages(*names: str) -> None:
    """Stop `import <name>` from binding to a namespace-package shadow.

    A directory named like the package but missing __init__.py on sys.path (a
    stray checkout, a partial clone, or a polluted PYTHONPATH) makes the path
    finder return a namespace package, so `from unsloth import FastLanguageModel`
    dies with "cannot import name ... (unknown location)". A normal
    site-packages install always wins, so only source/editable installs are
    exposed. Drop the offending entries, import the real packages, then restore
    sys.path so other modules on those entries keep importing.

    Pass dependency-first (e.g. ``"unsloth_zoo", "unsloth"``); the real imports
    are then performed dependency-last so each package's __init__ runs in order.
    """
    import importlib
    import importlib.util

    bad: set = set()
    shadowed: list = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        # a real package exposes its __init__ via spec.origin; a namespace
        # shadow has origin None/"namespace" and only search locations
        if spec is None or spec.origin not in (None, "namespace"):
            continue
        dirs = {os.path.realpath(d) for d in (spec.submodule_search_locations or [])}
        if not dirs:
            continue
        shadowed.append(name)
        for entry in sys.path:
            pkg = os.path.join(entry or os.getcwd(), name)
            if os.path.realpath(pkg) in dirs and not os.path.isfile(
                os.path.join(pkg, "__init__.py")
            ):
                bad.add(entry)
    if not bad:
        return
    saved = list(sys.path)
    sys.path[:] = [e for e in sys.path if e not in bad]
    for name in shadowed:
        for cached in [m for m in list(sys.modules) if m == name or m.startswith(name + ".")]:
            del sys.modules[cached]
    try:
        importlib.invalidate_caches()
        # Import unsloth before unsloth_zoo (names are dependency-first):
        # unsloth.__init__ runs ROCm/Windows bnb fixes before it imports zoo,
        # so importing zoo first here would skip them. Repeat import is a no-op.
        for name in reversed(names):
            importlib.import_module(name)
    finally:
        sys.path[:] = saved
