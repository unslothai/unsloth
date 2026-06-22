# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recover `unsloth`/`unsloth_zoo` from a namespace-package shadow. Stdlib-only."""

from __future__ import annotations

import os
import sys


def ensure_real_packages(*names: str) -> None:
    """Drop sys.path entries where a bare `<name>/` dir (no __init__.py) shadows
    the installed package as a namespace, import the real packages, restore
    sys.path. No-op without a shadow. Pass dependency-first (e.g. "unsloth_zoo",
    "unsloth"); imports run dependency-last."""
    import importlib
    import importlib.util

    bad: set = set()
    shadowed: list = []
    for name in names:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError):
            spec = None
        # real package -> spec.origin is its __init__; namespace shadow -> None/"namespace"
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
        # import unsloth before unsloth_zoo: unsloth.__init__ runs GPU/bnb fixes zoo relies on
        for name in reversed(names):
            importlib.import_module(name)
    finally:
        sys.path[:] = saved
