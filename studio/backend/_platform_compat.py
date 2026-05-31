# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Compatibility shim for Anaconda/conda-forge Python builds.

Anaconda modifies sys.version to include distributor metadata between pipe
characters, e.g. '3.12.4 | packaged by Anaconda, Inc. | (main, ...) [MSC ...]'.
Python's platform._sys_version() has a hardcoded regex that cannot parse this,
raising ValueError. CPython closed this as "not planned" (cpython#102396).

This module seeds platform._sys_version_cache so the stdlib parser never sees
the problematic string, fixing the import chain:
    structlog -> rich.pretty -> attrs._compat -> platform.python_implementation()

Import this module before any library imports that may trigger the above chain.
Safe to import multiple times (no-op if cache is already seeded or no pipes).
"""

import platform
import re
import sys


def _seed_sys_version_cache() -> None:
    """One-shot cache prime: parse a cleaned sys.version and seed the cache."""
    raw = sys.version

    # Strip paired |...| segments (Anaconda, conda-forge metadata)
    cleaned = re.sub(r"\s*\|[^|]*\|\s*", " ", raw).strip()

    # Format B: "ver (build) | label | (build_dup) \n[compiler]"
    # After pipe-strip, two consecutive (...) groups remain; drop the second.
    cleaned = re.sub(r"(\([^)]*\))\s+\([^)]*\)", r"\1", cleaned)

    if "|" in cleaned:
        # Unpaired pipe remaining -- keep version + everything from "(" onward
        m = re.match(r"([\w.+]+)\s*", cleaned)
        p = cleaned.find("(")
        if m and p > 0:
            cleaned = m.group(0) + cleaned[p:]

    if cleaned == raw:
        return  # Nothing to fix

    # Parse the cleaned string through the real stdlib parser
    try:
        result = platform._sys_version(cleaned)
    except ValueError:
        return  # Cleaning didn't produce a parseable string; don't make things worse

    # Seed the cache so future calls with the raw string skip parsing entirely
    cache = getattr(platform, "_sys_version_cache", None)
    if isinstance(cache, dict):
        cache[raw] = result


if "|" in sys.version:
    _seed_sys_version_cache()
