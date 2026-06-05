# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Compatibility shim for Anaconda/conda-forge Python builds.

Anaconda puts distributor metadata between pipes in sys.version, e.g.
'3.12.4 | packaged by Anaconda, Inc. | (main, ...) [MSC ...]'. The hardcoded
regex in platform._sys_version() can't parse this and raises ValueError;
CPython closed it as "not planned" (cpython#102396).

We seed platform._sys_version_cache so the stdlib parser never sees the bad
string, fixing the import chain:
    structlog -> rich.pretty -> attrs._compat -> platform.python_implementation()

Import before any library that may trigger that chain. Idempotent (no-op if the
cache is already seeded or there are no pipes).
"""

import platform
import re
import sys


def _seed_sys_version_cache() -> None:
    """One-shot cache prime: parse a cleaned sys.version and seed the cache."""
    raw = sys.version

    # Strip paired |...| segments (Anaconda, conda-forge metadata)
    cleaned = re.sub(r"\s*\|[^|]*\|\s*", " ", raw).strip()

    # Format B: "ver (build) | label | (build_dup) \n[compiler]" leaves two
    # consecutive (...) groups after pipe-strip; drop the second.
    cleaned = re.sub(r"(\([^)]*\))\s+\([^)]*\)", r"\1", cleaned)

    if "|" in cleaned:
        # Unpaired pipe left: keep version + everything from "(" onward
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
        return  # Still unparsable; don't make things worse

    # Seed the cache so future calls with the raw string skip parsing
    cache = getattr(platform, "_sys_version_cache", None)
    if isinstance(cache, dict):
        cache[raw] = result


if "|" in sys.version:
    _seed_sys_version_cache()
