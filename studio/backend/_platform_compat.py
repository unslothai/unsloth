# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Compatibility shim for Anaconda/conda-forge Python builds.

Anaconda puts distributor metadata between pipes in sys.version, e.g.
'3.12.4 | packaged by Anaconda, Inc. | (main, ...) [MSC ...]'. The regex in
platform._sys_version() can't parse this and raises ValueError (cpython#102396,
closed as "not planned").

We seed platform._sys_version_cache so the stdlib parser never sees the bad
string, fixing the import chain:
    structlog -> rich.pretty -> attrs._compat -> platform.python_implementation()

Import before any library that may trigger that chain. Idempotent.
"""

import platform
import re
import sys


def _seed_sys_version_cache() -> None:
    """Parse a cleaned sys.version and seed the cache once."""
    raw = sys.version

    # Strip paired |...| segments (Anaconda, conda-forge metadata)
    cleaned = re.sub(r"\s*\|[^|]*\|\s*", " ", raw).strip()

    # Pipe-strip can leave two consecutive (...) groups; drop the second.
    cleaned = re.sub(r"(\([^)]*\))\s+\([^)]*\)", r"\1", cleaned)

    if "|" in cleaned:
        # Unpaired pipe left: keep version + everything from "(" onward
        m = re.match(r"([\w.+]+)\s*", cleaned)
        p = cleaned.find("(")
        if m and p > 0:
            cleaned = m.group(0) + cleaned[p:]

    if cleaned == raw:
        return  # Nothing to fix

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
